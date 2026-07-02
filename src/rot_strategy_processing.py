"""
rot_strategy_processing.py
==========================
Section 2 of Rot_strategy_classification.ipynb.

Loads all three CSB feather files (2008-2024), assembles the full CDL
sequence for every polygon across all years, classifies each polygon
into a rotation strategy, and produces exploration visualizations.

Output
------
  output/rot_strategy_labeled.parquet  — one row per polygon with:
      CSBID, CNTYFIPS, CSBACRES, INSIDE_X, INSIDE_Y,
      Rotation_Type, Strategy_Category, <CDL2008 ... CDL2024>

  output/rot_strategy_top15.png
  output/rot_strategy_categories.png
  output/rot_strategy_class_balance.png
  output/rot_strategy_spatial_map.png
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.wkb import loads

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from config import CSB_SOURCES, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# All years covered across the three CSB releases
ALL_YEARS = list(range(2008, 2025))   # 2008-2024 inclusive

BASE_COLS = ["CSBID", "CNTYFIPS", "CSBACRES", "INSIDE_X", "INSIDE_Y"]

cdl_mapping = {
    1:   "Corn",       5:   "Soybeans",  36:  "Alfalfa",
    37:  "Other Hay",  176: "Grass/Pasture", 59: "Sod/Grass Seed",
    61:  "Fallow",     24:  "Winter Wheat",  121: "Developed",
    141: "Forest",     4:   "Sorghum",    21:  "Barley",
    23:  "Spring Wheat",
}

OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "rot_strategy_labeled.parquet")


def get_crop_name(code):
    return cdl_mapping.get(int(code) if not pd.isna(code) else -1, f"Other ({code})")


# ---------------------------------------------------------------------------
# Step 1 — Load CSB data and build one-row-per-polygon wide table
# ---------------------------------------------------------------------------

def load_wide_csb():
    """
    Load the three CSB feather files and merge them so every polygon
    has CDL columns for all years it appears in.  Returns a wide DataFrame
    with one row per unique CSBID and columns CDL2008 … CDL2024.
    """
    print("Loading CSB data (three-source strategy)...")

    # Start with base attributes from the first file that has all polygons
    all_frames = []

    for src in CSB_SOURCES:
        path  = src["path"]
        years = src["years"]
        if not os.path.exists(path):
            print(f"  Warning: {path} not found — skipping.")
            continue

        df = pd.read_feather(path)
        cdl_cols  = [f"CDL{y}" for y in years if f"CDL{y}" in df.columns]
        keep_cols = BASE_COLS + cdl_cols
        keep_cols = [c for c in keep_cols if c in df.columns]

        df_sub = df[keep_cols].copy()
        df_sub["CSBID"] = df_sub["CSBID"].astype(str)
        all_frames.append(df_sub)
        print(f"  {os.path.basename(path)}: {len(df_sub):,} polygons, "
              f"CDL years {min(years)}-{max(years)}")

    if not all_frames:
        raise FileNotFoundError("No CSB feather files found.")

    # Merge on CSBID — each source contributes different CDL year columns
    merged = all_frames[0]
    for frame in all_frames[1:]:
        new_cdl = [c for c in frame.columns if c.startswith("CDL") and c not in merged.columns]
        if new_cdl:
            merged = pd.merge(
                merged,
                frame[["CSBID"] + new_cdl],
                on="CSBID",
                how="outer",
            )
        # Also update BASE_COLS for polygons that appear only in later sources
        base_from_new = [c for c in BASE_COLS if c in frame.columns and c != "CSBID"]
        for col in base_from_new:
            if col not in merged.columns:
                merged = pd.merge(merged, frame[["CSBID", col]], on="CSBID", how="left")

    print(f"  Wide table shape: {merged.shape}")
    print(f"  CDL columns: {[c for c in merged.columns if c.startswith('CDL')]}")
    return merged


# ---------------------------------------------------------------------------
# Step 2 — Classify rotation strategy per polygon
# ---------------------------------------------------------------------------

def classify_row_rotation(row_values):
    """Classify the multi-year crop sequence into a rotation label."""
    valid = [v for v in row_values if not pd.isna(v)]
    if not valid:
        return "Unknown"
    unique = np.unique(valid)
    unique_names = sorted([get_crop_name(c) for c in unique])
    if len(unique) == 1:
        return f"Cont. {unique_names[0]}"
    elif len(unique) == 2:
        return f"Rot {unique_names[0]}-{unique_names[1]}"
    else:
        return "Complex/Mixed"


def classify_strategy_type(rotation_type):
    if rotation_type.startswith("Cont."):
        return "Continuous"
    elif rotation_type.startswith("Rot"):
        return "Rotation"
    else:
        return "Complex/Mixed"


def assign_strategy_labels(df):
    """Add Rotation_Type and Strategy_Category columns."""
    print("Classifying rotation strategies...")
    cdl_cols = sorted([c for c in df.columns if c.startswith("CDL")])
    crop_matrix = df[cdl_cols].values
    df = df.copy()
    df["Rotation_Type"]     = [classify_row_rotation(row) for row in crop_matrix]
    df["Strategy_Category"] = df["Rotation_Type"].apply(classify_strategy_type)

    print("  Strategy distribution:")
    for cat, cnt in df["Strategy_Category"].value_counts().items():
        print(f"    {cat}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Visualizations
# ---------------------------------------------------------------------------

def plot_top15_strategies(df):
    """Bar chart: top-15 rotation strategies by total acreage."""
    print("Plot 1: Top-15 rotation strategies by acreage...")
    totals = (
        df.groupby("Rotation_Type")["CSBACRES"]
        .sum()
        .reset_index()
        .nlargest(15, "CSBACRES")
    )
    plt.figure(figsize=(12, 7))
    sns.barplot(data=totals, x="Rotation_Type", y="CSBACRES", color="steelblue")
    plt.title("Top 15 Rotation Strategies by Total Acreage (2008-2024)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total Acres")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rot_strategy_top15.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()


def plot_category_totals(df):
    """Bar chart: 3-class category totals by acreage."""
    print("Plot 2: Strategy category totals...")
    cat_totals = df.groupby("Strategy_Category")["CSBACRES"].sum().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=cat_totals, x="Strategy_Category", y="CSBACRES", palette="viridis")
    plt.title("Total Acreage by Strategy Category (2008-2024)", fontsize=14)
    plt.ylabel("Total Acres")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rot_strategy_categories.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()


def plot_class_balance(df):
    """Pie chart: polygon count per Strategy_Category (class balance)."""
    print("Plot 3: Class balance (polygon counts)...")
    counts = df["Strategy_Category"].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#2196F3", "#4CAF50", "#FF5722"],
    )
    plt.title("Class Balance: Polygon Count per Strategy Category", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rot_strategy_class_balance.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()


def plot_spatial_map(df):
    """Spatial map: Strategy_Category per polygon, black background."""
    print("Plot 4: Spatial distribution map...")

    if "geometry" not in df.columns:
        print("  No geometry column — skipping spatial map.")
        return

    # Restore WKB geometry if needed
    sample = df["geometry"].iloc[0]
    if isinstance(sample, bytes):
        df = df.copy()
        df["geometry"] = df["geometry"].apply(lambda x: loads(x) if x else None)
    gdf = gpd.GeoDataFrame(df, geometry="geometry")

    # Sample for performance
    if len(gdf) > 50_000:
        print(f"  Sampling {len(gdf):,} -> 50,000 for visualization")
        gdf = gdf.sample(50_000, random_state=42)

    # CRS handling
    if gdf.crs is None:
        min_x = gdf.geometry.bounds["minx"].min()
        gdf.set_crs(epsg=5070 if (min_x < -180 or min_x > 180) else 4326, inplace=True)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    gdf.plot(
        column="Strategy_Category",
        legend=True,
        cmap="Set1",
        markersize=500,
        ax=ax,
        legend_kwds={"labelcolor": "white"},
    )
    ax.grid(True, color="white", linestyle="--", linewidth=0.4, alpha=0.3)
    plt.title("Spatial Distribution of Rotation Strategy Categories (NY, 2008-2024)",
              color="white", fontsize=15)
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("white")

    out = os.path.join(OUTPUT_DIR, "rot_strategy_spatial_map.png")
    plt.savefig(out, facecolor="black", dpi=150)
    print(f"  Saved {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load and classify
    wide_df  = load_wide_csb()
    labeled  = assign_strategy_labels(wide_df)

    # Save labeled parquet (drop raw geometry — not parquet-serialisable)
    save_df = labeled.drop(columns=["geometry"], errors="ignore")
    save_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved: {OUTPUT_PARQUET}  ({len(save_df):,} polygons)")

    # Visualizations
    plot_top15_strategies(labeled)
    plot_category_totals(labeled)
    plot_class_balance(labeled)
    plot_spatial_map(labeled)

    print("\n=== rot_strategy_processing.py complete ===")
    print("Outputs:")
    print(f"  {OUTPUT_PARQUET}")
    print(f"  output/rot_strategy_top15.png")
    print(f"  output/rot_strategy_categories.png")
    print(f"  output/rot_strategy_class_balance.png")
    print(f"  output/rot_strategy_spatial_map.png")


if __name__ == "__main__":
    main()
