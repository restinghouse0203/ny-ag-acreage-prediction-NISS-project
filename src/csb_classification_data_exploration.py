import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.wkb import loads
from statsmodels.tsa.stattools import acf
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers shared with processing script
# ---------------------------------------------------------------------------

def classify_strategy_type(rotation_type):
    if rotation_type.startswith("Cont."):
        return "Continuous"
    elif rotation_type.startswith("Rot"):
        return "Rotation"
    else:
        return "Complex/Mixed"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed_data():
    """Load outputs saved by csb_classification_data_processing.py."""
    print("Loading processed data...")

    rot_path = os.path.join(OUTPUT_DIR, "rotation_summary.csv")
    ts_path  = os.path.join(OUTPUT_DIR, "time_series_data.csv")

    if not os.path.exists(rot_path) or not os.path.exists(ts_path):
        raise FileNotFoundError(
            "Processed data not found. Run csb_classification_data_processing.py first."
        )

    rotation_summary = pd.read_csv(rot_path)
    time_series_data = pd.read_csv(ts_path)

    combined_dfs = []
    i = 0
    while True:
        p = os.path.join(OUTPUT_DIR, f"processed_csb_{i}.parquet")
        if not os.path.exists(p):
            break
        combined_dfs.append(pd.read_parquet(p))
        i += 1

    print(f"  Loaded rotation_summary ({len(rotation_summary):,} rows)")
    print(f"  Loaded time_series_data ({len(time_series_data):,} rows)")
    print(f"  Loaded {len(combined_dfs)} processed CSB file(s)")
    return rotation_summary, time_series_data, combined_dfs


# ---------------------------------------------------------------------------
# Rotation Strategy Analysis — 5 plots
# ---------------------------------------------------------------------------

def plot_rotation_by_window(rotation_summary):
    """Plot 1: Top 15 rotation strategies by time window (grouped bar)."""
    print("Generating Plot 1: Top Rotation Strategies by Window...")
    top_n = 15
    top_types = (
        rotation_summary.groupby("Rotation_Type")["CSBACRES"]
        .sum()
        .nlargest(top_n)
        .index
    )
    plot_data = rotation_summary[rotation_summary["Rotation_Type"].isin(top_types)]

    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_data, x="Rotation_Type", y="CSBACRES", hue="Window")
    plt.title(f"Top {top_n} Rotation Strategies by Acreage (By Window)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rotation_histogram_by_window.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()


def plot_total_rotation_strategies(rotation_summary):
    """Plot 2: Total rotation strategies (2008-2024 combined acreage)."""
    print("Generating Plot 2: Total Rotation Strategies (2008-2024)...")
    top_n = 15
    total_rot = rotation_summary.groupby("Rotation_Type")["CSBACRES"].sum().reset_index()
    top_total = total_rot.nlargest(top_n, "CSBACRES")

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_total, x="Rotation_Type", y="CSBACRES", color="tab:blue")
    plt.title(f"Top {top_n} Rotation Strategies by Total Acreage (2008-2024)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "total_rotation_strategies_2008_2024.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()


def plot_strategy_categories(rotation_summary):
    """Plot 3: High-level strategy categories (Continuous / Rotation / Complex)."""
    print("Generating Plot 3: Strategy Categories (2008-2024)...")
    rot = rotation_summary.copy()
    if "Strategy_Category" not in rot.columns:
        rot["Strategy_Category"] = rot["Rotation_Type"].apply(classify_strategy_type)
    cat_summary = rot.groupby("Strategy_Category")["CSBACRES"].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=cat_summary, x="Strategy_Category", y="CSBACRES", palette="viridis")
    plt.title("Total Acreage by Strategy Category (2008-2024)")
    plt.ylabel("Total Acres")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "strategy_categories_2008_2024.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()


def plot_early_window_focus(rotation_summary):
    """Plot 4: Top strategies for the earliest time window only."""
    print("Generating Plot 4: Top Rotation Strategies (earliest window)...")
    windows = sorted(rotation_summary["Window"].unique())
    early = windows[0]
    top_n = 15
    df_early = rotation_summary[rotation_summary["Window"] == early]
    top_early = df_early.nlargest(top_n, "CSBACRES")

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_early, x="Rotation_Type", y="CSBACRES", color="tab:orange")
    plt.title(f"Top {top_n} Rotation Strategies ({early})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    label_safe = early.replace("-", "_").replace(" ", "_")
    out = os.path.join(OUTPUT_DIR, f"rotation_strategies_{label_safe}.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Spatial Analysis — 2 maps
# ---------------------------------------------------------------------------

def plot_spatial_map(df, window_label):
    """
    Map the top-5 rotation strategies on a black-background NY map.
    Interprets dominant strategies for Western, Northern, and Central/Eastern NY.
    Samples to 50,000 polygons if dataset is larger.
    """
    print(f"Generating Spatial Map for {window_label}...")

    if "geometry" not in df.columns:
        print(f"  No geometry column in data for {window_label} — skipping map.")
        return

    # Restore geometry from WKB if needed
    sample_geom = df["geometry"].iloc[0]
    if isinstance(sample_geom, bytes):
        df = df.copy()
        df["geometry"] = df["geometry"].apply(lambda x: loads(x) if x else None)
    df = gpd.GeoDataFrame(df, geometry="geometry")

    # Keep top-5 rotation types
    top_5_rot = df["Rotation_Type"].value_counts().nlargest(5).index
    map_df = df[df["Rotation_Type"].isin(top_5_rot)].copy()

    if len(map_df) > 50_000:
        print(f"  Sampling {len(map_df):,} -> 50,000 for visualization")
        map_df = map_df.sample(50_000, random_state=42)

    # CRS handling
    if map_df.crs is None:
        min_x = map_df.geometry.bounds["minx"].min()
        epsg = 5070 if (min_x < -180 or min_x > 180) else 4326
        map_df.set_crs(epsg=epsg, inplace=True)
    if map_df.crs.to_epsg() != 4326:
        map_df = map_df.to_crs(epsg=4326)

    # Compute centroids for regional analysis
    map_df = map_df.copy()
    map_df["lon"] = map_df.geometry.centroid.x
    map_df["lat"] = map_df.geometry.centroid.y

    west_mask    = map_df["lon"] < -76.5
    north_mask   = (map_df["lat"] > 43.0) & (~west_mask)
    central_mask = (~west_mask) & (~north_mask)

    regions = {
        "Western NY":       map_df[west_mask],
        "Northern NY":      map_df[north_mask],
        "Central/Eastern NY": map_df[central_mask],
    }

    # Build interpretation text
    interp_text = f"Dominant Strategies ({window_label}):\n"
    for name, region_df in regions.items():
        if not region_df.empty:
            top_rot = region_df["Rotation_Type"].mode()
            if not top_rot.empty:
                tr    = top_rot.iloc[0]
                count = region_df["Rotation_Type"].value_counts().iloc[0]
                pct   = count / len(region_df) * 100
                interp_text += f"{name}: {tr} ({pct:.0f}%)\n"
            else:
                interp_text += f"{name}: Mixed\n"
        else:
            interp_text += f"{name}: N/A\n"

    # Black-background plot
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    map_df.plot(
        column="Rotation_Type",
        legend=True,
        cmap="Set1",
        markersize=1000,
        ax=ax,
        legend_kwds={"labelcolor": "white"},
    )
    ax.grid(True, which="both", color="white", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="white", linestyle=":", linewidth=0.3, alpha=0.2)

    props = dict(boxstyle="round", facecolor="black", alpha=0.7, edgecolor="white")
    ax.text(
        0.02, 0.02, interp_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="bottom", color="white", bbox=props,
    )
    plt.title(
        f"Spatial Distribution of Top 5 Rotation Strategies ({window_label} Sample)",
        color="white", fontsize=16,
    )
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("white")

    label_safe = window_label.replace("-", "_").replace(" ", "_")
    out = os.path.join(OUTPUT_DIR, f"rotation_map_{label_safe}.png")
    plt.savefig(out, facecolor="black", dpi=150)
    print(f"  Saved {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Time Series Analysis
# ---------------------------------------------------------------------------

def plot_crop_timeseries(ts_df):
    """
    Line plot of crop acreage trends for top-3 individual crops
    plus combined Hay/Grass. Returns (ts_plot_data, top_3_crops).
    """
    print("Generating Time Series: Crop Acreage Trends...")

    hay_grass_df = ts_df[ts_df["Crop_Name"].isin(["Other Hay", "Grass/Pasture"])].copy()
    combined_hay = hay_grass_df.groupby("Year")["CSBACRES"].sum().reset_index()
    combined_hay["Crop_Name"] = "Combined Hay/Grass"

    individual = ts_df[~ts_df["Crop_Name"].isin(["Other Hay", "Grass/Pasture"])]
    top_3_crops = individual.groupby("Crop_Name")["CSBACRES"].sum().nlargest(3).index

    ts_plot_data = ts_df[ts_df["Crop_Name"].isin(top_3_crops)]
    ts_plot_data = pd.concat([ts_plot_data, combined_hay], ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=ts_plot_data, x="Year", y="CSBACRES", hue="Crop_Name", marker="o")
    plt.title("Time Series of Crop Acreage (Top 3 Crops + Combined Hay/Grass)")
    plt.ylabel("Total Acres")
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "crop_acreage_timeseries_combined.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()

    return ts_plot_data, top_3_crops


# ---------------------------------------------------------------------------
# Auto-correlation Analysis
# ---------------------------------------------------------------------------

def plot_acf_analysis(ts_df, top_3_crops):
    """
    Compute and plot ACF (up to lag 5) for the top-3 crops and Combined Hay/Grass.
    Prints lag values and lag-1 summary to console.
    """
    print("\nGenerating ACF Analysis...")

    acf_data = {}
    for crop in top_3_crops:
        crop_ts = (
            ts_df[ts_df["Crop_Name"] == crop]
            .groupby("Year")["CSBACRES"]
            .sum()
            .sort_index()
        )
        acf_data[crop] = crop_ts

    hay_ts = (
        ts_df[ts_df["Crop_Name"].isin(["Other Hay", "Grass/Pasture"])]
        .groupby("Year")["CSBACRES"]
        .sum()
        .sort_index()
    )
    acf_data["Combined Hay/Grass"] = hay_ts

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    print("\n--- Auto-correlation Function (ACF) up to Lag 5 ---")
    for i, (crop_name, crop_ts) in enumerate(acf_data.items()):
        acf_values = acf(crop_ts, nlags=5, fft=False)

        print(f"\n{crop_name}:")
        for lag in range(6):
            print(f"  Lag {lag}: {acf_values[lag]:.4f}")

        ax   = axes[i]
        lags = range(len(acf_values))
        ax.bar(lags, acf_values, alpha=0.7, color=f"C{i}")
        ax.axhline(y=0,    color="black", linestyle="-",  alpha=0.3)
        ax.axhline(y=0.2,  color="red",   linestyle="--", alpha=0.5, label="±0.2")
        ax.axhline(y=-0.2, color="red",   linestyle="--", alpha=0.5)
        ax.set_title(f"ACF: {crop_name}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(lags)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "acf_plots_4_variables.png")
    plt.savefig(out, dpi=150)
    print(f"\n  Saved {out}")
    plt.show()

    print("\n--- Auto-correlation (Lag 1) Summary ---")
    for crop_name, crop_ts in acf_data.items():
        print(f"  {crop_name}: {crop_ts.autocorr(lag=1):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rotation_summary, time_series_data, combined_dfs = load_processed_data()

    print("\n=== Starting Visualization Generation ===\n")

    # Rotation Strategy Analysis (4 bar charts)
    plot_rotation_by_window(rotation_summary)
    plot_total_rotation_strategies(rotation_summary)
    plot_strategy_categories(rotation_summary)
    plot_early_window_focus(rotation_summary)

    # Spatial Analysis (one map per loaded dataset)
    for df in combined_dfs:
        window_label = df["Window"].iloc[0] if "Window" in df.columns else "2008-2024"
        plot_spatial_map(df.copy(), window_label)

    # Time Series Analysis
    ts_plot_data, top_3_crops = plot_crop_timeseries(time_series_data)

    # Auto-correlation Analysis
    plot_acf_analysis(time_series_data, top_3_crops)

    print("\n=== Visualization Complete ===")
    print(f"All outputs saved to {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. rotation_histogram_by_window.png")
    print("  2. total_rotation_strategies_2008_2024.png")
    print("  3. strategy_categories_2008_2024.png")
    print("  4. rotation_strategies_<window>.png")
    print("  5. rotation_map_<window>.png (spatial distribution)")
    print("  6. crop_acreage_timeseries_combined.png")
    print("  7. acf_plots_4_variables.png")


if __name__ == "__main__":
    main()
