"""
rot_strategy_feature_engineering.py
====================================
Section 3 of Rot_strategy_classification.ipynb.

Builds a polygon-level feature table (one row per polygon) for rotation
strategy prediction.  Key differences from feature_engineering.py:

  - Target is the polygon-level Strategy_Category label, not a
    point-in-time crop class.
  - Lag features extend to Lag-5 to capture full rotation cycles.
  - Weather features are time-averaged across 2008-2024 (mean + std)
    rather than year-specific.
  - County aggregations include County_Dominant_Strategy.
  - Final table is one row per polygon (not one row per polygon-year).

Inputs
------
  output/rot_strategy_labeled.parquet  — from rot_strategy_processing.py
  ny_weather_combined.feather          — monthly county weather

Output
------
  output/rot_strategy_features.parquet — model-ready, one row per polygon
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from config import OUTPUT_DIR, WEATHER_FEATHER

WEATHER_PATH = WEATHER_FEATHER
INPUT_PARQUET = OUTPUT_DIR / "rot_strategy_labeled.parquet"
OUTPUT_PARQUET = OUTPUT_DIR / "rot_strategy_features.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CDL_MAPPING_5 = {
    1: "Corn", 5: "Soybeans", 36: "Alfalfa",
    37: "Hay/Grass", 176: "Hay/Grass",
}


def map_crop_5(code):
    """Map a CDL code to one of 5 agronomic categories."""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "Other"
    return CDL_MAPPING_5.get(c, "Other")


# ---------------------------------------------------------------------------
# 1. Load labeled parquet and build long-format for lag computation
# ---------------------------------------------------------------------------

def load_labeled():
    """Load rot_strategy_labeled.parquet produced by rot_strategy_processing.py."""
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(
            f"{INPUT_PARQUET} not found. Run rot_strategy_processing.py first."
        )
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded labeled parquet: {df.shape}")
    return df


def build_long_format(wide_df):
    """
    Melt the wide CDL table (one row per polygon × one column per year)
    into long format (one row per polygon × year) for lag computation.
    """
    cdl_cols = sorted([c for c in wide_df.columns if c.startswith("CDL")])
    base_cols = ["CSBID", "CNTYFIPS", "CSBACRES", "INSIDE_X", "INSIDE_Y",
                 "Rotation_Type", "Strategy_Category"]
    base_cols = [c for c in base_cols if c in wide_df.columns]

    df_long = wide_df[base_cols + cdl_cols].melt(
        id_vars=base_cols,
        value_vars=cdl_cols,
        var_name="Year_Col",
        value_name="Crop_Code",
    )
    df_long["Year"] = df_long["Year_Col"].str.replace("CDL", "").astype(int)
    df_long.drop("Year_Col", axis=1, inplace=True)
    df_long = df_long.dropna(subset=["Crop_Code"])
    df_long.sort_values(["CSBID", "Year"], inplace=True)
    return df_long


# ---------------------------------------------------------------------------
# 2. Compute lag-based rotation pattern features
# ---------------------------------------------------------------------------

def compute_lag_features(df_long):
    """
    Compute Crop_Lag1 to Crop_Lag5 and derived rotation pattern features,
    then collapse back to one row per polygon using the most recent year
    that has all 5 lags available.
    """
    print("Computing lag features (Lag1-Lag5) per polygon...")
    df = df_long.copy()

    for lag in range(1, 6):
        df[f"Crop_Lag{lag}"] = df.groupby("CSBID")["Crop_Code"].shift(lag)

    # Crop type versions of lags (5-class)
    for lag in range(1, 6):
        df[f"Crop_Type_Lag{lag}"] = df[f"Crop_Lag{lag}"].apply(map_crop_5)

    # Drop rows without a full 5-year history
    lag_cols = [f"Crop_Lag{i}" for i in range(1, 6)]
    df_full = df.dropna(subset=lag_cols)

    # Rotation pattern features (computed per row, then collapsed to polygon)
    df_full = df_full.copy()

    # Number of unique crop types in last 5 years
    df_full["Crop_Diversity_L5"] = df_full[lag_cols].apply(
        lambda row: len(set(row.dropna().astype(int))), axis=1
    )

    # Did the crop change between last 2 years?
    df_full["Crop_Changed_L1_L2"] = (df_full["Crop_Lag1"] != df_full["Crop_Lag2"]).astype(int)

    # Classic alternating pattern: L1 == L3 but L1 != L2
    df_full["Is_Alternating_L1_L3"] = (
        (df_full["Crop_Lag1"] != df_full["Crop_Lag2"]) &
        (df_full["Crop_Lag1"] == df_full["Crop_Lag3"])
    ).astype(int)

    # Continuity streak: how many consecutive years the last crop was repeated
    def _streak(row):
        base = row["Crop_Lag1"]
        for i in range(2, 6):
            if row.get(f"Crop_Lag{i}") != base:
                return i - 1
        return 5

    df_full["Continuity_Streak"] = df_full.apply(_streak, axis=1)

    # Take the most recent observation per polygon (all lags populated)
    latest = df_full.groupby("CSBID").last().reset_index()

    lag_feature_cols = (lag_cols + [f"Crop_Type_Lag{i}" for i in range(1, 6)] +
                        ["Crop_Diversity_L5", "Crop_Changed_L1_L2",
                         "Is_Alternating_L1_L3", "Continuity_Streak"])
    keep = ["CSBID"] + [c for c in lag_feature_cols if c in latest.columns]
    print(f"  Lag features computed for {len(latest):,} polygons")
    return latest[keep]


# ---------------------------------------------------------------------------
# 3. Time-averaged weather features
# ---------------------------------------------------------------------------

def load_averaged_weather():
    """
    Load county-level monthly weather and aggregate to a single row per
    county representing the 2008-2024 climate normal (mean and std).

    Features produced per county:
        Mean_Planting_Precip, Std_Planting_Precip
        Mean_Growing_GDD,     Std_Growing_GDD
    """
    print("Loading and averaging weather data (2008-2024 climate normals)...")
    if not os.path.exists(WEATHER_PATH):
        print(f"  Warning: {WEATHER_PATH} not found. Weather features will be absent.")
        return None

    df = pd.read_feather(WEATHER_PATH)
    df["CNTYFIPS"]   = df["county"].str.replace("USNY", "").astype(int) % 1000
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["Year"]       = df["start_date"].dt.year
    df["Month"]      = df["start_date"].dt.month

    # Planting precip: April-May
    plant = (df[df["Month"].isin([4, 5])]
             .groupby(["Year", "CNTYFIPS"])["totr"].sum().reset_index()
             .rename(columns={"totr": "Planting_Precip"}))

    # Growing GDD: May-October
    grow = (df[df["Month"].isin([5, 6, 7, 8, 9, 10])]
            .groupby(["Year", "CNTYFIPS"])["gdd_b10"].sum().reset_index()
            .rename(columns={"gdd_b10": "Growing_GDD"}))

    annual = pd.merge(plant, grow, on=["Year", "CNTYFIPS"], how="outer")

    # Collapse to one row per county (time-average)
    climate = annual.groupby("CNTYFIPS").agg(
        Mean_Planting_Precip=("Planting_Precip", "mean"),
        Std_Planting_Precip=("Planting_Precip", "std"),
        Mean_Growing_GDD=("Growing_GDD", "mean"),
        Std_Growing_GDD=("Growing_GDD", "std"),
    ).reset_index()

    print(f"  Climate normals computed for {len(climate)} counties.")
    return climate


# ---------------------------------------------------------------------------
# 4. County-level strategy aggregations
# ---------------------------------------------------------------------------

def compute_county_features(labeled_df):
    """
    Compute county-level aggregations from the labeled polygon dataset.

    Features:
        County_Dominant_Strategy — most common Strategy_Category per county
        County_Crop_Diversity    — mean number of unique crop types per polygon
        County_Avg_Field_Size    — mean CSBACRES per county
    """
    print("Computing county-level features...")

    def _mode(s):
        m = s.mode()
        return m.iloc[0] if len(m) > 0 else "Complex/Mixed"

    labeled_df = labeled_df.copy()
    labeled_df["CNTYFIPS"] = pd.to_numeric(labeled_df["CNTYFIPS"], errors="coerce").astype("Int64")

    county_dominant = (labeled_df.groupby("CNTYFIPS")["Strategy_Category"]
                       .agg(_mode).reset_index()
                       .rename(columns={"Strategy_Category": "County_Dominant_Strategy"}))

    county_avg_size = (labeled_df.groupby("CNTYFIPS")["CSBACRES"]
                       .mean().reset_index()
                       .rename(columns={"CSBACRES": "County_Avg_Field_Size"}))

    # Crop diversity: approximate using number of CDL columns with distinct values
    cdl_cols = [c for c in labeled_df.columns if c.startswith("CDL")]
    labeled_df = labeled_df.copy()
    labeled_df["_n_unique_crops"] = labeled_df[cdl_cols].apply(
        lambda r: len(set(int(v) for v in r.dropna())), axis=1
    )
    county_div = (labeled_df.groupby("CNTYFIPS")["_n_unique_crops"]
                  .mean().reset_index()
                  .rename(columns={"_n_unique_crops": "County_Crop_Diversity"}))

    county_feats = (county_dominant
                    .merge(county_avg_size, on="CNTYFIPS", how="left")
                    .merge(county_div,      on="CNTYFIPS", how="left"))

    print(f"  County features for {len(county_feats)} counties.")
    return county_feats


# ---------------------------------------------------------------------------
# 5. Normalise coordinates
# ---------------------------------------------------------------------------

def normalise_coords(df):
    scaler = StandardScaler()
    df[["Longitude_Norm", "Latitude_Norm"]] = scaler.fit_transform(
        df[["INSIDE_X", "INSIDE_Y"]]
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Load labeled data
    # ------------------------------------------------------------------
    wide_df = load_labeled()

    # ------------------------------------------------------------------
    # Long-format → lag features (one row per polygon)
    # ------------------------------------------------------------------
    df_long  = build_long_format(wide_df)
    lag_feats = compute_lag_features(df_long)

    # ------------------------------------------------------------------
    # Merge lag features back onto labeled wide table
    # ------------------------------------------------------------------
    base_cols = ["CSBID", "CNTYFIPS", "CSBACRES", "INSIDE_X", "INSIDE_Y",
                 "Rotation_Type", "Strategy_Category"]
    base_cols = [c for c in base_cols if c in wide_df.columns]
    polygon_df = wide_df[base_cols].drop_duplicates("CSBID").copy()
    # Ensure CNTYFIPS is int64 so it matches the dtype produced by load_averaged_weather()
    polygon_df["CNTYFIPS"] = pd.to_numeric(polygon_df["CNTYFIPS"], errors="coerce").astype("Int64")
    polygon_df = polygon_df.merge(lag_feats, on="CSBID", how="left")

    # ------------------------------------------------------------------
    # Weather climate normals (one row per county)
    # ------------------------------------------------------------------
    climate = load_averaged_weather()
    if climate is not None:
        polygon_df = polygon_df.merge(climate, on="CNTYFIPS", how="left")
    else:
        for col in ["Mean_Planting_Precip", "Std_Planting_Precip",
                    "Mean_Growing_GDD", "Std_Growing_GDD"]:
            polygon_df[col] = np.nan

    # ------------------------------------------------------------------
    # County-level strategy aggregations
    # ------------------------------------------------------------------
    county_feats = compute_county_features(wide_df)
    polygon_df = polygon_df.merge(county_feats, on="CNTYFIPS", how="left")

    # ------------------------------------------------------------------
    # Normalise coordinates
    # ------------------------------------------------------------------
    polygon_df = normalise_coords(polygon_df)

    # ------------------------------------------------------------------
    # Quality summary
    # ------------------------------------------------------------------
    print("\n=== Feature Engineering Summary ===")
    print(f"  Final shape: {polygon_df.shape}")
    print(f"  Polygons: {len(polygon_df):,}")
    print(f"  Columns: {list(polygon_df.columns)}")
    print("\n  Strategy_Category distribution:")
    for cat, cnt in polygon_df["Strategy_Category"].value_counts().items():
        print(f"    {cat}: {cnt:,} ({cnt/len(polygon_df)*100:.1f}%)")

    null_counts = polygon_df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        print("\n  Columns with nulls:")
        for col, n in null_counts.items():
            print(f"    {col}: {n:,}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    polygon_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved: {OUTPUT_PARQUET}")
    print("\n=== rot_strategy_feature_engineering.py complete ===")


if __name__ == "__main__":
    main()
