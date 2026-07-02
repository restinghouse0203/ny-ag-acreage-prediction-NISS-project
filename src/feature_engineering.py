"""
feature_engineering.py
=======================
Data processing pipeline that produces the model-ready dataset.

Inputs
------
  CSB crop data  : merged_CSB_polygon.parquet  (2008-2024, includes 2016)
  Weather data   : ny_weather_combined.feather  (monthly county observations)
  Soil data      : soil_features_NY.csv         (gSSURGO map unit attributes)
  Soil lookup    : csbid_mukey_mapping.csv       (pre-computed CSBID -> MUKEY)

Output
------
  processed_dataset.parquet  —  one row per polygon × year, all features aligned

Run this script first; then run feature_exploration.py for visualisations.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    CSB_SOURCES,
    OUTPUT_DIR,
    PROCESSED_DATASET,
    SOIL_FEATURES,
    SOIL_MAP,
    WEATHER_FEATHER,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEATHER_PATH = WEATHER_FEATHER
SOIL_PATH = SOIL_FEATURES
SOIL_MAP_PATH = SOIL_MAP
OUTPUT_PATH = PROCESSED_DATASET

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 1. Crop Data Processing
# ===========================================================================

def load_and_process_csb():
    """
    Load the three CSB feather files and combine them into a
    polygon × year long-format DataFrame with lag features.

    Source strategy (mirrors CSB_combine_analysis.ipynb):
      - csb_20082015 : years 2008-2015
      - csb_20092016 : year  2016 only  (bridge file to fill the gap)
      - csb_20172024 : years 2017-2024

    Steps
    -----
    1. For each source, read feather, select only the requested CDL year columns
    2. Melt wide -> long (one row per CSBID × Year)
    3. Concatenate all sources and deduplicate on (CSBID, Year)
    4. Sort chronologically per polygon
    5. Create Crop_Lag1 and Crop_Lag2 (1- and 2-year history)
    6. Drop first 2 years per polygon (no complete lag history)
    """
    print("Loading CSB data (2008-2024, three-source strategy)...")
    base_cols = ["CSBID", "CNTYFIPS", "CSBACRES", "INSIDE_X", "INSIDE_Y"]
    all_long  = []

    for src in CSB_SOURCES:
        path  = src["path"]
        years = src["years"]

        if not os.path.exists(path):
            print(f"  Warning: {path} not found — skipping.")
            continue

        df = pd.read_feather(path)

        # Select only the years requested from this source
        year_cols = [f"CDL{y}" for y in years]
        available = [c for c in year_cols if c in df.columns]
        if not available:
            print(f"  Warning: no matching CDL columns in {path} — skipping.")
            continue

        missing_base = [c for c in base_cols if c not in df.columns]
        if missing_base:
            print(f"  Warning: missing base columns {missing_base} in {path} — skipping.")
            continue

        df["CNTYFIPS"] = df["CNTYFIPS"].astype(int)

        # Melt only the selected CDL columns to long format
        df_long = df[base_cols + available].melt(
            id_vars=base_cols,
            value_vars=available,
            var_name="Year_Col",
            value_name="Crop_Code",
        )
        df_long["Year"] = df_long["Year_Col"].str.replace("CDL", "").astype(int)
        df_long.drop("Year_Col", axis=1, inplace=True)

        print(f"  {path}: {len(df):,} polygons × {len(available)} years "
              f"({min(years)}-{max(years)})")
        all_long.append(df_long)

    if not all_long:
        raise ValueError("No valid CSB data loaded from any source.")

    # Combine and deduplicate (safety net for any overlapping years)
    df_combined = pd.concat(all_long, ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["CSBID", "Year"])

    print(f"  Combined shape before lags: {df_combined.shape}")
    print(f"  Year range: {df_combined['Year'].min()}-{df_combined['Year'].max()}")

    # Sort chronologically per polygon for correct lag calculation
    df_combined.sort_values(["CSBID", "Year"], inplace=True)

    print("Creating lag features...")
    df_combined["Crop_Lag1"] = df_combined.groupby("CSBID")["Crop_Code"].shift(1)
    df_combined["Crop_Lag2"] = df_combined.groupby("CSBID")["Crop_Code"].shift(2)

    # Drop rows without full lag history (first 2 years per polygon)
    df_model = df_combined.dropna(subset=["Crop_Lag1", "Crop_Lag2"])

    print(f"  Final shape after lag creation: {df_model.shape}")
    return df_model


def classify_crop_types(df):
    """
    Map raw CDL codes to 5 agronomic categories for modelling:
      Corn | Soybeans | Alfalfa | Combined Hay/Grass | Other

    Applies classification to current crop and both lag columns.
    """
    print("Classifying crop types into 5 categories...")

    def _map(code):
        if code == 1:
            return "Corn"
        elif code == 5:
            return "Soybeans"
        elif code == 36:
            return "Alfalfa"
        elif code in [37, 176]:          # Other Hay or Grass/Pasture
            return "Combined Hay/Grass"
        else:
            return "Other"

    df["Crop_Type"]      = df["Crop_Code"].apply(_map)
    df["Crop_Type_Lag1"] = df["Crop_Lag1"].apply(_map)
    df["Crop_Type_Lag2"] = df["Crop_Lag2"].apply(_map)

    print("  Crop type distribution:")
    for label, count in df["Crop_Type"].value_counts().items():
        print(f"    {label}: {count:,}")
    return df


# ===========================================================================
# 2. Weather Data Processing
# ===========================================================================

def load_and_process_weather():
    """
    Aggregate monthly county-level weather observations into annual
    growing-season features and create 1-5 year lags.

    Seasonal windows
    ----------------
    Planting_Precip : April-May  total precipitation  (totr)
    Growing_GDD     : May-October cumulative GDD base-10 (gdd_b10)

    Both raw and log1p-transformed versions are created.
    Lags 1-5 are generated for all four weather features.
    """
    print("Loading Weather data...")
    df = pd.read_feather(WEATHER_PATH)

    # Convert county identifier: USNY36001 -> 1  (3-digit FIPS to match CSB)
    df["CNTYFIPS"] = df["county"].str.replace("USNY", "").astype(int) % 1000
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["Year"]  = df["start_date"].dt.year
    df["Month"] = df["start_date"].dt.month

    print(f"  Weather year range: {df['Year'].min()}-{df['Year'].max()}")

    # Planting precipitation: April-May totals
    planting_df = (
        df[df["Month"].isin([4, 5])]
        .groupby(["Year", "CNTYFIPS"])["totr"]
        .sum()
        .reset_index()
        .rename(columns={"totr": "Planting_Precip"})
    )

    # Growing degree days: May-October cumulative
    growing_df = (
        df[df["Month"].isin([5, 6, 7, 8, 9, 10])]
        .groupby(["Year", "CNTYFIPS"])["gdd_b10"]
        .sum()
        .reset_index()
        .rename(columns={"gdd_b10": "Growing_GDD"})
    )

    weather_feat = pd.merge(planting_df, growing_df, on=["Year", "CNTYFIPS"], how="outer")

    # Log-transform to reduce positive skewness; log1p is safe for zero values
    print("  Creating log-transformed weather features...")
    weather_feat["Planting_Precip_Log"] = np.log1p(weather_feat["Planting_Precip"])
    weather_feat["Growing_GDD_Log"]     = np.log1p(weather_feat["Growing_GDD"])

    # Lags 1-5 for raw and log-transformed features
    print("  Creating lagged weather features (lags 1-5)...")
    weather_feat.sort_values(["CNTYFIPS", "Year"], inplace=True)
    for lag in range(1, 6):
        weather_feat[f"Planting_Precip_Lag{lag}"]     = weather_feat.groupby("CNTYFIPS")["Planting_Precip"].shift(lag)
        weather_feat[f"Growing_GDD_Lag{lag}"]          = weather_feat.groupby("CNTYFIPS")["Growing_GDD"].shift(lag)
        weather_feat[f"Planting_Precip_Log_Lag{lag}"] = weather_feat.groupby("CNTYFIPS")["Planting_Precip_Log"].shift(lag)
        weather_feat[f"Growing_GDD_Log_Lag{lag}"]     = weather_feat.groupby("CNTYFIPS")["Growing_GDD_Log"].shift(lag)

    return weather_feat


# ===========================================================================
# 3. Soil Data Processing
# ===========================================================================

def load_and_process_soil():
    """
    Load gSSURGO soil map unit attributes and encode categorical variables.

    Key difference from weather: soil features are TIME-INVARIANT.
    They are merged on MUKEY (spatial key) rather than Year (temporal key).
    Each CSB polygon is assigned a single MUKEY via a pre-computed spatial
    join stored in csbid_mukey_mapping.csv.

    Selected features
    -----------------
    slopegradwta   : weighted-average slope gradient (%)
    brockdepmin    : minimum bedrock depth (cm)
    wtdepannmin    : annual min water table depth (cm)
    wtdepaprjunmin : spring water table depth (cm)
    aws050wta      : available water storage 0-50 cm (cm)
    aws0100wta     : available water storage 0-100 cm (cm)
    drclassdcd     : dominant drainage class (categorical -> numeric)
    hydgrpdcd      : hydrologic group (categorical -> primary letter A/B/C/D)
    """
    print("Loading Soil data...")

    if not os.path.exists(SOIL_PATH):
        print(f"  Warning: soil file not found at {SOIL_PATH}. Skipping soil features.")
        return None

    soil_df = pd.read_csv(SOIL_PATH)

    # Rename long gSSURGO column headers to short names
    rename_map = {
        "gSSURGO_NY \u2014 muaggatt_slopegradwta":    "slopegradwta",
        "gSSURGO_NY \u2014 muaggatt_brockdepmin":     "brockdepmin",
        "gSSURGO_NY \u2014 muaggatt_wtdepannmin":     "wtdepannmin",
        "gSSURGO_NY \u2014 muaggatt_wtdepaprjunmin":  "wtdepaprjunmin",
        "gSSURGO_NY \u2014 muaggatt_aws050wta":       "aws050wta",
        "gSSURGO_NY \u2014 muaggatt_aws0100wta":      "aws0100wta",
        "gSSURGO_NY \u2014 muaggatt_drclassdcd":      "drclassdcd",
        "gSSURGO_NY \u2014 muaggatt_drclasswettest":  "drclasswettest",
        "gSSURGO_NY \u2014 muaggatt_hydgrpdcd":       "hydgrpdcd",
    }
    soil_df.rename(columns=rename_map, inplace=True)

    soil_features = [
        "MUKEY",
        "slopegradwta", "brockdepmin",
        "wtdepannmin",  "wtdepaprjunmin",
        "aws050wta",    "aws0100wta",
        "drclassdcd",   "hydgrpdcd",
    ]
    available = [c for c in soil_features if c in soil_df.columns]
    missing   = set(soil_features) - set(available)
    if missing:
        print(f"  Note: soil columns not found: {missing}")

    soil_df = soil_df[available].copy()

    # Clean MUKEY for consistent merging
    soil_df["MUKEY"] = soil_df["MUKEY"].astype(str).str.strip()

    # Convert numeric columns (may be read as ArrowDtype strings)
    numeric_cols = [
        "slopegradwta", "brockdepmin", "wtdepannmin",
        "wtdepaprjunmin", "aws050wta", "aws0100wta",
    ]
    for col in numeric_cols:
        if col in soil_df.columns:
            soil_df[col] = pd.to_numeric(soil_df[col], errors="coerce")

    # Numeric encoding of drainage class (ordinal: 1=worst to 5=best).
    # Includes all classes observed in the gSSURGO data.
    drainage_mapping = {
        "Excessively drained":           6,
        "Somewhat excessively drained":  5,
        "Well drained":                  4,
        "Moderately well drained":       3,
        "Somewhat poorly drained":       2,
        "Poorly drained":                1,
        "Very poorly drained":           0,
    }
    if "drclassdcd" in soil_df.columns:
        # Cast to plain str so .map() works regardless of ArrowDtype
        soil_df["drainage_numeric"] = (
            soil_df["drclassdcd"].astype(str).map(drainage_mapping)
        )

    # Primary hydrologic group letter (A / B / C / D).
    # Dual-group values like "B/D" yield the wettest (first) group letter.
    if "hydgrpdcd" in soil_df.columns:
        soil_df["hydro_group_primary"] = (
            soil_df["hydgrpdcd"].astype(str).str.strip().str[0]
            .replace("n", np.nan)   # "nan" -> NaN
        )

    # Aggregate duplicate MUKEYs (same map unit can appear in many rows)
    num_agg_cols = [c for c in numeric_cols + ["drainage_numeric"] if c in soil_df.columns]
    num_agg      = {c: "mean" for c in num_agg_cols}

    # For categoricals, aggregate separately to avoid lambda capture issues
    cat_cols = [c for c in ["drclassdcd", "hydgrpdcd", "hydro_group_primary"] if c in soil_df.columns]

    soil_num = soil_df.groupby("MUKEY")[num_agg_cols].mean().reset_index()

    soil_cat_parts = [soil_num]
    for col in cat_cols:
        def _mode(s):
            m = s.dropna().mode()
            return m.iloc[0] if len(m) > 0 else np.nan
        cat_agg = soil_df.groupby("MUKEY")[col].agg(_mode).reset_index()
        soil_cat_parts.append(cat_agg.set_index("MUKEY")[[col]])

    soil_agg = soil_num.copy()
    for part in soil_cat_parts[1:]:
        soil_agg = soil_agg.join(part, on="MUKEY", how="left")

    print(f"  Soil data loaded: {len(soil_agg):,} unique map units")
    print(f"  Drainage coverage: {soil_agg['drainage_numeric'].notna().mean()*100:.1f}% non-null")
    return soil_agg


# ===========================================================================
# 4. Feature Engineering — County-Level and Coordinate Features
# ===========================================================================

def add_county_level_features(df):
    """
    Compute county × year aggregations and normalise polygon coordinates.

    Features created
    ----------------
    County_Crop_Diversity  : number of distinct crop types grown per county-year
    County_Dominant_Crop   : most common crop type per county-year
    County_Avg_Field_Size  : mean CSBACRES per county-year (farm-scale proxy)
    Longitude_Norm         : standardised INSIDE_X
    Latitude_Norm          : standardised INSIDE_Y
    """
    print("Adding county-level features...")

    # Crop diversity
    county_diversity = (
        df.groupby(["CNTYFIPS", "Year"])["Crop_Type"]
        .nunique()
        .reset_index()
        .rename(columns={"Crop_Type": "County_Crop_Diversity"})
    )

    # Dominant crop type
    county_dominant = (
        df.groupby(["CNTYFIPS", "Year"])["Crop_Type"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Other")
        .reset_index()
        .rename(columns={"Crop_Type": "County_Dominant_Crop"})
    )

    # Average field size
    county_avg_size = (
        df.groupby(["CNTYFIPS", "Year"])["CSBACRES"]
        .mean()
        .reset_index()
        .rename(columns={"CSBACRES": "County_Avg_Field_Size"})
    )

    df = df.merge(county_diversity, on=["CNTYFIPS", "Year"], how="left")
    df = df.merge(county_dominant,  on=["CNTYFIPS", "Year"], how="left")
    df = df.merge(county_avg_size,  on=["CNTYFIPS", "Year"], how="left")

    # Standardise coordinates to zero mean / unit variance
    scaler = StandardScaler()
    df[["Longitude_Norm", "Latitude_Norm"]] = scaler.fit_transform(
        df[["INSIDE_X", "INSIDE_Y"]]
    )

    print("  County-level features added successfully.")
    return df


# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    # ------------------------------------------------------------------
    # Load and process each data source
    # ------------------------------------------------------------------
    csb_df     = load_and_process_csb()
    csb_df     = classify_crop_types(csb_df)
    weather_df = load_and_process_weather()
    soil_df    = load_and_process_soil()       # returns None if file missing

    # ------------------------------------------------------------------
    # Data overview
    # ------------------------------------------------------------------
    print("\n=== Data Overview ===")
    print(f"CSB years:    {sorted(csb_df['Year'].unique())}")
    print(f"Weather years:{sorted(weather_df['Year'].unique())}")
    print(f"CSB counties (sample): {sorted(csb_df['CNTYFIPS'].unique())[:5]}")

    # ------------------------------------------------------------------
    # Spatial soil join: CSBID -> MUKEY -> soil attributes  (static)
    # ------------------------------------------------------------------
    if soil_df is not None:
        if os.path.exists(SOIL_MAP_PATH):
            print("\nMerging soil features via CSBID-MUKEY lookup...")
            soil_map = pd.read_csv(SOIL_MAP_PATH)
            
            # Ensure data type compatibility for both CSBID and MUKEY columns
            soil_map["MUKEY"] = soil_map["MUKEY"].astype(str).str.strip()
            soil_map["CSBID"] = soil_map["CSBID"].astype(str).str.strip()
            csb_df["CSBID"] = csb_df["CSBID"].astype(str).str.strip()
            
            print(f"  Soil mapping file contains {len(soil_map):,} CSBID-MUKEY pairs")
            csb_df = csb_df.merge(soil_map, on="CSBID", how="left")
            csb_df = csb_df.merge(soil_df,  on="MUKEY",  how="left")
            matched = csb_df["MUKEY"].notna().mean() * 100
            print(f"  Soil match rate: {matched:.1f}% of polygons")
        else:
            print(
                f"\nWarning: CSBID-MUKEY lookup not found at {SOIL_MAP_PATH}.\n"
                "  Soil features will be absent from the final dataset.\n"
                "  Generate csbid_mukey_mapping.csv via a geopandas spatial join first."
            )
    else:
        print("\nSoil data unavailable — continuing without soil features.")

    # ------------------------------------------------------------------
    # Merge weather on [Year, CNTYFIPS]
    # ------------------------------------------------------------------
    print("\nMerging CSB and Weather data...")
    final_df = csb_df.merge(weather_df, on=["Year", "CNTYFIPS"], how="left")

    # ------------------------------------------------------------------
    # County-level features + coordinate normalisation
    # ------------------------------------------------------------------
    final_df = add_county_level_features(final_df)

    # ------------------------------------------------------------------
    # Drop rows without critical weather data
    # ------------------------------------------------------------------
    initial_shape = len(final_df)
    final_df = final_df.dropna(subset=["Planting_Precip", "Growing_GDD"])
    dropped  = initial_shape - len(final_df)
    print(f"Dropped {dropped:,} rows due to missing weather data.")
    print(f"Final dataset shape: {final_df.shape}")

    # ------------------------------------------------------------------
    # Data quality summary
    # ------------------------------------------------------------------
    print("\n=== Data Quality Summary ===")
    print(f"Year range:              {final_df['Year'].min()}-{final_df['Year'].max()}")
    print(f"Unique polygons:         {final_df['CSBID'].nunique():,}")
    print(f"Unique counties:         {final_df['CNTYFIPS'].nunique()}")
    print(f"Average records/polygon: {len(final_df) / final_df['CSBID'].nunique():.1f}")

    print("\n=== Geometric Data Check ===")
    for col in ["INSIDE_X", "INSIDE_Y", "Longitude_Norm", "Latitude_Norm"]:
        if col in final_df.columns:
            print(f"  {col}: {final_df[col].min():.2f} to {final_df[col].max():.2f}")
        else:
            print(f"  {col}: MISSING")

    print("\n=== Weather Features Check ===")
    for col in ["Planting_Precip", "Growing_GDD", "Planting_Precip_Log", "Growing_GDD_Log"]:
        if col in final_df.columns:
            print(f"  {col}: mean={final_df[col].mean():.2f}, std={final_df[col].std():.2f}")
        else:
            print(f"  {col}: MISSING")

    print("\n=== Soil Features Check ===")
    soil_numeric = ["slopegradwta", "aws050wta", "aws0100wta", "drainage_numeric"]
    for col in soil_numeric:
        if col in final_df.columns:
            non_null = final_df[col].notna().sum()
            print(f"  {col}: {non_null:,} non-null ({non_null/len(final_df)*100:.1f}%)")
        else:
            print(f"  {col}: NOT in dataset")

    # ------------------------------------------------------------------
    # Save model-ready dataset
    # ------------------------------------------------------------------
    print(f"\nSaving processed dataset to {OUTPUT_PATH}...")
    final_df.to_parquet(OUTPUT_PATH, index=False)
    print("Dataset saved successfully!")

    # ------------------------------------------------------------------
    # Log transformation summary
    # ------------------------------------------------------------------
    print("\n=== Log Transformation Summary ===")
    print(f"  Precip skewness: {final_df['Planting_Precip'].skew():.3f} -> {final_df['Planting_Precip_Log'].skew():.3f}")
    print(f"  GDD skewness:    {final_df['Growing_GDD'].skew():.3f} -> {final_df['Growing_GDD_Log'].skew():.3f}")
    print(f"  Precip std:      {final_df['Planting_Precip'].std():.2f} -> {final_df['Planting_Precip_Log'].std():.2f}")
    print(f"  GDD std:         {final_df['Growing_GDD'].std():.2f} -> {final_df['Growing_GDD_Log'].std():.2f}")

    print("\n=== Feature Engineering Complete ===")
    print("Generated features:")
    print("  1. Crop type classification (5 classes + lag 1-2)")
    print("  2. Weather: Planting_Precip, Growing_GDD (raw + log + lags 1-5)")
    print("  3. Soil: slope, drainage, AWS, hydrologic group (static per polygon)")
    print("  4. County aggregations: diversity, dominant crop, avg field size")
    print("  5. Normalised coordinates: Longitude_Norm, Latitude_Norm")


if __name__ == "__main__":
    main()
