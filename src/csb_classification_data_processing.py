import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.wkb import loads
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from config import CSB_SOURCES, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Three feather files, each supplying specific years to give a complete
# 2008-2024 time series with no gap at 2016.
datasets = [
    {**src, "label": label}
    for src, label in zip(
        CSB_SOURCES,
        ["2008-2015", "2016-Bridge", "2017-2024"],
    )
]

# ---------------------------------------------------------------------------
# Crop mapping
# ---------------------------------------------------------------------------
cdl_mapping = {
    1:   "Corn",
    5:   "Soybeans",
    36:  "Alfalfa",
    37:  "Other Hay",
    176: "Grass/Pasture",
    59:  "Sod/Grass Seed",
    61:  "Fallow",
    24:  "Winter Wheat",
    121: "Developed",
    141: "Forest",
    4:   "Sorghum",
    21:  "Barley",
    23:  "Spring Wheat",
}


def get_crop_name(code):
    return cdl_mapping.get(code, f"Other ({code})")


# ---------------------------------------------------------------------------
# Geometry handling
# ---------------------------------------------------------------------------

def process_geometry(df):
    """Convert WKB-encoded geometry column to shapely objects if needed."""
    if "geometry" not in df.columns:
        return df
    sample = df["geometry"].iloc[0]
    if isinstance(sample, bytes):
        print("  Converting WKB geometry to shapely objects...")
        df = df.copy()
        df["geometry"] = df["geometry"].apply(lambda x: loads(x) if x else None)
        df = gpd.GeoDataFrame(df, geometry="geometry")
    return df


# ---------------------------------------------------------------------------
# Rotation strategy classification
# ---------------------------------------------------------------------------

def classify_row_rotation(row_values):
    """
    Classify rotation strategy based on the crop sequence across all years:
      - "Cont. Corn"           (single unique crop)
      - "Rot Corn-Soybeans"    (exactly 2 unique crops)
      - "Complex/Mixed"        (3+ unique crops)
    """
    unique = np.unique(row_values)
    unique_names = sorted([get_crop_name(c) for c in unique])

    if len(unique) == 1:
        return f"Cont. {unique_names[0]}"
    elif len(unique) == 2:
        return f"Rot {unique_names[0]}-{unique_names[1]}"
    else:
        return "Complex/Mixed"


def classify_strategy_type(rotation_type):
    """Map a detailed rotation label to a broad category."""
    if rotation_type.startswith("Cont."):
        return "Continuous"
    elif rotation_type.startswith("Rot"):
        return "Rotation"
    else:
        return "Complex/Mixed"


# ---------------------------------------------------------------------------
# File loading and processing
# ---------------------------------------------------------------------------

def process_file(file_path, years, label):
    """
    Load a CSB feather file, keep only the requested year columns,
    handle geometry, and classify each polygon's rotation strategy.

    Parameters
    ----------
    file_path : str   – path to the feather file
    years     : list  – calendar years to extract (e.g. [2016])
    label     : str   – descriptive label for this time window

    Returns a DataFrame with Rotation_Type and Window columns added,
    plus all requested CDL columns (CDL<year>).
    """
    print(f"Processing {label} from {file_path}...")
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return None

    try:
        df = pd.read_feather(file_path)
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None

    # Build column list from requested years
    year_cols = [f"CDL{y}" for y in years]
    available_cols = [c for c in year_cols if c in df.columns]
    missing = set(year_cols) - set(available_cols)
    if missing:
        print(f"  Note: {len(missing)} requested CDL columns not found: {sorted(missing)}")

    if not available_cols:
        print("  No CDL columns found — skipping.")
        return None

    # Handle geometry
    df = process_geometry(df)

    # Classify rotation strategy using only the selected year columns
    crop_matrix = df[available_cols].values
    df = df.copy()
    df["Rotation_Type"] = [classify_row_rotation(row) for row in crop_matrix]
    df["Window"] = label

    print(f"  Loaded {len(df):,} polygons, {len(available_cols)} CDL years ({min(years)}-{max(years)})")
    return df


# ---------------------------------------------------------------------------
# Acreage aggregation
# ---------------------------------------------------------------------------

def aggregate_by_acreage(combined_dfs):
    """
    Sum CSBACRES for each (Rotation_Type, Window) pair across all datasets.

    Returns a summary DataFrame with columns: Rotation_Type, Window, CSBACRES.
    """
    print("Aggregating acreage by rotation type and window...")
    all_rotations = []
    for df in combined_dfs:
        counts = (
            df.groupby(["Rotation_Type", "Window"])["CSBACRES"]
            .sum()
            .reset_index()
        )
        all_rotations.append(counts)
    return pd.concat(all_rotations, ignore_index=True)


# ---------------------------------------------------------------------------
# Time series preparation (wide -> long)
# ---------------------------------------------------------------------------

def prepare_time_series(combined_dfs):
    """
    Reshape wide format (CDL2008, CDL2009, ...) to long format so each row
    represents one polygon × one year, suitable for temporal analysis.

    Returns a DataFrame with columns: Year, Crop_Code, CSBACRES, Crop_Name.
    """
    print("Preparing time series data (wide -> long)...")
    all_years_data = []

    for df in combined_dfs:
        year_cols = [c for c in df.columns if c.startswith("CDL")]
        melted = df.melt(
            id_vars=["CSBID", "CSBACRES"],
            value_vars=year_cols,
            var_name="Year_Col",
            value_name="Crop_Code",
        )
        melted["Year"] = melted["Year_Col"].str.replace("CDL", "").astype(int)
        melted.drop("Year_Col", axis=1, inplace=True)

        yearly_acres = (
            melted.groupby(["Year", "Crop_Code"])["CSBACRES"].sum().reset_index()
        )
        all_years_data.append(yearly_acres)

    ts_df = pd.concat(all_years_data, ignore_index=True)
    ts_df["Crop_Name"] = ts_df["Crop_Code"].apply(get_crop_name)
    return ts_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    combined_dfs = []

    for ds in datasets:
        df = process_file(ds["path"], ds["years"], ds["label"])
        if df is not None:
            combined_dfs.append(df)

    if not combined_dfs:
        print("No data processed. Exiting.")
        return

    # Generate aggregations
    rotation_summary = aggregate_by_acreage(combined_dfs)
    time_series_data = prepare_time_series(combined_dfs)

    # Add broad strategy category to rotation summary
    rotation_summary["Strategy_Category"] = rotation_summary["Rotation_Type"].apply(
        classify_strategy_type
    )

    # Save processed outputs for use by the exploration script
    print(f"\nSaving processed data to {OUTPUT_DIR}/...")
    rotation_summary.to_csv(
        os.path.join(OUTPUT_DIR, "rotation_summary.csv"), index=False
    )
    time_series_data.to_csv(
        os.path.join(OUTPUT_DIR, "time_series_data.csv"), index=False
    )

    for i, df in enumerate(combined_dfs):
        out_path = os.path.join(OUTPUT_DIR, f"processed_csb_{i}.parquet")
        # Drop raw geometry before saving to parquet (not always serialisable)
        save_df = df.drop(columns=["geometry"], errors="ignore")
        save_df.to_parquet(out_path, index=False)
        print(f"  Saved processed_csb_{i}.parquet ({len(save_df):,} rows)")

    print("\nData processing complete.")
    print(f"  Rotation types found: {rotation_summary['Rotation_Type'].nunique()}")
    print(f"  Time series years:    {sorted(time_series_data['Year'].unique())}")


if __name__ == "__main__":
    main()
