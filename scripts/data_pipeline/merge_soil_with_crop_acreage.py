import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CROP = os.path.join(BASE_DIR, "csb_crop_acreage_long_2017_2024.csv")
INPUT_SOIL = os.path.join(BASE_DIR, "soil_county_aggregated.csv")
OUTPUT_PANEL = os.path.join(BASE_DIR, "soil_crop_acreage_panel_2017_2024.csv")

def main():
    crop = pd.read_csv(INPUT_CROP)
    soil = pd.read_csv(INPUT_SOIL)

    # --- build merge key ---
    # crop already has county_fips
    crop["county_fips"] = crop["county_fips"].astype(str).str.zfill(5)

    # soil uses county_key -> county_fips
    if "county_key" not in soil.columns:
        raise ValueError("soil file missing 'county_key' column. Please check soil_county_aggregated.csv")

    soil["county_fips"] = soil["county_key"].astype(str).str.zfill(5)

    # --- merge ---
    panel = crop.merge(soil.drop(columns=["county_key"]), on="county_fips", how="left")

    # --- diagnostics ---
    print("Merged shape:", panel.shape)
    print("Years:", sorted(panel["year"].unique().tolist()))
    print("Unique crop codes:", panel["crop_code"].nunique())

    # check if any rows failed to match soil
    soil_cols = [c for c in soil.columns if c not in ["county_key", "county_fips"]]
    missing_all_soil = panel[soil_cols].isna().all(axis=1).sum()
    print("Rows with ALL soil features missing:", missing_all_soil)

    panel.to_csv(OUTPUT_PANEL, index=False)
    print("Saved:", OUTPUT_PANEL)

if __name__ == "__main__":
    main()
