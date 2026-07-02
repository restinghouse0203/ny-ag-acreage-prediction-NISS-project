import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSB = os.path.join(BASE_DIR, "ny_csb_20172024.csv")
OUTPUT_LONG = os.path.join(BASE_DIR, "csb_crop_acreage_long_2017_2024.csv")


def build_county_fips(df: pd.DataFrame) -> pd.Series:
    """
    Try to construct 5-digit county FIPS.
    Preferred: county_key already 5-digit.
    Otherwise: STATEFIPS (2 digits) + CNTYFIPS (3 digits).
    """
    if "county_key" in df.columns:
        # already county-level fips in many of your tables
        return df["county_key"].astype(str).str.zfill(5)

    if "CNTYFIPS" in df.columns and ("STATEFIPS" in df.columns or "STATEFP" in df.columns):
        state_col = "STATEFIPS" if "STATEFIPS" in df.columns else "STATEFP"
        state = pd.to_numeric(df[state_col], errors="coerce").astype("Int64")
        county = pd.to_numeric(df["CNTYFIPS"], errors="coerce").astype("Int64")

        if state.isna().any() or county.isna().any():
            raise ValueError("STATEFIPS/STATEFP or CNTYFIPS has missing/non-numeric values. Please check the input CSV.")

        return state.astype(int).astype(str).str.zfill(2) + county.astype(int).astype(str).str.zfill(3)

    raise ValueError("Cannot find columns to build county_fips. Need either county_key, or STATEFIPS/STATEFP + CNTYFIPS.")


def main() -> None:
    # 1) read
    csb = pd.read_csv(INPUT_CSB)

    # 2) basic column checks
    if "Shape_Area" not in csb.columns:
        raise ValueError("Missing Shape_Area in CSB file. Please check the CSV columns.")

    # 3) build county_fips
    csb["county_fips"] = build_county_fips(csb)

    # 4) clean Shape_Area
    csb["Shape_Area"] = pd.to_numeric(csb["Shape_Area"], errors="coerce")
    csb = csb.dropna(subset=["Shape_Area"])
    csb = csb[csb["Shape_Area"] > 0].copy()

    # 5) find year columns CDL2017..CDL2024
    year_cols = [c for c in csb.columns if c.startswith("CDL") and c[3:].isdigit()]
    year_cols = [c for c in year_cols if 2017 <= int(c[3:]) <= 2024]
    year_cols = sorted(year_cols, key=lambda x: int(x[3:]))

    if len(year_cols) == 0:
        raise ValueError("No CDL2017–CDL2024 columns found. Please verify column names like 'CDL2017'.")

    # 6) wide -> long
    long_df = csb.melt(
        id_vars=["county_fips", "Shape_Area"],
        value_vars=year_cols,
        var_name="year",
        value_name="crop_code"
    )
    long_df["year"] = long_df["year"].str.replace("CDL", "", regex=False).astype(int)

    long_df["crop_code"] = pd.to_numeric(long_df["crop_code"], errors="coerce")
    long_df = long_df.dropna(subset=["crop_code"])
    long_df["crop_code"] = long_df["crop_code"].astype(int)

    # 7) aggregate: county × year × crop_code
    out = (
        long_df
        .groupby(["county_fips", "year", "crop_code"], as_index=False)
        .agg(acreage_area_proxy=("Shape_Area", "sum"))
    )

    # 8) (optional) sort for readability
    out = out.sort_values(["county_fips", "year", "acreage_area_proxy"], ascending=[True, True, False])

    # 9) save
    out.to_csv(OUTPUT_LONG, index=False)

    # 10) print quick checks
    print("Saved:", OUTPUT_LONG)
    print("Output shape:", out.shape)
    print("\nYears:", sorted(out["year"].unique().tolist()))
    print("\nTop 10 crop codes by total area proxy:")
    top_codes = out.groupby("crop_code")["acreage_area_proxy"].sum().sort_values(ascending=False).head(10)
    print(top_codes)

if __name__ == "__main__":
    main()