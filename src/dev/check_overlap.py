import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WEATHER_FEATHER, interim_csb_path

CSB_PATH = interim_csb_path("20172024")
WEATHER_PATH = WEATHER_FEATHER


def check_overlap():
    print("Loading CSB...")
    csb = pd.read_feather(CSB_PATH)
    csb_years = [y for y in range(2017, 2025)]
    csb_counties = csb['CNTYFIPS'].unique().astype(int)
    print(f"CSB Counties: {len(csb_counties)} unique")
    print(f"CSB Counties Sample: {csb_counties[:10]}")
    print(f"CSB Years: {csb_years}")

    print("Loading Weather...")
    weather = pd.read_feather(WEATHER_PATH)
    weather['CNTYFIPS'] = weather['county'].str.replace('USNY', '').astype(int)
    weather['Year'] = pd.to_datetime(weather['start_date']).dt.year
    weather_years = weather['Year'].unique()
    weather_counties = weather['CNTYFIPS'].unique()

    print(f"Weather Counties: {len(weather_counties)} unique")
    print(f"Weather Counties Sample: {weather_counties[:10]}")
    print(f"Weather Years: {sorted(weather_years)}")

    common_counties = set(csb_counties).intersection(set(weather_counties))
    print(f"Common Counties: {len(common_counties)}")

    common_years = set(csb_years).intersection(set(weather_years))
    print(f"Common Years: {sorted(list(common_years))}")


if __name__ == "__main__":
    check_overlap()
