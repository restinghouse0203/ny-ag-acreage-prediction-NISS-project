import pandas as pd
import numpy as np
import geopandas as gpd


#  Load data

csb = pd.read_csv("ny_csb_20172024.csv")
soil = pd.read_csv("soil_features_NY.csv")
counties = gpd.read_file("tl_2025_us_county.shp")
counties.head()

#print("CSB shape:", csb.shape)
#print("Soil shape:", soil.shape)

#print(csb.head())
#print(soil.head())
#print(counties.head())

#Filter US counties to New York + create a county key
ny_counties = counties[counties["STATEFP"] == "36"].copy()
ny_counties["CNTYFIPS"] = ny_counties["COUNTYFP"].astype(str).str.zfill(3)
ny_counties["STATEFIPS"] = ny_counties["STATEFP"].astype(str).str.zfill(2)
ny_counties["county_key"] = ny_counties["STATEFIPS"] + ny_counties["CNTYFIPS"]

##print(ny_counties[["county_key","NAME"]].head())

soil_poly = gpd.read_file("gSSURGO_NY.gdb", layer="MUPOLYGON")
#print(soil_poly.columns)
#print(soil_poly[["MUKEY"]].head())

# ============================================================
# Step 1. Ensure both layers share the same CRS
# ============================================================
soil_poly = soil_poly.to_crs(ny_counties.crs)

soil_poly = soil_poly[["MUKEY", "geometry"]].copy()
ny_counties = ny_counties[["county_key", "geometry"]].copy()

# ============================================================
# Step 2. Spatial join to generate candidate soil–county pairs
# ============================================================
pairs = gpd.sjoin(
    soil_poly,
    ny_counties,
    how="inner",
    predicate="intersects"
).reset_index(drop=True)

# IMPORTANT: after sjoin, make sure county_key exists
# (it should come from ny_counties)
print("Columns after sjoin:", pairs.columns.tolist())

if "county_key" not in pairs.columns:
    # If county_key is missing, it means the join did not carry it over
    # (rare, but can happen if ny_counties columns were not kept correctly)
    raise ValueError("county_key is not present after sjoin. Please check ny_counties columns.")

print("Candidate soil–county pairs:", pairs.shape)

# ============================================================
# Step 3. Reproject to a projected CRS BEFORE computing areas
# ------------------------------------------------------------
# Area computations in a geographic CRS (degrees) are invalid.
# EPSG:3857 uses meters and is sufficient for area weights.
# ============================================================
pairs = pairs.to_crs(epsg=3857)
ny_counties_m = ny_counties.to_crs(epsg=3857)

# Attach county geometry for exact intersection
pairs = pairs.merge(
    ny_counties_m[["county_key", "geometry"]].rename(columns={"geometry": "county_geom"}),
    on="county_key",
    how="left"
)

# Compute exact intersection geometry
pairs["intersect_geom"] = pairs.geometry.intersection(pairs["county_geom"])

# Compute intersection area (square meters)
pairs["intersect_area"] = pairs["intersect_geom"].area

# Keep only positive-area overlaps
pairs = pairs[pairs["intersect_area"] > 0].copy()
print("Valid intersecting pairs:", pairs.shape)

# ============================================================
# Step 4. Compute area fractions within each MUKEY
# ============================================================
pairs["area_frac"] = (
    pairs["intersect_area"] /
    pairs.groupby("MUKEY")["intersect_area"].transform("sum")
)

# Final crosswalk table
crosswalk = pairs[["MUKEY", "county_key", "area_frac"]].copy()

print(crosswalk.head())
print("Crosswalk shape:", crosswalk.shape)

crosswalk.to_csv("mukey_county_crosswalk.csv", index=False)
print("Saved mukey_county_crosswalk.csv")

