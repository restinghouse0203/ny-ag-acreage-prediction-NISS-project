import numpy as np
import pandas as pd

# ============================================================
# Load precomputed crosswalk (from spatial steps)
# ============================================================
# ============================================================
# Step 5. Soil: county-level aggregation (area-weighted)
# ============================================================
crosswalk = pd.read_csv("mukey_county_crosswalk.csv")
soil_attr = pd.read_csv("soil_features_NY.csv")

# Treat MUKEY as an ID (string) to avoid dtype mismatch
crosswalk["MUKEY"] = crosswalk["MUKEY"].astype(str)
soil_attr["MUKEY"] = soil_attr["MUKEY"].astype(str)

# Keep only the columns we need to reduce memory
soil_num_cols = soil_attr.select_dtypes(include=[np.number]).columns.tolist()
soil_num_cols = [c for c in soil_num_cols if c != "MUKEY"]
soil_attr_small = soil_attr[["MUKEY"] + soil_num_cols].copy()

# Merge: MUKEY–county pairs + soil variables
soil_join = crosswalk.merge(soil_attr_small, on="MUKEY", how="left")

# ------------------------------------------------------------
# Vectorized area-weighted aggregation (NO groupby-apply)
# ------------------------------------------------------------
# 1) Multiply each soil variable by area_frac
w = pd.to_numeric(soil_join["area_frac"], errors="coerce").astype("float64").fillna(0.0)

for col in soil_num_cols:
    soil_join[col] = pd.to_numeric(soil_join[col], errors="coerce").astype("float64")
    soil_join[col] = soil_join[col] * w

# 2) Sum weighted values by county_key
soil_county_sum = soil_join.groupby("county_key", as_index=False)[soil_num_cols].sum()

# 3) Also compute sum of weights by county (for weighted mean)
w_sum = soil_join.groupby("county_key", as_index=False)["area_frac"].sum().rename(columns={"area_frac": "w_sum"})

# 4) Divide weighted sums by weight sums → weighted mean
soil_county = soil_county_sum.merge(w_sum, on="county_key", how="left")
for col in soil_num_cols:
    soil_county[col] = soil_county[col] / soil_county["w_sum"]

soil_county = soil_county.drop(columns=["w_sum"])

print("Soil county-level shape:", soil_county.shape)
soil_county.to_csv("soil_county_aggregated.csv", index=False)
print("Saved soil_county_aggregated.csv")



# ============================================================
# Step 6. CSB: county-level aggregation
# ============================================================
csb = pd.read_csv("ny_csb_20172024.csv")

csb["STATEFIPS"] = csb["STATEFIPS"].astype(str).str.zfill(2)
csb["CNTYFIPS"] = csb["CNTYFIPS"].astype(str).str.zfill(3)
csb["county_key"] = csb["STATEFIPS"] + csb["CNTYFIPS"]

csb_num_cols = csb.select_dtypes(include=[np.number]).columns.tolist()
csb_num_cols = [c for c in csb_num_cols if c != "CSBID"]

csb_county = csb.groupby("county_key", as_index=False)[csb_num_cols].mean()

# Force one row per county_key (safety)
csb_county_1 = csb_county.groupby("county_key", as_index=False).mean(numeric_only=True)

print("CSB county-level:", csb_county_1.shape)

# ============================================================
# Step 7. Merge (safe one-to-one)
# ============================================================

# Ensure county_key has the same dtype in both tables
csb_county_1["county_key"] = csb_county_1["county_key"].astype(str).str.zfill(5)
soil_county["county_key"] = soil_county["county_key"].astype(str).str.zfill(5)

final = csb_county_1.merge(soil_county, on="county_key", how="left", indicator=True)

print("Merge indicator counts:")
print(final["_merge"].value_counts())

final.to_csv("csb_soil_county_merged.csv", index=False)
print("Saved csb_soil_county_merged.csv")
