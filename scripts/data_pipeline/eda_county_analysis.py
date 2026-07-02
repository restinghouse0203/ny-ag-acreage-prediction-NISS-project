import pandas as pd
import numpy as np

# Load final merged county-level dataset
final = pd.read_csv("csb_soil_county_merged.csv")

print("Final dataset shape:", final.shape)
print(final.head())

# Missing rate by column
na_rate = final.isna().mean().sort_values(ascending=False)
print(na_rate)

soil_cols = [c for c in final.columns if "gSSURGO" in c]

soil_desc = final[soil_cols].describe().T
print(soil_desc)

# CSB outcome variables (you can adjust later)
csb_vars = [
    "CSBYEARS",
    "CSBACRES",
    "CDL2017", "CDL2018", "CDL2019",
    "CDL2020", "CDL2021", "CDL2022",
    "CDL2023", "CDL2024"
]

# Soil variables (all gSSURGO-derived)
soil_vars = [c for c in final.columns if "gSSURGO" in c]

print("Number of soil variables:", len(soil_vars))

# Subset and compute correlation
corr = final[soil_vars + csb_vars].corr()

# Only keep soil vs CSB block
soil_csb_corr = corr.loc[soil_vars, csb_vars]

top_corr = (
    soil_csb_corr["CSBACRES"]
    .sort_values(key=np.abs, ascending=False)
    .head(10)
)

print("Top soil variables correlated with CSBACRES:")
print(top_corr)
