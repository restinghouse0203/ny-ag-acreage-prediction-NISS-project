import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plot style
sns.set_theme(style="whitegrid")

# Define Paths
# Assuming script is run from script/ directory
csb_path = "../data/merged_data/merged_CSB_total.csv"

# 1. Load CSB Data
if os.path.exists(csb_path):
    csb_df = pd.read_csv(csb_path)
    print(f"CSB Data Loaded: {csb_df.shape}")
else:
    # Try absolute path or adjust if running from root
    if os.path.exists("data/merged_data/merged_CSB_total.csv"):
        csb_df = pd.read_csv("data/merged_data/merged_CSB_total.csv")
        print(f"CSB Data Loaded: {csb_df.shape}")
    else:
        print(f"Error: CSB file not found at {csb_path}")
        exit()

# ---------------------------------------------------------
# 3. Plant Cycle Analysis & Distribution
# ---------------------------------------------------------
# Discussion on Studying Plant Cycle with Annual Data:
# 1. Long-term Trends: Since data is annual, we look for trends over years rather than seasonal cycles within a year.
# 2. Inter-annual Variability: We can observe year-to-year fluctuations which may be driven by climate or market cycles.
# 3. Distribution Analysis: Agricultural data is often highly skewed (many small values, few large ones). 
#    Log-transformation is essential to study the underlying distribution and variance.

# 3.1 Time Series Plot of CSBACRES
# We aggregate by Year to see the total trend
plt.figure(figsize=(12, 6))
annual_acres = csb_df.groupby('Year')['CSBACRES'].sum().reset_index()
sns.lineplot(data=annual_acres, x='Year', y='CSBACRES', marker='o', linewidth=2.5)
plt.title('Total CSB Acreage Over Time (Annual Cycle)', fontsize=14)
plt.ylabel('Total Acres')
plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig('csb_acreage_time_series.png')
# print("Saved time series plot to csb_acreage_time_series.png")
plt.show()

# 3.2 Distribution Analysis: Histogram of CSBACRES vs Log(CSBACRES)
plt.figure(figsize=(14, 6))

# Raw CSBACRES
plt.subplot(1, 2, 1)
sns.histplot(csb_df['CSBACRES'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of CSBACRES (Raw)', fontsize=12)
plt.xlabel('Acres')
plt.ylabel('Frequency')

# Log-Transformed CSBACRES
# We use log1p (log(x+1)) to handle zero values safely
plt.subplot(1, 2, 2)
log_acres = np.log1p(csb_df['CSBACRES'])
sns.histplot(log_acres, bins=50, kde=True, color='green')
plt.title('Distribution of Log(CSBACRES)', fontsize=12)
plt.xlabel('Log(Acres + 1)')
plt.ylabel('Frequency')

plt.tight_layout()
# plt.savefig('csb_acreage_distribution.png')
# print("Saved distribution plot to csb_acreage_distribution.png")
plt.show()

# Summary Statistics
print("\nSummary Statistics (Raw):")
print(csb_df['CSBACRES'].describe())
print("\nSummary Statistics (Log-Transformed):")
print(log_acres.describe())
