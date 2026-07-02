import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from shapely.wkb import loads
from statsmodels.tsa.stattools import acf

from config import OUTPUT_DIR, interim_csb_path

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Crop Mapping
cdl_mapping = {
    1: 'Corn', 5: 'Soybeans', 36: 'Alfalfa', 37: 'Other Hay', 
    176: 'Grass/Pasture', 59: 'Sod/Grass Seed', 61: 'Fallow', 
    24: 'Winter Wheat', 121: 'Developed', 141: 'Forest',
    4: 'Sorghum', 21: 'Barley', 23: 'Spring Wheat'
}

def get_crop_name(code):
    return cdl_mapping.get(code, f"Other ({code})")

def classify_row_rotation(row_values):
    """
    Classify the rotation strategy based on the sequence of crops.
    row_values: list or array of crop codes
    """
    unique = np.unique(row_values)
    unique_names = sorted([get_crop_name(c) for c in unique])
    
    if len(unique) == 1:
        return f"Cont. {unique_names[0]}"
    
    if len(unique) == 2:
        return f"Rot {unique_names[0]}-{unique_names[1]}"
    
    return "Complex/Mixed"

def classify_strategy_type(rotation_type):
    if rotation_type.startswith("Cont."):
        return "Continuous"
    elif rotation_type.startswith("Rot"):
        return "Rotation"
    else:
        return "Complex/Mixed"

def process_file(file_path, year_cols, label):
    print(f"Processing {label} from {file_path}...")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        df = pd.read_feather(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    # Extract crop columns
    crop_matrix = df[year_cols].values
    
    # Classify rotations
    rotation_types = [classify_row_rotation(row) for row in crop_matrix]
    
    df['Rotation_Type'] = rotation_types
    df['Window'] = label
    
    return df

# Define datasets to process
datasets = [
    {
        "path": interim_csb_path("20082015"),
        "label": "2008-2015",
        "cols": [f"CDL{y}" for y in range(2008, 2016)]
    },
    {
        "path": interim_csb_path("20172024"),
        "label": "2017-2024",
        "cols": [f"CDL{y}" for y in range(2017, 2025)]
    }
]

combined_dfs = []

for ds in datasets:
    df = process_file(ds["path"], ds["cols"], ds["label"])
    if df is not None:
        combined_dfs.append(df)

if not combined_dfs:
    print("No data processed.")
    exit()

# Combine for summary stats
print("Generating Summaries...")
all_rotations = []
for df in combined_dfs:
    counts = df.groupby(['Rotation_Type', 'Window'])['CSBACRES'].sum().reset_index()
    all_rotations.append(counts)

rot_summary = pd.concat(all_rotations)

# --- Plot 1: Histogram of Rotation Types (Original - by Window) ---
print("Generating Plot 1: Top Rotation Strategies by Window...")
top_n = 15
top_types = rot_summary.groupby('Rotation_Type')['CSBACRES'].sum().nlargest(top_n).index
plot_data = rot_summary[rot_summary['Rotation_Type'].isin(top_types)]

plt.figure(figsize=(12, 8))
sns.barplot(data=plot_data, x='Rotation_Type', y='CSBACRES', hue='Window')
plt.title(f'Top {top_n} Rotation Strategies by Acreage (By Window)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rotation_histogram_by_window.png"))
print(f"Saved rotation_histogram_by_window.png")
plt.show()


# --- Plot 2: Total Rotation Strategies by Acreage (2008-2024) ---
print("Generating Plot 2: Total Rotation Strategies (2008-2024)...")
total_rot = rot_summary.groupby('Rotation_Type')['CSBACRES'].sum().reset_index()
top_total_types = total_rot.nlargest(top_n, 'CSBACRES')

plt.figure(figsize=(12, 8))
sns.barplot(data=top_total_types, x='Rotation_Type', y='CSBACRES', color='tab:blue')
plt.title(f'Top {top_n} Rotation Strategies by Total Acreage (2008-2024)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "total_rotation_strategies_2008_2024.png"))
print(f"Saved total_rotation_strategies_2008_2024.png")
plt.show()


# --- Plot 3: Total Rotation 3 Types (Complex/Mixed, Continuous, Rotation) ---
print("Generating Plot 3: 3 Types of Strategies (2008-2024)...")
rot_summary['Strategy_Category'] = rot_summary['Rotation_Type'].apply(classify_strategy_type)
cat_summary = rot_summary.groupby('Strategy_Category')['CSBACRES'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=cat_summary, x='Strategy_Category', y='CSBACRES', palette='viridis')
plt.title('Total Acreage by Strategy Category (2008-2024)')
plt.ylabel('Total Acres')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "strategy_categories_2008_2024.png"))
print(f"Saved strategy_categories_2008_2024.png")
plt.show()


# --- Plot 4: 2008-2015 Specific Plot ---
print("Generating Plot 4: Top Rotation Strategies (2008-2015)...")
df_2008_2015 = rot_summary[rot_summary['Window'] == '2008-2015']
top_2008_types = df_2008_2015.nlargest(top_n, 'CSBACRES')

plt.figure(figsize=(12, 8))
sns.barplot(data=top_2008_types, x='Rotation_Type', y='CSBACRES', color='tab:orange')
plt.title(f'Top {top_n} Rotation Strategies (2008-2015)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rotation_strategies_2008_2015.png"))
print(f"Saved rotation_strategies_2008_2015.png")
plt.show()


# --- Plot 5 & 6: Map of Dominant Strategies (2008-2015 and 2017-2024) ---
def plot_spatial_map(df, window_label):
    print(f"Generating Map for {window_label} (Black Background)...")
    
    # Check geometry type
    sample_geom = df['geometry'].iloc[0]
    if isinstance(sample_geom, bytes):
        df['geometry'] = df['geometry'].apply(lambda x: loads(x) if x else None)
        df = gpd.GeoDataFrame(df, geometry='geometry')

    # Filter for top 5 rotations
    top_5_rot = df['Rotation_Type'].value_counts().nlargest(5).index
    map_df = df[df['Rotation_Type'].isin(top_5_rot)].copy()

    # Sample if too large
    if len(map_df) > 50000:
        map_df = map_df.sample(50000)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Use a brighter colormap and larger marker size (increased to 1000)
    map_df.plot(column='Rotation_Type', legend=True, cmap='Set1', markersize=1000, ax=ax, legend_kwds={'labelcolor': 'white'})

    # Add grid lines (white, transparent, finer)
    ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which='minor', color='white', linestyle=':', linewidth=0.3, alpha=0.2)

    # Interpretation
    if map_df.crs is None:
        min_x = map_df.geometry.bounds['minx'].min()
        if min_x < -180 or min_x > 180:
            map_df.set_crs(epsg=5070, inplace=True)
        else:
            map_df.set_crs(epsg=4326, inplace=True)

    if map_df.crs.to_epsg() != 4326:
        map_df = map_df.to_crs(epsg=4326)

    map_df['centroid'] = map_df.geometry.centroid
    map_df['lon'] = map_df.centroid.x
    map_df['lat'] = map_df.centroid.y

    west_mask = map_df['lon'] < -76.5
    north_mask = (map_df['lat'] > 43.0) & (~west_mask)
    central_mask = (~west_mask) & (~north_mask)

    regions = {
        "Western NY": map_df[west_mask],
        "Northern NY": map_df[north_mask],
        "Central/Eastern NY": map_df[central_mask]
    }

    interpretation_text = f"Dominant Strategies ({window_label}):\n"
    for name, region_df in regions.items():
        if not region_df.empty:
            top_rot = region_df['Rotation_Type'].mode()
            if not top_rot.empty:
                top_rot = top_rot.iloc[0]
                count = region_df['Rotation_Type'].value_counts().iloc[0]
                total = len(region_df)
                pct = (count / total) * 100
                interpretation_text += f"{name}: {top_rot} ({pct:.0f}%)\n"
            else:
                interpretation_text += f"{name}: Mixed\n"
        else:
            interpretation_text += f"{name}: N/A\n"

    props = dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white')
    ax.text(0.02, 0.02, interpretation_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', color='white', bbox=props)

    plt.title(f'Spatial Distribution of Top 5 Rotation Strategies ({window_label} Sample)', color='white', fontsize=16)
    ax.tick_params(colors='white', which='both')
    for spine in ax.spines.values():
        spine.set_color('white')

    filename = f"rotation_map_{window_label.replace('-', '_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), facecolor='black')
    print(f"Saved {filename} to {OUTPUT_DIR}")
    plt.show()

plot_spatial_map(combined_dfs[0].copy(), "2008-2015")
plot_spatial_map(combined_dfs[-1].copy(), "2017-2024")

# --- Plot 7: Time Series of Crop Acreage and Auto-correlation ---
print("Generating Time Series and Auto-correlation...")
all_years_data = []
for df in combined_dfs:
    year_cols = [c for c in df.columns if c.startswith('CDL')]
    melted = df.melt(id_vars=['CSBID', 'CSBACRES'], value_vars=year_cols, var_name='Year_Col', value_name='Crop_Code')
    melted['Year'] = melted['Year_Col'].str.replace('CDL', '').astype(int)
    yearly_acres = melted.groupby(['Year', 'Crop_Code'])['CSBACRES'].sum().reset_index()
    all_years_data.append(yearly_acres)

ts_df = pd.concat(all_years_data)
ts_df['Crop_Name'] = ts_df['Crop_Code'].apply(get_crop_name)

# Create combined hay and grass data
hay_grass_df = ts_df[ts_df['Crop_Name'].isin(['Other Hay', 'Grass/Pasture'])].copy()
combined_hay_grass = hay_grass_df.groupby('Year')['CSBACRES'].sum().reset_index()
combined_hay_grass['Crop_Name'] = 'Combined Hay/Grass'

# Get top 3 individual crops (excluding hay and grass since we'll combine them)
individual_crops = ts_df[~ts_df['Crop_Name'].isin(['Other Hay', 'Grass/Pasture'])]
top_3_crops = individual_crops.groupby('Crop_Name')['CSBACRES'].sum().nlargest(3).index

# Prepare data for plotting (top 3 individual crops + combined hay/grass)
ts_plot_data = ts_df[ts_df['Crop_Name'].isin(top_3_crops)]
ts_plot_data = pd.concat([ts_plot_data, combined_hay_grass])

plt.figure(figsize=(12, 6))
sns.lineplot(data=ts_plot_data, x='Year', y='CSBACRES', hue='Crop_Name', marker='o')
plt.title('Time Series of Crop Acreage (Top 3 Crops + Combined Hay/Grass)')
plt.ylabel('Total Acres')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "crop_acreage_timeseries_combined.png"))
print(f"Saved crop_acreage_timeseries_combined.png to {OUTPUT_DIR}")
plt.show()

# --- ACF Analysis and Plots ---
print("\nGenerating ACF Analysis...")

# Prepare time series data for ACF analysis
crop_names_for_acf = list(top_3_crops) + ['Combined Hay/Grass']
acf_data = {}

# Get individual crop time series
for crop in top_3_crops:
    crop_ts = ts_df[ts_df['Crop_Name'] == crop].groupby('Year')['CSBACRES'].sum().sort_index()
    acf_data[crop] = crop_ts

# Add combined hay/grass time series
acf_data['Combined Hay/Grass'] = combined_hay_grass.set_index('Year')['CSBACRES'].sort_index()

# Create ACF plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

print("\n--- Auto-correlation Function (ACF) up to Lag 5 ---")

for i, (crop_name, crop_ts) in enumerate(acf_data.items()):
    # Calculate ACF up to lag 5
    acf_values = acf(crop_ts, nlags=5, fft=False)
    
    # Print ACF values
    print(f"\n{crop_name}:")
    for lag in range(6):  # 0 to 5
        print(f"  Lag {lag}: {acf_values[lag]:.4f}")
    
    # Plot ACF
    ax = axes[i]
    lags = range(len(acf_values))
    ax.bar(lags, acf_values, alpha=0.7, color=f'C{i}')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='±0.2')
    ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
    ax.set_title(f'ACF: {crop_name}')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(lags)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "acf_plots_4_variables.png"))
print(f"\nSaved acf_plots_4_variables.png to {OUTPUT_DIR}")
plt.show()

print("\n--- Auto-correlation (Lag 1) Summary ---")
for crop_name, crop_ts in acf_data.items():
    autocorr = crop_ts.autocorr(lag=1)
    print(f"{crop_name}: {autocorr:.4f}")

