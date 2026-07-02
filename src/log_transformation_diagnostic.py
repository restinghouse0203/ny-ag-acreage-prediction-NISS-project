import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import OUTPUT_DIR, WEATHER_FEATHER

WEATHER_PATH = WEATHER_FEATHER
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_weather_data():
    """Load and process weather data to examine log transformation"""
    print("Loading Weather data for diagnostic...")
    df = pd.read_feather(WEATHER_PATH)
    
    # Extract FIPS from 'county' 
    df['CNTYFIPS'] = df['county'].str.replace('USNY', '').astype(int) % 1000
    
    # Convert dates
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['Year'] = df['start_date'].dt.year
    df['Month'] = df['start_date'].dt.month
    
    # Create weather features
    planting_mask = df['Month'].isin([4, 5])
    planting_df = df[planting_mask].groupby(['Year', 'CNTYFIPS'])['totr'].sum().reset_index()
    planting_df.rename(columns={'totr': 'Planting_Precip'}, inplace=True)
    
    growing_mask = df['Month'].isin([5, 6, 7, 8, 9, 10])
    growing_df = df[growing_mask].groupby(['Year', 'CNTYFIPS'])['gdd_b10'].sum().reset_index()
    growing_df.rename(columns={'gdd_b10': 'Growing_GDD'}, inplace=True)
    
    # Merge weather features
    weather_feat = pd.merge(planting_df, growing_df, on=['Year', 'CNTYFIPS'], how='outer')
    
    # Add log-transformed features
    weather_feat['Planting_Precip_Log'] = np.log1p(weather_feat['Planting_Precip'])
    weather_feat['Growing_GDD_Log'] = np.log1p(weather_feat['Growing_GDD'])
    
    return weather_feat

def diagnostic_analysis():
    """Comprehensive diagnostic of log transformation"""
    weather_df = load_weather_data()
    
    print("=== LOG TRANSFORMATION DIAGNOSTIC ===\n")
    
    # 1. Basic Statistics
    print("1. BASIC STATISTICS:")
    print("\nPlanting Precipitation:")
    print(f"  Raw - Min: {weather_df['Planting_Precip'].min():.2f}, Max: {weather_df['Planting_Precip'].max():.2f}")
    print(f"  Raw - Mean: {weather_df['Planting_Precip'].mean():.2f}, Std: {weather_df['Planting_Precip'].std():.2f}")
    print(f"  Log - Min: {weather_df['Planting_Precip_Log'].min():.2f}, Max: {weather_df['Planting_Precip_Log'].max():.2f}")
    print(f"  Log - Mean: {weather_df['Planting_Precip_Log'].mean():.2f}, Std: {weather_df['Planting_Precip_Log'].std():.2f}")
    
    print("\nGrowing Degree Days:")
    print(f"  Raw - Min: {weather_df['Growing_GDD'].min():.2f}, Max: {weather_df['Growing_GDD'].max():.2f}")
    print(f"  Raw - Mean: {weather_df['Growing_GDD'].mean():.2f}, Std: {weather_df['Growing_GDD'].std():.2f}")
    print(f"  Log - Min: {weather_df['Growing_GDD_Log'].min():.2f}, Max: {weather_df['Growing_GDD_Log'].max():.2f}")
    print(f"  Log - Mean: {weather_df['Growing_GDD_Log'].mean():.2f}, Std: {weather_df['Growing_GDD_Log'].std():.2f}")
    
    # 2. Check for zero values
    print(f"\n2. ZERO VALUES CHECK:")
    print(f"  Planting_Precip zeros: {(weather_df['Planting_Precip'] == 0).sum()}")
    print(f"  Growing_GDD zeros: {(weather_df['Growing_GDD'] == 0).sum()}")
    
    # 3. Distribution analysis
    print(f"\n3. DISTRIBUTION ANALYSIS:")
    print(f"  Planting_Precip skewness: Raw={weather_df['Planting_Precip'].skew():.3f}, Log={weather_df['Planting_Precip_Log'].skew():.3f}")
    print(f"  Growing_GDD skewness: Raw={weather_df['Growing_GDD'].skew():.3f}, Log={weather_df['Growing_GDD_Log'].skew():.3f}")
    
    # 4. Sample values comparison
    print(f"\n4. SAMPLE VALUES COMPARISON (first 10 rows):")
    sample_df = weather_df[['Year', 'CNTYFIPS', 'Planting_Precip', 'Planting_Precip_Log', 
                           'Growing_GDD', 'Growing_GDD_Log']].head(10)
    print(sample_df.to_string(index=False))
    
    # 5. Time series aggregation for plotting
    yearly_stats = weather_df.groupby('Year').agg({
        'Planting_Precip': ['mean', 'std', 'min', 'max'],
        'Planting_Precip_Log': ['mean', 'std', 'min', 'max'],
        'Growing_GDD': ['mean', 'std', 'min', 'max'],
        'Growing_GDD_Log': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    print(f"\n5. YEARLY AGGREGATED STATISTICS:")
    print(yearly_stats)
    
    # 6. Visualization
    create_diagnostic_plots(weather_df)
    
    return weather_df

def create_diagnostic_plots(weather_df):
    """Create comprehensive diagnostic plots"""
    
    # 1. Distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Precipitation distributions
    axes[0,0].hist(weather_df['Planting_Precip'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].set_title('Distribution: Raw Planting Precipitation')
    axes[0,0].set_xlabel('Precipitation (mm)')
    axes[0,0].set_ylabel('Frequency')
    
    axes[0,1].hist(weather_df['Planting_Precip_Log'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0,1].set_title('Distribution: Log Planting Precipitation')
    axes[0,1].set_xlabel('Log(1 + Precipitation)')
    axes[0,1].set_ylabel('Frequency')
    
    # GDD distributions
    axes[1,0].hist(weather_df['Growing_GDD'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1,0].set_title('Distribution: Raw Growing Degree Days')
    axes[1,0].set_xlabel('Growing Degree Days')
    axes[1,0].set_ylabel('Frequency')
    
    axes[1,1].hist(weather_df['Growing_GDD_Log'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1,1].set_title('Distribution: Log Growing Degree Days')
    axes[1,1].set_xlabel('Log(1 + GDD)')
    axes[1,1].set_ylabel('Frequency')
    
    plt.suptitle('Distribution Comparison: Raw vs Log-transformed Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "log_transformation_distributions.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Scatter plots: Raw vs Log
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(weather_df['Planting_Precip'], weather_df['Planting_Precip_Log'], alpha=0.6, color='blue')
    ax1.set_xlabel('Raw Planting Precipitation')
    ax1.set_ylabel('Log(1 + Precipitation)')
    ax1.set_title('Raw vs Log: Planting Precipitation')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(weather_df['Growing_GDD'], weather_df['Growing_GDD_Log'], alpha=0.6, color='red')
    ax2.set_xlabel('Raw Growing Degree Days')
    ax2.set_ylabel('Log(1 + GDD)')
    ax2.set_title('Raw vs Log: Growing Degree Days')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "raw_vs_log_scatter.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Time series comparison with proper scaling
    weather_ts = weather_df.groupby('Year')[['Planting_Precip', 'Planting_Precip_Log', 
                                            'Growing_GDD', 'Growing_GDD_Log']].mean().reset_index()
    
    # Create normalized versions for better comparison
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Normalize for comparison
    weather_ts_norm = weather_ts.copy()
    features_to_norm = ['Planting_Precip', 'Planting_Precip_Log', 'Growing_GDD', 'Growing_GDD_Log']
    weather_ts_norm[features_to_norm] = scaler.fit_transform(weather_ts[features_to_norm])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Raw time series
    axes[0,0].plot(weather_ts['Year'], weather_ts['Planting_Precip'], 'b-o', linewidth=2, label='Raw')
    axes[0,0].set_title('Raw Planting Precipitation Time Series')
    axes[0,0].set_ylabel('Precipitation (mm)')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(weather_ts['Year'], weather_ts['Planting_Precip_Log'], 'b-s', linewidth=2, label='Log')
    axes[0,1].set_title('Log Planting Precipitation Time Series')
    axes[0,1].set_ylabel('Log(1 + Precipitation)')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(weather_ts['Year'], weather_ts['Growing_GDD'], 'r-o', linewidth=2, label='Raw')
    axes[1,0].set_title('Raw Growing Degree Days Time Series')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('Growing Degree Days')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(weather_ts['Year'], weather_ts['Growing_GDD_Log'], 'r-s', linewidth=2, label='Log')
    axes[1,1].set_title('Log Growing Degree Days Time Series')
    axes[1,1].set_xlabel('Year')
    axes[1,1].set_ylabel('Log(1 + GDD)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('Time Series Comparison: Raw vs Log-transformed (Separate Scales)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "time_series_separate_scales.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Normalized comparison on same plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(weather_ts_norm['Year'], weather_ts_norm['Planting_Precip'], 'b-o', linewidth=2, label='Raw (normalized)')
    ax1.plot(weather_ts_norm['Year'], weather_ts_norm['Planting_Precip_Log'], 'b--s', linewidth=2, label='Log (normalized)')
    ax1.set_title('Normalized Comparison: Planting Precipitation')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Standardized Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(weather_ts_norm['Year'], weather_ts_norm['Growing_GDD'], 'r-o', linewidth=2, label='Raw (normalized)')
    ax2.plot(weather_ts_norm['Year'], weather_ts_norm['Growing_GDD_Log'], 'r--s', linewidth=2, label='Log (normalized)')
    ax2.set_title('Normalized Comparison: Growing Degree Days')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Standardized Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Normalized Time Series: Raw vs Log-transformed (Same Scale)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "time_series_normalized_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    weather_data = diagnostic_analysis()
    
    print(f"\n=== CONCLUSION ===")
    print("Check the generated diagnostic plots in the output folder:")
    print("1. log_transformation_distributions.png - Shows distribution changes")
    print("2. raw_vs_log_scatter.png - Shows relationship between raw and log values")
    print("3. time_series_separate_scales.png - Time series on separate scales")
    print("4. time_series_normalized_comparison.png - Normalized comparison on same scale")