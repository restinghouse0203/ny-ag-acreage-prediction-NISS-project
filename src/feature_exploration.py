"""
feature_exploration.py
======================
Diagnostic visualisations and validation for the model-ready dataset.

Input
-----
  processed_dataset.parquet — output of feature_engineering.py

Outputs (saved to OUTPUT_DIR)
------------------------------
  1. crop_types_timeseries.png
  2. weather_features_timeseries_raw.png
  3. weather_features_timeseries_log.png
  4. weather_features_comparison.png
  5. weather_features_normalized_comparison.png
  6. weather_features_distributions.png
  7. weather_acf_analysis_comparison.png

Run feature_engineering.py first to generate processed_dataset.parquet.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from config import OUTPUT_DIR, PROCESSED_DATASET

INPUT_PATH = PROCESSED_DATASET
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed_data():
    """Load the dataset produced by feature_engineering.py."""
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found at {INPUT_PATH}.\n"
            "Run feature_engineering.py first."
        )
    print(f"Loading processed dataset from {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"  Year range: {df['Year'].min()}-{df['Year'].max()}")
    return df


# ===========================================================================
# 3.1  Time Series Analysis — 6 visualisation sets
# ===========================================================================

def plot_crop_types_timeseries(df):
    """
    Plot 1: Line chart of total acreage per year for all 5 crop categories.
    Covers 2008-2024 (including 2016 from the merged dataset).
    """
    print("Plot 1: Crop type time series...")
    crop_ts = df.groupby(["Year", "Crop_Type"])["CSBACRES"].sum().reset_index()
    crop_pivot = (
        crop_ts.pivot(index="Year", columns="Crop_Type", values="CSBACRES").fillna(0)
    )

    plt.figure(figsize=(14, 8))
    for crop in crop_pivot.columns:
        plt.plot(crop_pivot.index, crop_pivot[crop], marker="o", label=crop, linewidth=2)

    plt.title("Time Series of Crop Types (Total Acreage)", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Acreage", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "crop_types_timeseries.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.show()


def plot_weather_raw_timeseries(df):
    """
    Plot 2: Two-panel time series for raw Planting_Precip and Growing_GDD.
    Shows inter-annual variability in precipitation and thermal accumulation.
    """
    print("Plot 2: Raw weather time series...")
    weather_ts = df.groupby("Year")[["Planting_Precip", "Growing_GDD"]].mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(weather_ts["Year"], weather_ts["Planting_Precip"], "b-", marker="o", linewidth=2)
    ax1.set_title("Planting Precipitation (April-May) — Raw", fontsize=14)
    ax1.set_ylabel("Precipitation (mm)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(weather_ts["Year"], weather_ts["Growing_GDD"], "r-", marker="o", linewidth=2)
    ax2.set_title("Growing Degree Days (May-October) — Raw", fontsize=14)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Growing Degree Days", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "weather_features_timeseries_raw.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.show()


def plot_weather_log_timeseries(df):
    """
    Plot 3: Two-panel time series for log-transformed weather features.
    Confirms temporal trends are preserved after log1p transformation.
    """
    print("Plot 3: Log-transformed weather time series...")
    weather_ts_log = df.groupby("Year")[["Planting_Precip_Log", "Growing_GDD_Log"]].mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(weather_ts_log["Year"], weather_ts_log["Planting_Precip_Log"], "b-", marker="s", linewidth=2)
    ax1.set_title("Planting Precipitation (April-May) — Log-transformed", fontsize=14)
    ax1.set_ylabel("Log(1 + Precipitation)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(weather_ts_log["Year"], weather_ts_log["Growing_GDD_Log"], "r-", marker="s", linewidth=2)
    ax2.set_title("Growing Degree Days (May-October) — Log-transformed", fontsize=14)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Log(1 + GDD)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "weather_features_timeseries_log.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.show()


def plot_weather_comparison(df):
    """
    Plot 4: Side-by-side 2x2 grid — Raw Precip | Log Precip / Raw GDD | Log GDD.
    Highlights scale compression and variance reduction from log transformation.
    """
    print("Plot 4: Raw vs log-transformed comparison...")
    weather_ts     = df.groupby("Year")[["Planting_Precip", "Growing_GDD"]].mean().reset_index()
    weather_ts_log = df.groupby("Year")[["Planting_Precip_Log", "Growing_GDD_Log"]].mean().reset_index()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    ax1.plot(weather_ts["Year"], weather_ts["Planting_Precip"], "b-", marker="o", linewidth=2)
    ax1.set_title("Raw Planting Precipitation", fontsize=12)
    ax1.set_ylabel("Precipitation (mm)", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(weather_ts_log["Year"], weather_ts_log["Planting_Precip_Log"], "b-", marker="s", linewidth=2)
    ax2.set_title("Log-transformed Planting Precipitation", fontsize=12)
    ax2.set_ylabel("Log(1 + Precipitation)", fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3.plot(weather_ts["Year"], weather_ts["Growing_GDD"], "r-", marker="o", linewidth=2)
    ax3.set_title("Raw Growing Degree Days", fontsize=12)
    ax3.set_xlabel("Year", fontsize=10)
    ax3.set_ylabel("Growing Degree Days", fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4.plot(weather_ts_log["Year"], weather_ts_log["Growing_GDD_Log"], "r-", marker="s", linewidth=2)
    ax4.set_title("Log-transformed Growing Degree Days", fontsize=12)
    ax4.set_xlabel("Year", fontsize=10)
    ax4.set_ylabel("Log(1 + GDD)", fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Comparison: Raw vs Log-transformed Weather Features", fontsize=16, y=0.98)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "weather_features_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.show()


def plot_normalized_comparison(df):
    """
    Plot 5: Normalised overlay — raw and log series standardised to
    zero mean / unit variance, revealing true shape differences.
    """
    print("Plot 5: Normalized comparison...")
    weather_ts     = df.groupby("Year")[["Planting_Precip", "Growing_GDD"]].mean().reset_index()
    weather_ts_log = df.groupby("Year")[["Planting_Precip_Log", "Growing_GDD_Log"]].mean().reset_index()

    scaler = StandardScaler()
    wn = weather_ts.copy()
    wl = weather_ts_log.copy()
    wn["Planting_Precip_Norm"]     = scaler.fit_transform(weather_ts[["Planting_Precip"]])[:, 0]
    wl["Planting_Precip_Log_Norm"] = scaler.fit_transform(weather_ts_log[["Planting_Precip_Log"]])[:, 0]
    wn["Growing_GDD_Norm"]         = scaler.fit_transform(weather_ts[["Growing_GDD"]])[:, 0]
    wl["Growing_GDD_Log_Norm"]     = scaler.fit_transform(weather_ts_log[["Growing_GDD_Log"]])[:, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(wn["Year"], wn["Planting_Precip_Norm"],     "b-o",  linewidth=2, label="Raw (normalised)",  markersize=6)
    ax1.plot(wl["Year"], wl["Planting_Precip_Log_Norm"], "b--s", linewidth=2, label="Log (normalised)",  markersize=6, alpha=0.8)
    ax1.set_title("Normalised: Planting Precipitation", fontsize=14)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Standardised Values", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(wn["Year"], wn["Growing_GDD_Norm"],         "r-o",  linewidth=2, label="Raw (normalised)",  markersize=6)
    ax2.plot(wl["Year"], wl["Growing_GDD_Log_Norm"],     "r--s", linewidth=2, label="Log (normalised)",  markersize=6, alpha=0.8)
    ax2.set_title("Normalised: Growing Degree Days", fontsize=14)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Standardised Values", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Normalised Time Series: Differences Between Raw and Log-transformed", fontsize=16)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "weather_features_normalized_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.show()


def plot_distribution_comparison(df):
    """
    Plot 6: 2x2 histogram grid (Raw Precip | Log Precip / Raw GDD | Log GDD).
    Mean line overlay demonstrates right-skew reduction from log transformation.
    """
    print("Plot 6: Distribution comparison (histograms)...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    def _hist(ax, series, title, xlabel, color):
        ax.hist(series.dropna(), bins=30, alpha=0.7, color=color, edgecolor="black")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        m = series.mean()
        ax.axvline(m, color="red", linestyle="--", label=f"Mean: {m:.2f}")
        ax.legend()

    _hist(ax1, df["Planting_Precip"],     "Distribution: Raw Planting Precipitation",     "Precipitation (mm)",        "blue")
    _hist(ax2, df["Planting_Precip_Log"], "Distribution: Log Planting Precipitation",     "Log(1 + Precipitation)",    "blue")
    _hist(ax3, df["Growing_GDD"],         "Distribution: Raw Growing Degree Days",         "Growing Degree Days",       "red")
    _hist(ax4, df["Growing_GDD_Log"],     "Distribution: Log Growing Degree Days",         "Log(1 + GDD)",              "red")

    plt.suptitle("Distribution Comparison: Raw vs Log-transformed Features", fontsize=16)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "weather_features_distributions.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.show()


# ===========================================================================
# 3.2  Auto-correlation Function (ACF) Analysis
# ===========================================================================

def plot_acf_analysis(df):
    """
    Compute county-level ACF (up to lag 5) for raw and log-transformed
    weather features, average across counties, and plot a 2x2 comparison grid.

    County-level averaging prevents any single county from dominating the
    temporal correlation signal.
    """
    print("\nGenerating ACF Analysis for weather features...")

    weather_cols = [
        "Planting_Precip", "Growing_GDD",
        "Planting_Precip_Log", "Growing_GDD_Log",
    ]

    county_weather = (
        df.groupby(["CNTYFIPS", "Year"])[weather_cols]
        .mean()
        .reset_index()
    )

    acf_results = {c: [] for c in weather_cols}

    for county in county_weather["CNTYFIPS"].unique():
        county_data = county_weather[county_weather["CNTYFIPS"] == county].sort_values("Year")
        if len(county_data) < 8:          # require sufficient observations
            continue
        for feat in weather_cols:
            ts_data = county_data[feat].dropna()
            if len(ts_data) < 6:
                continue
            try:
                acf_vals = acf(ts_data, nlags=5, fft=False)
                acf_results[feat].append(acf_vals)
            except Exception:
                continue

    avg_acf = {}
    for feat in weather_cols:
        if acf_results[feat]:
            avg_acf[feat] = np.mean(acf_results[feat], axis=0)
        else:
            avg_acf[feat] = np.array([1, 0, 0, 0, 0, 0])

    # Console output
    print("\n--- ACF Results (averaged across counties) ---")
    print("\nRaw Features:")
    for feat in ["Planting_Precip", "Growing_GDD"]:
        print(f"\n  {feat}:")
        for lag in range(4):
            print(f"    Lag {lag}: {avg_acf[feat][lag]:.4f}")

    print("\nLog-transformed Features:")
    for feat in ["Planting_Precip_Log", "Growing_GDD_Log"]:
        print(f"\n  {feat}:")
        for lag in range(4):
            print(f"    Lag {lag}: {avg_acf[feat][lag]:.4f}")

    # 2x2 grid: top row = raw, bottom row = log
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, feat in enumerate(["Planting_Precip", "Growing_GDD"]):
        lags = range(len(avg_acf[feat]))
        axes[0, i].bar(lags, avg_acf[feat], alpha=0.7, color=f"C{i}")
        axes[0, i].axhline(y=0,    color="black", linestyle="-",  alpha=0.3)
        axes[0, i].axhline(y=0.2,  color="red",   linestyle="--", alpha=0.5, label="±0.2")
        axes[0, i].axhline(y=-0.2, color="red",   linestyle="--", alpha=0.5)
        axes[0, i].set_title(f"ACF: {feat} (Raw)")
        axes[0, i].set_xlabel("Lag")
        axes[0, i].set_ylabel("Autocorrelation")
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].legend()
        axes[0, i].set_xticks(lags)

    for i, feat in enumerate(["Planting_Precip_Log", "Growing_GDD_Log"]):
        lags = range(len(avg_acf[feat]))
        axes[1, i].bar(lags, avg_acf[feat], alpha=0.7, color=f"C{i}")
        axes[1, i].axhline(y=0,    color="black", linestyle="-",  alpha=0.3)
        axes[1, i].axhline(y=0.2,  color="red",   linestyle="--", alpha=0.5, label="±0.2")
        axes[1, i].axhline(y=-0.2, color="red",   linestyle="--", alpha=0.5)
        axes[1, i].set_title(f"ACF: {feat} (Log)")
        axes[1, i].set_xlabel("Lag")
        axes[1, i].set_ylabel("Autocorrelation")
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()
        axes[1, i].set_xticks(lags)

    plt.suptitle("ACF Analysis: Raw vs Log-transformed Weather Features", fontsize=16)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "weather_acf_analysis_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n  Saved {out}")
    plt.show()


# ===========================================================================
# 3.3  Data Quality & Validation
# ===========================================================================

def run_data_quality_report(df):
    """
    Comprehensive validation report: geometric features, weather coverage,
    soil match rate, and log-transformation effectiveness.
    """
    print("\n=== Data Quality & Validation Report ===")

    # Year / polygon / county coverage
    print(f"\n  Year range:              {df['Year'].min()}-{df['Year'].max()}")
    print(f"  Unique polygons (CSBID): {df['CSBID'].nunique():,}")
    print(f"  Unique counties:         {df['CNTYFIPS'].nunique()}")
    print(f"  Records per polygon:     {len(df) / df['CSBID'].nunique():.1f} (avg)")

    # Geometric feature verification
    print("\n  Geometric features:")
    for col in ["INSIDE_X", "INSIDE_Y", "Longitude_Norm", "Latitude_Norm"]:
        if col in df.columns:
            print(f"    {col}: {df[col].min():.3f} to {df[col].max():.3f}  "
                  f"(null={df[col].isna().sum():,})")
        else:
            print(f"    {col}: MISSING")

    # Weather feature validation
    print("\n  Weather features (mean ± std):")
    for col in ["Planting_Precip", "Growing_GDD", "Planting_Precip_Log", "Growing_GDD_Log"]:
        if col in df.columns:
            print(f"    {col}: {df[col].mean():.2f} ± {df[col].std():.2f}  "
                  f"(null={df[col].isna().sum():,})")
        else:
            print(f"    {col}: MISSING")

    # CNTYFIPS match rate between CSB and weather
    weather_nulls = df["Planting_Precip"].isna().sum()
    match_rate    = (1 - weather_nulls / len(df)) * 100
    print(f"\n  Weather county-year match rate: {match_rate:.1f}%")

    # Soil feature validation
    soil_cols = ["slopegradwta", "aws050wta", "aws0100wta", "drainage_numeric",
                 "drclassdcd", "hydro_group_primary"]
    present_soil = [c for c in soil_cols if c in df.columns]
    if present_soil:
        print("\n  Soil features:")
        for col in present_soil:
            non_null = df[col].notna().sum()
            print(f"    {col}: {non_null:,} non-null ({non_null/len(df)*100:.1f}%)")
        if "drclassdcd" in df.columns:
            print("\n  Drainage class distribution:")
            for cls, cnt in df["drclassdcd"].value_counts().head(6).items():
                print(f"      {cls}: {cnt:,}")
    else:
        print("\n  Soil features: NOT in dataset (csbid_mukey_mapping.csv may be missing)")

    # Log transformation effectiveness
    print("\n  Log transformation effectiveness:")
    for raw, log in [("Planting_Precip", "Planting_Precip_Log"),
                     ("Growing_GDD",     "Growing_GDD_Log")]:
        if raw in df.columns and log in df.columns:
            print(f"    {raw}: skewness {df[raw].skew():.3f} -> {df[log].skew():.3f}  |  "
                  f"std {df[raw].std():.2f} -> {df[log].std():.2f}")

    # Missing data summary
    print("\n  Missing value counts (top columns):")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    if not null_counts.empty:
        for col, cnt in null_counts.head(10).items():
            print(f"    {col}: {cnt:,}")
    else:
        print("    None — dataset is complete.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    df = load_processed_data()

    print("\n=== Starting Feature Exploration ===\n")

    # 3.1 Time Series (6 plots)
    plot_crop_types_timeseries(df)
    plot_weather_raw_timeseries(df)
    plot_weather_log_timeseries(df)
    plot_weather_comparison(df)
    plot_normalized_comparison(df)
    plot_distribution_comparison(df)

    # 3.2 ACF Analysis
    plot_acf_analysis(df)

    # 3.3 Data Quality
    run_data_quality_report(df)

    print("\n=== Feature Exploration Complete ===")
    print(f"All outputs saved to {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. crop_types_timeseries.png")
    print("  2. weather_features_timeseries_raw.png")
    print("  3. weather_features_timeseries_log.png")
    print("  4. weather_features_comparison.png")
    print("  5. weather_features_normalized_comparison.png")
    print("  6. weather_features_distributions.png")
    print("  7. weather_acf_analysis_comparison.png")


if __name__ == "__main__":
    main()
