"""
Central path configuration for the capstone pipeline.

All scripts and notebooks should import paths from here rather than
hardcoding machine-specific absolute paths.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
WEATHER_DIR = DATA_DIR / "weather"
SOIL_DIR = DATA_DIR / "soil"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Key data files (post Phase 0 layout)
PROCESSED_DATASET = PROCESSED_DIR / "processed_dataset.parquet"
WEATHER_FEATHER = WEATHER_DIR / "ny_weather_combined.feather"
SOIL_FEATURES = SOIL_DIR / "soil_features_NY.csv"
SOIL_MAP = SOIL_DIR / "csbid_mukey_mapping.csv"
SOIL_CHECKPOINT = SOIL_DIR / "csbid_mukey_checkpoint.csv"
MERGED_DATA_DIR = PROCESSED_DIR / "merged_data"


def interim_csb_path(window: str) -> Path:
    """Path to CSB feather for a rolling window label, e.g. '20082015'."""
    return INTERIM_DIR / f"csb_{window}" / f"ny_csb.feather_{window}"


# Three feather files covering 2008-2024 with no gap (see CSB_combine_analysis.ipynb)
CSB_SOURCES = [
    {"path": interim_csb_path("20082015"), "years": list(range(2008, 2016))},
    {"path": interim_csb_path("20092016"), "years": [2016]},
    {"path": interim_csb_path("20172024"), "years": list(range(2017, 2025))},
]

CSB_FEATHER_PATHS = [src["path"] for src in CSB_SOURCES]
