# Data Guide

This repository does **not** ship the full USDA Crop Sequence Boundaries (CSB) geodatabases. After cloning, you must download raw inputs locally and run the pipeline notebooks/scripts to regenerate interim and processed tables.

For field-level descriptions of CSB and gSSURGO soil attributes, see [`Data_description.md`](Data_description.md).

---

## Directory Layout

```
data/
├── raw/           # USDA CSB geodatabases (local only — never committed)
├── interim/       # Per-window NY CSB subsets (feather/parquet/csv)
├── processed/     # Merged + feature-engineered model tables
├── weather/       # HDD text files + combined county weather feather
└── soil/          # gSSURGO features + CSBID→MUKEY mapping
```

Paths are centralized in [`src/config.py`](../src/config.py).

---

## What Goes on GitHub

| Asset | Approx. size | Policy |
|-------|-------------|--------|
| `raw/NationalCSB_*` GDB folders | 5–8 GB each | **Never commit** — download from USDA (see below) |
| `soil/csbid_mukey_checkpoint.csv` | ~1.4 GB | **Excluded** (`.gitignore`) — regenerable API checkpoint |
| `interim/csb_*/ny_csb.feather_*` | 330–470 MB each | **Git LFS** (`*.feather`) |
| `processed/processed_dataset.parquet` | ~130 MB | **Git LFS** (`*.parquet`) |
| `soil/csbid_mukey_mapping.csv` | ~33 MB | Commit or LFS (regenerable via `src/generate_soil_mapping.py`) |
| `soil/soil_features_NY.csv` | ~170 MB | Local / LFS — derived from gSSURGO |
| `weather/HDD/*.txt` | Small | **Commit** |
| `weather/ny_weather_combined.feather` | ~3 MB | **Git LFS** |
| `output/*.parquet`, `output/*.pth` | Varies | **Excluded** or LFS — regenerate from `src/` scripts |

Large processed artifacts are tracked via **Git LFS** in this repo (not hosted on Zenodo).

---

## 1. Download Raw CSB (required)

**Official source:** [USDA NASS Crop Sequence Boundaries](https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/)

Download the 8-year Geodatabase (`.gdb`) releases you need and extract under `data/raw/`. This project uses rolling windows from 2008–2024; at minimum you need:

| Window | Download label on NASS site | Expected folder | GDB file |
|--------|----------------------------|-----------------|----------|
| 2008–2015 | Crop Sequence Boundaries 2008–2015 | `NationalCSB_2008-2015_rev23/` | `CSB0815.gdb` |
| 2009–2016 | Crop Sequence Boundaries 2009–2016 | `NationalCSB_2009-2016_rev23/` | `CSB0916.gdb` |
| 2016–2023 | Crop Sequence Boundaries 2016–2023 | `NationalCSB_2016-2023_rev23/` | `CSB1623.gdb` |
| 2017–2024 | Crop Sequence Boundaries 2017–2024 | `NationalCSB_2017-2024_rev23/` | `CSB1724.gdb` |

> **Naming note:** NASS folder names use the 8-year span (e.g. `2009-2016`). Our interim folders use the end-year pair label (e.g. `csb_20092016/`). See notebook `01_csb_data_download.ipynb` for the exact mapping logic.

**Algorithm reference:** [USDA-REE-NASS/crop-sequence-boundaries](https://github.com/USDA-REE-NASS/crop-sequence-boundaries)

**Disk space:** allow ~15–20 GB for two full CONUS GDB releases plus derived feathers.

---

## 2. Weather Data

| File | Location | Source |
|------|----------|--------|
| HDD county/month text files | `weather/HDD/` | NOAA/NCEI heating & cooling degree-day summaries (committed) |
| County JSON shares | `weather/json/` | Team-derived monthly weather extracts |
| Combined feather | `weather/ny_weather_combined.feather` | Built in `notebooks/04_csb_weather_analysis.ipynb` |

---

## 3. Soil Data (gSSURGO)

| File | Location | How to obtain |
|------|----------|---------------|
| `soil_features_NY.csv` | `soil/` | gSSURGO NY — polygon join of MUPOLYGON + muaggatt (see `Data_description.md`) |
| `csbid_mukey_mapping.csv` | `soil/` | Run `python src/generate_soil_mapping.py` (USDA Soil Data Access API) |
| `csbid_mukey_checkpoint.csv` | `soil/` | Auto-written during mapping; **not committed**; delete to restart from scratch |

Soil integration notebook: `notebooks/05_csb_soil_integration.ipynb`

---

## 4. Regeneration Pipeline

Run from the **repository root** after `pip install -r requirements.txt` (or `mamba env create -f environment.yml`).

### Step A — CSB ingest (notebooks)

| Step | Notebook | Output |
|------|----------|--------|
| A1 | `notebooks/01_csb_data_download.ipynb` | `interim/csb_<window>/` feathers per rolling window |
| A2 | `notebooks/02_csb_data_transform.ipynb` | Additional NY subsets (optional windows) |
| A3 | `notebooks/03_csb_combine_analysis.ipynb` | Rotation EDA on combined windows |

For the classification pipeline you need these three interim feathers (already produced in a full run):

- `interim/csb_20082015/ny_csb.feather_20082015`
- `interim/csb_20092016/ny_csb.feather_20092016` *(2016 bridge year only)*
- `interim/csb_20172024/ny_csb.feather_20172024`

### Step B — Weather & soil (notebooks)

| Step | Notebook | Output |
|------|----------|--------|
| B1 | `notebooks/04_csb_weather_analysis.ipynb` | `weather/ny_weather_combined.feather` |
| B2 | `notebooks/05_csb_soil_integration.ipynb` | Soil joins on merged acreage tables |

### Step C — Feature engineering & models (`src/`)

```bash
# Crop-type classification features (5-class)
python src/generate_soil_mapping.py      # once, if mapping missing (~hours; resumable)
python src/feature_engineering.py        # → data/processed/processed_dataset.parquet
python src/model_baseline.py             # → output/ model plots & CSVs

# Rotation strategy (3-class)
python src/rot_strategy_processing.py           # → output/rot_strategy_labeled.parquet
python src/rot_strategy_feature_engineering.py  # → output/rot_strategy_features.parquet
python src/rot_strategy_model_baseline.py       # → output/ confusion matrices, CSVs
python src/deep_rot_strategy.py                 # → output/ LSTM results (optional)
```

Equivalent narrative notebooks: `06_crop_type_classification.ipynb`, `07_rotation_strategy_models.ipynb`.

### Step D — Legacy county/soil scripts (optional)

R scripts and county-level merges live in `scripts/data_pipeline/` (see `crosswalk_build.py`, `merge_soil_with_crop_acreage.py`, etc.).

---

## 5. Minimum Reproducibility Path

If you only want to **re-run models** (not rebuild from raw GDB):

1. Pull Git LFS objects: `git lfs pull`
2. Confirm these exist:
   - `data/processed/processed_dataset.parquet`
   - `output/rot_strategy_features.parquet`
3. Run `python src/model_baseline.py` or `python src/rot_strategy_model_baseline.py`

Scripts subsample to 50K–100K rows by default for runtime; set `SAMPLE_SIZE = None` in the script for full-data training.

---

## 6. Licenses & Attribution

- **USDA CSB / CDL:** Public domain; follow [NASS terms](https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/) for redistribution.
- **gSSURGO:** USDA NRCS; attribute NRCS when publishing maps or derivatives.
- **Code in this repo:** See root `LICENSE` (to be added in Phase 4).
