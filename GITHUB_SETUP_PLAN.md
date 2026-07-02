# GitHub Repository Setup Plan

**Master repo:** local clone of this repository  
**Target GitHub repo:** [Acreage-Forecasting-and-Uncertainty-Quantification-in-NY-State](https://github.com/restinghouse0203/Acreage-Forecasting-and-Uncertainty-Quantification-in-NY-State)  
**Merged source:** `ny-ag-acreage-prediction-NISS-project` (legacy NISS repo, now merged into this tree)  
**Career profile:** `acreage-estimation-capstone.yaml` (external job-app portfolio entry)

---

## Project Summary

Cornell MPS capstone (Spring 2026): predict field-level crop planting strategy and county/state acreage from 1.55M USDA Crop Sequence Boundaries (CSB) polygons (2008–2024), with uncertainty quantification.

| Task | Best model | Key metric |
|------|------------|------------|
| Planting strategy (3-class) | CatBoost | 87% acc, 96.9% macro ROC AUC |
| Planting strategy (3-class) | LSTM | 91.9% acc |
| Crop type (5-class) | CatBoost | >91% acc, 96% F1 |
| Acreage regression | Per-crop Random Forest | R² 0.82–0.98; 94.9% PI coverage |

**Team:** Tony Au, Wenlin Huang, Sirui Cao, Yuheng Shen, Yiyou Jin  
**Advisors:** Luca Sartore (USDA/NISS), David Ruppert (Cornell)

---

## Target Repository Structure

```
mps project/                          # ← master local repo
├── GITHUB_SETUP_PLAN.md              # this file
├── README.md                         # public landing page (Phase 4)
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitattributes                    # LFS rules
├── .gitignore
│
├── docs/
│   ├── report/                       # PDF capstone report
│   ├── poster/                       # poster PPTX + PDF export
│   ├── slides/                       # presentation PPTX + PDF export
│   ├── admin/                        # peer review, project charter
│   ├── figures/                      # curated plots for README
│   ├── html_exports/                 # notebook HTML exports
│   └── technical_notes/              # FEATURE_ENGINEERING_SUMMARY, etc.
│
├── data/
│   ├── README.md                     # how to download/regenerate data
│   ├── raw/                          # USDA GDB metadata (not full GDB in git)
│   ├── interim/                      # per-window CSB subsets (csb_20082015 …)
│   ├── processed/                    # merged + feature-engineered tables
│   ├── weather/                      # HDD, feather, JSON shares
│   └── soil/                         # soil features, mukey mapping
│
├── notebooks/                        # numbered pipeline narrative
│   ├── 01_csb_data_download.ipynb
│   ├── 02_csb_data_transform.ipynb
│   ├── 03_csb_combine_analysis.ipynb
│   ├── 04_csb_weather_analysis.ipynb
│   ├── 05_csb_soil_integration.ipynb
│   ├── 06_crop_type_classification.ipynb
│   └── 07_rotation_strategy_models.ipynb
│
├── src/                              # importable pipeline modules (from NISS project)
│   ├── data/                         # (future) ingest helpers
│   ├── features/
│   ├── models/
│   └── dev/                          # inspect_*, check_* utilities
│
├── scripts/                          # thin CLI + legacy pipeline scripts
│   ├── data_pipeline/                # py_script originals
│   └── dev/
│
├── output/                           # regenerated model artifacts
│   └── README.md
│
├── archive/                          # superseded copies (not for GitHub)
│   ├── root_notebooks_duplicates/
│   ├── nested_niss_project/
│   └── bin/
│
└── tests/                            # (future) unit tests
```

---

## Phase 0 — Local Consolidation (file moves only)

> **Status:** ✅ Complete (verified 2026-07-02). File moves/copies only — no source edits.

### 0.1 Organize master repo folders

- [x] Create target directory tree (`docs/`, `notebooks/`, `src/`, `scripts/`, `archive/`, etc.)
- [x] Move root-level notebooks → `notebooks/` with numbered names
- [x] Move `script/` notebooks → `notebooks/` (deduplicate into `archive/`)
- [x] Move `doc/` → `docs/report|poster|slides|admin`
- [x] Move `pdf/` → `docs/html_exports/`
- [x] Move `raw data/` → `data/raw/`
- [x] Reorganize `data/` into `interim/`, `processed/`, `weather/`, `soil/`
- [x] Move `py_script/` → `scripts/data_pipeline/`
- [x] Move `weather_data_to_share/` → `data/weather/json/`
- [x] Move `bin/` → `archive/bin/`
- [x] Move nested `ny-ag-acreage-prediction-NISS-project/` → `archive/nested_niss_project/`
- [x] Move loose admin files → `docs/admin/`

### 0.2 Merge NISS project into master

Source: `ny-ag-acreage-prediction-NISS-project` (merged; see `archive/nested_niss_project/` for superseded copy)

- [x] Copy `src/*.py` → `src/` (and `src/dev/` for inspect/check scripts)
- [x] Copy `analysis_notebook.ipynb` → `notebooks/06_crop_type_classification.ipynb`
- [x] Copy `Rot_strategy_classification.ipynb` → `notebooks/07_rotation_strategy_models.ipynb`
- [x] Merge `output/` → `output/` (NISS outputs take precedence on conflict)
- [x] Copy `data/ny_csb.*` → `data/interim/ny_csb_20172024/`
- [x] Copy `README.md`, `FEATURE_ENGINEERING_SUMMARY.md`, `PARALLELIZATION_SUMMARY.md` → `docs/technical_notes/`
- [x] Copy `.gitattributes` → repo root
- [x] Move `src/output/` from NISS → `archive/src_output_duplicate/`

### 0.3 Post-merge verification

- [x] Confirm no stray notebooks remain at repo root (except plan/README)
- [x] Confirm `src/` has all 20+ Python modules from NISS project (14 in `src/`, 7 in `src/dev/`)
- [x] Confirm `output/` contains rotation + classification plots (64 artifacts)
- [x] Confirm `docs/report/` has capstone PDF
- [x] Skim `archive/` — superseded copies only; canonical files in main tree

**Note:** `data/.git` nested repo removed during Phase 2 (2026-07-02).

---

## Phase 1 — Path & Config Cleanup (requires editing)

> **Status:** ✅ Complete (verified 2026-07-02).

- [x] Create `src/config.py` with `PROJECT_ROOT = Path(__file__).resolve().parents[1]`
- [x] Replace hardcoded absolute `data/` paths in all `src/*.py`
- [x] Replace hardcoded paths in notebooks (or add first cell with `PROJECT_ROOT` setup)
- [x] Update `DATA_DIR` in `model_baseline.py`, `feature_engineering.py`, etc.
- [x] Smoke-test: `python src/rot_strategy_model_baseline.py` from repo root

**Files with known hardcoded paths:**

| File | Current path reference |
|------|------------------------|
| `src/model_baseline.py` | `DATA_DIR = ".../mps project/data"` |
| `src/feature_engineering.py` | same pattern |
| `src/rot_strategy_*.py` | `output/` relative (OK) |
| `src/generate_soil_mapping.py` | checkpoint paths |

---

## Phase 2 — Dependencies & Git Hygiene

> **Status:** ✅ Complete (verified 2026-07-02).

- [x] Create `requirements.txt`:
  ```
  geopandas pandas pyarrow catboost scikit-learn matplotlib seaborn torch fiona shapely pyproj
  ```
- [x] Create `environment.yml` for conda/mamba (Apple Silicon geopandas)
- [x] Create `.gitignore`:
  ```
  .venv/
  __pycache__/
  .DS_Store
  catboost_info/
  data/raw/NationalCSB*/
  data/soil/csbid_mukey_checkpoint.csv
  output/*.parquet
  output/*.pth
  archive/
  .cursor/
  ```
- [x] Extend `.gitattributes` for LFS:
  ```
  *.parquet filter=lfs diff=lfs merge=lfs -text
  *.feather filter=lfs diff=lfs merge=lfs -text
  *.pth filter=lfs diff=lfs merge=lfs -text
  ```
- [x] Initialize git at repo root (if not already): `git init`
- [x] Install Git LFS: `git lfs install`

---

## Phase 3 — Data Policy (what goes on GitHub)

> **Status:** ✅ Complete (verified 2026-07-02). Zenodo upload deferred; interim sample deferred pending discussion.

| Asset | Size | Action |
|-------|------|--------|
| Raw CSB GDB | Multi-GB | **Never commit** — link in `data/README.md` |
| `csbid_mukey_checkpoint.csv` | ~1.4 GB | **Exclude** — regenerable checkpoint |
| `processed_dataset.parquet` | ~130 MB | Git LFS |
| `rot_strategy_features.parquet` | ~40 MB | Git LFS (in `output/`, excluded from plain git) |
| `ny_csb` subsets (csv/feather) | 50–150 MB | Git LFS |
| HDD weather text files | Small | Commit |
| `rotation_summary.csv`, model CSVs | Small | Commit |
| Model weights `.pth` | Small | LFS or exclude |

- [x] Write `data/README.md` with USDA CSB download URL and regeneration steps
- [x] ~~(Optional) Upload large processed files to Zenodo~~ — **skipped**; using Git LFS only
- [ ] Create `data/interim/sample/` with 1000-row parquet for demo/CI — **deferred** (see discussion below)

---

## Phase 4 — README & Documentation

> **Status:** ✅ Complete (verified 2026-07-02).

- [x] Write root `README.md` (capstone title, results table, pipeline diagram, quick start)
- [x] Add hero image from `docs/figures/` (pick 1 rotation map + 1 confusion matrix)
- [x] Export poster and slides PPTX → PDF into `docs/poster/` and `docs/slides/`
- [x] Curate 5–8 key plots from `output/` → `docs/figures/`
- [x] Write `output/README.md`: "Regenerate via `python src/rot_strategy_model_baseline.py`"
- [x] Add `CONTRIBUTORS.md` crediting all 5 teammates + advisors
- [x] Fix filename typo: `MPS Acreage Forcasting report.pdf` → `MPS_Acreage_Forecasting_Report.pdf`

**README sections checklist:**

- [x] Title + one-line description
- [x] Results-at-a-glance table
- [x] Links to report / poster / slides PDFs
- [x] Mermaid pipeline diagram
- [x] Quick start (clone, env, run)
- [x] Repository layout table
- [x] Data sources (CSB, weather, gSSURGO)
- [x] Methods summary (3 bullets)
- [x] Reproducibility notes (50K sample constraint)
- [x] Future work
- [x] License

---

## Phase 5 — GitHub Publish

> **Status:** ✅ Mostly complete (2026-07-02). Topics, pin, and release require `gh auth login`.

- [x] Decide canonical repo name: `Acreage-Forecasting-and-Uncertainty-Quantification-in-NY-State`
- [ ] Archive or redirect old `ny-ag-acreage-prediction-NISS-project` repo (manual on GitHub)
- [x] Update `acreage-estimation-capstone.yaml` `link:` to canonical URL
- [x] First commit (exclude `archive/`, `.venv/`, large checkpoints)
- [x] Push to GitHub (`force-with-lease` replaced old remote layout)
- [ ] Add repo topics: `machine-learning`, `agriculture`, `geospatial`, `uncertainty-quantification`, `catboost`, `cornell`
- [ ] Pin repo on GitHub profile
- [ ] Create release `v1.0-capstone` with report PDF attached

---

## Phase 6 — Code Quality (optional, post-publish)

- [ ] Extract notebook logic into `src/` subpackages (`data/`, `features/`, `models/`)
- [ ] Add `scripts/run_crop_models.py` and `scripts/run_rotation_models.py` CLI wrappers
- [ ] Add `tests/test_feature_engineering.py`
- [ ] Add GitHub Actions smoke test on sample data
- [ ] Add `CITATION.cff` for academic citation

---

## File Mapping Reference

### Notebooks (master ← sources)

| Target | Source (preferred) |
|--------|-------------------|
| `notebooks/01_csb_data_download.ipynb` | root `CSB_data_download.ipynb` |
| `notebooks/02_csb_data_transform.ipynb` | root `CSB_data_transform.ipynb` |
| `notebooks/03_csb_combine_analysis.ipynb` | root `CSB_combine_analysis.ipynb` |
| `notebooks/04_csb_weather_analysis.ipynb` | `script/CSB_weather_analysis.ipynb` |
| `notebooks/05_csb_soil_integration.ipynb` | root `csb_soil.ipynb` |
| `notebooks/06_crop_type_classification.ipynb` | NISS `analysis_notebook.ipynb` |
| `notebooks/07_rotation_strategy_models.ipynb` | NISS `Rot_strategy_classification.ipynb` |

### src/ modules (from NISS project)

| Module | Purpose |
|--------|---------|
| `feature_engineering.py` | Crop type features, lags, county aggregates |
| `model_baseline.py` | KNN / CatBoost crop classification |
| `model_advanced.py` | CNN / RNN geometric models |
| `rot_strategy_processing.py` | 3-class rotation strategy labels |
| `rot_strategy_feature_engineering.py` | Rotation model features |
| `rot_strategy_model_baseline.py` | KNN / CatBoost rotation models |
| `deep_rot_strategy.py` | LSTM rotation model |
| `generate_soil_mapping.py` | CSBID → mukey soil API mapping |
| `eda_rotation.py` | Rotation EDA plots |
| `csb_classification_data_*.py` | Classification data prep |
| `src/dev/inspect_*.py` | Data inspection utilities |

### scripts/data_pipeline/ (from py_script/)

| Script | Purpose |
|--------|---------|
| `build_csb_crop_acreage_long.py` | Long-format acreage table |
| `crosswalk_build.py` | County/crop crosswalk |
| `merge_soil_with_crop_acreage.py` | Soil + acreage merge |
| `postprocess_county_merge.py` | County-level postprocessing |
| `eda_county_analysis.py` | County EDA |

---

## Notes & Warnings

1. **Paths will break** after Phase 0 until Phase 1 is done. Run scripts from repo root only after updating `config.py`.
2. **`archive/` is local-only** — do not push to GitHub.
3. **Two GitHub repos exist today** — consolidate to one canonical URL before sharing on resume.
4. **Team attribution** — confirm with teammates before making repo public.
5. **USDA data license** — CSB data has USDA terms; code can be MIT, data links only.

---

## Progress Log

| Date | Action |
|------|--------|
| 2026-06-16 | Created `GITHUB_SETUP_PLAN.md` |
| 2026-06-16 | Phase 0 complete: reorganized master repo + merged NISS project (moves only) |
| 2026-07-02 | Phase 0 re-verified: idempotent NISS merge pass, all 0.1–0.3 checks passed |
| 2026-07-02 | Phase 1 complete: `src/config.py`, all `src/*.py` + notebooks updated; smoke test passed |
| 2026-07-02 | Phase 2 complete: `requirements.txt`, `environment.yml`, `.gitignore`, LFS; root git init; `data/.git` removed |
| 2026-07-02 | Phase 3 complete: `data/README.md` (download URLs, regeneration steps, LFS policy); Zenodo skipped; sample deferred |
| 2026-07-02 | Phase 4 complete: root README, CONTRIBUTORS, output/README, 8 curated figures, poster/slides PDFs, report rename |
| 2026-07-02 | Path sanitization: cleared notebook outputs, fixed absolute paths; expanded LFS patterns for `*.feather_*` |
| 2026-07-02 | Phase 5: initial commit + push to GitHub (4.8 GB LFS); career profile link updated |

### Current layout snapshot (after Phase 0)

```
mps project/
├── README.md
├── LICENSE
├── CONTRIBUTORS.md
├── GITHUB_SETUP_PLAN.md
├── .gitattributes
├── docs/          report, poster, slides, admin, figures, html_exports, technical_notes
├── notebooks/     10 notebooks (01–07 + supplementary 01b–04b)
├── src/           14 modules + dev/ (7 inspect/check utilities)
├── scripts/       data_pipeline/ (8 scripts) + dev/
├── data/          raw, interim (10 CSB windows + ny_csb_20172024), processed, weather, soil
├── output/        64 model plots, CSVs, parquets
├── archive/       superseded copies (local only)
└── tests/         (empty, for Phase 6)
```
