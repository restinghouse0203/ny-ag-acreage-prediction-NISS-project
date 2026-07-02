# Model Outputs

This directory holds plots, CSV summaries, and model artifacts produced by the `src/` pipeline. Files here are **regenerated locally** — large `.parquet` and `.pth` files are excluded from plain git (see root `.gitignore`).

## Regenerate

From the repository root, after installing dependencies and ensuring input data exist (see [`data/README.md`](../data/README.md)):

```bash
# Rotation strategy (3-class) — CatBoost + KNN
python src/rot_strategy_processing.py
python src/rot_strategy_feature_engineering.py
python src/rot_strategy_model_baseline.py

# Optional: LSTM sequence model
python src/deep_rot_strategy.py

# Crop type classification (5-class)
python src/feature_engineering.py
python src/model_baseline.py

# Optional: geometric deep learning models
python src/model_advanced.py
```

Equivalent narrative: `notebooks/06_crop_type_classification.ipynb`, `notebooks/07_rotation_strategy_models.ipynb`.

## Key artifacts

| File | Produced by |
|------|-------------|
| `rot_strategy_*.png`, `rot_strategy_model_comparison.csv` | `rot_strategy_model_baseline.py` |
| `rot_strategy_lstm_*.png`, `rot_strategy_lstm_results.csv` | `deep_rot_strategy.py` |
| `*_confusion_matrix.png`, `model_comparison_summary.csv` | `model_baseline.py` |
| `geometric_*.png`, `geometric_models_comparison.csv` | `model_advanced.py` |
| `rotation_summary.csv`, `crop_types_timeseries.png` | `csb_classification_data_processing.py` / EDA scripts |

Curated figures for the public README live in [`docs/figures/`](../docs/figures/).
