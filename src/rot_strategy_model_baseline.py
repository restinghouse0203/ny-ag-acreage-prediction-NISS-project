"""
rot_strategy_model_baseline.py
================================
Section 4 of Rot_strategy_classification.ipynb.

Trains KNN and CatBoost classifiers to predict the 3-class rotation
strategy label (Continuous / Rotation / Complex/Mixed) from polygon-level
features.

Key differences from model_baseline.py:
  - Target: Strategy_Category (3 classes) rather than 5-class crop type.
  - Train/test split: random 80/20 on polygons (not by year).
  - Features include extended lag history (Lag1-5) and rotation pattern
    features (Crop_Diversity_L5, Is_Alternating_L1_L3, etc.).
  - No soil features (excluded per plan).

Input
-----
  output/rot_strategy_features.parquet — from rot_strategy_feature_engineering.py

Outputs
-------
  output/rot_strategy_confusion_<model>.png
  output/rot_strategy_roc_curves.png
  output/rot_strategy_feature_importance.png
  output/rot_strategy_model_comparison.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available — install with: pip install catboost")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from config import OUTPUT_DIR

INPUT_PARQUET = OUTPUT_DIR / "rot_strategy_features.parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE   = 100_000   # subsample for speed; set None to use full dataset
RANDOM_STATE  = 42

# Strategy label mapping
STRATEGY_MAPPING = {"Continuous": 0, "Rotation": 1, "Complex/Mixed": 2}
STRATEGY_NAMES   = {v: k for k, v in STRATEGY_MAPPING.items()}


# ---------------------------------------------------------------------------
# 1. Load and preprocess
# ---------------------------------------------------------------------------

def load_data():
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(
            f"{INPUT_PARQUET} not found. Run rot_strategy_feature_engineering.py first."
        )
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded features: {df.shape}")
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        # DataFrame.sample() has no stratify argument; use train_test_split to
        # draw a stratified subsample of exactly SAMPLE_SIZE rows.
        df, _ = train_test_split(
            df,
            train_size=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
            stratify=df["Strategy_Category"],
        )
        print(f"Sampled to {len(df):,} rows (stratified by Strategy_Category).")
    return df


def preprocess(df):
    """Encode target, build feature lists, split train/test."""
    print("Preprocessing...")

    # Target
    df = df.dropna(subset=["Strategy_Category"])
    df["Target"] = df["Strategy_Category"].map(STRATEGY_MAPPING)
    df = df.dropna(subset=["Target"])
    df["Target"] = df["Target"].astype(int)

    # Encode lag crop codes as numeric
    for i in range(1, 6):
        col = f"Crop_Lag{i}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    # Numeric features
    numeric_features = []
    candidates_num = [
        "Crop_Lag1", "Crop_Lag2", "Crop_Lag3", "Crop_Lag4", "Crop_Lag5",
        "Crop_Diversity_L5", "Crop_Changed_L1_L2", "Is_Alternating_L1_L3",
        "Continuity_Streak",
        "Mean_Planting_Precip", "Std_Planting_Precip",
        "Mean_Growing_GDD",     "Std_Growing_GDD",
        "Longitude_Norm", "Latitude_Norm",
        "County_Crop_Diversity", "County_Avg_Field_Size",
        "CNTYFIPS",
    ]
    for col in candidates_num:
        if col in df.columns and df[col].notna().any():
            numeric_features.append(col)

    # Categorical features
    cat_candidates = ["County_Dominant_Strategy"]
    categorical_features = [c for c in cat_candidates if c in df.columns and df[c].notna().any()]

    # Fill missing numeric with median
    for col in numeric_features:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical with "Unknown"
    for col in categorical_features:
        df[col] = df[col].fillna("Unknown").astype(str)

    print(f"  Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"  Categorical features: {categorical_features}")
    print(f"\n  Target distribution:")
    for num, name in STRATEGY_NAMES.items():
        cnt = (df["Target"] == num).sum()
        print(f"    {name} ({num}): {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # Train / test split by polygon (random 80/20, stratified)
    X = df[numeric_features + categorical_features]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {len(X_train):,}   Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


# ---------------------------------------------------------------------------
# 2. Model training
# ---------------------------------------------------------------------------

def train_knn(X_train, X_test, y_train, y_test, numeric_features, categorical_features):
    print("\n" + "=" * 50)
    print("TRAINING K-NEAREST NEIGHBORS (k=5)")
    print("=" * 50)

    transformers = [("num", StandardScaler(), numeric_features)]
    if categorical_features:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ))
    preprocessor = ColumnTransformer(transformers)

    # Stratified subsample for KNN efficiency
    if len(X_train) > 30_000:
        from sklearn.model_selection import train_test_split as tts
        X_knn, _, y_knn, _ = tts(
            X_train, y_train, train_size=30_000, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_knn, y_knn = X_train, y_train

    pipe = Pipeline([("preprocessor", preprocessor),
                     ("classifier", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])
    pipe.fit(X_knn, y_knn)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy: {acc:.4f}")
    return {"predictions": y_pred, "probabilities": y_proba, "model": pipe}


def train_catboost(X_train, X_test, y_train, y_test, categorical_features):
    print("\n" + "=" * 50)
    print("TRAINING CATBOOST")
    print("=" * 50)

    if not CATBOOST_AVAILABLE:
        print("Skipping — CatBoost not installed.")
        return None

    # Subsample for speed
    if len(X_train) > 30_000:
        from sklearn.model_selection import train_test_split as tts
        X_cat, _, y_cat, _ = tts(
            X_train, y_train, train_size=30_000, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_cat, y_cat = X_train, y_train

    X_cat  = X_cat.copy()
    X_test = X_test.copy()
    for col in categorical_features:
        if col in X_cat.columns:
            X_cat[col]  = X_cat[col].fillna("Unknown").astype(str)
            X_test[col] = X_test[col].fillna("Unknown").astype(str)

    all_cols       = list(X_cat.columns)
    cat_feat_idx   = [all_cols.index(c) for c in categorical_features if c in all_cols]

    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=4,
        verbose=False,
        random_state=RANDOM_STATE,
        auto_class_weights="Balanced",
        cat_features=cat_feat_idx if cat_feat_idx else None,
    )
    model.fit(X_cat, y_cat)
    y_pred  = model.predict(X_test).flatten().astype(int)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"CatBoost Accuracy: {acc:.4f}")
    return {"predictions": y_pred, "probabilities": y_proba, "model": model}


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm     = confusion_matrix(y_test, y_pred)
    labels = [STRATEGY_NAMES[i] for i in sorted(STRATEGY_NAMES)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Count"})
    plt.title(f"Confusion Matrix — {model_name} (Rotation Strategy)", fontsize=13)
    plt.xlabel("Predicted Strategy")
    plt.ylabel("True Strategy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"rot_strategy_confusion_{model_name.lower()}.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()


def plot_roc_curves(results, y_test):
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_test)
    n_classes = len(STRATEGY_NAMES)
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

    fig, axes = plt.subplots(1, n_classes, figsize=(15, 5))
    for ci in range(n_classes):
        ax = axes[ci]
        for mi, (mname, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_bin[:, ci], res["probabilities"][:, ci])
            auc = roc_auc_score(y_bin[:, ci], res["probabilities"][:, ci])
            ax.plot(fpr, tpr, color=colors[mi], lw=2, label=f"{mname} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"ROC — {STRATEGY_NAMES[ci]}", fontsize=11)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle("ROC Curves by Strategy Class", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rot_strategy_roc_curves.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()


def plot_feature_importance(results, numeric_features):
    catboost_res = results.get("CatBoost")
    if catboost_res is None:
        return
    model = catboost_res["model"]
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_

    # Use the feature names the model was actually trained on (includes
    # categorical features) rather than the numeric-only list, which would
    # have a different length and cause a ValueError in pd.DataFrame().
    if hasattr(model, "feature_names_"):
        feat_names = list(model.feature_names_)
    else:
        feat_names = numeric_features[:len(importances)]

    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(fi_df["Feature"], fi_df["Importance"])
    plt.title("CatBoost Feature Importance — Rotation Strategy", fontsize=13)
    plt.xlabel("Importance")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rot_strategy_feature_importance.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.show()


def print_model_comparison(results, y_test):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY — Rotation Strategy Classification")
    print("=" * 70)

    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_test)

    rows = []
    for mname, res in results.items():
        y_pred = res["predictions"]
        acc    = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        try:
            auc = roc_auc_score(y_bin, res["probabilities"], average="macro", multi_class="ovr")
        except Exception:
            auc = float("nan")

        rows.append({
            "Model":             mname,
            "Accuracy":          f"{acc:.4f}",
            "Precision (Macro)": f"{prec:.4f}",
            "Recall (Macro)":    f"{rec:.4f}",
            "F1-Score (Macro)":  f"{f1:.4f}",
            "ROC AUC (Macro)":   f"{auc:.4f}" if not np.isnan(auc) else "N/A",
        })

        print(f"\n{'-'*20} {mname} {'-'*20}")
        print(classification_report(
            y_test, y_pred,
            target_names=[STRATEGY_NAMES[i] for i in sorted(STRATEGY_NAMES)],
            digits=4,
        ))

    comp = pd.DataFrame(rows)
    print(comp.to_string(index=False))
    out = os.path.join(OUTPUT_DIR, "rot_strategy_model_comparison.csv")
    comp.to_csv(out, index=False)
    print(f"\nSaved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()
    X_train, X_test, y_train, y_test, num_feats, cat_feats = preprocess(df)

    results = {}

    knn_res = train_knn(X_train, X_test, y_train, y_test, num_feats, cat_feats)
    if knn_res:
        results["KNN"] = knn_res

    cat_res = train_catboost(X_train, X_test, y_train, y_test, cat_feats)
    if cat_res:
        results["CatBoost"] = cat_res

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for mname, res in results.items():
        plot_confusion_matrix(y_test, res["predictions"], mname)

    if results:
        plot_roc_curves(results, y_test)
        plot_feature_importance(results, num_feats)
        print_model_comparison(results, y_test)

    print("\n=== rot_strategy_model_baseline.py complete ===")


if __name__ == "__main__":
    main()
