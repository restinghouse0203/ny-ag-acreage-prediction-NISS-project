"""
rot_strategy_model_advanced.py
================================
Section 5 of Rot_strategy_classification.ipynb.

LSTM sequence classifier for rotation strategy prediction.

Each polygon's ordered crop sequence (2008-2024) is treated as a time
series.  For each year, a feature vector of:
    (crop_code_normalised, planting_precip, growing_gdd)
is assembled.  The LSTM reads this sequence and classifies the polygon
into one of 3 rotation strategy categories.

Architecture
------------
  Input  → LSTM (hidden_size=64, num_layers=2, dropout=0.3)
         → final hidden state
         → Linear classifier → 3-class softmax

Inputs
------
  output/rot_strategy_labeled.parquet    — polygon-level labels + CDL columns
  ny_weather_combined.feather            — monthly county weather

Outputs
-------
  output/rot_strategy_lstm_loss_curve.png
  output/rot_strategy_lstm_accuracy_curve.png
  output/rot_strategy_lstm_confusion.png
  output/rot_strategy_lstm_results.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available — install with: pip install torch")

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    confusion_matrix, roc_auc_score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from config import OUTPUT_DIR, WEATHER_FEATHER

WEATHER_PATH = WEATHER_FEATHER
INPUT_PARQUET = OUTPUT_DIR / "rot_strategy_labeled.parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HYPERPARAMS = {
    "hidden_size":    64,
    "num_layers":     2,
    "dropout":        0.3,
    "learning_rate":  0.001,
    "num_epochs":     500,
    "batch_size":     64,
    "gradient_clip":  1.0,
    "sample_size":    50_000,
    "sequence_length": 17,   # 2008-2024 = 17 years
    "input_size":     3,     # crop_code, planting_precip, growing_gdd
}

STRATEGY_MAPPING = {"Continuous": 0, "Rotation": 1, "Complex/Mixed": 2}
STRATEGY_NAMES   = {v: k for k, v in STRATEGY_MAPPING.items()}
ALL_YEARS        = list(range(2008, 2025))
TRAIN_YEARS      = list(range(2008, 2023))  # 2008-2022
VAL_YEARS        = [2023, 2024]             # 2023-2024
RANDOM_STATE     = 42


# ---------------------------------------------------------------------------
# 1. Build per-polygon sequences
# ---------------------------------------------------------------------------

def load_weather_annual():
    """Annual county-level weather: dict[(CNTYFIPS, Year)] -> (precip, gdd)."""
    if not os.path.exists(WEATHER_PATH):
        print("  Weather file not found — sequences will use zeros for weather.")
        return {}

    df = pd.read_feather(WEATHER_PATH)
    df["CNTYFIPS"]   = df["county"].str.replace("USNY", "").astype(int) % 1000
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["Year"]       = df["start_date"].dt.year
    df["Month"]      = df["start_date"].dt.month

    plant = (df[df["Month"].isin([4, 5])]
             .groupby(["Year", "CNTYFIPS"])["totr"].sum().reset_index()
             .rename(columns={"totr": "Planting_Precip"}))
    grow  = (df[df["Month"].isin([5, 6, 7, 8, 9, 10])]
             .groupby(["Year", "CNTYFIPS"])["gdd_b10"].sum().reset_index()
             .rename(columns={"gdd_b10": "Growing_GDD"}))
    annual = pd.merge(plant, grow, on=["Year", "CNTYFIPS"], how="outer").fillna(0)

    # Normalise weather per county
    scaler_p = StandardScaler()
    scaler_g = StandardScaler()
    annual["Planting_Precip"] = scaler_p.fit_transform(annual[["Planting_Precip"]])
    annual["Growing_GDD"]     = scaler_g.fit_transform(annual[["Growing_GDD"]])

    weather = {}
    for _, row in annual.iterrows():
        weather[(int(row["CNTYFIPS"]), int(row["Year"]))] = (
            float(row["Planting_Precip"]),
            float(row["Growing_GDD"]),
        )
    return weather


def build_sequences(labeled_df, weather_lookup, sample_size=None):
    """
    Build (sequence, label) pairs.

    Each sequence has shape (T, 3):
        T = HYPERPARAMS["sequence_length"] years
        Features per year: [crop_code_norm, planting_precip, growing_gdd]

    Missing crop years are filled with 0.  Returns:
        sequences — np.ndarray of shape (N, T, 3)
        labels    — np.ndarray of shape (N,)
    """
    print("Building LSTM input sequences...")

    # Drop rows without a strategy label
    df = labeled_df.dropna(subset=["Strategy_Category"]).copy()
    df = df[df["Strategy_Category"].isin(STRATEGY_MAPPING)]

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE,
                       weights=None)  # not stratified here — fast
        print(f"  Sampled to {len(df):,} polygons")

    cdl_cols_avail = {int(c.replace("CDL", "")): c
                      for c in df.columns if c.startswith("CDL")}

    # Normalise crop codes: divide by 200 to bring to ~[0, 1] range
    T = HYPERPARAMS["sequence_length"]
    sequences = np.zeros((len(df), T, 3), dtype=np.float32)
    labels    = np.zeros(len(df), dtype=np.int64)

    for idx, (_, row) in enumerate(df.iterrows()):
        county = int(row["CNTYFIPS"]) if pd.notna(row.get("CNTYFIPS")) else -1
        label  = STRATEGY_MAPPING[row["Strategy_Category"]]
        labels[idx] = label

        for ti, year in enumerate(ALL_YEARS):
            crop_code = 0.0
            if year in cdl_cols_avail:
                val = row.get(cdl_cols_avail[year])
                if pd.notna(val):
                    crop_code = float(val) / 200.0   # normalise

            precip, gdd = weather_lookup.get((county, year), (0.0, 0.0))
            sequences[idx, ti, 0] = crop_code
            sequences[idx, ti, 1] = precip
            sequences[idx, ti, 2] = gdd

    print(f"  Built {len(sequences):,} sequences of shape "
          f"{sequences.shape[1]}×{sequences.shape[2]}")
    return sequences, labels


# ---------------------------------------------------------------------------
# 2. PyTorch Dataset and Model
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class RotationDataset(Dataset):
        def __init__(self, sequences, labels):
            self.X = torch.tensor(sequences, dtype=torch.float32)
            self.y = torch.tensor(labels,    dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class RotationLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout    = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.classifier(self.dropout(h_n[-1]))


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train_lstm(sequences, labels):
    if not TORCH_AVAILABLE:
        print("PyTorch not available — skipping LSTM training.")
        return None

    hp = HYPERPARAMS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("ROTATION STRATEGY LSTM TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Hyperparameters: {hp}\n")

    # Timeline split:
    # - training timesteps:   2008-2022 (15 years)
    # - validation timesteps: 2023-2024 (2 years)
    train_steps = len(TRAIN_YEARS)
    val_steps = len(VAL_YEARS)
    total_steps = train_steps + val_steps
    if sequences.shape[1] != total_steps:
        raise ValueError(
            f"Expected sequence length {total_steps} from years 2008-2024, "
            f"got {sequences.shape[1]}."
        )

    train_sequences = sequences[:, :train_steps, :]
    val_sequences   = sequences[:, train_steps:, :]

    train_ds = RotationDataset(train_sequences, labels)
    val_ds   = RotationDataset(val_sequences, labels)

    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=hp["batch_size"])
    train_pct = (train_steps / total_steps) * 100
    val_pct = (val_steps / total_steps) * 100
    print(f"Train samples: {len(train_ds):,}  Val samples: {len(val_ds):,}")
    print(
        f"Timeline proportion -> Train (2008-2022): {train_pct:.2f}%  |  "
        f"Val (2023-2024): {val_pct:.2f}%"
    )

    # Class weights for imbalance
    class_counts = np.bincount(labels, minlength=3)
    class_weights = torch.tensor(
        1.0 / (class_counts + 1e-6), dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * 3
    class_weights = class_weights.to(device)

    model = RotationLSTM(
        input_size  = hp["input_size"],
        hidden_size = hp["hidden_size"],
        num_layers  = hp["num_layers"],
        num_classes = len(STRATEGY_NAMES),
        dropout     = hp["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(hp["num_epochs"]):
        # --- Training ---
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["gradient_clip"])
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total += len(y_batch)
        train_losses.append(total_loss / len(train_ds))
        train_accuracies.append(train_correct / train_total if train_total else 0.0)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_logits = model(X_batch)
                val_loss += criterion(val_logits, y_batch).item() * len(y_batch)
                val_correct += (val_logits.argmax(dim=1) == y_batch).sum().item()
                val_total += len(y_batch)
        val_losses.append(val_loss / len(val_ds))
        val_accuracies.append(val_correct / val_total if val_total else 0.0)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{hp['num_epochs']} | "
                  f"Train loss: {train_losses[-1]:.4f} | "
                  f"Val loss:   {val_losses[-1]:.4f} | "
                  f"Train acc: {train_accuracies[-1]:.4f} | "
                  f"Val acc: {val_accuracies[-1]:.4f}")

    # --- Validation evaluation ---
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in val_dl:
            logits = model(X_batch.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    y_val_arr   = np.array(all_labels)
    y_pred_arr  = np.array(all_preds)
    y_proba_arr = np.array(all_probs)

    acc = accuracy_score(y_val_arr, y_pred_arr)
    f1  = f1_score(y_val_arr, y_pred_arr, average="macro")
    print(f"\nValidation Accuracy: {acc:.4f}   Macro F1: {f1:.4f}")
    print(classification_report(
        y_val_arr, y_pred_arr,
        target_names=[STRATEGY_NAMES[i] for i in sorted(STRATEGY_NAMES)],
        digits=4,
    ))

    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "y_val":        y_val_arr,
        "y_pred":       y_pred_arr,
        "y_proba":      y_proba_arr,
        "accuracy":     acc,
        "f1":           f1,
    }


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def plot_loss_curve(res):
    plt.figure(figsize=(9, 5))
    plt.plot(res["train_losses"], label="Train loss", linewidth=2)
    plt.plot(res["val_losses"],   label="Val loss",   linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("LSTM Training / Validation Loss — Rotation Strategy", fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rot_strategy_lstm_loss_curve.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


def plot_accuracy_curve(res):
    plt.figure(figsize=(9, 5))
    plt.plot(res["train_accuracies"], label="Train accuracy", linewidth=2)
    plt.plot(res["val_accuracies"],   label="Val accuracy",   linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM Training / Validation Accuracy — Rotation Strategy", fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rot_strategy_lstm_accuracy_curve.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


def plot_lstm_confusion(res):
    cm     = confusion_matrix(res["y_val"], res["y_pred"])
    labels = [STRATEGY_NAMES[i] for i in sorted(STRATEGY_NAMES)]
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=labels, yticklabels=labels)
    plt.title("LSTM Confusion Matrix — Rotation Strategy", fontsize=13)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rot_strategy_lstm_confusion.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


def save_results_table(res, baseline_csv=None):
    """Append LSTM row to the baseline comparison CSV if it exists."""
    row = pd.DataFrame([{
        "Model":             "LSTM",
        "Accuracy":          f"{res['accuracy']:.4f}",
        "Macro F1":          f"{res['f1']:.4f}",
    }])

    out = os.path.join(OUTPUT_DIR, "rot_strategy_lstm_results.csv")
    row.to_csv(out, index=False)
    print(f"Saved {out}")

    # If baseline comparison CSV exists, append LSTM row
    baseline = os.path.join(OUTPUT_DIR, "rot_strategy_model_comparison.csv")
    if os.path.exists(baseline):
        base_df = pd.read_csv(baseline)
        combined = pd.concat([base_df, row], ignore_index=True)
        combined.to_csv(baseline, index=False)
        print(f"Updated {baseline} with LSTM row")
        print(combined.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Hyperparameters:", HYPERPARAMS)

    # Load data
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(
            f"{INPUT_PARQUET} not found. Run rot_strategy_processing.py first."
        )
    labeled_df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded labeled parquet: {labeled_df.shape}")

    # Load weather lookup
    weather_lookup = load_weather_annual()

    # Build sequences
    sequences, labels = build_sequences(
        labeled_df, weather_lookup,
        sample_size=HYPERPARAMS["sample_size"]
    )

    # Train LSTM
    results = train_lstm(sequences, labels)

    if results is None:
        print("Training skipped (PyTorch unavailable).")
        return

    # Visualize
    plot_loss_curve(results)
    plot_accuracy_curve(results)
    plot_lstm_confusion(results)
    save_results_table(results)

    print("\n=== rot_strategy_model_advanced.py complete ===")


if __name__ == "__main__":
    main()
