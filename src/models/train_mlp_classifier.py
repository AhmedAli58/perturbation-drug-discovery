"""
MLP perturbation classifier — first neural model in the pipeline.

Architecture : FC(2000→512) → ReLU → Dropout(0.3)
               FC(512→256)  → ReLU → Dropout(0.3)
               FC(256→237)  → logits
Loss         : CrossEntropyLoss
Optimizer    : Adam  lr=1e-3
Batch size   : 512
Epochs       : 20

Inputs : data/processed/norman2019_processed.h5ad
Outputs: data/results/mlp_metrics.json
         data/models/mlp_classifier.pt
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from time import time

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "norman2019_processed.h5ad"
RESULTS_DIR    = PROJECT_ROOT / "data" / "results"
MODELS_DIR     = PROJECT_ROOT / "data" / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH   = RESULTS_DIR / "mlp_metrics.json"
MODEL_PATH     = MODELS_DIR  / "mlp_classifier.pt"
BASELINE_PATH  = RESULTS_DIR / "baseline_metrics.json"

# ── Hyperparameters ────────────────────────────────────────────────────────
LR         = 1e-3
BATCH_SIZE = 512
EPOCHS     = 20
DROPOUT    = 0.3
HIDDEN     = [512, 256]
SEED       = 42

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", DEVICE)

# ── 1. Load dataset ────────────────────────────────────────────────────────
logger.info("Loading %s …", PROCESSED_PATH)
adata = sc.read_h5ad(PROCESSED_PATH)
logger.info("AnnData: %d cells × %d genes", adata.n_obs, adata.n_vars)

# ── 2. Extract X, y, split ─────────────────────────────────────────────────
X_raw = adata.X
if sp.issparse(X_raw):
    X_raw = X_raw.toarray()
X_raw = X_raw.astype(np.float32)

perturbation_col = adata.obs["perturbation"].astype(str)
split_col        = adata.obs["split"]

# ── 3. Integer label encoding ──────────────────────────────────────────────
if "perturbation_encoding" in adata.uns:
    pert_to_idx: dict[str, int] = adata.uns["perturbation_encoding"]["pert_to_idx"]
    classes = sorted(pert_to_idx, key=pert_to_idx.get)
    y_all = np.array([pert_to_idx[p] for p in perturbation_col], dtype=np.int64)
    logger.info("Using stored encoding — %d classes", len(classes))
else:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_all = le.fit_transform(perturbation_col).astype(np.int64)
    classes = list(le.classes_)
    logger.info("Fitted LabelEncoder — %d classes", len(classes))

n_features  = X_raw.shape[1]
n_classes   = len(classes)

# ── 4. Train / test masks ──────────────────────────────────────────────────
train_mask = (split_col == "train").values
test_mask  = (split_col == "test").values

X_train, y_train = X_raw[train_mask], y_all[train_mask]
X_test,  y_test  = X_raw[test_mask],  y_all[test_mask]
logger.info("Train: %d  Test: %d", len(y_train), len(y_test))

# ── PyTorch tensors & DataLoaders ──────────────────────────────────────────
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE.type == "cuda"))
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE.type == "cuda"))

# ── 5. Model architecture ──────────────────────────────────────────────────
class PerturbationMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = PerturbationMLP(n_features, HIDDEN, n_classes, DROPOUT).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Model parameters: %s", f"{n_params:,}")
print(model)

# ── 6. Optimizer & loss ────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ── 7. Training loop ───────────────────────────────────────────────────────
def evaluate(loader: DataLoader) -> tuple[float, float]:
    """Return (top1_acc, top5_acc) over a DataLoader."""
    model.eval()
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            top5 = logits.topk(5, dim=1).indices
            correct1 += (logits.argmax(1) == yb).sum().item()
            correct5 += (top5 == yb.unsqueeze(1)).any(1).sum().item()
            total += yb.size(0)
    return correct1 / total, correct5 / total


history: list[dict] = []
print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Top1':>10}  {'Test Top1':>9}  {'Test Top5':>9}  {'Time':>6}")
print("─" * 62)

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * yb.size(0)

    avg_loss = total_loss / len(y_train)
    train_top1, train_top5 = evaluate(train_loader)
    test_top1,  test_top5  = evaluate(test_loader)
    elapsed = time() - t0

    print(f"{epoch:>5}  {avg_loss:>10.4f}  {train_top1:>10.4f}  "
          f"{test_top1:>9.4f}  {test_top5:>9.4f}  {elapsed:>5.1f}s")

    history.append({
        "epoch": epoch,
        "train_loss": round(avg_loss, 6),
        "train_top1": round(train_top1, 6),
        "test_top1":  round(test_top1,  6),
        "test_top5":  round(test_top5,  6),
    })

# ── 8. Final evaluation — per-class accuracy ──────────────────────────────
logger.info("Computing per-class accuracy …")
model.eval()
all_preds = []
all_labels = []
all_top5 = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_top5.append(logits.topk(5, dim=1).indices.cpu().numpy())
        all_labels.append(yb.numpy())

y_pred   = np.concatenate(all_preds)
y_top5   = np.concatenate(all_top5, axis=0)
y_true   = np.concatenate(all_labels)

final_top1 = float((y_pred == y_true).mean())
final_top5 = float(np.array([y_true[i] in y_top5[i] for i in range(len(y_true))]).mean())

per_class_acc: dict[str, float] = {}
for idx, pert in enumerate(classes):
    mask = y_true == idx
    if mask.sum() > 0:
        per_class_acc[pert] = round(float((y_pred[mask] == idx).mean()), 6)

mean_per_class = float(np.mean(list(per_class_acc.values())))

# ── 9. Save metrics ────────────────────────────────────────────────────────
metrics = {
    "model": "mlp_classifier",
    "architecture": {
        "input_dim": n_features,
        "hidden_dims": HIDDEN,
        "output_dim": n_classes,
        "dropout": DROPOUT,
        "n_params": n_params,
    },
    "training": {
        "optimizer": "adam",
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    },
    "n_perturbations": n_classes,
    "train_cells": int(train_mask.sum()),
    "test_cells":  int(test_mask.sum()),
    "accuracy":              round(final_top1, 6),
    "top5_accuracy":         round(final_top5, 6),
    "mean_per_class_accuracy": round(mean_per_class, 6),
    "per_perturbation_accuracy": per_class_acc,
    "history": history,
}
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
logger.info("Metrics saved → %s", METRICS_PATH)

# ── 10. Save model ─────────────────────────────────────────────────────────
torch.save({
    "model_state_dict": model.state_dict(),
    "classes":          classes,
    "n_features":       n_features,
    "n_classes":        n_classes,
    "hidden":           HIDDEN,
    "dropout":          DROPOUT,
    "hyperparams":      {"lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS},
}, MODEL_PATH)
logger.info("Model saved → %s", MODEL_PATH)

# ── 11. Comparison with logistic baseline ──────────────────────────────────
print("\n── Model Comparison ──────────────────────────────────────────────")
print(f"  {'Metric':<26}  {'Logistic Baseline':>18}  {'MLP':>10}")
print("  " + "─" * 58)

if BASELINE_PATH.exists():
    baseline = json.loads(BASELINE_PATH.read_text())
    b_top1 = baseline.get("accuracy", float("nan"))
    b_top5 = baseline.get("top5_accuracy", float("nan"))
    b_mpc  = baseline.get("mean_per_class_accuracy", float("nan"))
    delta1 = final_top1 - b_top1
    delta5 = final_top5 - b_top5
    print(f"  {'Top-1 Accuracy':<26}  {b_top1:>17.4f}   {final_top1:>9.4f}  (Δ {delta1:+.4f})")
    print(f"  {'Top-5 Accuracy':<26}  {b_top5:>17.4f}   {final_top5:>9.4f}  (Δ {delta5:+.4f})")
    print(f"  {'Mean Per-Class Accuracy':<26}  {b_mpc:>17.4f}   {mean_per_class:>9.4f}")
else:
    print(f"  {'Top-1 Accuracy':<26}  {'N/A':>18}  {final_top1:>10.4f}")
    print(f"  {'Top-5 Accuracy':<26}  {'N/A':>18}  {final_top5:>10.4f}")
    print(f"  {'Mean Per-Class Accuracy':<26}  {'N/A':>18}  {mean_per_class:>10.4f}")

print("  " + "─" * 58)

sorted_perts = sorted(per_class_acc.items(), key=lambda x: x[1])
print("\n  Hardest perturbations (MLP):")
for p, a in sorted_perts[:5]:
    print(f"    {p:<34s}  {a*100:.1f}%")
print("  Easiest perturbations (MLP):")
for p, a in sorted_perts[-5:]:
    print(f"    {p:<34s}  {a*100:.1f}%")

print("\nMLP training complete.")
