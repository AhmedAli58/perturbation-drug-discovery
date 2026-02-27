"""
Perturbation effect prediction model.

Task : (control cell expression, perturbation ID) → perturbed expression
Loss : MSELoss
This is the core AI drug-discovery task, analogous to scGen / CPA / GEARS.

Architecture
────────────
  control_expr (2000) ──► Linear(2000→512) ──► ReLU ──────────────────┐
                                                                        ├─ cat([512+64])
  pert_id ──► Embedding(n_perts, 64) ─────────────────────────────────┘
              ──► Linear(576→512) ──► ReLU ──► Dropout(0.3)
              ──► Linear(512→2000)
              ──► predicted perturbed expression

Training pairs: (random_control_cell, pert_id) → perturbed_cell
Evaluation    : (mean_control, pert_id) → vs. mean_perturbed per perturbation

Inputs : data/processed/norman2019_processed.h5ad
Outputs: data/results/perturbation_effect_metrics.json
         data/models/perturbation_effect_model.pt
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
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "norman2019_processed.h5ad"
RESULTS_DIR    = PROJECT_ROOT / "data" / "results"
MODELS_DIR     = PROJECT_ROOT / "data" / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "perturbation_effect_metrics.json"
MODEL_PATH   = MODELS_DIR  / "perturbation_effect_model.pt"

# ── Hyperparameters ────────────────────────────────────────────────────────
CONTROL_LABEL  = "control"
EMBED_DIM      = 64
HIDDEN1        = 512
LR             = 1e-3
BATCH_SIZE     = 256
EPOCHS         = 30
DROPOUT        = 0.3
SEED           = 42

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", DEVICE)

# ── 1. Load dataset ────────────────────────────────────────────────────────
logger.info("Loading %s …", PROCESSED_PATH)
adata = sc.read_h5ad(PROCESSED_PATH)
logger.info("AnnData: %d cells × %d genes", adata.n_obs, adata.n_vars)

X_all = adata.X
if sp.issparse(X_all):
    X_all = X_all.toarray()
X_all = X_all.astype(np.float32)

perts     = adata.obs["perturbation"].astype(str).values
split_col = adata.obs["split"].values
n_genes   = X_all.shape[1]

# ── 2. Perturbation encoding (exclude "control" from pert classes) ─────────
if "perturbation_encoding" in adata.uns:
    pert_to_idx: dict[str, int] = dict(adata.uns["perturbation_encoding"]["pert_to_idx"])
else:
    unique_perts = sorted(set(perts))
    pert_to_idx  = {p: i for i, p in enumerate(unique_perts)}

classes   = sorted(pert_to_idx, key=pert_to_idx.get)
n_classes = len(classes)
logger.info("Perturbation classes (incl. control): %d", n_classes)

# ── 2. Identify control cells ──────────────────────────────────────────────
ctrl_mask  = perts == CONTROL_LABEL
n_ctrl     = int(ctrl_mask.sum())
X_ctrl     = X_all[ctrl_mask]
mean_ctrl  = X_ctrl.mean(axis=0)          # (n_genes,) — reference state

logger.info("Control cells: %d  |  mean_ctrl shape: %s", n_ctrl, mean_ctrl.shape)

# ── 3. Build training / test indices for perturbed cells ───────────────────
train_pert_mask = (~ctrl_mask) & (split_col == "train")
test_pert_mask  = (~ctrl_mask) & (split_col == "test")

X_train_pert = X_all[train_pert_mask]        # target expressions
y_train_pert = np.array([pert_to_idx[p] for p in perts[train_pert_mask]], dtype=np.int64)

X_test_pert  = X_all[test_pert_mask]
y_test_pert  = np.array([pert_to_idx[p] for p in perts[test_pert_mask]],  dtype=np.int64)

logger.info("Train perturbed cells: %d  |  Test: %d",
            len(y_train_pert), len(y_test_pert))

# ── Custom Dataset: samples a random control cell as input per step ─────────
class PerturbationPairDataset(Dataset):
    """
    Each item: (control_expr, pert_idx, target_expr)
    The control cell is sampled uniformly at random from X_ctrl each epoch,
    giving the model diverse context rather than a fixed mean.
    """
    def __init__(self, X_ctrl: np.ndarray, X_pert: np.ndarray, y_pert: np.ndarray):
        self.X_ctrl = torch.from_numpy(X_ctrl)
        self.X_pert = torch.from_numpy(X_pert)
        self.y_pert = torch.from_numpy(y_pert)
        self.n_ctrl = len(X_ctrl)

    def __len__(self) -> int:
        return len(self.X_pert)

    def __getitem__(self, idx: int):
        ctrl_idx = torch.randint(self.n_ctrl, (1,)).item()
        return self.X_ctrl[ctrl_idx], self.y_pert[idx], self.X_pert[idx]


train_ds = PerturbationPairDataset(X_ctrl, X_train_pert, y_train_pert)
test_ds  = PerturbationPairDataset(X_ctrl, X_test_pert,  y_test_pert)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 4. Model architecture ──────────────────────────────────────────────────
class PerturbationEffectModel(nn.Module):
    """
    control_expr ──► encoder ──► h (512)  ┐
    pert_id      ──► Embedding ──► e (64) ┤ cat → decoder → pred_expr (2000)
    """
    def __init__(self, n_genes: int, n_perts: int, embed_dim: int,
                 hidden: int, dropout: float):
        super().__init__()
        self.pert_emb = nn.Embedding(n_perts, embed_dim)

        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden + embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, ctrl_expr: torch.Tensor, pert_idx: torch.Tensor) -> torch.Tensor:
        h = self.encoder(ctrl_expr)                 # (B, 512)
        e = self.pert_emb(pert_idx)                 # (B, 64)
        return self.decoder(torch.cat([h, e], dim=1))   # (B, 2000)


model = PerturbationEffectModel(
    n_genes=n_genes, n_perts=n_classes,
    embed_dim=EMBED_DIM, hidden=HIDDEN1, dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Model parameters: %s", f"{n_params:,}")
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ── 5. Training loop ───────────────────────────────────────────────────────
def eval_mse(loader: DataLoader) -> float:
    model.eval()
    total_loss = total_n = 0
    with torch.no_grad():
        for ctrl, pert_idx, target in loader:
            ctrl, pert_idx, target = ctrl.to(DEVICE), pert_idx.to(DEVICE), target.to(DEVICE)
            pred = model(ctrl, pert_idx)
            total_loss += criterion(pred, target).item() * target.size(0)
            total_n    += target.size(0)
    return total_loss / total_n


history: list[dict] = []
print(f"\n{'Epoch':>5}  {'Train MSE':>10}  {'Test MSE':>9}  {'Time':>6}")
print("─" * 36)

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time()
    running_loss = 0.0

    for ctrl, pert_idx, target in train_loader:
        ctrl, pert_idx, target = ctrl.to(DEVICE), pert_idx.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        pred = model(ctrl, pert_idx)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * target.size(0)

    train_mse = running_loss / len(y_train_pert)
    test_mse  = eval_mse(test_loader)
    elapsed   = time() - t0

    print(f"{epoch:>5}  {train_mse:>10.4f}  {test_mse:>9.4f}  {elapsed:>5.1f}s")
    history.append({"epoch": epoch,
                    "train_mse": round(train_mse, 6),
                    "test_mse":  round(test_mse,  6)})

# ── 6. Evaluation — per-cell and per-gene Pearson correlation ───────────────
logger.info("Computing Pearson correlations …")
model.eval()

# Use mean_ctrl as a fixed reference for evaluation (deterministic)
mean_ctrl_t = torch.from_numpy(mean_ctrl).unsqueeze(0).to(DEVICE)   # (1, 2000)

all_pred   = []
all_target = []

with torch.no_grad():
    for ctrl, pert_idx, target in test_loader:
        # Replace random ctrl with mean_ctrl for deterministic eval
        mc = mean_ctrl_t.expand(target.size(0), -1)
        pred = model(mc, pert_idx.to(DEVICE))
        all_pred.append(pred.cpu().numpy())
        all_target.append(target.numpy())

pred_arr   = np.concatenate(all_pred,   axis=0)   # (N_test, 2000)
target_arr = np.concatenate(all_target, axis=0)

# Per-cell Pearson: correlation across genes for each cell
cell_cors = np.array([
    pearsonr(pred_arr[i], target_arr[i])[0]
    for i in range(len(pred_arr))
])
mean_cell_cor = float(np.nanmean(cell_cors))

# Per-gene Pearson: correlation across cells for each gene
gene_cors = np.array([
    pearsonr(pred_arr[:, g], target_arr[:, g])[0]
    for g in range(n_genes)
])
mean_gene_cor = float(np.nanmean(gene_cors))

# Per-perturbation MSE & correlation (aggregate eval)
pert_eval: dict[str, dict] = {}
unique_test_perts = np.unique(y_test_pert)
mean_ctrl_t_cpu   = mean_ctrl_t.cpu()

with torch.no_grad():
    for pidx in unique_test_perts:
        pert_name = classes[pidx]
        mask      = y_test_pert == pidx
        target_p  = X_test_pert[mask]                              # (n_cells, 2000)
        mean_tgt  = target_p.mean(axis=0)                          # mean perturbed

        mc   = mean_ctrl_t.expand(1, -1)
        pidx_t = torch.tensor([pidx], dtype=torch.long).to(DEVICE)
        pred_p = model(mc, pidx_t).squeeze(0).cpu().numpy()        # (2000,)

        mse_p = float(np.mean((pred_p - mean_tgt) ** 2))
        cor_p, _ = pearsonr(pred_p, mean_tgt)

        pert_eval[pert_name] = {
            "n_cells":     int(mask.sum()),
            "mse":         round(mse_p, 6),
            "pearson_r":   round(float(cor_p), 6),
        }

mean_pert_cor = float(np.mean([v["pearson_r"] for v in pert_eval.values()
                                if not np.isnan(v["pearson_r"])]))

test_mse_final = eval_mse(test_loader)

# ── 7. Save metrics ────────────────────────────────────────────────────────
metrics = {
    "model": "perturbation_effect_model",
    "architecture": {
        "n_genes":    n_genes,
        "n_perts":    n_classes,
        "embed_dim":  EMBED_DIM,
        "hidden1":    HIDDEN1,
        "dropout":    DROPOUT,
        "n_params":   n_params,
    },
    "training": {
        "optimizer":    "adam",
        "lr":           LR,
        "batch_size":   BATCH_SIZE,
        "epochs":       EPOCHS,
        "loss":         "MSELoss",
        "control_label": CONTROL_LABEL,
        "n_ctrl_cells": n_ctrl,
    },
    "test_mse":                   round(test_mse_final, 6),
    "mean_cell_pearson_r":        round(mean_cell_cor,  6),
    "mean_gene_pearson_r":        round(mean_gene_cor,  6),
    "mean_per_pert_pearson_r":    round(mean_pert_cor,  6),
    "per_perturbation_eval":      pert_eval,
    "history":                    history,
}
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
logger.info("Metrics saved → %s", METRICS_PATH)

# ── 8. Save model ──────────────────────────────────────────────────────────
torch.save({
    "model_state_dict": model.state_dict(),
    "classes":          classes,
    "n_genes":          n_genes,
    "n_classes":        n_classes,
    "embed_dim":        EMBED_DIM,
    "hidden1":          HIDDEN1,
    "dropout":          DROPOUT,
    "mean_ctrl":        mean_ctrl,
}, MODEL_PATH)
logger.info("Model saved → %s", MODEL_PATH)

# ── Summary print ──────────────────────────────────────────────────────────
print("\n── Perturbation Effect Model — Results ───────────────────────────")
print(f"  Test MSE                     : {test_mse_final:.4f}")
print(f"  Mean cell-level Pearson r    : {mean_cell_cor:.4f}")
print(f"  Mean gene-level Pearson r    : {mean_gene_cor:.4f}")
print(f"  Mean per-perturbation r      : {mean_pert_cor:.4f}")
print()

sorted_perts = sorted(pert_eval.items(), key=lambda x: x[1]["pearson_r"])
print("  Hardest perturbations to predict (lowest r):")
for p, v in sorted_perts[:5]:
    print(f"    {p:<34s}  r={v['pearson_r']:+.3f}  n={v['n_cells']}")
print("  Easiest perturbations to predict (highest r):")
for p, v in sorted_perts[-5:]:
    print(f"    {p:<34s}  r={v['pearson_r']:+.3f}  n={v['n_cells']}")

print(f"\n  Training loss curve (every 5 epochs):")
for h in history[::5]:
    print(f"    epoch {h['epoch']:>2}  train={h['train_mse']:.4f}  test={h['test_mse']:.4f}")

print("\nPerturbation effect model training complete.")
