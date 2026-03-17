"""
scGen-style Variational Autoencoder for perturbation effect prediction.

Key idea
────────
Learn a smooth latent space where:
  • The encoder maps any cell's expression → μ, σ (latent dim 128)
  • A perturbation embedding (64-d) shifts the decoder input
  • At inference: encode(ctrl_cell) + pert_emb[p] → decode → predicted expression

This mirrors the scGen / CPA approach: perturbations are additive operators
in a disentangled latent space, making them compositionally interpretable.

Architecture
────────────
Encoder  : Linear(2000→512)→ReLU→Linear(512→256)→ReLU
           → μ(256→128),  logvar(256→128)
Reparam  : z = μ + ε·exp(½logvar),  ε ~ N(0,I)
Decoder  : concat(z:128, pert_emb:64) → Linear(192→256)→ReLU
                                       → Linear(256→512)→ReLU
                                       → Linear(512→2000)

Loss     : MSE(recon, target) + β·KL   β annealed 0→1 over first 10 epochs

Inputs : data/processed/norman2019_processed.h5ad
Outputs: data/results/scgen_metrics.json
         data/models/scgen_model.pt
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
import torch.nn.functional as F
from scipy.stats import pearsonr
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

METRICS_PATH = RESULTS_DIR / "scgen_metrics.json"
MODEL_PATH   = MODELS_DIR  / "scgen_model.pt"

# ── Hyperparameters ────────────────────────────────────────────────────────
LATENT_DIM   = 128
PERT_EMB_DIM = 64
ENC_H1, ENC_H2 = 512, 256
DEC_H1, DEC_H2 = 256, 512
LR           = 1e-3
BATCH_SIZE   = 256
EPOCHS       = 40
KL_ANNEAL    = 10          # epochs over which β goes 0 → KL_MAX
KL_MAX       = 1e-4        # final β weight (keeps KL from dominating)
from src.constants import SEED, CONTROL_LABEL  # noqa: E402

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", DEVICE)

# ══════════════════════════════════════════════════════════════════════════
# 1. Load dataset
# ══════════════════════════════════════════════════════════════════════════
if not PROCESSED_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found at {PROCESSED_PATH}. Run: make preprocess"
    )
logger.info("Loading AnnData …")
adata = sc.read_h5ad(PROCESSED_PATH)
logger.info("AnnData: %d cells × %d genes", adata.n_obs, adata.n_vars)

X_all = adata.X
if sp.issparse(X_all):
    X_all = X_all.toarray()
X_all = X_all.astype(np.float32)

n_genes   = adata.n_vars
perts_col = adata.obs["perturbation"].astype(str).values
split_col = adata.obs["split"].values

# ══════════════════════════════════════════════════════════════════════════
# 2. Perturbation encoding (reuse stored mapping for consistency)
# ══════════════════════════════════════════════════════════════════════════
if "perturbation_encoding" in adata.uns:
    pert_to_idx: dict[str, int] = dict(adata.uns["perturbation_encoding"]["pert_to_idx"])
else:
    unique_perts = sorted(set(perts_col))
    pert_to_idx  = {p: i for i, p in enumerate(unique_perts)}

classes    = sorted(pert_to_idx, key=pert_to_idx.get)
n_classes  = len(classes)
ctrl_idx   = pert_to_idx[CONTROL_LABEL]
logger.info("Classes: %d  |  control idx: %d", n_classes, ctrl_idx)

y_all = np.array([pert_to_idx[p] for p in perts_col], dtype=np.int64)

# ══════════════════════════════════════════════════════════════════════════
# 3. Train / test splits  (all cells, including control)
# ══════════════════════════════════════════════════════════════════════════
train_mask = split_col == "train"
val_mask   = split_col == "val"
test_mask  = split_col == "test"

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
logger.info("Train: %d  Val: %d  Test: %d", len(y_train), len(y_val), len(y_test))

# Reference state: mean of control cells (used at inference)
ctrl_mask = perts_col == CONTROL_LABEL
X_ctrl    = X_all[ctrl_mask]
mean_ctrl = X_ctrl.mean(axis=0)                     # (2000,)
logger.info("Control cells: %d", int(ctrl_mask.sum()))

train_ds = TensorDataset(
    torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(
    torch.from_numpy(X_val),   torch.from_numpy(y_val))
test_ds  = TensorDataset(
    torch.from_numpy(X_test),  torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

# ══════════════════════════════════════════════════════════════════════════
# 4. Model
# ══════════════════════════════════════════════════════════════════════════
class ScGenVAE(nn.Module):
    """
    VAE where the decoder is conditioned on a perturbation embedding.
    During training: encode(perturbed_cell) → z → decode(z, pert_emb) → recon.
    At inference:    encode(ctrl_cell)      → z → decode(z, pert_emb) → pred.
    """
    def __init__(
        self,
        n_genes:      int,
        n_perts:      int,
        latent_dim:   int = LATENT_DIM,
        pert_emb_dim: int = PERT_EMB_DIM,
        enc_h1:       int = ENC_H1,
        enc_h2:       int = ENC_H2,
        dec_h1:       int = DEC_H1,
        dec_h2:       int = DEC_H2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, enc_h1), nn.ReLU(),
            nn.Linear(enc_h1,  enc_h2), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(enc_h2, latent_dim)
        self.fc_logvar = nn.Linear(enc_h2, latent_dim)

        # ── Perturbation embedding ────────────────────────────────────────
        self.pert_emb = nn.Embedding(n_perts, pert_emb_dim)

        # ── Decoder (z + pert_emb) ────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + pert_emb_dim, dec_h1), nn.ReLU(),
            nn.Linear(dec_h1,  dec_h2), nn.ReLU(),
            nn.Linear(dec_h2,  n_genes),
        )

    # ── Encoder forward ───────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    # ── Reparameterisation ────────────────────────────────────────────────
    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu                                   # deterministic at eval

    # ── Decoder forward ───────────────────────────────────────────────────
    def decode(
        self, z: torch.Tensor, pert_idx: torch.Tensor
    ) -> torch.Tensor:
        e = self.pert_emb(pert_idx)                 # (B, 64)
        return self.decoder(torch.cat([z, e], dim=1))

    def forward(
        self, x: torch.Tensor, pert_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decode(z, pert_idx)
        return recon, mu, logvar


model = ScGenVAE(n_genes=n_genes, n_perts=n_classes).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Parameters: %s", f"{n_params:,}")
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ══════════════════════════════════════════════════════════════════════════
# 5. Loss — ELBO = MSE + β·KL
# ══════════════════════════════════════════════════════════════════════════
def elbo(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, float, float]:
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl_loss    = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    total = recon_loss + beta * kl_loss
    return total, recon_loss.item(), kl_loss.item()


def beta_schedule(epoch: int) -> float:
    """Linear warm-up from 0 to KL_MAX over KL_ANNEAL epochs."""
    return KL_MAX * min(epoch / KL_ANNEAL, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# 6. Training loop
# ══════════════════════════════════════════════════════════════════════════
def eval_loss(loader: DataLoader, beta: float) -> tuple[float, float, float]:
    model.eval()
    tot_recon = tot_kl = tot_n = 0
    with torch.no_grad():
        for x, pidx in loader:
            x, pidx = x.to(DEVICE), pidx.to(DEVICE)
            recon, mu, logvar = model(x, pidx)
            _, rl, kl = elbo(recon, x, mu, logvar, beta)
            tot_recon += rl * x.size(0)
            tot_kl    += kl * x.size(0)
            tot_n     += x.size(0)
    r = tot_recon / tot_n
    k = tot_kl    / tot_n
    return r + beta * k, r, k


history: list[dict] = []
header = (f"{'Ep':>3}  {'β':>7}  {'Tr-MSE':>8}  {'Tr-KL':>8}  "
          f"{'Va-MSE':>8}  {'Va-KL':>8}  {'Time':>5}")
print(f"\n{header}")
print("─" * len(header))

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0   = time()
    beta = beta_schedule(epoch)
    tot_recon = tot_kl = tot_n = 0

    for x, pidx in train_loader:
        x, pidx = x.to(DEVICE), pidx.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = model(x, pidx)
        loss, rl, kl = elbo(recon, x, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        tot_recon += rl * x.size(0)
        tot_kl    += kl * x.size(0)
        tot_n     += x.size(0)

    tr_mse = tot_recon / tot_n
    tr_kl  = tot_kl    / tot_n
    _, va_mse, va_kl = eval_loss(val_loader, beta)   # val only — test stays held-out
    elapsed = time() - t0

    print(f"{epoch:>3}  {beta:>7.1e}  {tr_mse:>8.4f}  {tr_kl:>8.2f}  "
          f"{va_mse:>8.4f}  {va_kl:>8.2f}  {elapsed:>4.1f}s")
    history.append({
        "epoch": epoch, "beta": beta,
        "train_mse": round(tr_mse, 6), "train_kl": round(tr_kl, 4),
        "val_mse":   round(va_mse, 6), "val_kl":   round(va_kl, 4),
    })

# ══════════════════════════════════════════════════════════════════════════
# 7. Evaluation  (scGen-style inference: ctrl + pert_emb → predicted)
# ══════════════════════════════════════════════════════════════════════════
logger.info("Evaluating in scGen inference mode (ctrl → pert) …")
model.eval()

mean_ctrl_t = torch.from_numpy(mean_ctrl).unsqueeze(0).to(DEVICE)  # (1, 2000)

# Encode mean ctrl once → fixed latent reference
with torch.no_grad():
    z_ctrl_mu, _ = model.encode(mean_ctrl_t)       # (1, 128) — deterministic

# ── Cell-level evaluation: for each test cell predict via ctrl+pert ───────
all_pred, all_tgt = [], []
with torch.no_grad():
    for x, pidx in test_loader:
        pidx_d = pidx.to(DEVICE)
        z_rep  = z_ctrl_mu.expand(x.size(0), -1)   # broadcast ctrl latent
        pred   = model.decode(z_rep, pidx_d)
        all_pred.append(pred.cpu().numpy())
        all_tgt.append(x.numpy())

pred_arr = np.concatenate(all_pred, axis=0)         # (N_test, 2000)
tgt_arr  = np.concatenate(all_tgt,  axis=0)

# MSE
test_mse = float(np.mean((pred_arr - tgt_arr) ** 2))

# Cell-level Pearson r (across 2000 genes per cell)
cell_cors = np.array([pearsonr(pred_arr[i], tgt_arr[i])[0]
                       for i in range(len(pred_arr))])
mean_cell_cor = float(np.nanmean(cell_cors))

# Gene-level Pearson r (across test cells per gene)
gene_cors = np.array([pearsonr(pred_arr[:, g], tgt_arr[:, g])[0]
                       for g in range(n_genes)])
mean_gene_cor = float(np.nanmean(gene_cors))

# Per-perturbation: predicted vs. observed mean expression
pert_eval: dict[str, dict] = {}
test_perts_col = perts_col[test_mask]

with torch.no_grad():
    for pert_name, pidx in pert_to_idx.items():
        if pert_name == CONTROL_LABEL:
            continue
        mask = test_perts_col == pert_name
        if mask.sum() == 0:
            continue
        mean_tgt = X_all[test_mask][mask].mean(axis=0)         # (2000,)

        pidx_t = torch.tensor([pidx], dtype=torch.long).to(DEVICE)
        pred_p = model.decode(z_ctrl_mu, pidx_t).squeeze(0).cpu().numpy()

        mse_p    = float(np.mean((pred_p - mean_tgt) ** 2))
        r_p, _   = pearsonr(pred_p, mean_tgt)
        delta_p  = pred_p - mean_ctrl   # predicted shift in expression

        pert_eval[pert_name] = {
            "n_cells":   int(mask.sum()),
            "mse":       round(mse_p, 6),
            "pearson_r": round(float(r_p), 6),
        }

mean_pert_cor = float(np.nanmean([v["pearson_r"] for v in pert_eval.values()]))

# ══════════════════════════════════════════════════════════════════════════
# 8. Save metrics
# ══════════════════════════════════════════════════════════════════════════
metrics = {
    "model": "scgen_vae",
    "architecture": {
        "n_genes": n_genes, "n_perts": n_classes,
        "latent_dim": LATENT_DIM, "pert_emb_dim": PERT_EMB_DIM,
        "encoder": [ENC_H1, ENC_H2], "decoder": [DEC_H1, DEC_H2],
        "n_params": n_params,
    },
    "training": {
        "optimizer": "adam", "lr": LR, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "kl_max": KL_MAX, "kl_anneal_epochs": KL_ANNEAL,
    },
    "inference_mode": "encode(mean_ctrl) → decode(z, pert_emb)",
    "test_mse":                round(test_mse,      6),
    "mean_cell_pearson_r":     round(mean_cell_cor, 6),
    "mean_gene_pearson_r":     round(mean_gene_cor, 6),
    "mean_per_pert_pearson_r": round(mean_pert_cor, 6),
    "per_perturbation_eval":   pert_eval,
    "history":                 history,
}
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
logger.info("Metrics → %s", METRICS_PATH)

# ══════════════════════════════════════════════════════════════════════════
# 9. Save model
# ══════════════════════════════════════════════════════════════════════════
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes, "n_genes": n_genes, "n_classes": n_classes,
    "latent_dim": LATENT_DIM, "pert_emb_dim": PERT_EMB_DIM,
    "enc_h1": ENC_H1, "enc_h2": ENC_H2, "dec_h1": DEC_H1, "dec_h2": DEC_H2,
    "ctrl_idx": ctrl_idx, "mean_ctrl": mean_ctrl,
}, MODEL_PATH)
logger.info("Model → %s", MODEL_PATH)

# ══════════════════════════════════════════════════════════════════════════
# 10. Three-way comparison
# ══════════════════════════════════════════════════════════════════════════
prev_files = {
    "MLP (vanilla)": RESULTS_DIR / "perturbation_effect_metrics.json",
    "Graph GCN":     RESULTS_DIR / "graph_model_metrics.json",
}
prev_results: dict[str, dict] = {}
for label, path in prev_files.items():
    if path.exists():
        prev_results[label] = json.loads(path.read_text())

all_models   = list(prev_results.items()) + [("scGen VAE", metrics)]
col_w        = 14
metric_keys  = [
    ("test_mse",                "Test MSE",          False),
    ("mean_cell_pearson_r",     "Cell Pearson r",     True),
    ("mean_gene_pearson_r",     "Gene Pearson r",     True),
    ("mean_per_pert_pearson_r", "Per-pert Pearson r", True),
]

header_row = f"  {'Metric':<26}" + "".join(f"{n:>{col_w}}" for n, _ in
             [(name, _) for name, _ in [(m[0], 0) for m in all_models]])
print("\n── Three-Way Model Comparison ────────────────────────────────────────")
print(f"  {'Metric':<26}" +
      "".join(f"{name:>{col_w}}" for name, _ in all_models))
print("  " + "─" * (26 + col_w * len(all_models)))

for key, label, higher_better in metric_keys:
    vals = [d.get(key, float("nan")) for _, d in all_models]
    best = max(vals, key=lambda v: v if higher_better else -v)
    row  = f"  {label:<26}"
    for v in vals:
        marker = " ★" if abs(v - best) < 1e-9 else "  "
        row += f"{v:>{col_w - 2}.4f}{marker}"
    print(row)

print("  " + "─" * (26 + col_w * len(all_models)))

# Highlight gene-level improvement
scgen_gene_r = metrics["mean_gene_pearson_r"]
best_prev_gene_r = max(
    (d.get("mean_gene_pearson_r", -999) for _, d in prev_results.items()),
    default=float("nan"),
)
improvement = scgen_gene_r - best_prev_gene_r
print(f"\n  Gene-level Pearson r improvement over best prior model: "
      f"{improvement:+.4f}  "
      f"({'▲ better' if improvement > 0 else '▼ worse'})")

# Perturbation prediction extremes
sorted_perts = sorted(pert_eval.items(), key=lambda x: x[1]["pearson_r"])
print(f"\n  Hardest perturbations to predict:")
for p, v in sorted_perts[:5]:
    print(f"    {p:<36s}  r={v['pearson_r']:+.3f}  n={v['n_cells']}")
print(f"  Easiest perturbations to predict:")
for p, v in sorted_perts[-5:]:
    print(f"    {p:<36s}  r={v['pearson_r']:+.3f}  n={v['n_cells']}")

print("\nscGen-style VAE training complete.")
