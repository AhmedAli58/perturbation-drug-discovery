"""
Graph-aware perturbation effect model.

Key idea (GEARS-inspired)
─────────────────────────
Instead of a generic learnable embedding per perturbation, we represent
each perturbation as the GCN-updated embeddings of its HVG *neighbours*
in the STRING PPI network.

  perturbed gene (e.g. ARID1A) ─► STRING neighbours ∩ HVGs
                                 ─► average their GCN embeddings
                                 ─► pert_graph_feat (64-d)

The GCN itself operates on the 2000-node HVG gene graph with STRING
HVG–HVG edges, learning interaction-aware gene representations.

Architecture
────────────
  gene_emb (2000, 64) ─┐
  GCNLayer(64→128)     │  ─► gene_feat (2000, 64)
  GCNLayer(128→64)     ┘

  control_expr (B,2000) ─► Linear(2000,512) ─► ctrl_feat (B,512)
  pert_idx    ─► lookup pert_graph_feat matrix ─► (B,64)

  cat([ctrl_feat, pert_graph_feat]) ─► (B,576)
  ─► Linear(576,512)→ReLU→Dropout(0.3)
  ─► Linear(512,2000)  ─► predicted expression (B,2000)

No external dependencies beyond PyTorch + scipy.

Inputs : data/processed/norman2019_processed.h5ad
         data/external/string_ppi_edges.tsv
Outputs: data/results/graph_model_metrics.json
         data/models/graph_perturbation_model.pt
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
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "norman2019_processed.h5ad"
STRING_PATH    = PROJECT_ROOT / "data" / "external"  / "string_ppi_edges.tsv"
RESULTS_DIR    = PROJECT_ROOT / "data" / "results"
MODELS_DIR     = PROJECT_ROOT / "data" / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "graph_model_metrics.json"
MODEL_PATH   = MODELS_DIR  / "graph_perturbation_model.pt"
PREV_METRICS = RESULTS_DIR / "perturbation_effect_metrics.json"

from src.constants import SEED, CONTROL_LABEL  # noqa: E402

# ── Hyperparameters ────────────────────────────────────────────────────────
GENE_EMB_DIM  = 64
GCN_HIDDEN    = 128
GCN_OUT       = 64
CTRL_HIDDEN   = 512
LR            = 1e-3
BATCH_SIZE    = 256
EPOCHS        = 40
DROPOUT       = 0.3

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
rng    = np.random.default_rng(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", DEVICE)

# ══════════════════════════════════════════════════════════════════════════
# 1. Load dataset
# ══════════════════════════════════════════════════════════════════════════
if not PROCESSED_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found at {PROCESSED_PATH}. Run: make preprocess"
    )
if not STRING_PATH.exists():
    raise FileNotFoundError(
        f"STRING PPI file not found at {STRING_PATH}. Run: make data"
    )
logger.info("Loading AnnData …")
adata = sc.read_h5ad(PROCESSED_PATH)
logger.info("AnnData: %d cells × %d genes", adata.n_obs, adata.n_vars)

X_all = adata.X
if sp.issparse(X_all):
    X_all = X_all.toarray()
X_all = X_all.astype(np.float32)

n_genes    = adata.n_vars
hvg_genes  = list(adata.var_names)
gene_to_idx = {g: i for i, g in enumerate(hvg_genes)}

perts     = adata.obs["perturbation"].astype(str).values
split_col = adata.obs["split"].values

if "perturbation_encoding" in adata.uns:
    pert_to_idx: dict[str, int] = dict(adata.uns["perturbation_encoding"]["pert_to_idx"])
else:
    unique_perts = sorted(set(perts))
    pert_to_idx  = {p: i for i, p in enumerate(unique_perts)}

classes   = sorted(pert_to_idx, key=pert_to_idx.get)
n_classes = len(classes)
logger.info("Perturbation classes: %d", n_classes)

# ══════════════════════════════════════════════════════════════════════════
# 2. Control cells
# ══════════════════════════════════════════════════════════════════════════
ctrl_mask = perts == CONTROL_LABEL
X_ctrl    = X_all[ctrl_mask]
mean_ctrl = X_ctrl.mean(axis=0)
logger.info("Control cells: %d", int(ctrl_mask.sum()))

# ══════════════════════════════════════════════════════════════════════════
# 3. Load STRING PPI and build normalised adjacency
# ══════════════════════════════════════════════════════════════════════════
logger.info("Loading STRING PPI from %s …", STRING_PATH)

hvg_set  = set(hvg_genes)
rows, cols, weights = [], [], []
pert_gene_edges: dict[str, list[int]] = {c: [] for c in classes}  # pert_name → HVG indices

with open(STRING_PATH) as f:
    next(f)   # skip header
    for line in f:
        a, b, w = line.strip().split("\t")
        w = float(w)
        a_in = a in gene_to_idx
        b_in = b in gene_to_idx

        # HVG–HVG edges → GCN adjacency
        if a_in and b_in:
            i, j = gene_to_idx[a], gene_to_idx[b]
            rows.extend([i, j]); cols.extend([j, i]); weights.extend([w, w])

        # Edges involving a perturbed (non-HVG) gene → pert neighbour map
        # Format: geneA is the perturbed gene, geneB is a HVG (or vice versa)
        if not a_in and b_in:
            for pert in classes:
                if a in pert.split("_"):
                    pert_gene_edges[pert].append(gene_to_idx[b])
        if a_in and not b_in:
            for pert in classes:
                if b in pert.split("_"):
                    pert_gene_edges[pert].append(gene_to_idx[a])

        # HVG target gene perturbed → also add its own HVG neighbours as context
        if a_in and b_in:
            for pert in classes:
                genes_in_pert = pert.split("_")
                if a in genes_in_pert:
                    pert_gene_edges[pert].append(gene_to_idx[b])
                if b in genes_in_pert:
                    pert_gene_edges[pert].append(gene_to_idx[a])

n_hvg_edges = len(rows) // 2
logger.info("HVG–HVG edges loaded: %d", n_hvg_edges)

# Normalised adjacency Ã = D^{-½}(A+I)D^{-½}  (dense, 2000×2000 = 16 MB)
A = torch.zeros(n_genes, n_genes)
if rows:
    A[rows, cols] = torch.tensor(weights, dtype=torch.float32)
A = A + torch.eye(n_genes)                           # self-loops
deg      = A.sum(1)
d_inv_sq = deg.pow(-0.5)
d_inv_sq[deg == 0] = 0.0
D        = torch.diag(d_inv_sq)
A_norm   = (D @ A @ D).to(DEVICE)                    # (2000, 2000)
logger.info("Normalised adjacency built: %s, device=%s", A_norm.shape, A_norm.device)

# ══════════════════════════════════════════════════════════════════════════
# 4. Build perturbation → HVG-neighbour mask  (n_perts × n_genes)
# ══════════════════════════════════════════════════════════════════════════
pert_neighbor_mask = torch.zeros(n_classes, n_genes)

for pert, neighbours in pert_gene_edges.items():
    if pert not in pert_to_idx:
        continue
    pidx = pert_to_idx[pert]
    if neighbours:
        for ni in set(neighbours):
            pert_neighbor_mask[pidx, ni] = 1.0

# Normalise rows (mean pooling); fall back to uniform for no-neighbour perts
row_sums = pert_neighbor_mask.sum(1, keepdim=True)
no_nbr   = (row_sums == 0).squeeze()
pert_neighbor_mask[~no_nbr] = pert_neighbor_mask[~no_nbr] / row_sums[~no_nbr]
# For perts with no STRING neighbours: uniform over all genes (use GCN global mean)
pert_neighbor_mask[no_nbr] = 1.0 / n_genes

n_with_nbrs = int((~no_nbr).sum())
logger.info("Perturbations with STRING neighbours: %d / %d", n_with_nbrs, n_classes)

pert_neighbor_mask = pert_neighbor_mask.to(DEVICE)   # (n_perts, n_genes)

# ══════════════════════════════════════════════════════════════════════════
# 5. Datasets & DataLoaders
# ══════════════════════════════════════════════════════════════════════════
train_pert_mask = (~ctrl_mask) & (split_col == "train")
val_pert_mask   = (~ctrl_mask) & (split_col == "val")
test_pert_mask  = (~ctrl_mask) & (split_col == "test")

X_train_pert = X_all[train_pert_mask]
y_train_pert = np.array([pert_to_idx[p] for p in perts[train_pert_mask]], dtype=np.int64)

X_val_pert   = X_all[val_pert_mask]
y_val_pert   = np.array([pert_to_idx[p] for p in perts[val_pert_mask]],   dtype=np.int64)

X_test_pert  = X_all[test_pert_mask]
y_test_pert  = np.array([pert_to_idx[p] for p in perts[test_pert_mask]],  dtype=np.int64)

logger.info("Train: %d  Val: %d  Test: %d perturbed cells",
            len(y_train_pert), len(y_val_pert), len(y_test_pert))


class PerturbationPairDataset(Dataset):
    def __init__(self, X_ctrl, X_pert, y_pert):
        self.X_ctrl = torch.from_numpy(X_ctrl)
        self.X_pert = torch.from_numpy(X_pert)
        self.y_pert = torch.from_numpy(y_pert)
        self.n_ctrl = len(X_ctrl)

    def __len__(self):
        return len(self.X_pert)

    def __getitem__(self, idx):
        ctrl_idx = torch.randint(self.n_ctrl, (1,)).item()
        return self.X_ctrl[ctrl_idx], self.y_pert[idx], self.X_pert[idx]


train_ds = PerturbationPairDataset(X_ctrl, X_train_pert, y_train_pert)
val_ds   = PerturbationPairDataset(X_ctrl, X_val_pert,   y_val_pert)
test_ds  = PerturbationPairDataset(X_ctrl, X_test_pert,  y_test_pert)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ══════════════════════════════════════════════════════════════════════════
# 6. Model
# ══════════════════════════════════════════════════════════════════════════
class ManualGCNLayer(nn.Module):
    """Â_norm @ X @ W  — manual graph convolution (no PyG needed)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (n_nodes, in_dim)  adj: (n_nodes, n_nodes)
        return self.linear(adj @ x)


class GraphPerturbationModel(nn.Module):
    def __init__(
        self,
        n_genes:     int,
        n_perts:     int,
        adj_norm:    torch.Tensor,           # (n_genes, n_genes) precomputed
        nbr_mask:    torch.Tensor,           # (n_perts, n_genes) normalised
        gene_emb_dim: int = GENE_EMB_DIM,
        gcn_hidden:   int = GCN_HIDDEN,
        gcn_out:      int = GCN_OUT,
        ctrl_hidden:  int = CTRL_HIDDEN,
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        self.n_genes = n_genes

        # Register fixed buffers (moved to device with model)
        self.register_buffer("adj_norm", adj_norm)
        self.register_buffer("nbr_mask", nbr_mask)

        # Gene graph branch
        self.gene_emb = nn.Embedding(n_genes, gene_emb_dim)
        self.gcn1     = ManualGCNLayer(gene_emb_dim, gcn_hidden)
        self.gcn2     = ManualGCNLayer(gcn_hidden,   gcn_out)
        self.gcn_drop = nn.Dropout(dropout)

        # Control expression encoder
        self.ctrl_encoder = nn.Sequential(
            nn.Linear(n_genes, ctrl_hidden),
            nn.ReLU(),
        )

        # Decoder: ctrl_feat (512) + pert_graph_feat (64)
        in_dim = ctrl_hidden + gcn_out
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, ctrl_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ctrl_hidden, n_genes),
        )

    # ── GCN: run on full gene graph (once per forward call) ───────────────
    def _gene_features(self) -> torch.Tensor:
        """Returns GCN-updated gene embeddings (n_genes, gcn_out)."""
        x = self.gene_emb.weight                            # (n, 64)
        x = F.relu(self.gcn1(x, self.adj_norm))            # (n, 128)
        x = self.gcn_drop(x)
        x = F.relu(self.gcn2(x, self.adj_norm))            # (n, 64)
        return x

    def forward(
        self,
        ctrl_expr:  torch.Tensor,   # (B, n_genes)
        pert_idx:   torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        # Gene-graph features
        gene_feat = self._gene_features()                   # (n_genes, 64)

        # Perturbation-specific graph features via precomputed neighbour mask
        # nbr_mask[p] selects & averages the HVG-neighbours of pert p
        pert_repr_all = self.nbr_mask @ gene_feat           # (n_perts, 64)
        pert_graph    = pert_repr_all[pert_idx]             # (B, 64)

        # Control encoder
        ctrl_feat = self.ctrl_encoder(ctrl_expr)            # (B, 512)

        # Decode
        x = torch.cat([ctrl_feat, pert_graph], dim=1)      # (B, 576)
        return self.decoder(x)                              # (B, n_genes)


model = GraphPerturbationModel(
    n_genes=n_genes, n_perts=n_classes,
    adj_norm=A_norm, nbr_mask=pert_neighbor_mask,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Model parameters: %s", f"{n_params:,}")
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ══════════════════════════════════════════════════════════════════════════
# 7. Training loop
# ══════════════════════════════════════════════════════════════════════════
def eval_mse(loader: DataLoader) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for ctrl, pidx, tgt in loader:
            ctrl, pidx, tgt = ctrl.to(DEVICE), pidx.to(DEVICE), tgt.to(DEVICE)
            pred = model(ctrl, pidx)
            total += criterion(pred, tgt).item() * tgt.size(0)
            n    += tgt.size(0)
    return total / n


history: list[dict] = []
print(f"\n{'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>8}  {'Time':>6}")
print("─" * 36)

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time()
    running = 0.0

    for ctrl, pidx, tgt in train_loader:
        ctrl, pidx, tgt = ctrl.to(DEVICE), pidx.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        pred = model(ctrl, pidx)
        loss = criterion(pred, tgt)
        loss.backward()
        optimizer.step()
        running += loss.item() * tgt.size(0)

    train_mse = running / len(y_train_pert)
    val_mse   = eval_mse(val_loader)   # val only — test stays held-out until final eval
    elapsed   = time() - t0

    print(f"{epoch:>5}  {train_mse:>10.4f}  {val_mse:>8.4f}  {elapsed:>5.1f}s")
    history.append({"epoch": epoch,
                    "train_mse": round(train_mse, 6),
                    "val_mse":   round(val_mse,   6)})

# ══════════════════════════════════════════════════════════════════════════
# 8. Evaluation — MSE, cell-level r, gene-level r, per-perturbation r
# ══════════════════════════════════════════════════════════════════════════
logger.info("Computing correlations …")
model.eval()
mean_ctrl_t = torch.from_numpy(mean_ctrl).unsqueeze(0).to(DEVICE)

all_pred, all_tgt = [], []
with torch.no_grad():
    for ctrl, pidx, tgt in test_loader:
        mc   = mean_ctrl_t.expand(tgt.size(0), -1)
        pred = model(mc, pidx.to(DEVICE))
        all_pred.append(pred.cpu().numpy())
        all_tgt.append(tgt.numpy())

pred_arr = np.concatenate(all_pred, axis=0)
tgt_arr  = np.concatenate(all_tgt,  axis=0)

# Cell-level Pearson (across genes per cell)
cell_cors = np.array([pearsonr(pred_arr[i], tgt_arr[i])[0]
                       for i in range(len(pred_arr))])
mean_cell_cor = float(np.nanmean(cell_cors))

# Gene-level Pearson (across test cells per gene)
gene_cors = np.array([pearsonr(pred_arr[:, g], tgt_arr[:, g])[0]
                       for g in range(n_genes)])
mean_gene_cor = float(np.nanmean(gene_cors))

# Per-perturbation: predicted vs. observed mean expression
pert_eval: dict[str, dict] = {}
with torch.no_grad():
    for pidx in np.unique(y_test_pert):
        pert_name = classes[pidx]
        mask      = y_test_pert == pidx
        mean_tgt  = X_test_pert[mask].mean(axis=0)

        mc   = mean_ctrl_t.expand(1, -1)
        pt   = torch.tensor([pidx], dtype=torch.long).to(DEVICE)
        pred_p = model(mc, pt).squeeze(0).cpu().numpy()

        mse_p = float(np.mean((pred_p - mean_tgt) ** 2))
        r_p, _ = pearsonr(pred_p, mean_tgt)
        pert_eval[pert_name] = {
            "n_cells":   int(mask.sum()),
            "mse":       round(mse_p, 6),
            "pearson_r": round(float(r_p), 6),
        }

mean_pert_cor = float(np.nanmean([v["pearson_r"] for v in pert_eval.values()]))
test_mse_final = eval_mse(test_loader)

# ══════════════════════════════════════════════════════════════════════════
# 9. Save metrics
# ══════════════════════════════════════════════════════════════════════════
metrics = {
    "model": "graph_perturbation_model",
    "architecture": {
        "n_genes": n_genes, "n_perts": n_classes,
        "gene_emb_dim": GENE_EMB_DIM, "gcn_hidden": GCN_HIDDEN,
        "gcn_out": GCN_OUT, "ctrl_hidden": CTRL_HIDDEN,
        "dropout": DROPOUT, "n_params": n_params,
        "gcn_edges": n_hvg_edges,
    },
    "training": {"optimizer": "adam", "lr": LR, "batch_size": BATCH_SIZE,
                 "epochs": EPOCHS, "loss": "MSELoss"},
    "test_mse":                round(test_mse_final, 6),
    "mean_cell_pearson_r":     round(mean_cell_cor,  6),
    "mean_gene_pearson_r":     round(mean_gene_cor,  6),
    "mean_per_pert_pearson_r": round(mean_pert_cor,  6),
    "per_perturbation_eval":   pert_eval,
    "history":                 history,
}
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
logger.info("Metrics saved → %s", METRICS_PATH)

# ══════════════════════════════════════════════════════════════════════════
# 10. Save model
# ══════════════════════════════════════════════════════════════════════════
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes, "n_genes": n_genes, "n_classes": n_classes,
    "gene_emb_dim": GENE_EMB_DIM, "gcn_hidden": GCN_HIDDEN,
    "gcn_out": GCN_OUT, "ctrl_hidden": CTRL_HIDDEN, "dropout": DROPOUT,
    "mean_ctrl": mean_ctrl,
}, MODEL_PATH)
logger.info("Model saved → %s", MODEL_PATH)

# ══════════════════════════════════════════════════════════════════════════
# 11. Comparison with MLP baseline
# ══════════════════════════════════════════════════════════════════════════
print("\n── Model Comparison ─────────────────────────────────────────────────")
print(f"  {'Metric':<30}  {'MLP Baseline':>14}  {'Graph Model':>12}  {'Δ':>8}")
print("  " + "─" * 70)

if PREV_METRICS.exists():
    prev = json.loads(PREV_METRICS.read_text())
    rows_cmp = [
        ("Test MSE",               prev["test_mse"],               test_mse_final, False),
        ("Cell-level Pearson r",   prev["mean_cell_pearson_r"],     mean_cell_cor,  True),
        ("Gene-level Pearson r",   prev["mean_gene_pearson_r"],     mean_gene_cor,  True),
        ("Per-perturbation r",     prev["mean_per_pert_pearson_r"], mean_pert_cor,  True),
    ]
    for label, b, g, higher_better in rows_cmp:
        delta = g - b
        sign  = "+" if delta >= 0 else ""
        mark  = "▲" if (delta > 0) == higher_better else ("▼" if delta != 0 else "")
        print(f"  {label:<30}  {b:>14.4f}  {g:>12.4f}  {sign}{delta:.4f} {mark}")
else:
    print(f"  {'Test MSE':<30}  {'—':>14}  {test_mse_final:>12.4f}")
    print(f"  {'Cell-level Pearson r':<30}  {'—':>14}  {mean_cell_cor:>12.4f}")
    print(f"  {'Gene-level Pearson r':<30}  {'—':>14}  {mean_gene_cor:>12.4f}")
    print(f"  {'Per-perturbation r':<30}  {'—':>14}  {mean_pert_cor:>12.4f}")

print("  " + "─" * 70)

sorted_perts = sorted(pert_eval.items(), key=lambda x: x[1]["pearson_r"])
print(f"\n  Hardest perturbations (graph model):")
for p, v in sorted_perts[:5]:
    print(f"    {p:<36s}  r={v['pearson_r']:+.3f}  n={v['n_cells']}")
print(f"  Easiest perturbations (graph model):")
for p, v in sorted_perts[-5:]:
    print(f"    {p:<36s}  r={v['pearson_r']:+.3f}  n={v['n_cells']}")

print("\nGraph perturbation model training complete.")
