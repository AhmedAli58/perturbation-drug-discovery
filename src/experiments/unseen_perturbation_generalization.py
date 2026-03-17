"""
Unseen perturbation generalization experiment.

Key question: can the model predict the transcriptional response to a
perturbation it has never seen during training?

Approach (using the existing trained scGen VAE)
────────────────────────────────────────────────
1. Filter perturbations to those with ≥100 cells.
2. Randomly assign 20% as "unseen" (held-out), 80% as "seen".
3. For each unseen perturbation p:
   - Find its nearest seen perturbation (by STRING PPI gene proximity,
     falling back to expression-space cosine similarity in training data).
   - VAE zero-shot  : decode(z_ctrl, emb_nearest_seen)
   - VAE oracle     : decode(z_ctrl, emb_true_unseen)   ← upper bound
   - Baseline       : mean_ctrl + Δ_expression_nearest_seen
4. Evaluate all three against real unseen expression.
5. Report per-perturbation Pearson r, gene-level r, cell-level r, MSE.

Inputs  : data/processed/norman2019_processed.h5ad
          data/models/scgen_model.pt
          data/external/string_ppi_edges.tsv  (optional — used for nearest-nbr)
Output  : data/results/unseen_perturbation_metrics.json
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "norman2019_processed.h5ad"
MODEL_PATH     = PROJECT_ROOT / "data" / "models"   / "scgen_model.pt"
STRING_PATH    = PROJECT_ROOT / "data" / "external"  / "string_ppi_edges.tsv"
OUT_PATH       = PROJECT_ROOT / "data" / "results"   / "unseen_perturbation_metrics.json"

from src.constants import SEED, CONTROL_LABEL  # noqa: E402

MIN_CELLS     = 100
UNSEEN_FRAC   = 0.20
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# ══════════════════════════════════════════════════════════════════════════
# 1. Minimal VAE definition (matches saved checkpoint exactly)
# ══════════════════════════════════════════════════════════════════════════
class ScGenVAE(nn.Module):
    def __init__(self, n_genes, n_perts, latent_dim, pert_emb_dim,
                 enc_h1, enc_h2, dec_h1, dec_h2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, enc_h1), nn.ReLU(),
            nn.Linear(enc_h1,  enc_h2), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(enc_h2, latent_dim)
        self.fc_logvar = nn.Linear(enc_h2, latent_dim)
        self.pert_emb  = nn.Embedding(n_perts, pert_emb_dim)
        self.decoder   = nn.Sequential(
            nn.Linear(latent_dim + pert_emb_dim, dec_h1), nn.ReLU(),
            nn.Linear(dec_h1, dec_h2), nn.ReLU(),
            nn.Linear(dec_h2, n_genes),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, pert_idx):
        e = self.pert_emb(pert_idx)
        return self.decoder(torch.cat([z, e], dim=1))

# ══════════════════════════════════════════════════════════════════════════
# 2. Load model checkpoint
# ══════════════════════════════════════════════════════════════════════════
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"scGen model not found at {MODEL_PATH}. Run: make train_scgen"
    )
if not PROCESSED_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found at {PROCESSED_PATH}. Run: make preprocess"
    )
logger.info("Loading checkpoint from %s …", MODEL_PATH)
ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

model = ScGenVAE(
    n_genes     = ck["n_genes"],
    n_perts     = ck["n_classes"],
    latent_dim  = ck["latent_dim"],
    pert_emb_dim= ck["pert_emb_dim"],
    enc_h1      = ck["enc_h1"],
    enc_h2      = ck["enc_h2"],
    dec_h1      = ck["dec_h1"],
    dec_h2      = ck["dec_h2"],
)
model.load_state_dict(ck["model_state_dict"])
model.eval()

classes     = ck["classes"]                          # list[str], length 237
n_classes   = ck["n_classes"]
mean_ctrl   = ck["mean_ctrl"]                        # np.ndarray (2000,)
ctrl_idx    = ck["ctrl_idx"]
pert_to_idx = {p: i for i, p in enumerate(classes)}

logger.info("Model loaded — %d perturbation classes", n_classes)

# ══════════════════════════════════════════════════════════════════════════
# 3. Load dataset
# ══════════════════════════════════════════════════════════════════════════
logger.info("Loading AnnData …")
adata = sc.read_h5ad(PROCESSED_PATH)

X_all = adata.X
if sp.issparse(X_all):
    X_all = X_all.toarray()
X_all = X_all.astype(np.float32)

perts_col = adata.obs["perturbation"].astype(str).values
split_col = adata.obs["split"].values
n_genes   = adata.n_vars

# ══════════════════════════════════════════════════════════════════════════
# 4. Filter & split perturbations into seen / unseen
# ══════════════════════════════════════════════════════════════════════════
pert_counts = {
    p: int((perts_col == p).sum())
    for p in classes if p != CONTROL_LABEL
}
eligible = sorted(
    [p for p, n in pert_counts.items() if n >= MIN_CELLS]
)

n_unseen = max(1, int(len(eligible) * UNSEEN_FRAC))
unseen_perts = set(rng.choice(eligible, size=n_unseen, replace=False).tolist())
seen_perts   = set(eligible) - unseen_perts

logger.info(
    "Eligible perts: %d  |  seen: %d  |  unseen (held-out): %d",
    len(eligible), len(seen_perts), len(unseen_perts),
)

# ══════════════════════════════════════════════════════════════════════════
# 5. Build nearest-seen-perturbation map via STRING PPI
#    (fallback: expression cosine similarity in training data)
# ══════════════════════════════════════════════════════════════════════════

# 5a. Load STRING adjacency if available
string_nbrs: dict[str, set[str]] = defaultdict(set)
if STRING_PATH.exists():
    with open(STRING_PATH) as f:
        next(f)
        for line in f:
            a, b, _ = line.strip().split("\t")
            string_nbrs[a].add(b)
            string_nbrs[b].add(a)
    logger.info("STRING PPI loaded: %d genes with neighbours", len(string_nbrs))
else:
    logger.warning("STRING PPI not found — will use expression fallback only")


def ppi_score(pert_a: str, pert_b: str) -> int:
    """Shared PPI neighbours between target genes of two perturbations."""
    genes_a = set(pert_a.split("_"))
    genes_b = set(pert_b.split("_"))
    score = 0
    for ga in genes_a:
        for gb in genes_b:
            if gb in string_nbrs.get(ga, set()):
                score += 10          # direct interaction — high reward
            score += len(string_nbrs.get(ga, set()) & string_nbrs.get(gb, set()))
    return score


# 5b. Compute training mean expression per seen perturbation (for fallback)
train_mask = split_col == "train"
seen_mean_expr: dict[str, np.ndarray] = {}
for p in seen_perts:
    mask = (perts_col == p) & train_mask
    if mask.sum() > 0:
        seen_mean_expr[p] = X_all[mask].mean(axis=0)

ctrl_mask    = perts_col == CONTROL_LABEL
mean_ctrl_arr = X_all[ctrl_mask].mean(axis=0)      # (2000,) — matches ck["mean_ctrl"]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-8))


def nearest_seen(unseen_pert: str) -> str:
    """Find the seen perturbation most similar to `unseen_pert`.

    Priority:
    1. STRING PPI score > 0 → highest PPI score wins.
    2. Fallback: cosine similarity of training-mean expression shifts.
    """
    best_pert, best_score = None, -1

    # --- PPI-based ---
    for sp_name in seen_perts:
        sc_ppi = ppi_score(unseen_pert, sp_name)
        if sc_ppi > best_score:
            best_score = sc_ppi
            best_pert  = sp_name

    if best_score > 0:
        return best_pert   # type: ignore[return-value]

    # --- Expression cosine similarity fallback ---
    best_pert, best_cos = None, -2.0
    unseen_genes = set(unseen_pert.split("_"))
    # Represent unseen pert by the *control* vector (all we know without test data)
    for sp_name, sp_expr in seen_mean_expr.items():
        cos = cosine_sim(sp_expr - mean_ctrl_arr, mean_ctrl_arr)  # shift similarity
        if cos > best_cos:
            best_cos  = cos
            best_pert = sp_name

    return best_pert or list(seen_perts)[0]


nearest_map: dict[str, str] = {}
for up in unseen_perts:
    nearest_map[up] = nearest_seen(up)

logger.info("Nearest-seen map built (sample):")
for up in sorted(unseen_perts)[:5]:
    logger.info("  %s  →  %s", up, nearest_map[up])

# ══════════════════════════════════════════════════════════════════════════
# 6. Encode mean control → fixed latent reference z_ctrl
# ══════════════════════════════════════════════════════════════════════════
mean_ctrl_t = torch.from_numpy(mean_ctrl_arr).unsqueeze(0)   # (1, 2000)
with torch.no_grad():
    z_ctrl, _ = model.encode(mean_ctrl_t)                    # (1, 128)

# ══════════════════════════════════════════════════════════════════════════
# 7. Evaluate on unseen perturbations — three predictors
# ══════════════════════════════════════════════════════════════════════════
results_per_pert: dict[str, dict] = {}

all_pred_zeroshot, all_pred_oracle, all_pred_baseline, all_true = [], [], [], []

for up in sorted(unseen_perts):
    # True cells for this unseen perturbation (all splits — none used in training)
    mask      = perts_col == up
    X_true    = X_all[mask]                          # (n_cells, 2000)
    mean_true = X_true.mean(axis=0)                  # (2000,)

    pidx_true    = pert_to_idx[up]
    pidx_nearest = pert_to_idx[nearest_map[up]]

    with torch.no_grad():
        # VAE zero-shot: use nearest seen embedding
        pred_zs = model.decode(
            z_ctrl,
            torch.tensor([pidx_nearest], dtype=torch.long),
        ).squeeze(0).numpy()                          # (2000,)

        # VAE oracle: use the perturbation's own (true) embedding
        pred_or = model.decode(
            z_ctrl,
            torch.tensor([pidx_true], dtype=torch.long),
        ).squeeze(0).numpy()

    # Baseline: mean_ctrl + expression delta of nearest seen pert
    delta_nearest = seen_mean_expr.get(nearest_map[up], mean_ctrl_arr) - mean_ctrl_arr
    pred_bl       = mean_ctrl_arr + delta_nearest     # (2000,)

    # Metrics vs mean_true (per-perturbation Pearson)
    def safe_r(a, b):
        try:
            r, _ = pearsonr(a, b)
            return float(r) if not np.isnan(r) else 0.0
        except Exception:
            return 0.0

    r_zs = safe_r(pred_zs, mean_true)
    r_or = safe_r(pred_or, mean_true)
    r_bl = safe_r(pred_bl, mean_true)

    results_per_pert[up] = {
        "nearest_seen":          nearest_map[up],
        "n_cells":               int(mask.sum()),
        "pearson_r_vae_zeroshot":round(r_zs, 6),
        "pearson_r_vae_oracle":  round(r_or, 6),
        "pearson_r_baseline":    round(r_bl, 6),
        "mse_vae_zeroshot":      round(float(np.mean((pred_zs - mean_true)**2)), 6),
        "mse_vae_oracle":        round(float(np.mean((pred_or - mean_true)**2)), 6),
        "mse_baseline":          round(float(np.mean((pred_bl - mean_true)**2)), 6),
    }

    # Accumulate cell-level arrays (replicate prediction for each true cell)
    n_cells = X_true.shape[0]
    all_pred_zeroshot.append(np.tile(pred_zs, (n_cells, 1)))
    all_pred_oracle.append(  np.tile(pred_or, (n_cells, 1)))
    all_pred_baseline.append(np.tile(pred_bl, (n_cells, 1)))
    all_true.append(X_true)

# Stack all cells
P_zs  = np.concatenate(all_pred_zeroshot, axis=0)
P_or  = np.concatenate(all_pred_oracle,   axis=0)
P_bl  = np.concatenate(all_pred_baseline, axis=0)
Y     = np.concatenate(all_true,          axis=0)

# Cell-level Pearson r (across genes per cell)
def mean_cell_cor(pred, true):
    return float(np.nanmean([safe_r(pred[i], true[i]) for i in range(len(pred))]))

# Gene-level Pearson r (across unseen test cells per gene)
def mean_gene_cor(pred, true):
    cors = []
    for g in range(true.shape[1]):
        if true[:, g].std() > 0 and pred[:, g].std() > 0:
            cors.append(safe_r(pred[:, g], true[:, g]))
    return float(np.nanmean(cors)) if cors else 0.0

# Per-perturbation aggregate metrics
def agg(key):
    return float(np.mean([v[key] for v in results_per_pert.values()]))

# ══════════════════════════════════════════════════════════════════════════
# 8. Save results
# ══════════════════════════════════════════════════════════════════════════
summary = {
    "experiment": "unseen_perturbation_generalization",
    "config": {
        "min_cells_per_pert": MIN_CELLS,
        "unseen_fraction":    UNSEEN_FRAC,
        "seed":               SEED,
        "nearest_method":     "STRING PPI score (fallback: expression cosine)",
    },
    "split": {
        "n_eligible_perts": len(eligible),
        "n_seen_perts":     len(seen_perts),
        "n_unseen_perts":   len(unseen_perts),
    },
    "aggregate_metrics": {
        "vae_zeroshot": {
            "mean_per_pert_pearson_r": round(agg("pearson_r_vae_zeroshot"), 6),
            "mean_cell_pearson_r":     round(mean_cell_cor(P_zs, Y), 6),
            "mean_gene_pearson_r":     round(mean_gene_cor(P_zs, Y), 6),
            "mean_mse":                round(agg("mse_vae_zeroshot"),   6),
        },
        "vae_oracle": {
            "mean_per_pert_pearson_r": round(agg("pearson_r_vae_oracle"), 6),
            "mean_cell_pearson_r":     round(mean_cell_cor(P_or, Y), 6),
            "mean_gene_pearson_r":     round(mean_gene_cor(P_or, Y), 6),
            "mean_mse":                round(agg("mse_vae_oracle"),   6),
        },
        "baseline_nearest_seen": {
            "mean_per_pert_pearson_r": round(agg("pearson_r_baseline"), 6),
            "mean_cell_pearson_r":     round(mean_cell_cor(P_bl, Y), 6),
            "mean_gene_pearson_r":     round(mean_gene_cor(P_bl, Y), 6),
            "mean_mse":                round(agg("mse_baseline"),     6),
        },
    },
    "per_perturbation": results_per_pert,
}
OUT_PATH.write_text(json.dumps(summary, indent=2))
logger.info("Results saved → %s", OUT_PATH)

# ══════════════════════════════════════════════════════════════════════════
# 9. Print summary
# ══════════════════════════════════════════════════════════════════════════
agg_m = summary["aggregate_metrics"]

print("\n── Unseen Perturbation Generalization ────────────────────────────────")
print(f"  Eligible perturbations (≥{MIN_CELLS} cells) : {len(eligible)}")
print(f"  Training (seen)  perts                : {len(seen_perts)}")
print(f"  Held-out (unseen) perts               : {len(unseen_perts)}")

col = 18
print(f"\n  {'Metric':<30}  {'VAE zero-shot':>{col}}  {'VAE oracle':>{col}}  {'Baseline':>{col}}")
print("  " + "─" * (30 + 3 * (col + 2)))

rows = [
    ("Per-pert Pearson r",  "mean_per_pert_pearson_r"),
    ("Cell-level Pearson r","mean_cell_pearson_r"),
    ("Gene-level Pearson r","mean_gene_pearson_r"),
    ("MSE",                 "mean_mse"),
]
for label, key in rows:
    zs = agg_m["vae_zeroshot"][key]
    or_ = agg_m["vae_oracle"][key]
    bl  = agg_m["baseline_nearest_seen"][key]
    print(f"  {label:<30}  {zs:>{col}.4f}  {or_:>{col}.4f}  {bl:>{col}.4f}")

print("\n  Nearest-seen perturbation map (all held-out perts):")
print(f"  {'Unseen pert':<30}  {'Nearest seen':<30}  {'r_zeroshot':>10}  {'r_oracle':>8}  {'r_baseline':>10}")
print("  " + "─" * 94)
for up, v in sorted(results_per_pert.items(), key=lambda x: x[1]["pearson_r_vae_zeroshot"]):
    print(f"  {up:<30}  {v['nearest_seen']:<30}  "
          f"{v['pearson_r_vae_zeroshot']:>10.3f}  "
          f"{v['pearson_r_vae_oracle']:>8.3f}  "
          f"{v['pearson_r_baseline']:>10.3f}")

# Generalisation gap
gap = agg_m["vae_oracle"]["mean_per_pert_pearson_r"] - agg_m["vae_zeroshot"]["mean_per_pert_pearson_r"]
zs_vs_bl = agg_m["vae_zeroshot"]["mean_per_pert_pearson_r"] - agg_m["baseline_nearest_seen"]["mean_per_pert_pearson_r"]

print(f"\n  Generalization gap (oracle − zero-shot) : {gap:+.4f}")
print(f"  VAE zero-shot vs baseline (Δr)         : {zs_vs_bl:+.4f}  "
      f"({'VAE better' if zs_vs_bl > 0 else 'Baseline better'})")
print("\nUnseen perturbation generalization experiment complete.")
