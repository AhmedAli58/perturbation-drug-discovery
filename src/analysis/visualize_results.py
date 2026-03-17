"""
Essential visual validation — 3 plots.

1. pred_vs_real.png       — predicted vs actual mean gene expression
                            (scGen VAE, 6 selected perturbations)
2. model_comparison.png   — bar chart comparing all models across tasks
3. unseen_generalization.png — zero-shot / oracle / baseline per-pert r

All figures saved to reports/figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RES  = PROJECT_ROOT / "data" / "results"
MODS = PROJECT_ROOT / "data" / "models"

# ── Shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":      150,
})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3"]

# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def load_json(name: str) -> dict:
    p = RES / name
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}. Run the relevant training script first.")
    return json.loads(p.read_text())


class ScGenVAE(nn.Module):
    """Minimal definition matching the saved checkpoint."""
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

    def decode(self, z, pidx):
        return self.decoder(torch.cat([z, self.pert_emb(pidx)], dim=1))


# ════════════════════════════════════════════════════════════════════════════
# Load shared data (used by plots 1 and 3)
# ════════════════════════════════════════════════════════════════════════════
_adata_path = PROJECT_ROOT / "data" / "processed" / "norman2019_processed.h5ad"
_model_path = MODS / "scgen_model.pt"
if not _adata_path.exists():
    raise FileNotFoundError(f"Processed dataset not found at {_adata_path}. Run: make preprocess")
if not _model_path.exists():
    raise FileNotFoundError(f"scGen model not found at {_model_path}. Run: make train_scgen")

print("Loading AnnData …")
adata = sc.read_h5ad(_adata_path)
X_all = adata.X
if sp.issparse(X_all):
    X_all = X_all.toarray()
X_all = X_all.astype(np.float32)

perts_col = adata.obs["perturbation"].astype(str).values
n_genes   = adata.n_vars

print("Loading scGen VAE …")
ck = torch.load(_model_path, map_location="cpu", weights_only=False)
model = ScGenVAE(
    n_genes=ck["n_genes"], n_perts=ck["n_classes"],
    latent_dim=ck["latent_dim"], pert_emb_dim=ck["pert_emb_dim"],
    enc_h1=ck["enc_h1"], enc_h2=ck["enc_h2"],
    dec_h1=ck["dec_h1"], dec_h2=ck["dec_h2"],
)
model.load_state_dict(ck["model_state_dict"])
model.eval()

classes     = ck["classes"]
pert_to_idx = {p: i for i, p in enumerate(classes)}
mean_ctrl   = ck["mean_ctrl"]

ctrl_mask  = perts_col == "control"
mean_ctrl_t = torch.from_numpy(X_all[ctrl_mask].mean(axis=0)).unsqueeze(0)

with torch.no_grad():
    z_ctrl, _ = model.encode(mean_ctrl_t)      # (1, 128) — fixed reference

# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Predicted vs Actual mean gene expression
# ════════════════════════════════════════════════════════════════════════════
print("Generating Plot 1: pred_vs_real …")

# Pick 6 perturbations spanning a range of per-pert r values
scgen_m = load_json("scgen_metrics.json")
pert_r  = {
    p: v["pearson_r"]
    for p, v in scgen_m["per_perturbation_eval"].items()
    if v["n_cells"] >= 100
}
sorted_perts = sorted(pert_r, key=pert_r.get)
n_p = len(sorted_perts)
selected = [
    sorted_perts[int(n_p * f)]
    for f in [0.02, 0.18, 0.35, 0.55, 0.75, 0.97]
]

fig, ax = plt.subplots(figsize=(6, 5.5))

for i, pert in enumerate(selected):
    mask     = perts_col == pert
    mean_obs = X_all[mask].mean(axis=0)          # (2000,)

    pidx = torch.tensor([pert_to_idx[pert]], dtype=torch.long)
    with torch.no_grad():
        mean_pred = model.decode(z_ctrl, pidx).squeeze(0).numpy()

    r, _ = pearsonr(mean_pred, mean_obs)

    # Sub-sample genes for speed (plot 500 random genes)
    rng  = np.random.default_rng(42 + i)
    idx  = rng.choice(n_genes, size=500, replace=False)

    ax.scatter(
        mean_obs[idx], mean_pred[idx],
        s=8, alpha=0.45, color=PALETTE[i],
        label=f"{pert}  (r={r:.3f})",
        linewidths=0,
    )

# Identity line
lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1]),
]
ax.plot(lims, lims, "k--", lw=1, alpha=0.5, zorder=0)

ax.set_xlabel("Observed mean expression", fontsize=12)
ax.set_ylabel("Predicted mean expression (scGen VAE)", fontsize=12)
ax.set_title("Predicted vs Actual Gene Expression\n(6 perturbations, 500 genes each)", fontsize=13)
ax.legend(fontsize=8, framealpha=0.7, loc="upper left",
          title="Perturbation", title_fontsize=8)

plt.tight_layout()
out1 = FIG_DIR / "pred_vs_real.png"
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out1}")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Model comparison bar chart (two task groups)
# ════════════════════════════════════════════════════════════════════════════
print("Generating Plot 2: model_comparison …")

bl_m    = load_json("baseline_metrics.json")
mlp_m   = load_json("mlp_metrics.json")
eff_m   = load_json("perturbation_effect_metrics.json")
graph_m = load_json("graph_model_metrics.json")

# Group A — Classification (accuracy)
clf_models  = ["Logistic\nRegression", "MLP\nClassifier"]
clf_top1    = [bl_m["accuracy"],       mlp_m["accuracy"]]
clf_top5    = [bl_m["top5_accuracy"],  mlp_m["top5_accuracy"]]

# Group B — Perturbation effect prediction (per-pert Pearson r)
eff_models  = ["Vanilla\nMLP", "Graph\nGCN", "scGen\nVAE"]
eff_r       = [
    eff_m["mean_per_pert_pearson_r"],
    graph_m["mean_per_pert_pearson_r"],
    scgen_m["mean_per_pert_pearson_r"],
]
eff_gene_r  = [
    eff_m["mean_gene_pearson_r"],
    graph_m["mean_gene_pearson_r"],
    scgen_m["mean_gene_pearson_r"],
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                gridspec_kw={"wspace": 0.38})

# ── Left: Classification accuracy ──────────────────────────────────────────
x1    = np.arange(len(clf_models))
w     = 0.34
bars1 = ax1.bar(x1 - w/2, [v * 100 for v in clf_top1], w,
                label="Top-1 accuracy", color=PALETTE[0], zorder=3)
bars2 = ax1.bar(x1 + w/2, [v * 100 for v in clf_top5], w,
                label="Top-5 accuracy", color=PALETTE[1], zorder=3)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
             f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

ax1.set_xticks(x1); ax1.set_xticklabels(clf_models, fontsize=10)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.set_title("Task 1 — Perturbation Classification", fontsize=12)
ax1.set_ylim(0, 85)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.axhline(100 / 237, color="gray", ls=":", lw=1.2, label="Random")
ax1.text(1.55, 100 / 237 + 1, "Random\nchance", fontsize=7.5, color="gray", va="bottom")

# ── Right: Per-perturbation Pearson r ──────────────────────────────────────
x2 = np.arange(len(eff_models))
bars3 = ax2.bar(x2 - w/2, eff_r,      w, label="Per-pert Pearson r",  color=PALETTE[2], zorder=3)
bars4 = ax2.bar(x2 + w/2, eff_gene_r, w, label="Gene-level Pearson r", color=PALETTE[3], zorder=3)

for bar in list(bars3) + list(bars4):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
             f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

ax2.set_xticks(x2); ax2.set_xticklabels(eff_models, fontsize=10)
ax2.set_ylabel("Pearson r", fontsize=11)
ax2.set_title("Task 2 — Perturbation Effect Prediction", fontsize=12)
ax2.set_ylim(0, 1.09)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3, zorder=0)

plt.suptitle("Model Comparison — Norman 2019 Perturb-seq", fontsize=13, y=1.02)
plt.tight_layout()
out2 = FIG_DIR / "model_comparison.png"
fig.savefig(out2, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out2}")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Unseen perturbation generalization
# ════════════════════════════════════════════════════════════════════════════
print("Generating Plot 3: unseen_generalization …")

unseen_m  = load_json("unseen_perturbation_metrics.json")
per_pert  = unseen_m["per_perturbation"]
agg       = unseen_m["aggregate_metrics"]

perts_sorted = sorted(per_pert, key=lambda p: per_pert[p]["pearson_r_vae_zeroshot"])
r_zs = [per_pert[p]["pearson_r_vae_zeroshot"] for p in perts_sorted]
r_or = [per_pert[p]["pearson_r_vae_oracle"]   for p in perts_sorted]
r_bl = [per_pert[p]["pearson_r_baseline"]      for p in perts_sorted]
xx   = np.arange(len(perts_sorted))

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(12, 7.5),
    gridspec_kw={"height_ratios": [2.8, 1], "hspace": 0.38},
)

# ── Top: per-perturbation dot strip ────────────────────────────────────────
ax_top.scatter(xx, r_bl, s=48, marker="s", color=PALETTE[3], alpha=0.85,
               label="Baseline (nearest-seen Δ expr)", zorder=3)
ax_top.scatter(xx, r_zs, s=52, marker="o", color=PALETTE[0], alpha=0.85,
               label="VAE zero-shot (nearest-seen emb)", zorder=4)
ax_top.scatter(xx, r_or, s=28, marker="^", color=PALETTE[2], alpha=0.75,
               label="VAE oracle (true emb, upper bound)", zorder=5)

# Annotate the hardest and easiest
for idx, p in [(0, perts_sorted[0]), (-1, perts_sorted[-1])]:
    ax_top.annotate(
        p,
        xy=(xx[idx], r_zs[idx]),
        xytext=(xx[idx] + (1.5 if idx == 0 else -1.5), r_zs[idx] - 0.014),
        fontsize=7.5, color="#333",
        arrowprops=dict(arrowstyle="-", lw=0.6, color="#777"),
    )

ax_top.set_xticks(xx)
ax_top.set_xticklabels(perts_sorted, rotation=65, ha="right", fontsize=7)
ax_top.set_ylabel("Per-perturbation Pearson r", fontsize=11)
ax_top.set_title(
    "Unseen Perturbation Generalization — 44 held-out perturbations\n"
    "Norman 2019 Perturb-seq  ·  scGen VAE",
    fontsize=12,
)
ax_top.set_ylim(0.92, 1.005)
ax_top.legend(fontsize=9, loc="lower right")
ax_top.grid(axis="y", alpha=0.25)

# ── Bottom: aggregate summary bar chart ────────────────────────────────────
conditions = ["VAE\nzero-shot", "VAE\noracle", "Baseline\n(nearest Δ)"]
metrics_to_show = {
    "Per-pert r":  [agg["vae_zeroshot"]["mean_per_pert_pearson_r"],
                    agg["vae_oracle"]["mean_per_pert_pearson_r"],
                    agg["baseline_nearest_seen"]["mean_per_pert_pearson_r"]],
    "Cell-level r":[agg["vae_zeroshot"]["mean_cell_pearson_r"],
                    agg["vae_oracle"]["mean_cell_pearson_r"],
                    agg["baseline_nearest_seen"]["mean_cell_pearson_r"]],
}

n_cond   = len(conditions)
n_met    = len(metrics_to_show)
bw       = 0.28
offsets  = np.linspace(-(n_met - 1) * bw / 2, (n_met - 1) * bw / 2, n_met)
x_cond   = np.arange(n_cond)

for k, ((label, vals), offset, color) in enumerate(
    zip(metrics_to_show.items(), offsets, PALETTE[4:])
):
    bars = ax_bot.bar(x_cond + offset, vals, bw,
                      label=label, color=color, zorder=3)
    for bar, v in zip(bars, vals):
        ax_bot.text(bar.get_x() + bar.get_width() / 2, v + 0.0005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

ax_bot.set_xticks(x_cond)
ax_bot.set_xticklabels(conditions, fontsize=10)
ax_bot.set_ylabel("Pearson r", fontsize=11)
ax_bot.set_title("Aggregate metrics across 44 unseen perturbations", fontsize=11)
ax_bot.set_ylim(0.855, 0.905)
ax_bot.legend(fontsize=9, loc="lower right")
ax_bot.grid(axis="y", alpha=0.25)

plt.tight_layout()
out3 = FIG_DIR / "unseen_generalization.png"
fig.savefig(out3, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out3}")

print("\nAll 3 figures saved to reports/figures/")
print(f"  {out1.name}")
print(f"  {out2.name}")
print(f"  {out3.name}")
