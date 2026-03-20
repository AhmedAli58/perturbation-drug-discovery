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
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.stats import pearsonr, gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RES  = PROJECT_ROOT / "data" / "results"
MODS = PROJECT_ROOT / "data" / "models"

# ── Publication-grade style (Nature / Cell Research standard) ───────────────
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.size":            11,
    "axes.linewidth":       0.8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.25,
    "grid.linewidth":       0.5,
    "grid.color":           "#CCCCCC",
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
    "legend.frameon":       True,
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#CCCCCC",
    "legend.fontsize":      9,
})

# Paul Tol "bright" palette — colorblind-friendly, Nature-journal standard
TOL = ["#0077BB", "#CC3311", "#009988", "#EE7733",
       "#AA3377", "#33BBEE", "#BBBBBB"]


def add_panel_label(ax, label, x=-0.12, y=1.05):
    """Add bold A/B/C panel label, standard for pharma publications."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="right")

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
# PLOT 1 — Density-coloured scatter: Predicted vs Observed (2×3 panel)
# ════════════════════════════════════════════════════════════════════════════
print("Generating Plot 1: pred_vs_real …")

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

fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharex=False, sharey=False)
fig.patch.set_facecolor("white")
axes_flat = axes.flatten()
panel_labels = ["A", "B", "C", "D", "E", "F"]

for i, (pert, ax) in enumerate(zip(selected, axes_flat)):
    mask     = perts_col == pert
    mean_obs = X_all[mask].mean(axis=0)

    pidx = torch.tensor([pert_to_idx[pert]], dtype=torch.long)
    with torch.no_grad():
        mean_pred = model.decode(z_ctrl, pidx).squeeze(0).numpy()

    r, _ = pearsonr(mean_pred, mean_obs)

    # KDE density colouring — standard in pharma/genomics publications
    xy    = np.vstack([mean_obs, mean_pred])
    z     = gaussian_kde(xy)(xy)
    order = z.argsort()   # plot dense points on top

    sc_plot = ax.scatter(
        mean_obs[order], mean_pred[order],
        c=z[order], cmap="YlOrRd",
        s=12, alpha=0.85, linewidths=0, rasterized=True,
    )

    # Identity line
    lo = min(mean_obs.min(), mean_pred.min()) - 0.1
    hi = max(mean_obs.max(), mean_pred.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], color="#444444", lw=1.2,
            ls="--", alpha=0.6, zorder=0, label="Identity")

    # Styled r-value box (top-left)
    ax.text(0.05, 0.95, f"$r$ = {r:.4f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left", color="#1A1A1A",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#CCCCCC", alpha=0.9))

    # Cell/gene count (bottom-right, muted)
    ax.text(0.97, 0.04,
            f"n = {int(mask.sum()):,} cells  ·  2,000 genes",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            color="#888888")

    ax.set_title(pert, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Observed mean expression", fontsize=10)
    ax.set_ylabel("Predicted mean expression", fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    add_panel_label(ax, panel_labels[i])

# Shared colorbar
cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="YlOrRd")
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("Point density", fontsize=10)
cb.ax.tick_params(labelsize=9)

fig.suptitle(
    "scGen VAE — Predicted vs Observed Mean Gene Expression",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
out1 = FIG_DIR / "pred_vs_real.png"
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out1}")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Three-panel model comparison (pharma publication style)
# ════════════════════════════════════════════════════════════════════════════
print("Generating Plot 2: model_comparison …")

bl_m    = load_json("baseline_metrics.json")
mlp_m   = load_json("mlp_metrics.json")
eff_m   = load_json("perturbation_effect_metrics.json")
graph_m = load_json("graph_model_metrics.json")

# Extract per-perturbation r distributions (not just means)
def pert_r_dist(metrics: dict) -> list[float]:
    return [v["pearson_r"] for v in metrics["per_perturbation_eval"].values()
            if not np.isnan(v["pearson_r"])]

eff_dist   = pert_r_dist(eff_m)
graph_dist = pert_r_dist(graph_m)
scgen_dist = pert_r_dist(scgen_m)
naive_r    = 0.9829   # from baseline_naive.json

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(14, 5),
    gridspec_kw={"wspace": 0.42},
)
fig.patch.set_facecolor("white")

# ── Panel A: Classification accuracy ───────────────────────────────────────
clf_labels = ["Logistic\nRegression", "MLP\nClassifier"]
top1 = [bl_m["accuracy"] * 100, mlp_m["accuracy"] * 100]
top5 = [bl_m["top5_accuracy"] * 100, mlp_m["top5_accuracy"] * 100]

x = np.arange(len(clf_labels))
w = 0.33
b1 = ax1.bar(x - w/2, top1, w, color=TOL[0], label="Top-1", zorder=3,
             linewidth=0, alpha=0.88)
b2 = ax1.bar(x + w/2, top5, w, color=TOL[1], label="Top-5", zorder=3,
             linewidth=0, alpha=0.88)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
             f"{h:.1f}%", ha="center", va="bottom", fontsize=9,
             fontweight="bold", color="#333")

# Random-chance reference
ax1.axhline(100 / 237, color="#999", ls="--", lw=1.2, zorder=2)
ax1.text(1.47, 100 / 237 + 0.8, "Random\nchance",
         fontsize=8, color="#888", va="bottom", ha="right")

ax1.set_xticks(x)
ax1.set_xticklabels(clf_labels, fontsize=10)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.set_ylim(0, 88)
ax1.set_title("Perturbation\nClassification", fontsize=11, fontweight="bold", pad=8)
ax1.legend(loc="upper left", fontsize=9)
add_panel_label(ax1, "A")

# ── Panel B: Per-perturbation Pearson r — DISTRIBUTION (box + strip) ───────
model_names  = ["Effect\nMLP", "Graph\nGCN", "scGen\nVAE"]
model_colors = [TOL[0], TOL[2], TOL[3]]
dists        = [eff_dist, graph_dist, scgen_dist]

bp = ax2.boxplot(
    dists,
    patch_artist=True,
    widths=0.45,
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(color="#555", linewidth=0.9),
    capprops=dict(color="#555", linewidth=0.9),
    flierprops=dict(marker="o", markersize=4, alpha=0.5,
                    markerfacecolor="#999", markeredgewidth=0),
    zorder=3,
)
for patch, color in zip(bp["boxes"], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Jitter strip
rng = np.random.default_rng(42)
for xi, (dist, color) in enumerate(zip(dists, model_colors), start=1):
    jitter = rng.uniform(-0.12, 0.12, len(dist))
    ax2.scatter(np.full(len(dist), xi) + jitter, dist,
                s=6, alpha=0.35, color=color, linewidths=0, zorder=2)

# Naive baseline reference
ax2.axhline(naive_r, color="#CC3311", ls="--", lw=1.5, zorder=4,
            label=f"Naive baseline  r = {naive_r}")
ax2.text(3.55, naive_r + 0.0005, f"{naive_r}", fontsize=8.5,
         color="#CC3311", va="bottom", ha="right")

ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(model_names, fontsize=10)
ax2.set_ylabel("Per-perturbation Pearson r", fontsize=11)
ax2.set_ylim(0.93, 1.005)
ax2.set_title("Expression Prediction\n(per-pert r distribution)", fontsize=11,
              fontweight="bold", pad=8)
ax2.legend(loc="lower right", fontsize=8.5)
add_panel_label(ax2, "B")

# ── Panel C: Gene-level Pearson r — horizontal bars ────────────────────────
gene_r_vals  = [eff_m["mean_gene_pearson_r"],
                graph_m["mean_gene_pearson_r"],
                scgen_m["mean_gene_pearson_r"]]
gene_labels  = ["Effect MLP", "Graph GCN", "scGen VAE"]

y = np.arange(len(gene_labels))
bars = ax3.barh(y, gene_r_vals, color=model_colors, alpha=0.85,
                linewidth=0, zorder=3)

for bar, val in zip(bars, gene_r_vals):
    ax3.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left",
             fontsize=10, fontweight="bold", color="#333")

ax3.set_yticks(y)
ax3.set_yticklabels(gene_labels, fontsize=10)
ax3.set_xlabel("Gene-level Pearson r", fontsize=11)
ax3.set_xlim(0, 0.16)
ax3.set_title("Gene-level Prediction\n(ranking individual genes)", fontsize=11,
              fontweight="bold", pad=8)
ax3.invert_yaxis()
add_panel_label(ax3, "C")

fig.suptitle("Model Comparison — Norman 2019 Perturb-seq  ·  237 CRISPR Perturbations",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
out2 = FIG_DIR / "model_comparison.png"
fig.savefig(out2)
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
ax_top.scatter(xx, r_bl, s=40, marker="s", color=TOL[6], alpha=0.75,
               label="Baseline (nearest-seen Δ expr)", zorder=3)
ax_top.scatter(xx, r_zs, s=48, marker="o", color=TOL[0], alpha=0.85,
               label="VAE zero-shot (nearest-seen emb)", zorder=4)
ax_top.scatter(xx, r_or, s=28, marker="^", color=TOL[2], alpha=0.75,
               label="VAE oracle (true emb — upper bound)", zorder=5)

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
    zip(metrics_to_show.items(), offsets, TOL[4:])
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
