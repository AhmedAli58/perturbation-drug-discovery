"""Generate a publication-quality pipeline diagram saved to reports/figures/pipeline.png"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "data":    "#1B4F72",   # dark navy
    "prep":    "#1A7A6E",   # teal
    "split":   "#5D6D7E",   # slate
    "clf":     "#2980B9",   # blue  (classification models)
    "eff":     "#27AE60",   # green (expression models)
    "vae":     "#8E44AD",   # purple (VAE / zero-shot)
    "eval":    "#C0392B",   # red (evaluation)
    "ppi":     "#E67E22",   # orange (STRING PPI)
    "white":   "#FFFFFF",
    "text_l":  "#FFFFFF",   # light text (on dark bg)
    "text_d":  "#1C1C1C",   # dark text
    "arrow":   "#555555",
    "bg":      "#F8F9FA",
}

fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")


def box(ax, x, y, w, h, label, sublabel=None, color="#2980B9",
        fontsize=11, radius=0.3, text_color="#FFFFFF"):
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3,
    )
    ax.add_patch(patch)
    if sublabel:
        ax.text(x, y + 0.13, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(x, y - 0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 2.5, color=text_color, alpha=0.85, zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)


def section_bg(ax, x, y, w, h, color, alpha=0.07, label="", label_x=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.25",
        facecolor=color, edgecolor=color, linewidth=1.2,
        alpha=alpha, zorder=1,
    )
    ax.add_patch(patch)
    if label:
        ax.text(label_x or (x + w/2), y + h + 0.08, label,
                ha="center", va="bottom", fontsize=8.5,
                color=color, fontweight="bold", alpha=0.8)


# ── Section backgrounds ───────────────────────────────────────────────────────
section_bg(ax, 0.3, 7.8, 15.4, 1.7,  C["prep"],  label="Preprocessing",  label_x=8)
section_bg(ax, 0.3, 4.0, 4.8,  3.4,  C["clf"],   label="Classification",  label_x=2.7)
section_bg(ax, 5.4, 4.0, 5.8,  3.4,  C["eff"],   label="Expression Prediction", label_x=8.3)
section_bg(ax, 11.5,4.0, 4.2,  3.4,  C["vae"],   label="Zero-shot",       label_x=13.6)

# ── Data source ───────────────────────────────────────────────────────────────
box(ax, 8, 9.3, 5.0, 0.85,
    "Norman 2019 Perturb-seq",
    "111,391 cells  ·  237 CRISPR perturbations  ·  K562 cell line",
    color=C["data"], fontsize=11)

arrow(ax, 8, 8.88, 8, 8.6)

# ── Preprocessing ─────────────────────────────────────────────────────────────
box(ax, 4.2, 8.2, 3.8, 0.7, "QC Filtering",
    "min genes = 200  ·  max mito = 20%",
    color=C["prep"], fontsize=10)

box(ax, 8.0, 8.2, 3.6, 0.7, "Normalise + log1p",
    "library-size 10,000  ·  Scanpy",
    color=C["prep"], fontsize=10)

box(ax, 11.9, 8.2, 3.4, 0.7, "2,000 HVGs",
    "Seurat v3 flavour",
    color=C["prep"], fontsize=10)

# Preprocessing arrows (horizontal)
arrow(ax, 6.1, 8.2, 6.2, 8.2)
arrow(ax, 9.8, 8.2, 10.2, 8.2)

# Down to split
arrow(ax, 8, 7.85, 8, 7.55)

# ── Split ─────────────────────────────────────────────────────────────────────
box(ax, 8, 7.25, 4.4, 0.55,
    "Stratified 80 / 10 / 10 Split  —  by perturbation group",
    color=C["split"], fontsize=10)

# ── Branching arrows ──────────────────────────────────────────────────────────
#  to LogReg (1.6), MLP (3.2), EffMLP (6.2), GraphGCN (8.5), scGenVAE (13.6)
branch_targets = [1.6, 3.2, 6.2, 8.8, 13.6]
for bx in branch_targets:
    ax.annotate("", xy=(bx, 6.85), xytext=(8, 6.97),
                arrowprops=dict(arrowstyle="->", color=C["arrow"],
                                lw=1.6, connectionstyle="arc3,rad=0.0"),
                zorder=2)

# ── Model boxes ───────────────────────────────────────────────────────────────
# Classification
box(ax, 1.6, 6.3, 2.6, 1.0, "Logistic Regression",
    "sklearn · lbfgs · linear baseline",
    color=C["clf"], fontsize=9.5)

box(ax, 3.2, 6.3, 2.2, 1.0, "MLP Classifier",
    "3 layers · 512→256→237",
    color=C["clf"], fontsize=9.5)

# Expression prediction
box(ax, 6.2, 6.3, 2.6, 1.0, "Effect MLP",
    "control + pert emb → expr",
    color=C["eff"], fontsize=9.5)

box(ax, 8.8, 6.3, 2.6, 1.0, "Graph GCN",
    "STRING PPI prior · 2-layer GCN",
    color=C["eff"], fontsize=9.5)

# VAE
box(ax, 13.6, 6.3, 2.6, 1.0, "scGen VAE",
    "KL annealing · latent arithmetic",
    color=C["vae"], fontsize=9.5)

# STRING PPI reference
box(ax, 11.2, 6.3, 1.7, 0.6, "STRING PPI",
    "v12.0 · score ≥ 700",
    color=C["ppi"], fontsize=8.5)
ax.annotate("", xy=(9.4, 6.15), xytext=(11.2, 6.15),
            arrowprops=dict(arrowstyle="->", color=C["ppi"],
                            lw=1.4, linestyle="dashed",
                            connectionstyle="arc3,rad=0.0"), zorder=2)

# ── Evaluation boxes ──────────────────────────────────────────────────────────
# Classification eval
arrow(ax, 1.6, 5.8, 1.6, 5.2)
arrow(ax, 3.2, 5.8, 3.2, 5.2)
box(ax, 2.4, 4.85, 3.4, 0.65,
    "Top-1: 37% → 46%",
    "237 classes  ·  random = 0.4%",
    color=C["eval"], fontsize=9.5)

# Expression eval
arrow(ax, 6.2, 5.8, 6.2, 5.2)
arrow(ax, 8.8, 5.8, 8.8, 5.2)
box(ax, 7.5, 4.85, 3.6, 0.65,
    "Per-pert r: 0.9829 → 0.9957",
    "naive baseline → Effect MLP",
    color=C["eval"], fontsize=9.5)

# Zero-shot eval
arrow(ax, 13.6, 5.8, 13.6, 5.2)
box(ax, 13.6, 4.85, 3.0, 0.65,
    "Zero-shot  r = 0.9843",
    "44 unseen perturbations",
    color=C["eval"], fontsize=9.5)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C["data"],  "Raw data"),
    (C["prep"],  "Preprocessing"),
    (C["clf"],   "Classification models"),
    (C["eff"],   "Expression prediction models"),
    (C["vae"],   "scGen VAE (zero-shot)"),
    (C["ppi"],   "External biological network"),
    (C["eval"],  "Key results"),
]
for i, (color, label) in enumerate(legend_items):
    px = 0.55 + i * 2.22
    patch = FancyBboxPatch((px, 0.18), 0.28, 0.28,
                           boxstyle="round,pad=0.02,rounding_size=0.05",
                           facecolor=color, edgecolor="white", lw=0.8)
    ax.add_patch(patch)
    ax.text(px + 0.38, 0.32, label, va="center", fontsize=8.5, color="#333")

ax.text(8, 9.85,
        "Computational Pipeline — Perturbation-Based Drug Target Discovery",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1C1C1C")

plt.tight_layout(pad=0.2)
out = FIG_DIR / "pipeline.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=C["bg"])
plt.close(fig)
print(f"Saved → {out}")
