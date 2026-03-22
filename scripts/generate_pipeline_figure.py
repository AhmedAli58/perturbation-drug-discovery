"""Generate a clean, minimal pipeline diagram saved to reports/figures/pipeline.png"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
SLATE   = "#4A5568"   # data / preprocessing
BLUE    = "#3182CE"   # classification
GREEN   = "#2F855A"   # expression prediction
PURPLE  = "#6B46C1"   # zero-shot VAE
RESULT  = "#1A202C"   # result labels
ARROW   = "#718096"

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 11)
ax.set_ylim(0, 9)
ax.axis("off")


def rbox(ax, cx, cy, w, h, label, color, fontsize=10.5, alpha=1.0):
    """Rounded rectangle centered at (cx, cy)."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor=color, edgecolor="white",
        linewidth=0, alpha=alpha, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="semibold",
            color="white", zorder=4, wrap=False)


def down_arrow(ax, cx, y1, y2):
    ax.annotate("", xy=(cx, y2), xytext=(cx, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=1.4, mutation_scale=12),
                zorder=2)


def split_arrow(ax, x1, y1, x2, y2):
    """Diagonal arrow from split box down to model column."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=1.4, mutation_scale=12,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(5.5, 8.7, "Perturbation-Based Drug Target Discovery  —  Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold", color=RESULT)

# ── 1. Data ───────────────────────────────────────────────────────────────────
rbox(ax, 5.5, 8.1, 5.8, 0.65,
     "Norman 2019 Perturb-seq   ·   111,391 cells   ·   237 perturbations",
     SLATE, fontsize=10)

down_arrow(ax, 5.5, 7.77, 7.52)

# ── 2. Preprocessing (single merged box) ─────────────────────────────────────
rbox(ax, 5.5, 7.25, 5.8, 0.55,
     "QC filtering   →   Normalise + log1p   →   2,000 HVGs",
     SLATE, fontsize=10)

down_arrow(ax, 5.5, 6.97, 6.67)

# ── 3. Split ──────────────────────────────────────────────────────────────────
rbox(ax, 5.5, 6.4, 4.4, 0.52,
     "Stratified split   ·   80% train  /  10% val  /  10% test",
     SLATE, fontsize=10)

# ── Branching arrows from split → 3 columns ──────────────────────────────────
COL = [1.8, 5.5, 9.2]   # column x positions
for cx in COL:
    split_arrow(ax, 5.5, 6.14, cx, 5.62)

# ── Column headers ────────────────────────────────────────────────────────────
ax.text(COL[0], 5.48, "Classification", ha="center", va="top",
        fontsize=9.5, fontweight="bold", color=BLUE)
ax.text(COL[1], 5.48, "Expression Prediction", ha="center", va="top",
        fontsize=9.5, fontweight="bold", color=GREEN)
ax.text(COL[2], 5.48, "Zero-shot", ha="center", va="top",
        fontsize=9.5, fontweight="bold", color=PURPLE)

# ── 4. Models ─────────────────────────────────────────────────────────────────
# Classification
rbox(ax, COL[0], 4.85, 2.9, 0.52, "Logistic Regression", BLUE, fontsize=9.5)
down_arrow(ax, COL[0], 4.59, 4.29)
rbox(ax, COL[0], 4.0, 2.9, 0.52, "MLP Classifier", BLUE, fontsize=9.5)

# Expression Prediction
rbox(ax, COL[1], 4.85, 2.9, 0.52, "Effect MLP", GREEN, fontsize=9.5)
down_arrow(ax, COL[1], 4.59, 4.29)
rbox(ax, COL[1], 4.0, 2.9, 0.52, "Graph GCN  (STRING PPI)", GREEN, fontsize=9.5)

# Zero-shot
rbox(ax, COL[2], 4.42, 2.9, 0.52, "scGen VAE", PURPLE, fontsize=9.5)

# ── Arrows down to results ────────────────────────────────────────────────────
down_arrow(ax, COL[0], 3.74, 3.32)
down_arrow(ax, COL[1], 3.74, 3.32)
down_arrow(ax, COL[2], 4.16, 3.32)

# ── 5. Results ────────────────────────────────────────────────────────────────
def result_box(ax, cx, cy, w, h, line1, line2, color):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor=color, edgecolor="white",
        linewidth=0, alpha=0.12, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(cx, cy + 0.14, line1, ha="center", va="center",
            fontsize=11, fontweight="bold", color=color, zorder=4)
    ax.text(cx, cy - 0.18, line2, ha="center", va="center",
            fontsize=8.5, color=RESULT, alpha=0.65, zorder=4)


result_box(ax, COL[0], 2.95, 2.9, 0.82,
           "Top-1  46%  ·  Top-5  71%",
           "random chance = 0.4%",
           BLUE)

result_box(ax, COL[1], 2.95, 2.9, 0.82,
           "Per-pert r = 0.9957",
           "naive baseline r = 0.9829",
           GREEN)

result_box(ax, COL[2], 2.95, 2.9, 0.82,
           "Zero-shot r = 0.9843",
           "44 unseen perturbations",
           PURPLE)

# ── Thin divider line between models and results ──────────────────────────────
ax.axhline(3.45, xmin=0.06, xmax=0.94, color="#E2E8F0", lw=1.2, zorder=1)

plt.tight_layout(pad=0.3)
out = FIG_DIR / "pipeline.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out}")
