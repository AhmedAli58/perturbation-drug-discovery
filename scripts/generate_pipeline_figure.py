"""
Concourse CI-style pipeline diagram.
Dark navy · monospace · colored job boxes · clean connector lines.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BG      = "#1d2532"
RESOURCE= "#2c3750"
RES_BDR = "#3d4f6e"
JOB_GRN = "#4caf50"
JOB_PRP = "#7c5cbf"
JOB_BLU = "#3a7bd5"
LINE    = "#6b7fa3"
LINE_DIM= "#4a5f80"
TXT     = "#e8eaf0"
TXT_DIM = "#8899bb"
MONO    = "monospace"

# ── canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6.2))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(1.9, 6.65)
ax.axis("off")

# ── geometry ──────────────────────────────────────────────────────────────────
JOB_W, JOB_H = 1.35, 1.0     # job box dims (shorter → gap between boxes)
RES_W, RES_H = 2.1,  0.50
OUT_W        = 2.55

IX   = 1.1    # input resources x
PJX  = 3.1    # preprocess job x
PRX  = 5.15   # processed.h5ad x
TRX  = 6.5    # vertical trunk x
JX   = 7.85   # job boxes x
OX   = 10.35  # output resources x

# 4 jobs, evenly spaced with 0.3 gap between boxes
JYS  = [6.05, 4.85, 3.65, 2.45]   # classify / effect-mlp / graph-gcn / scgen-vae
PY   = (JYS[0] + JYS[-1]) / 2     # preprocess y  ≈ 4.25
NY   = 5.15   # norman-2019 y
SY   = 3.35   # string-ppi y


# ── helpers ───────────────────────────────────────────────────────────────────
def rbox(cx, cy, label, w=RES_W, h=RES_H, color=RESOURCE,
         border=RES_BDR, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.09",
        facecolor=color, edgecolor=border, linewidth=1.2, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fs, color=TXT, fontfamily=MONO,
            fontweight="bold" if bold else "normal", zorder=4)


def jbox(cx, cy, label, color=JOB_GRN):
    ax.add_patch(FancyBboxPatch(
        (cx - JOB_W/2, cy - JOB_H/2), JOB_W, JOB_H,
        boxstyle="round,pad=0.0,rounding_size=0.09",
        facecolor=color, edgecolor="none", linewidth=0, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=9.5, color="white", fontfamily=MONO,
            fontweight="bold", zorder=4)


def hline(x1, x2, y, color=LINE, lw=1.5, dash=False):
    ax.plot([x1, x2], [y, y], color=color, lw=lw,
            linestyle=(0, (5, 3)) if dash else "solid",
            solid_capstyle="round", zorder=2)


def vline(x, y1, y2, color=LINE, lw=1.5):
    ax.plot([x, x], [y1, y2], color=color, lw=lw,
            solid_capstyle="round", zorder=2)


def elbow(x1, y1, xm, y2, color=LINE, lw=1.5, dash=False):
    """Horizontal then vertical L-shape."""
    ax.plot([x1, xm, xm], [y1, y1, y2], color=color, lw=lw,
            linestyle=(0, (5, 3)) if dash else "solid",
            solid_capstyle="round", zorder=2)


# ── inputs ────────────────────────────────────────────────────────────────────
rbox(IX, NY, "norman-2019")
rbox(IX, SY, "string-ppi-v12")

# ── inputs → preprocess ───────────────────────────────────────────────────────
MX = PJX - JOB_W/2 - 0.05    # merge x (just left of preprocess)
# both inputs elbow into merge x then down/up to PY
elbow(IX + RES_W/2, NY, MX, PY)
elbow(IX + RES_W/2, SY, MX, PY)

# ── preprocess job ────────────────────────────────────────────────────────────
jbox(PJX, PY, "preprocess")

# ── preprocess → processed.h5ad ───────────────────────────────────────────────
hline(PJX + JOB_W/2, PRX - RES_W/2, PY)
rbox(PRX, PY, "processed.h5ad", w=2.3)

# ── processed.h5ad → trunk ────────────────────────────────────────────────────
hline(PRX + 2.3/2, TRX, PY)

# ── vertical trunk ────────────────────────────────────────────────────────────
vline(TRX, JYS[-1], JYS[0])

# ── branches trunk → jobs ─────────────────────────────────────────────────────
for y in JYS:
    hline(TRX, JX - JOB_W/2, y)

# ── job boxes ─────────────────────────────────────────────────────────────────
jbox(JX, JYS[0], "classify",   color=JOB_BLU)
jbox(JX, JYS[1], "effect-mlp", color=JOB_GRN)
jbox(JX, JYS[2], "graph-gcn",  color=JOB_GRN)
jbox(JX, JYS[3], "scgen-vae",  color=JOB_PRP)

# ── string-ppi dashed → graph-gcn ─────────────────────────────────────────────
# route horizontally at SY (below preprocess bottom) then elbow up to graph-gcn
elbow(IX + RES_W/2, SY, TRX - 0.08, JYS[2],
      color=LINE_DIM, lw=1.2, dash=True)

# ── jobs → output resources ───────────────────────────────────────────────────
colors = [JOB_BLU, JOB_GRN, JOB_GRN, JOB_PRP]
labels = [
    "top-1: 46%  ·  top-5: 71%",
    "per-pert r = 0.9957",
    "per-pert r = 0.9903",
    "zero-shot r = 0.9843",
]
for y, color, label in zip(JYS, colors, labels):
    hline(JX + JOB_W/2, OX - OUT_W/2, y, color=color, lw=1.4)
    rbox(OX, y, label, w=OUT_W)

# ── header ────────────────────────────────────────────────────────────────────
ax.text(0.25, 6.55, "perturbation-drug-discovery  /  main",
        ha="left", va="center", fontsize=8.5, color=TXT_DIM, fontfamily=MONO)

plt.tight_layout(pad=0.15)
out = FIG_DIR / "pipeline.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved → {out}")
