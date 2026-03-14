"""
PerturbExplorer  ·  Norman 2019 Perturb-seq
Single-page perturbation discovery dashboard

Run:  streamlit run app.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.sparse as sp
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

ROOT            = Path(__file__).parent
ADATA_PATH      = ROOT / "data" / "processed" / "norman2019_processed.h5ad"
RESULTS_DIR     = ROOT / "data" / "results"
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "embeddings_20k.npz"
STATS_PATH      = ROOT / "data" / "processed" / "precomputed_stats.npz"

MODEL_KEYS = {"Effect MLP": "effect", "scGen VAE": "scgen", "Graph GCN": "graph"}

# dark scientific palette
C_BG     = "#0d1117"
C_PANEL  = "#161b22"
C_BORDER = "#30363d"
C_TEXT   = "#c9d1d9"
C_MUTED  = "#8b949e"
C_ACCENT = "#58a6ff"
C_UP     = "#f97583"    # upregulated
C_DOWN   = "#79c0ff"    # downregulated
C_GRAY   = "#3b424e"    # muted / other
C_CTRL   = "#4a7fa8"    # control cells (muted teal-blue)

# stable categorical palette (26 + 10 + 9 = 45 distinct colors before cycling)
_PAL = (
    px.colors.qualitative.Alphabet
    + px.colors.qualitative.D3
    + px.colors.qualitative.Set1
)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PerturbExplorer · Norman 2019",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  —  dark scientific theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
/* ── global background ────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {{
    background: {C_BG} !important;
    color: {C_TEXT};
}}
[data-testid="stHeader"] {{
    background: {C_BG};
    border-bottom: 1px solid {C_BORDER};
}}

/* ── hide sidebar entirely ────────────────────────────────────────────────── */
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] {{
    display: none !important;
}}

/* ── body spacing ─────────────────────────────────────────────────────────── */
.block-container {{
    padding: 0.55rem 1.3rem 1rem !important;
    max-width: 100% !important;
}}

/* ── control bar ──────────────────────────────────────────────────────────── */
.ctrl-bar {{
    background: {C_PANEL};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 8px 16px 4px;
    margin-bottom: 8px;
}}

/* ── panel section header ─────────────────────────────────────────────────── */
.ph {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {C_MUTED};
    border-bottom: 1px solid {C_BORDER};
    padding-bottom: 3px;
    margin: 10px 0 5px;
}}

/* ── info pill ────────────────────────────────────────────────────────────── */
.pill {{
    display: inline-block;
    background: {C_PANEL};
    border: 1px solid {C_BORDER};
    border-radius: 5px;
    padding: 2px 9px;
    font-size: 11px;
    color: {C_TEXT};
    margin: 2px 2px 2px 0;
}}
.pill b {{ color: {C_ACCENT}; }}

/* ── widget label color ───────────────────────────────────────────────────── */
label,
.stSelectbox label,
.stRadio label,
[data-testid="stWidgetLabel"] {{
    color: {C_MUTED} !important;
    font-size: 11px !important;
}}

/* ── select box dark ──────────────────────────────────────────────────────── */
[data-baseweb="select"] > div {{
    background: {C_PANEL} !important;
    border-color: {C_BORDER} !important;
    color: {C_TEXT} !important;
}}
[data-baseweb="popover"] li,
[data-baseweb="popover"] ul {{
    background: {C_PANEL} !important;
    color: {C_TEXT} !important;
}}
[data-baseweb="popover"] li:hover {{
    background: {C_BORDER} !important;
}}

/* ── radio ────────────────────────────────────────────────────────────────── */
[data-testid="stRadio"] > div > label {{
    color: {C_TEXT} !important;
    font-size: 12px !important;
}}

/* ── button ───────────────────────────────────────────────────────────────── */
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {{
    background: {C_PANEL} !important;
    border-color: {C_BORDER} !important;
    color: {C_TEXT} !important;
    font-size: 12px !important;
}}
button[kind="secondary"]:hover {{
    border-color: {C_ACCENT} !important;
    color: {C_ACCENT} !important;
}}

/* ── hide streamlit chrome ────────────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stToolbar"] {{
    visibility: hidden;
    display: none;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY BASE — dark theme
# ══════════════════════════════════════════════════════════════════════════════

_PL = dict(
    paper_bgcolor=C_PANEL,
    plot_bgcolor=C_PANEL,
    font=dict(color=C_TEXT, size=10, family="Inter, ui-sans-serif, system-ui, sans-serif"),
    margin=dict(l=42, r=10, t=30, b=36),
)


def _ax(**kw):
    """Dark-themed axis dict."""
    return dict(
        gridcolor=C_BORDER,
        zerolinecolor=C_BORDER,
        linecolor=C_BORDER,
        tickfont=dict(color=C_MUTED, size=9),
        title_font=dict(color=C_MUTED, size=10),
        **kw,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_adata():
    if not ADATA_PATH.exists():
        return None, f"Not found: {ADATA_PATH}"
    try:
        import anndata as ad
        return ad.read_h5ad(str(ADATA_PATH)), None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def load_results():
    fmap = {
        "baseline": "baseline_metrics.json",
        "mlp":      "mlp_metrics.json",
        "effect":   "perturbation_effect_metrics.json",
        "graph":    "graph_model_metrics.json",
        "scgen":    "scgen_metrics.json",
        "unseen":   "unseen_perturbation_metrics.json",
    }
    return {
        k: (json.loads((RESULTS_DIR / v).read_text())
            if (RESULTS_DIR / v).exists() else {})
        for k, v in fmap.items()
    }


@st.cache_data(show_spinner=False)
def load_precomputed_embeddings():
    if not EMBEDDINGS_PATH.exists():
        return None
    try:
        f = np.load(str(EMBEDDINGS_PATH), allow_pickle=True)
        return dict(
            idx=f["idx"].astype(int),
            umap3d=f["umap3d"],
            umap2d=f["umap2d"],
            tsne2d=f["tsne2d"],
            perturbation=f["perturbation"].astype(str),
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_gene_expr(_adata, gene):
    """Dense 1-D expression vector for one gene, all cells."""
    if gene not in _adata.var_names:
        return None
    j = _adata.var_names.get_loc(gene)
    col = _adata.X[:, j]
    return (np.asarray(col.todense()).flatten()
            if sp.issparse(col) else np.asarray(col).flatten())


@st.cache_data(show_spinner=False)
def load_precomputed_stats():
    """Load disk-cached perturbation stats. Returns None if file not found.

    Run scripts/precompute_stats.py once to generate the file.
    Returns dict with keys: gene_names, control_mean, perturbation_means,
    delta, cell_counts — all O(1) to look up by perturbation name.
    """
    if not STATS_PATH.exists():
        return None
    data       = np.load(str(STATS_PATH), allow_pickle=True)
    pert_names = list(data["perturbation_names"].astype(str))
    pm_arr     = data["perturbation_means"]   # (n_perts, n_genes)
    delta_arr  = data["delta"]                # (n_perts, n_genes)
    counts_arr = data["cell_counts"]          # (n_perts,)
    return dict(
        gene_names         = list(data["gene_names"].astype(str)),
        control_mean       = data["control_mean"],
        perturbation_means = {p: pm_arr[i]    for i, p in enumerate(pert_names)},
        delta              = {p: delta_arr[i] for i, p in enumerate(pert_names)},
        cell_counts        = {p: int(counts_arr[i]) for i, p in enumerate(pert_names)},
    )


@st.cache_data(show_spinner=False)
def get_obs_labels(_adata, pc_col):
    """Cache obs perturbation labels as a plain numpy array (read once)."""
    return _adata.obs[pc_col].values.copy()


@st.cache_data(show_spinner=False)
def get_all_cell_ids(_adata):
    """Cache all cell IDs as a string array (indexed by cell position)."""
    return _adata.obs.index.astype(str).values


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _detect_pert_col(obs):
    for c in ("perturbation", "gene", "condition", "pert_label"):
        if c in obs.columns:
            return c
    return next((c for c in obs.columns if obs[c].dtype == object), None)


def _safe(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def get_embedding_for_view(embedding_type: str, dimension: str, embeddings: dict):
    """Return (coords, idx, render_dim, label).

    render_dim is derived from the ACTUAL array shape, never from the UI toggle.
    This is the only safe source of truth for the 2D/3D rendering branch.

    Fallback rules:
      t-SNE 3D  → warns, uses tsne2d  → render_dim = 2
      UMAP 3D shape mismatch → warns, uses umap2d → render_dim = 2
      UMAP 3D missing        → warns, uses umap2d → render_dim = 2

    Exits via st.stop() (no crash) if an embedding is missing or malformed.
    """
    if embeddings is None:
        st.warning("Embeddings dict is None — cannot render.")
        st.stop()

    idx = embeddings["idx"]

    # ── t-SNE ────────────────────────────────────────────────────────────────
    if embedding_type == "t-SNE":
        if dimension == "3D":
            st.warning("t-SNE 3D is not precomputed — falling back to t-SNE 2D.")
        coords = embeddings.get("tsne2d")
        if coords is None:
            st.warning("t-SNE 2D not found in precomputed embeddings file.")
            st.stop()
        if coords.ndim != 2 or coords.shape[1] < 2:
            st.warning(f"t-SNE embedding has unexpected shape {coords.shape}.")
            st.stop()
        return coords, idx, 2, "t-SNE"

    # ── UMAP 3D ──────────────────────────────────────────────────────────────
    if dimension == "3D":
        coords = embeddings.get("umap3d")
        if coords is not None:
            if coords.ndim == 2 and coords.shape[1] == 3:
                return coords, idx, 3, "UMAP"
            # array exists but shape is wrong — warn and fall through to 2D
            st.warning(
                f"UMAP 3D array has shape {coords.shape} (expected (n, 3)) — "
                "falling back to UMAP 2D."
            )
        else:
            st.warning("UMAP 3D not found in precomputed embeddings — falling back to UMAP 2D.")

    # ── UMAP 2D (requested or fallback) ──────────────────────────────────────
    coords = embeddings.get("umap2d")
    if coords is None:
        st.warning("UMAP 2D not found in precomputed embeddings file.")
        st.stop()
    if coords.ndim != 2 or coords.shape[1] < 2:
        st.warning(f"UMAP 2D embedding has unexpected shape {coords.shape}.")
        st.stop()
    return coords, idx, 2, "UMAP"


def _color_map(names):
    """Stable alphabetical name → color dict."""
    return {p: _PAL[i % len(_PAL)] for i, p in enumerate(sorted(set(names)))}


def safe_rgba(hex_color: str, alpha: float) -> str:
    """Convert a CSS hex color + alpha to a valid rgba() string for Plotly.

    safe_rgba("#3b424e", 0.27)  ->  "rgba(59,66,78,0.27)"
    Never passes 8-digit hex or bare alpha-appended strings.
    """
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

adata,    adata_err = load_adata()
results             = load_results()
emb_dict            = load_precomputed_embeddings()

pc        = _detect_pert_col(adata.obs) if adata is not None else None
pert_list = sorted(adata.obs[pc].unique().tolist()) if (adata is not None and pc) else ["(no data)"]
gene_list = list(adata.var_names) if adata is not None else ["(no data)"]

# ── load precomputed stats (O(1) lookup; run scripts/precompute_stats.py once) ──
precomp: dict | None = load_precomputed_stats()
if precomp is None and adata is not None and pc is not None:
    st.warning(
        "Precomputed stats not found. "
        "Run `python scripts/precompute_stats.py` for instant interactions. "
        "Computing in-memory for this session…",
        icon="⚠️",
    )
    # in-memory fallback: compute once per session via st.cache_data
    @st.cache_data(show_spinner="Computing perturbation statistics (one-time)…")
    def _build_precomp(_adata, pc_col):
        import scipy.sparse as _sp
        obs_col   = _adata.obs[pc_col].values
        gene_names = list(_adata.var_names)
        ctrl_mask  = obs_col == "control"
        def _mf32(mask):
            X = _adata.X[mask]
            v = np.asarray(X.mean(axis=0) if _sp.issparse(X) else X.mean(axis=0))
            return v.flatten().astype(np.float32)
        ctrl_mean  = _mf32(ctrl_mask)
        perts      = sorted(p for p in np.unique(obs_col) if p != "control")
        pm_dict, d_dict, cnt_dict = {}, {}, {}
        for p in perts:
            mask = obs_col == p
            if not mask.any():
                continue
            pm = _mf32(mask)
            pm_dict[p] = pm
            d_dict[p]  = pm - ctrl_mean
            cnt_dict[p] = int(mask.sum())
        return dict(gene_names=gene_names, control_mean=ctrl_mean,
                    perturbation_means=pm_dict, delta=d_dict, cell_counts=cnt_dict)
    precomp = _build_precomp(adata, pc)

# ── cache obs labels + cell IDs (avoid re-reading AnnData on every render) ──
obs_labels  = get_obs_labels(adata, pc) if (adata is not None and pc) else None
all_cell_ids = get_all_cell_ids(adata)  if adata is not None else None

# ══════════════════════════════════════════════════════════════════════════════
#  TOP CONTROL BAR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="ctrl-bar">', unsafe_allow_html=True)

cb1, cb2, cb3, cb4, cb5, cb6, cb7 = st.columns([1.0, 0.9, 1.4, 2.4, 2.4, 1.5, 0.65])

with cb1:
    sel_emb = st.selectbox("Embedding", ["UMAP", "t-SNE"], key="sel_emb")

with cb2:
    sel_dims = st.selectbox("Dims", ["3D", "2D"], key="sel_dims")

with cb3:
    color_mode = st.selectbox(
        "Color by",
        ["Perturbation", "Gene Expr", "Model Residual"],
        key="color_mode",
    )

with cb4:
    sel_gene = st.selectbox("Gene", gene_list, key="sel_gene")

with cb5:
    default_pert_idx = min(10, max(0, len(pert_list) - 1))
    sel_pert = st.selectbox(
        "Perturbation", pert_list, index=default_pert_idx, key="sel_pert"
    )

with cb6:
    sel_model = st.selectbox("Model", list(MODEL_KEYS.keys()), key="sel_model")

with cb7:
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    reset_view = st.button("⟳", help="Reset view", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

MODEL_KEY = MODEL_KEYS[sel_model]

# thin separator
st.markdown(
    f"<hr style='border:none;border-top:1px solid {C_BORDER};margin:0 0 6px'>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT  —  left 70 %  |  right 30 %
# ══════════════════════════════════════════════════════════════════════════════

main_col, right_col = st.columns([7, 3], gap="medium")


# ──────────────────────────────────────────────────────────────────────────────
#  LEFT  —  EMBEDDING SCATTER
# ──────────────────────────────────────────────────────────────────────────────

with main_col:

    if adata is None:
        st.error(f"Dataset unavailable: {adata_err}")

    elif emb_dict is None:
        st.error(
            f"Precomputed embeddings not found at "
            f"`{EMBEDDINGS_PATH.relative_to(ROOT)}`. "
            "Run the embedding script first."
        )

    else:
        emb, s_idx, render_dim, emb_src = get_embedding_for_view(
            sel_emb, sel_dims, emb_dict
        )
        # render_dim is derived from emb.shape[1] — safe source of truth
        # (get_embedding_for_view calls st.stop() on unrecoverable errors,
        #  so execution never reaches here with a None emb)

        if emb is None:
            st.error(f"Embedding mode '{sel_emb} {sel_dims}' could not be resolved.")

        else:
            # ── shared setup ──────────────────────────────────────────────────
            embed_perts = emb_dict["perturbation"]   # (20k,) pert label per cell

            # semantic masks — used by perturbation mode and for size arrays
            ctrl_m = embed_perts == "control"
            hi_m   = embed_perts == sel_pert
            bg_m   = ~ctrl_m & ~hi_m

            # fallback banner when 3D was requested but 2D is rendered
            fallback_note = ""
            if sel_dims == "3D" and render_dim == 2:
                st.info(
                    f"⚠ {sel_emb} 3D not available — rendering {sel_emb} 2D",
                    icon=None,
                )
                fallback_note = " (2D fallback)"

            # ── hover data (aligned to s_idx) ─────────────────────────────────
            expr_all   = get_gene_expr(adata, sel_gene)
            expr_hover = (expr_all[s_idx] if expr_all is not None
                          else np.zeros(len(s_idx)))
            cell_ids   = all_cell_ids[s_idx]
            custom     = np.stack(
                [cell_ids, embed_perts, expr_hover.round(3)], axis=1
            )
            htmpl = (
                "<b>Cell:</b> %{customdata[0]}<br>"
                "<b>Pert:</b> %{customdata[1]}<br>"
                f"<b>{sel_gene}:</b> %{{customdata[2]}}<extra></extra>"
            )

            H_MAIN = 700 if render_dim == 3 else 660

            # ── shared layout fragments ────────────────────────────────────────
            _legend = dict(
                bgcolor=C_PANEL, bordercolor=C_BORDER, borderwidth=1,
                font=dict(size=10, color=C_TEXT), itemsizing="constant",
            )
            _cb = dict(
                thickness=10, len=0.65,
                tickfont=dict(color=C_MUTED, size=8),
            )

            # ══════════════════════════════════════════════════════════════════
            #  PERTURBATION MODE  —  3 semantic traces only
            #  background (gray) / control (muted blue) / selected (accent)
            # ══════════════════════════════════════════════════════════════════
            if color_mode == "Perturbation":
                n_sel     = int(hi_m.sum())
                n_ctrl    = int(ctrl_m.sum())
                title_txt = (
                    f"{emb_src} {render_dim}D{fallback_note}  ·  "
                    f"<b>{sel_pert}</b>  ({n_sel:,} cells)  ·  "
                    f"control ({n_ctrl:,})  ·  {len(s_idx):,} total"
                )

                fig = go.Figure()

                # layer 1 — background (all other perts, very muted)
                if bg_m.any():
                    if render_dim == 3:
                        fig.add_trace(go.Scatter3d(
                            x=emb[bg_m, 0], y=emb[bg_m, 1], z=emb[bg_m, 2],
                            mode="markers", name="other cells",
                            marker=dict(size=1.8, color=C_GRAY, opacity=0.18),
                            customdata=custom[bg_m], hovertemplate=htmpl,
                        ))
                    else:
                        fig.add_trace(go.Scattergl(
                            x=emb[bg_m, 0], y=emb[bg_m, 1],
                            mode="markers", name="other cells",
                            marker=dict(size=2, color=C_GRAY, opacity=0.18),
                            customdata=custom[bg_m], hovertemplate=htmpl,
                        ))

                # layer 2 — control cells
                if ctrl_m.any():
                    if render_dim == 3:
                        fig.add_trace(go.Scatter3d(
                            x=emb[ctrl_m, 0], y=emb[ctrl_m, 1], z=emb[ctrl_m, 2],
                            mode="markers", name="control",
                            marker=dict(size=2, color=C_CTRL, opacity=0.55),
                            customdata=custom[ctrl_m], hovertemplate=htmpl,
                        ))
                    else:
                        fig.add_trace(go.Scattergl(
                            x=emb[ctrl_m, 0], y=emb[ctrl_m, 1],
                            mode="markers", name="control",
                            marker=dict(size=2.5, color=C_CTRL, opacity=0.55),
                            customdata=custom[ctrl_m], hovertemplate=htmpl,
                        ))

                # layer 3 — selected perturbation (on top, full accent)
                if hi_m.any():
                    if render_dim == 3:
                        fig.add_trace(go.Scatter3d(
                            x=emb[hi_m, 0], y=emb[hi_m, 1], z=emb[hi_m, 2],
                            mode="markers", name=sel_pert,
                            marker=dict(
                                size=4, color=C_ACCENT, opacity=1.0,
                                line=dict(width=0.5, color="white"),
                            ),
                            customdata=custom[hi_m], hovertemplate=htmpl,
                        ))
                    else:
                        fig.add_trace(go.Scattergl(
                            x=emb[hi_m, 0], y=emb[hi_m, 1],
                            mode="markers", name=sel_pert,
                            marker=dict(
                                size=6, color=C_ACCENT, opacity=1.0,
                                line=dict(width=0.8, color="white"),
                            ),
                            customdata=custom[hi_m], hovertemplate=htmpl,
                        ))

            # ══════════════════════════════════════════════════════════════════
            #  CONTINUOUS COLOR MODES  —  Gene Expr / Model Residual
            # ══════════════════════════════════════════════════════════════════
            else:
                if color_mode == "Gene Expr":
                    expr_full = get_gene_expr(adata, sel_gene)
                    cvals  = (expr_full[s_idx] if expr_full is not None
                              else np.zeros(len(s_idx)))
                    clabel = sel_gene
                    cscale = "Plasma"
                else:  # Model Residual
                    ppe_raw = _safe(results, MODEL_KEY, "per_perturbation_eval") or {}
                    r_map   = {p: d.get("pearson_r", np.nan)
                               for p, d in ppe_raw.items()}
                    cvals   = np.array([r_map.get(p, np.nan) for p in embed_perts])
                    fill_r  = np.nanmean(cvals) if not np.all(np.isnan(cvals)) else 0.0
                    cvals   = np.where(np.isnan(cvals), fill_r, cvals)
                    clabel  = f"{sel_model} r"
                    cscale  = "RdYlGn"

                # selected pert cells slightly larger for orientation
                sizes = np.where(hi_m, 4.0, 2.0)

                title_txt = (
                    f"{emb_src} {render_dim}D{fallback_note}  ·  "
                    f"color: {clabel}  ·  {sel_pert} enlarged  ·  "
                    f"{len(s_idx):,} cells"
                )

                cb_spec = dict(
                    title=dict(text=clabel, font=dict(color=C_MUTED, size=9)),
                    **_cb,
                )

                if render_dim == 3:
                    fig = go.Figure(go.Scatter3d(
                        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                        mode="markers",
                        marker=dict(
                            size=sizes.tolist(), color=cvals,
                            colorscale=cscale, opacity=0.85,
                            colorbar=cb_spec,
                        ),
                        customdata=custom, hovertemplate=htmpl,
                    ))
                else:
                    fig = go.Figure(go.Scattergl(
                        x=emb[:, 0], y=emb[:, 1],
                        mode="markers",
                        marker=dict(
                            size=sizes.tolist(), color=cvals,
                            colorscale=cscale, opacity=0.85,
                            colorbar=cb_spec,
                        ),
                        customdata=custom, hovertemplate=htmpl,
                    ))

            # ── layout (shared, branch on render_dim only) ────────────────────
            title_cfg = dict(
                text=title_txt,
                font=dict(size=11, color=C_TEXT),
                x=0,
            )

            if render_dim == 3:
                fig.update_layout(
                    **_PL,
                    height=H_MAIN,
                    title=title_cfg,
                    scene=dict(
                        bgcolor=C_PANEL,
                        xaxis=dict(backgroundcolor=C_PANEL, gridcolor=C_BORDER,
                                   showbackground=True,
                                   title=dict(text=f"{emb_src}1",
                                              font=dict(color=C_MUTED)),
                                   tickfont=dict(color=C_MUTED, size=8)),
                        yaxis=dict(backgroundcolor=C_PANEL, gridcolor=C_BORDER,
                                   showbackground=True,
                                   title=dict(text=f"{emb_src}2",
                                              font=dict(color=C_MUTED)),
                                   tickfont=dict(color=C_MUTED, size=8)),
                        zaxis=dict(backgroundcolor=C_PANEL, gridcolor=C_BORDER,
                                   showbackground=True,
                                   title=dict(text=f"{emb_src}3",
                                              font=dict(color=C_MUTED)),
                                   tickfont=dict(color=C_MUTED, size=8)),
                    ),
                    legend=_legend,
                )
            else:
                fig.update_layout(
                    **_PL,
                    height=H_MAIN,
                    title=title_cfg,
                    xaxis=_ax(title=f"{emb_src}1", showgrid=True),
                    yaxis=_ax(title=f"{emb_src}2", showgrid=True),
                    legend=_legend,
                )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=dict(
                    displayModeBar=True,
                    modeBarButtonsToRemove=["select2d", "lasso2d", "toImage"],
                    displaylogo=False,
                ),
            )


# ──────────────────────────────────────────────────────────────────────────────
#  RIGHT  —  STACKED INSIGHT PANELS
# ──────────────────────────────────────────────────────────────────────────────

with right_col:

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL A  —  Gene Signal
    #  Selected gene expression distribution across top perturbed populations
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown(
        f'<div class="ph">Gene Signal &nbsp;·&nbsp; {sel_gene}</div>',
        unsafe_allow_html=True,
    )

    if adata is not None and pc is not None:
        expr_full_a = get_gene_expr(adata, sel_gene)

        if expr_full_a is None:
            st.warning(f"Gene `{sel_gene}` not found in dataset.")
        else:
            labels_a = obs_labels
            rng_a    = np.random.default_rng(42)

            # 3 groups: selected pert / control / all others pooled
            def _sample(arr, n=500):
                return arr[rng_a.choice(len(arr), n, replace=False)] if len(arr) > n else arr

            groups = {
                "other cells": _sample(
                    expr_full_a[(labels_a != "control") & (labels_a != sel_pert)]
                ),
                "control":     _sample(expr_full_a[labels_a == "control"]),
                sel_pert:      _sample(expr_full_a[labels_a == sel_pert]),
            }
            group_colors = {
                "other cells": C_GRAY,
                "control":     C_CTRL,
                sel_pert:      C_ACCENT,
            }

            fig_a = go.Figure()
            for gname, vals in groups.items():
                if len(vals) < 3:
                    continue
                col = group_colors[gname]
                fig_a.add_trace(go.Violin(
                    y=vals,
                    name=gname,
                    side="both",
                    box_visible=True,
                    meanline_visible=True,
                    meanline=dict(visible=True, color="white", width=1),
                    points=False,
                    line_color=col,
                    fillcolor=safe_rgba(col, 0.35 if gname == sel_pert else 0.20),
                    opacity=1.0,
                ))

            n_pert_cells = int((labels_a == sel_pert).sum())
            fig_a.update_layout(
                **_PL,
                height=240,
                showlegend=True,
                violingap=0.25,
                violingroupgap=0.1,
                title=dict(
                    text=f"{sel_gene}  ·  {sel_pert} vs control vs others",
                    font=dict(size=10, color=C_TEXT), x=0,
                ),
                xaxis=dict(
                    tickfont=dict(color=C_TEXT, size=9),
                    gridcolor=C_BORDER,
                    linecolor=C_BORDER,
                ),
                yaxis=_ax(title="log expr"),
                legend=dict(
                    bgcolor=C_PANEL, bordercolor=C_BORDER, borderwidth=1,
                    font=dict(size=9, color=C_TEXT), orientation="h",
                    y=-0.18, x=0,
                ),
            )

            st.plotly_chart(
                fig_a, use_container_width=True,
                config=dict(displayModeBar=False),
            )
            st.markdown(
                f'<div class="pill">{sel_pert} <b>{n_pert_cells:,} cells</b></div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL B  —  Perturbation Effect
    #  Top genes by Δ expression for selected perturbation vs control
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown(
        f'<div class="ph">Perturbation Effect &nbsp;·&nbsp; {sel_pert}</div>',
        unsafe_allow_html=True,
    )

    if adata is not None and pc is not None:
        if precomp and sel_pert in precomp["delta"]:
            mc_b    = precomp["control_mean"]
            mp_b    = precomp["perturbation_means"][sel_pert]
            delta_b = precomp["delta"][sel_pert]
            gnames_b = precomp["gene_names"]
        else:
            mc_b = mp_b = delta_b = None
            gnames_b = list(adata.var_names)

        if delta_b is None:
            st.warning(f"No perturbation data for `{sel_pert}`.")
        else:
            order_b  = np.argsort(np.abs(delta_b))[::-1][:10]
            top_d    = delta_b[order_b]
            top_g    = [gnames_b[i] for i in order_b]
            bar_cols = [C_UP if v > 0 else C_DOWN for v in top_d]

            fig_b = go.Figure()

            fig_b.add_trace(go.Bar(
                x=top_d,
                y=top_g,
                orientation="h",
                marker=dict(color=bar_cols, line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>Δ = %{x:.4f}<extra></extra>",
            ))

            # highlight selected gene if present
            if sel_gene in top_g:
                gi   = top_g.index(sel_gene)
                xpos = top_d[gi]
                fig_b.add_shape(
                    type="line",
                    x0=xpos, x1=xpos, y0=-0.5, y1=len(top_g) - 0.5,
                    line=dict(color=C_ACCENT, width=1.5, dash="dot"),
                )
                fig_b.add_annotation(
                    x=xpos, y=gi,
                    text=f" {sel_gene}",
                    showarrow=False,
                    font=dict(color=C_ACCENT, size=8),
                    xanchor="left",
                )

            # model pearson_r for this pert as subtitle
            r_b = _safe(results, MODEL_KEY, "per_perturbation_eval", sel_pert, "pearson_r")
            title_b = f"Top 10 genes Δ vs control"
            if r_b is not None:
                title_b += f"  ·  model r = {r_b:.4f}"

            n_cells_b = precomp["cell_counts"].get(sel_pert, 0) if precomp else 0

            fig_b.update_layout(
                **_PL,
                height=270,
                title=dict(text=title_b, font=dict(size=10, color=C_MUTED), x=0),
                xaxis=_ax(title="Δ log expression", zeroline=True, zerolinewidth=1),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(color=C_TEXT, size=9),
                    gridcolor=C_BORDER,
                    linecolor=C_BORDER,
                ),
                bargap=0.22,
            )

            st.plotly_chart(
                fig_b, use_container_width=True,
                config=dict(displayModeBar=False),
            )

            st.markdown(
                f'<div class="pill">n cells <b>{n_cells_b:,}</b></div>'
                + (f'<div class="pill">model r <b>{r_b:.4f}</b></div>'
                   if r_b is not None else ""),
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL C  —  Model Insight
    #  Per-perturbation Pearson r landscape, selected pert highlighted
    #  + zero-shot / oracle pills if applicable
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown(
        f'<div class="ph">Model Insight &nbsp;·&nbsp; {sel_model}</div>',
        unsafe_allow_html=True,
    )

    if adata is not None and pc is not None:
        if precomp and sel_pert in precomp["delta"]:
            mc_c    = precomp["control_mean"]
            mp_c    = precomp["perturbation_means"][sel_pert]
            delta_c = precomp["delta"][sel_pert]
            gnames_c = precomp["gene_names"]
        else:
            mc_c = mp_c = delta_c = gnames_c = None

        if mc_c is None:
            st.warning(f"No expression data for `{sel_pert}`.")
        else:
            # top 60 genes by |delta| — enough for a meaningful scatter
            top_idx_c = np.argsort(np.abs(delta_c))[::-1][:60]
            x_c = mc_c[top_idx_c]       # control mean expression
            y_c = mp_c[top_idx_c]       # perturbed mean expression
            d_c = delta_c[top_idx_c]
            g_c = [gnames_c[i] for i in top_idx_c]

            pt_colors = [C_UP if d > 0 else C_DOWN for d in d_c]
            # selected gene larger if it appears
            pt_sizes  = [8 if g == sel_gene else 4 for g in g_c]

            fig_c = go.Figure()

            # diagonal reference line y = x (no perturbation effect)
            xy_min = float(min(x_c.min(), y_c.min())) * 0.95
            xy_max = float(max(x_c.max(), y_c.max())) * 1.05
            fig_c.add_shape(
                type="line",
                x0=xy_min, y0=xy_min, x1=xy_max, y1=xy_max,
                line=dict(color=C_MUTED, dash="dash", width=1),
            )

            # gene points
            fig_c.add_trace(go.Scatter(
                x=x_c, y=y_c,
                mode="markers",
                marker=dict(color=pt_colors, size=pt_sizes, opacity=0.85,
                            line=dict(width=0.4, color=C_PANEL)),
                customdata=[[g, f"{d:.4f}"] for g, d in zip(g_c, d_c)],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "ctrl = %{x:.3f}  pert = %{y:.3f}<br>"
                    "Δ = %{customdata[1]}<extra></extra>"
                ),
                showlegend=False,
            ))

            # annotate selected gene if in top 60
            if sel_gene in g_c:
                gi_c = g_c.index(sel_gene)
                fig_c.add_annotation(
                    x=x_c[gi_c], y=y_c[gi_c],
                    text=f" {sel_gene}",
                    showarrow=False,
                    font=dict(color=C_ACCENT, size=8),
                    xanchor="left",
                )

            # model metric annotation
            r_c   = _safe(results, MODEL_KEY, "per_perturbation_eval", sel_pert, "pearson_r")
            mse_c_val = _safe(results, MODEL_KEY, "per_perturbation_eval", sel_pert, "mse")
            metric_txt = ""
            if r_c is not None:
                metric_txt += f"model r = {r_c:.4f}"
            if mse_c_val is not None:
                metric_txt += f"  ·  MSE = {mse_c_val:.4f}"
            if metric_txt:
                fig_c.add_annotation(
                    x=0.98, y=0.04, xref="paper", yref="paper",
                    text=metric_txt,
                    showarrow=False,
                    font=dict(color=C_ACCENT, size=9),
                    xanchor="right", bgcolor=C_PANEL,
                    bordercolor=C_BORDER, borderwidth=1, borderpad=4,
                )

            fig_c.update_layout(
                **_PL,
                height=235,
                title=dict(
                    text=f"Control vs Perturbed  ·  {sel_pert}  ·  top 60 genes",
                    font=dict(size=10, color=C_TEXT), x=0,
                ),
                xaxis=_ax(title="mean ctrl expression"),
                yaxis=_ax(title=f"mean {sel_pert}"),
                showlegend=False,
            )

            st.plotly_chart(
                fig_c, use_container_width=True,
                config=dict(displayModeBar=False),
            )

            # summary pills
            r_mean_c = _safe(results, MODEL_KEY, "mean_per_pert_pearson_r")
            mse_glob = _safe(results, MODEL_KEY, "test_mse")
            pills_c  = ""
            if r_c is not None:
                pills_c += f'<div class="pill">this pert r <b>{r_c:.4f}</b></div>'
            if r_mean_c is not None:
                pills_c += f'<div class="pill">model mean r <b>{r_mean_c:.4f}</b></div>'
            if mse_glob is not None:
                pills_c += f'<div class="pill">MSE <b>{mse_glob:.4f}</b></div>'
            if pills_c:
                st.markdown(pills_c, unsafe_allow_html=True)

            # zero-shot pills if this pert was in the unseen experiment
            unseen_pp = _safe(results, "unseen", "per_perturbation") or {}
            if sel_pert in unseen_pp:
                ud  = unseen_pp[sel_pert]
                zsr = ud.get("pearson_r_vae_zeroshot")
                orr = ud.get("pearson_r_vae_oracle")
                nn  = ud.get("nearest_seen", "—")
                if zsr and orr:
                    st.markdown(
                        f'<div class="pill">zero-shot <b>{zsr:.4f}</b></div>'
                        f'<div class="pill">oracle <b>{orr:.4f}</b></div>'
                        f'<div class="pill">nearest: <b>{nn}</b></div>',
                        unsafe_allow_html=True,
                    )
