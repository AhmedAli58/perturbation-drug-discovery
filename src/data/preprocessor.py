"""
Standard single-cell preprocessing pipeline.
"""

from __future__ import annotations

import logging

import anndata as ad
import scanpy as sc

logger = logging.getLogger(__name__)


def basic_qc(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
) -> ad.AnnData:
    """Filter cells and genes by basic QC thresholds.

    Args:
        adata: Input AnnData object.
        min_genes: Minimum number of genes expressed per cell.
        min_cells: Minimum number of cells a gene must appear in.
        max_pct_mito: Maximum percentage of mitochondrial counts per cell.

    Returns:
        Filtered AnnData (copy).
    """
    adata = adata.copy()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    before = adata.n_obs
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()
    logger.info(
        "QC filter: %d → %d cells (removed %d high-mito)",
        before, adata.n_obs, before - adata.n_obs,
    )
    return adata


def normalize_and_log(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    """Normalize total counts per cell and apply log1p transform.

    Stores raw counts in `adata.layers['counts']` before normalizing.

    Args:
        adata: Input AnnData object (modified in-place).
        target_sum: Target count sum per cell after normalization.

    Returns:
        The same AnnData object (modified in-place).
    """
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    logger.info("Normalized to %g counts/cell and applied log1p", target_sum)
    return adata


def select_hvgs(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
) -> ad.AnnData:
    """Select highly variable genes.

    Args:
        adata: Input AnnData (must contain raw counts in `layers['counts']`
               when using seurat_v3 flavor).
        n_top_genes: Number of HVGs to select.
        flavor: HVG selection method ('seurat_v3', 'seurat', 'cell_ranger').

    Returns:
        AnnData subset to HVGs.
    """
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        layer="counts" if flavor == "seurat_v3" else None,
        subset=False,
    )
    n_hvg = adata.var["highly_variable"].sum()
    logger.info("Selected %d highly variable genes (flavor=%s)", n_hvg, flavor)
    return adata[:, adata.var["highly_variable"]].copy()


def preprocess(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
) -> ad.AnnData:
    """End-to-end preprocessing: QC → normalize → HVG selection.

    Args:
        adata: Raw AnnData object.

    Returns:
        Preprocessed AnnData ready for modeling.
    """
    adata = basic_qc(adata, min_genes=min_genes, min_cells=min_cells, max_pct_mito=max_pct_mito)
    adata = normalize_and_log(adata, target_sum=target_sum)
    adata = select_hvgs(adata, n_top_genes=n_top_genes)
    return adata
