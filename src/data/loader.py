"""
Single-cell data loader for perturbation datasets.

Supports:
  - AnnData HDF5 (.h5ad)
  - 10x Genomics sparse matrix directories (.mtx + barcodes + features)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


def load_h5ad(path: str | Path) -> ad.AnnData:
    """Load an AnnData object from a .h5ad file.

    Args:
        path: Path to the .h5ad file.

    Returns:
        Loaded AnnData object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix != ".h5ad":
        raise ValueError(f"Expected .h5ad file, got: {path.suffix}")

    try:
        import anndata as ad
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "anndata is required to load .h5ad files. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    logger.info("Loading .h5ad from %s", path)
    adata = ad.read_h5ad(path)
    logger.info("Loaded AnnData: %d cells × %d genes", *adata.shape)
    return adata


def load_10x_mtx(directory: str | Path) -> ad.AnnData:
    """Load a 10x Genomics sparse matrix directory.

    Expected files inside `directory`:
      - matrix.mtx (or matrix.mtx.gz)
      - barcodes.tsv (or barcodes.tsv.gz)
      - features.tsv / genes.tsv (or .gz variants)

    Args:
        directory: Path to the 10x output directory.

    Returns:
        Loaded AnnData object.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    try:
        import scanpy as sc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scanpy is required to load 10x MTX directories. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    logger.info("Loading 10x MTX from %s", directory)
    adata = sc.read_10x_mtx(directory, var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    logger.info("Loaded AnnData: %d cells × %d genes", *adata.shape)
    return adata


def load_dataset(path: str | Path) -> ad.AnnData:
    """Auto-detect format and load a single-cell dataset.

    Dispatches to `load_h5ad` for .h5ad files and `load_10x_mtx` for
    directories containing a matrix.mtx file.

    Args:
        path: Path to a .h5ad file or a 10x MTX directory.

    Returns:
        Loaded AnnData object.

    Raises:
        ValueError: If the format cannot be determined.
    """
    path = Path(path)

    if path.is_file() and path.suffix == ".h5ad":
        return load_h5ad(path)

    if path.is_dir():
        mtx_files = list(path.glob("matrix.mtx*"))
        if mtx_files:
            return load_10x_mtx(path)

    raise ValueError(
        f"Cannot determine dataset format for: {path}\n"
        "Expected a .h5ad file or a 10x MTX directory containing matrix.mtx."
    )


def summarize(adata: ad.AnnData) -> None:
    """Print a brief summary of an AnnData object."""
    print(f"AnnData object: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  obs columns : {list(adata.obs.columns)}")
    print(f"  var columns : {list(adata.var.columns)}")
    print(f"  obsm keys   : {list(adata.obsm.keys())}")
    print(f"  layers      : {list(adata.layers.keys())}")
    print(f"  uns keys    : {list(adata.uns.keys())}")
