"""Preparation utilities with train-only HVG selection."""

from __future__ import annotations

from collections.abc import Iterable

import anndata as ad
import scanpy as sc

from perturbation_dd.config import DatasetConfig


def basic_qc(adata: ad.AnnData, dataset_config: DatasetConfig) -> ad.AnnData:
    filtered = adata.copy()
    sc.pp.filter_cells(filtered, min_genes=dataset_config.min_genes)
    sc.pp.filter_genes(filtered, min_cells=dataset_config.min_cells)
    filtered.var["mt"] = filtered.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        filtered,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    filtered = filtered[filtered.obs["pct_counts_mt"] < dataset_config.max_pct_mito].copy()
    return filtered


def normalize_all_cells(adata: ad.AnnData, dataset_config: DatasetConfig) -> ad.AnnData:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=dataset_config.target_sum)
    sc.pp.log1p(adata)
    return adata


def select_train_hvgs(
    adata: ad.AnnData,
    train_mask: Iterable[bool],
    dataset_config: DatasetConfig,
) -> tuple[ad.AnnData, list[str]]:
    train_subset = adata[list(train_mask)].copy()
    sc.pp.highly_variable_genes(
        train_subset,
        n_top_genes=dataset_config.n_top_genes,
        flavor=dataset_config.hvg_flavor,
        layer="counts" if dataset_config.hvg_flavor == "seurat_v3" else None,
        subset=False,
    )
    hvg_genes = train_subset.var_names[train_subset.var["highly_variable"]].tolist()
    return adata[:, hvg_genes].copy(), hvg_genes
