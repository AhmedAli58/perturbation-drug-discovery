"""Prepared dataset orchestration and manifest writing."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import scanpy as sc

from perturbation_dd.config import ProjectConfig
from perturbation_dd.data.preprocessing import basic_qc, normalize_all_cells, select_train_hvgs
from perturbation_dd.splits.registry import build_split_assignment
from perturbation_dd.types import PreparedDatasetManifest
from perturbation_dd.utils.io import sha256_file, utc_now


def prepare_dataset(
    config: ProjectConfig,
    project_root: Path,
    split_name: str,
) -> PreparedDatasetManifest:
    raw_path = _resolve_path(project_root, config.paths.raw_dataset)
    artifact_root = _resolve_path(project_root, config.paths.artifact_root)
    prepared_dir = artifact_root / "prepared" / config.dataset.name / split_name
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / "prepared.h5ad"
    manifest_path = prepared_dir / "manifest.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    adata = sc.read_h5ad(raw_path)
    if config.dataset.perturbation_key not in adata.obs.columns:
        raise KeyError(
            f"Expected perturbation column '{config.dataset.perturbation_key}' in obs."
        )

    adata = basic_qc(adata, config.dataset)

    perturbations = adata.obs[config.dataset.perturbation_key].astype(str).to_numpy()
    assignment = build_split_assignment(
        perturbations=perturbations,
        split_name=split_name,
        control_label=config.dataset.control_label,
        split_config=config.split,
    )

    adata.obs["split"] = assignment.labels
    adata = normalize_all_cells(adata, config.dataset)
    train_mask = adata.obs["split"].to_numpy() == "train"
    adata, hvg_genes = select_train_hvgs(
        adata,
        train_mask=train_mask,
        dataset_config=config.dataset,
    )

    perturbations = adata.obs[config.dataset.perturbation_key].astype(str)
    unique_perts = sorted(perturbations.unique().tolist())
    pert_to_idx = {pert: idx for idx, pert in enumerate(unique_perts)}
    adata.obs["perturbation_idx"] = perturbations.map(pert_to_idx).astype(int)
    adata.uns["perturbation_encoding"] = {
        "pert_to_idx": pert_to_idx,
        "idx_to_pert": {str(idx): pert for pert, idx in pert_to_idx.items()},
    }
    adata.uns["prepared_manifest_hint"] = str(manifest_path)

    adata.write_h5ad(prepared_path, compression="gzip")

    split_counts = Counter(adata.obs["split"].astype(str).tolist())
    split_perts = {
        split: sorted(
            adata.obs.loc[adata.obs["split"] == split, config.dataset.perturbation_key]
            .astype(str)
            .unique()
            .tolist()
        )
        for split in ("train", "val", "test")
    }
    train_control_cells = int(
        (
            (adata.obs["split"] == "train")
            & (
                adata.obs[config.dataset.perturbation_key].astype(str)
                == config.dataset.control_label
            )
        ).sum()
    )

    manifest = PreparedDatasetManifest(
        dataset_name=config.dataset.name,
        split_name=split_name,
        raw_dataset_path=str(raw_path),
        raw_dataset_sha256=sha256_file(raw_path),
        prepared_dataset_path=str(prepared_path),
        prepared_dataset_sha256=sha256_file(prepared_path),
        perturbation_key=config.dataset.perturbation_key,
        control_label=config.dataset.control_label,
        split_seed=config.split.seed,
        control_policy="train_controls_only",
        hvg_method=f"{config.dataset.hvg_flavor}_train_only",
        hvg_genes=hvg_genes,
        split_counts=dict(split_counts),
        perturbation_counts=dict(Counter(perturbations.tolist())),
        split_perturbations=split_perts,
        heldout_perturbations=assignment.heldout_perturbations,
        train_control_cells=train_control_cells,
        created_at=utc_now(),
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


def _resolve_path(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else project_root / maybe_relative
