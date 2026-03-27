"""Helpers for reading prepared artifacts and manifests."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from perturbation_dd.config import ProjectConfig
from perturbation_dd.types import PreparedDatasetManifest
from perturbation_dd.utils.io import read_json


def resolve_project_path(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else project_root / maybe_relative


def prepared_split_dir(
    project_root: Path,
    config: ProjectConfig,
    split_name: str,
) -> Path:
    artifact_root = resolve_project_path(project_root, config.paths.artifact_root)
    return artifact_root / "prepared" / config.dataset.name / split_name


def prepared_manifest_path(
    project_root: Path,
    config: ProjectConfig,
    split_name: str,
) -> Path:
    return prepared_split_dir(project_root, config, split_name) / "manifest.json"


def load_prepared_manifest(
    project_root: Path,
    config: ProjectConfig,
    split_name: str,
) -> PreparedDatasetManifest:
    manifest_path = prepared_manifest_path(project_root, config, split_name)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Prepared manifest not found at {manifest_path}. "
            f"Run prepare-data for split={split_name}."
        )
    return PreparedDatasetManifest.model_validate(read_json(manifest_path))


def load_prepared_adata(manifest: PreparedDatasetManifest) -> ad.AnnData:
    path = Path(manifest.prepared_dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found at {path}.")
    return ad.read_h5ad(path)
