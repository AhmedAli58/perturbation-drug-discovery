"""Shared manifest and benchmark types."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PreparedDatasetManifest(BaseModel):
    manifest_version: int = 1
    dataset_name: str
    split_name: str
    raw_dataset_path: str
    raw_dataset_sha256: str
    prepared_dataset_path: str
    prepared_dataset_sha256: str
    perturbation_key: str
    control_label: str
    split_seed: int
    control_policy: str
    hvg_method: str
    hvg_genes: list[str]
    split_counts: dict[str, int]
    perturbation_counts: dict[str, int]
    split_perturbations: dict[str, list[str]]
    heldout_perturbations: list[str] = Field(default_factory=list)
    train_control_cells: int
    created_at: str
