"""Shared manifest and benchmark types."""

from __future__ import annotations

from typing import Optional

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


class RunManifest(BaseModel):
    manifest_version: int = 1
    run_id: str
    task: str
    model: str
    split_name: str
    prepared_manifest_path: str
    prepared_dataset_path: str
    results_dir: str
    models_dir: str
    training_metrics_path: str
    model_artifact_path: str
    status: str
    supports_unseen_direct: bool = False
    mlflow_run_id: Optional[str] = None
    evaluation_path: Optional[str] = None
    report_path: Optional[str] = None
    notes: list[str] = Field(default_factory=list)
    created_at: str
    completed_at: Optional[str] = None


class EvaluationManifest(BaseModel):
    manifest_version: int = 1
    run_id: str
    task: str
    model: str
    split_name: str
    baselines: dict[str, dict]
    model_metrics: dict[str, object]
    comparison: dict[str, object]
    created_at: str


class RankingRecord(BaseModel):
    candidate: str
    status: str
    proxy_source: Optional[str] = None
    effect_strength: Optional[float] = None
    model_agreement: Optional[float] = None
    support_cells: Optional[float] = None
    confidence: Optional[float] = None
    priority_score: Optional[float] = None
    notes: list[str] = Field(default_factory=list)
