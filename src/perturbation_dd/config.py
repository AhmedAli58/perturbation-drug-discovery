"""Pydantic-backed project configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    raw_dataset: Path = Path("data/raw/norman2019.h5ad")
    string_network: Path = Path("data/external/string_ppi_edges.tsv")
    artifact_root: Path = Path("artifacts")
    mlflow_tracking_uri: str = "mlruns"


class DatasetConfig(BaseModel):
    name: str = "norman2019"
    perturbation_key: str = "perturbation"
    control_label: str = "control"
    min_genes: int = 200
    min_cells: int = 3
    max_pct_mito: float = 20.0
    target_sum: float = 10000.0
    n_top_genes: int = 2000
    hvg_flavor: str = "seurat_v3"


class SplitConfig(BaseModel):
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    heldout_frac: float = 0.2
    min_cells_per_pert: int = 100
    seed: int = 42

    @model_validator(mode="after")
    def _validate_fractions(self) -> SplitConfig:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
        if not 0 < self.heldout_frac < 1:
            raise ValueError("heldout_frac must be between 0 and 1")
        return self


class RankingConfig(BaseModel):
    effect_strength_weight: float = 0.5
    model_agreement_weight: float = 0.3
    support_cells_weight: float = 0.2
    confidence_agreement_weight: float = 0.6
    confidence_support_weight: float = 0.4

    @model_validator(mode="after")
    def _validate_weights(self) -> RankingConfig:
        priority_total = (
            self.effect_strength_weight
            + self.model_agreement_weight
            + self.support_cells_weight
        )
        confidence_total = self.confidence_agreement_weight + self.confidence_support_weight
        if abs(priority_total - 1.0) > 1e-6:
            raise ValueError("priority score weights must sum to 1.0")
        if abs(confidence_total - 1.0) > 1e-6:
            raise ValueError("confidence weights must sum to 1.0")
        return self


class ProjectConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)


def load_project_config(config_path: Path) -> ProjectConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    return ProjectConfig.model_validate(payload)
