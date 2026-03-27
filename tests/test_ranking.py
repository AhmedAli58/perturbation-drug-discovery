from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from perturbation_dd.config import ProjectConfig
from perturbation_dd.ranking import service
from perturbation_dd.types import RunManifest
from perturbation_dd.utils.io import write_json


class StubPredictor:
    def __init__(self, vector: np.ndarray):
        self.vector = vector

    def predict(self, candidate: str):
        from perturbation_dd.models.inference import PredictionResult

        if candidate != "A":
            return PredictionResult(vector=None, supported=False, notes=["unsupported candidate"])
        return PredictionResult(vector=self.vector.copy(), supported=True, notes=[])


def test_rank_candidates_marks_unseen_candidates_as_proxy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared_dir = artifact_root / "prepared" / "testset" / "pert_split_v1"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / "prepared.h5ad"
    manifest_path = prepared_dir / "manifest.json"
    string_path = tmp_path / "string.tsv"

    adata = ad.AnnData(
        X=np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [3.0, 1.0],
                [3.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    adata.obs["perturbation"] = ["control", "control", "A", "A"]
    adata.obs["split"] = ["train", "train", "train", "train"]
    adata.var_names = ["G1", "G2"]
    adata.write_h5ad(prepared_path)

    write_json(
        manifest_path,
        {
            "manifest_version": 1,
            "dataset_name": "testset",
            "split_name": "pert_split_v1",
            "raw_dataset_path": str(tmp_path / "raw.h5ad"),
            "raw_dataset_sha256": "raw",
            "prepared_dataset_path": str(prepared_path),
            "prepared_dataset_sha256": "prepared",
            "perturbation_key": "perturbation",
            "control_label": "control",
            "split_seed": 7,
            "control_policy": "train_controls_only",
            "hvg_method": "seurat_train_only",
            "hvg_genes": ["G1", "G2"],
            "split_counts": {"train": 4},
            "perturbation_counts": {"control": 2, "A": 2},
            "split_perturbations": {"train": ["A", "control"], "val": [], "test": []},
            "heldout_perturbations": ["B"],
            "train_control_cells": 2,
            "created_at": "2026-03-27T00:00:00+00:00",
        },
    )
    string_path.write_text("protein1\tprotein2\tscore\nB\tA\t900\n")

    evaluation_payload = {
        "comparison": {"preferred_ranking_mode": "proxy"},
        "model_metrics": {"mean_per_pert_pearson_r": 0.3},
    }

    def fake_latest_run_for(*args, **kwargs):
        model_name = kwargs["model"]
        run_dir = artifact_root / "runs" / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        evaluation_path = run_dir / "evaluation.json"
        write_json(evaluation_path, evaluation_payload)
        return RunManifest(
            run_id=model_name,
            task=kwargs["task"],
            model=model_name,
            split_name=kwargs["split_name"],
            prepared_manifest_path=str(manifest_path),
            prepared_dataset_path=str(prepared_path),
            results_dir=str(run_dir / "results"),
            models_dir=str(run_dir / "models"),
            training_metrics_path=str(run_dir / "results" / "metrics.json"),
            model_artifact_path=str(run_dir / "models" / "model.pt"),
            status="completed",
            supports_unseen_direct=False,
            evaluation_path=str(evaluation_path),
            created_at="2026-03-27T00:00:00+00:00",
        )

    monkeypatch.setattr(service, "latest_run_for", fake_latest_run_for)
    monkeypatch.setattr(
        service,
        "load_predictor",
        lambda *args, **kwargs: StubPredictor(np.array([3.0, 1.0], dtype=np.float32)),
    )

    input_path = tmp_path / "candidates.json"
    output_path = tmp_path / "ranked.json"
    input_path.write_text('{"candidates": ["B"]}')

    config = ProjectConfig.model_validate(
        {
            "paths": {
                "raw_dataset": tmp_path / "raw.h5ad",
                "artifact_root": artifact_root,
                "string_network": string_path,
            },
            "dataset": {
                "name": "testset",
                "perturbation_key": "perturbation",
                "control_label": "control",
            },
        }
    )
    ranking = service.rank_candidates(
        config,
        tmp_path,
        input_path=input_path,
        output_path=output_path,
        split_name="pert_split_v1",
    )

    assert ranking[0].status == "proxy"
    assert ranking[0].proxy_source == "A"
    assert ranking[0].priority_score == 1.0
    assert output_path.exists()
