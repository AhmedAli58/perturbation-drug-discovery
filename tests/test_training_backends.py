from __future__ import annotations

import subprocess
from pathlib import Path

import anndata as ad
import numpy as np

from perturbation_dd.config import ProjectConfig
from perturbation_dd.training import backends
from perturbation_dd.utils.io import write_json


def test_train_backend_run_writes_run_manifest(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared_dir = artifact_root / "prepared" / "testset" / "cell_split_v1"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / "prepared.h5ad"
    manifest_path = prepared_dir / "manifest.json"

    adata = ad.AnnData(X=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32))
    adata.obs["perturbation"] = ["control", "A"]
    adata.obs["split"] = ["train", "test"]
    adata.var_names = ["G1", "G2"]
    adata.write_h5ad(prepared_path)

    write_json(
        manifest_path,
        {
            "manifest_version": 1,
            "dataset_name": "testset",
            "split_name": "cell_split_v1",
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
            "split_counts": {"train": 1, "test": 1},
            "perturbation_counts": {"control": 1, "A": 1},
            "split_perturbations": {"train": ["control"], "val": [], "test": ["A"]},
            "heldout_perturbations": [],
            "train_control_cells": 1,
            "created_at": "2026-03-27T00:00:00+00:00",
        },
    )

    config = ProjectConfig.model_validate(
        {
            "paths": {
                "raw_dataset": tmp_path / "raw.h5ad",
                "artifact_root": artifact_root,
                "string_network": tmp_path / "string.tsv",
                "mlflow_tracking_uri": str(tmp_path / "mlruns"),
            },
            "dataset": {
                "name": "testset",
                "perturbation_key": "perturbation",
                "control_label": "control",
            },
        }
    )

    def fake_run(command, cwd, env, capture_output, text, check):
        results_dir = Path(env["PDD_RESULTS_DIR"])
        models_dir = Path(env["PDD_MODELS_DIR"])
        results_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "baseline_metrics.json").write_text('{"accuracy": 0.5}')
        (models_dir / "baseline_logreg.pkl").write_bytes(b"stub")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(backends.subprocess, "run", fake_run)
    monkeypatch.setattr(backends, "_log_run_to_mlflow", lambda **kwargs: "mlflow-run")

    run_manifest = backends.train_backend_run(
        config,
        tmp_path,
        task="classification",
        model="logreg",
        split_name="cell_split_v1",
    )

    assert run_manifest.status == "completed"
    assert run_manifest.mlflow_run_id == "mlflow-run"
    assert Path(run_manifest.training_metrics_path).exists()
    assert Path(run_manifest.model_artifact_path).exists()
