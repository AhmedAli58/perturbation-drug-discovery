from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from perturbation_dd.config import ProjectConfig
from perturbation_dd.evaluation.baselines import compute_response_baselines
from perturbation_dd.types import PreparedDatasetManifest


def test_response_heldout_baseline_uses_nearest_seen_proxy(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.h5ad"
    string_path = tmp_path / "string.tsv"

    adata = ad.AnnData(
        X=np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    adata.obs["perturbation"] = ["control", "control", "A", "A", "B", "B"]
    adata.obs["split"] = ["train", "train", "train", "train", "test", "test"]
    adata.var_names = ["G1", "G2", "G3"]
    adata.write_h5ad(prepared_path)

    string_path.write_text("protein1\tprotein2\tscore\nB\tA\t900\n")

    config = ProjectConfig.model_validate(
        {
            "paths": {
                "raw_dataset": tmp_path / "raw.h5ad",
                "artifact_root": tmp_path / "artifacts",
                "string_network": string_path,
            },
            "dataset": {
                "name": "testset",
                "perturbation_key": "perturbation",
                "control_label": "control",
            },
        }
    )
    manifest = PreparedDatasetManifest(
        dataset_name="testset",
        split_name="pert_split_v1",
        raw_dataset_path=str(tmp_path / "raw.h5ad"),
        raw_dataset_sha256="raw",
        prepared_dataset_path=str(prepared_path),
        prepared_dataset_sha256="prepared",
        perturbation_key="perturbation",
        control_label="control",
        split_seed=11,
        control_policy="train_controls_only",
        hvg_method="seurat_train_only",
        hvg_genes=["G1", "G2", "G3"],
        split_counts={"train": 4, "test": 2},
        perturbation_counts={"control": 2, "A": 2, "B": 2},
        split_perturbations={"train": ["A", "control"], "val": [], "test": ["B"]},
        heldout_perturbations=["B"],
        train_control_cells=2,
        created_at="2026-03-27T00:00:00+00:00",
    )

    baselines = compute_response_baselines(config, tmp_path, manifest, task="response_heldout")

    assert baselines["nearest_seen"]["proxy_map"]["B"] == "A"
    assert baselines["nearest_seen"]["test_mse"] < baselines["naive_control"]["test_mse"]
    assert baselines["nearest_seen"]["mean_per_pert_pearson_r"] >= (
        baselines["naive_control"]["mean_per_pert_pearson_r"]
    )
