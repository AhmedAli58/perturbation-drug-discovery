from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from perturbation_dd.cli import app

pytest.importorskip("scanpy")


def test_prepare_data_cli_writes_manifest(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5ad"
    artifact_root = tmp_path / "artifacts"
    config_path = tmp_path / "base.yaml"

    obs = {
        "perturbation": [
            "control",
            "control",
            "control",
            "control",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
        ]
    }
    var_names = [f"GENE_{idx}" for idx in range(6)]
    matrix = np.abs(np.arange(72, dtype=np.float32).reshape(12, 6)) + 1
    adata = ad.AnnData(X=matrix)
    adata.obs["perturbation"] = obs["perturbation"]
    adata.var_names = var_names
    adata.write_h5ad(raw_path)

    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  raw_dataset: {raw_path}",
                f"  artifact_root: {artifact_root}",
                "dataset:",
                "  name: testset",
                "  perturbation_key: perturbation",
                "  control_label: control",
                "  min_genes: 0",
                "  min_cells: 1",
                "  max_pct_mito: 100.0",
                "  target_sum: 10000.0",
                "  n_top_genes: 4",
                "  hvg_flavor: seurat",
                "split:",
                "  train_frac: 0.5",
                "  val_frac: 0.25",
                "  test_frac: 0.25",
                "  heldout_frac: 0.5",
                "  min_cells_per_pert: 2",
                "  seed: 11",
            ]
        )
    )

    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["prepare-data", "--config", str(config_path), "--split", "cell_split_v1"],
    )
    assert result.exit_code == 0, result.stdout

    manifest_path = artifact_root / "prepared" / "testset" / "cell_split_v1" / "manifest.json"
    prepared_path = artifact_root / "prepared" / "testset" / "cell_split_v1" / "prepared.h5ad"

    assert prepared_path.exists()
    assert manifest_path.exists()
    manifest_text = manifest_path.read_text()
    assert '"split_name": "cell_split_v1"' in manifest_text
    assert '"control_policy": "train_controls_only"' in manifest_text
