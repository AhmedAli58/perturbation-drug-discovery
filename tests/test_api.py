from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from perturbation_dd.serving.api import create_app
from perturbation_dd.utils.io import write_json


def test_api_health_and_runs_endpoint(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    artifact_root = tmp_path / "artifacts"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "base.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  raw_dataset: {tmp_path / 'raw.h5ad'}",
                f"  artifact_root: {artifact_root}",
                f"  string_network: {tmp_path / 'string.tsv'}",
            ]
        )
    )

    run_id = "demo-run"
    run_path = artifact_root / "runs" / run_id / "run.json"
    write_json(
        run_path,
        {
            "manifest_version": 1,
            "run_id": run_id,
            "task": "classification",
            "model": "logreg",
            "split_name": "cell_split_v1",
            "prepared_manifest_path": str(tmp_path / "manifest.json"),
            "prepared_dataset_path": str(tmp_path / "prepared.h5ad"),
            "results_dir": str(tmp_path / "results"),
            "models_dir": str(tmp_path / "models"),
            "training_metrics_path": str(tmp_path / "metrics.json"),
            "model_artifact_path": str(tmp_path / "model.pkl"),
            "status": "completed",
            "supports_unseen_direct": False,
            "created_at": "2026-03-27T00:00:00+00:00",
        },
    )

    client = TestClient(create_app(config_path=config_path, project_root=tmp_path))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    run = client.get(f"/runs/{run_id}")
    assert run.status_code == 200
    assert run.json()["run_id"] == run_id
