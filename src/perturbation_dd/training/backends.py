"""Common run contract for legacy training backends."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow

from perturbation_dd.config import ProjectConfig
from perturbation_dd.data.access import load_prepared_manifest, resolve_project_path
from perturbation_dd.training.runs import runs_root
from perturbation_dd.types import RunManifest
from perturbation_dd.utils.io import read_json, utc_now, write_json


@dataclass(frozen=True)
class LegacyBackend:
    task: str
    model: str
    script_path: str
    metrics_filename: str
    model_filename: str
    supports_unseen_direct: bool = False


BACKENDS = {
    ("classification", "logreg"): LegacyBackend(
        task="classification",
        model="logreg",
        script_path="src/models/train_baseline_classifier.py",
        metrics_filename="baseline_metrics.json",
        model_filename="baseline_logreg.pkl",
    ),
    ("classification", "mlp"): LegacyBackend(
        task="classification",
        model="mlp",
        script_path="src/models/train_mlp_classifier.py",
        metrics_filename="mlp_metrics.json",
        model_filename="mlp_classifier.pt",
    ),
    ("response_known", "effect_mlp"): LegacyBackend(
        task="response_known",
        model="effect_mlp",
        script_path="src/models/train_perturbation_effect_model.py",
        metrics_filename="perturbation_effect_metrics.json",
        model_filename="perturbation_effect_model.pt",
    ),
    ("response_known", "graph_gcn"): LegacyBackend(
        task="response_known",
        model="graph_gcn",
        script_path="src/models/train_graph_perturbation_model.py",
        metrics_filename="graph_model_metrics.json",
        model_filename="graph_perturbation_model.pt",
        supports_unseen_direct=True,
    ),
    ("response_known", "scgen"): LegacyBackend(
        task="response_known",
        model="scgen",
        script_path="src/models/train_scgen_style_model.py",
        metrics_filename="scgen_metrics.json",
        model_filename="scgen_model.pt",
    ),
    ("response_heldout", "graph_gcn"): LegacyBackend(
        task="response_heldout",
        model="graph_gcn",
        script_path="src/models/train_graph_perturbation_model.py",
        metrics_filename="graph_model_metrics.json",
        model_filename="graph_perturbation_model.pt",
        supports_unseen_direct=True,
    ),
}


def train_backend_run(
    config: ProjectConfig,
    project_root: Path,
    *,
    task: str,
    model: str,
    split_name: str,
) -> RunManifest:
    backend = BACKENDS.get((task, model))
    if backend is None:
        raise ValueError(f"Unsupported task/model combination: task={task}, model={model}")

    prepared_manifest = load_prepared_manifest(project_root, config, split_name)
    run_id = _make_run_id(task=task, model=model, split_name=split_name)
    root = runs_root(project_root, config)
    run_dir = root / run_id
    results_dir = run_dir / "results"
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    for directory in (results_dir, models_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_manifest = RunManifest(
        run_id=run_id,
        task=task,
        model=model,
        split_name=split_name,
        prepared_manifest_path=str(
            Path(prepared_manifest.prepared_dataset_path).parent / "manifest.json"
        ),
        prepared_dataset_path=prepared_manifest.prepared_dataset_path,
        results_dir=str(results_dir),
        models_dir=str(models_dir),
        training_metrics_path=str(results_dir / backend.metrics_filename),
        model_artifact_path=str(models_dir / backend.model_filename),
        status="running",
        supports_unseen_direct=backend.supports_unseen_direct,
        notes=[],
        created_at=utc_now(),
    )
    write_json(run_dir / "run.json", run_manifest.model_dump())

    command = [sys.executable, str(project_root / backend.script_path)]
    env = os.environ.copy()
    env.update(
        {
            "PDD_PROCESSED_PATH": prepared_manifest.prepared_dataset_path,
            "PDD_RESULTS_DIR": str(results_dir),
            "PDD_MODELS_DIR": str(models_dir),
            "PDD_STRING_PATH": str(resolve_project_path(project_root, config.paths.string_network)),
        }
    )

    result = subprocess.run(
        command,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    (logs_dir / "stdout.log").write_text(result.stdout)
    (logs_dir / "stderr.log").write_text(result.stderr)

    if result.returncode != 0:
        run_manifest.status = "failed"
        run_manifest.notes.append("Legacy training backend exited non-zero.")
        run_manifest.completed_at = utc_now()
        write_json(run_dir / "run.json", run_manifest.model_dump())
        raise RuntimeError(
            f"Training failed for run={run_id} with exit code {result.returncode}.\n"
            f"stderr tail:\n{_tail(result.stderr)}"
        )

    metrics_path = Path(run_manifest.training_metrics_path)
    model_path = Path(run_manifest.model_artifact_path)
    if not metrics_path.exists() or not model_path.exists():
        run_manifest.status = "failed"
        run_manifest.notes.append("Expected backend artifacts were not produced.")
        run_manifest.completed_at = utc_now()
        write_json(run_dir / "run.json", run_manifest.model_dump())
        raise FileNotFoundError(
            f"Expected training artifacts missing for run={run_id}: "
            f"metrics={metrics_path.exists()} model={model_path.exists()}"
        )

    run_manifest.status = "completed"
    run_manifest.completed_at = utc_now()
    run_manifest.mlflow_run_id = _log_run_to_mlflow(
        config=config,
        project_root=project_root,
        run_manifest=run_manifest,
    )
    write_json(run_dir / "run.json", run_manifest.model_dump())
    return run_manifest


def _make_run_id(*, task: str, model: str, split_name: str) -> str:
    timestamp = utc_now().replace(":", "").replace("-", "").replace(".", "")
    return f"{timestamp}-{task}-{model}-{split_name}"


def _log_run_to_mlflow(
    *,
    config: ProjectConfig,
    project_root: Path,
    run_manifest: RunManifest,
) -> str | None:
    tracking_uri = resolve_project_path(project_root, Path(config.paths.mlflow_tracking_uri))
    tracking_uri.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(tracking_uri))

    metrics_payload = read_json(Path(run_manifest.training_metrics_path))
    with mlflow.start_run(run_name=run_manifest.run_id) as run:
        mlflow.log_params(
            {
                "task": run_manifest.task,
                "model": run_manifest.model,
                "split_name": run_manifest.split_name,
                "prepared_dataset_path": run_manifest.prepared_dataset_path,
                "supports_unseen_direct": run_manifest.supports_unseen_direct,
            }
        )
        for key, value in _flatten_numeric_metrics(metrics_payload).items():
            mlflow.log_metric(key, value)
        mlflow.log_artifact(run_manifest.training_metrics_path, artifact_path="metrics")
        mlflow.log_artifact(run_manifest.model_artifact_path, artifact_path="models")
        return run.info.run_id


def _flatten_numeric_metrics(payload: dict, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in payload.items():
        metric_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flat[metric_key] = float(value)
        elif isinstance(value, dict):
            flat.update(_flatten_numeric_metrics(value, prefix=metric_key))
    return flat


def _tail(text: str, lines: int = 20) -> str:
    parts = text.strip().splitlines()
    if not parts:
        return "<empty stderr>"
    return "\n".join(parts[-lines:])
