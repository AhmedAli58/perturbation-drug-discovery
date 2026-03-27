"""Evaluation manifests and markdown benchmark reports."""

from __future__ import annotations

from pathlib import Path

import mlflow

from perturbation_dd.config import ProjectConfig
from perturbation_dd.data.access import resolve_project_path
from perturbation_dd.evaluation.baselines import compute_response_baselines
from perturbation_dd.training.runs import load_run_manifest, run_manifest_path
from perturbation_dd.types import EvaluationManifest
from perturbation_dd.utils.io import read_json, utc_now, write_json


def evaluate_run(
    config: ProjectConfig,
    project_root: Path,
    *,
    run_id: str,
) -> EvaluationManifest:
    run_manifest = load_run_manifest(project_root, config, run_id)
    metrics = read_json(Path(run_manifest.training_metrics_path))

    baselines: dict[str, dict] = {}
    if run_manifest.task.startswith("response_"):
        prepared_manifest = read_json(Path(run_manifest.prepared_manifest_path))
        baselines = compute_response_baselines(
            config=config,
            project_root=project_root,
            manifest=_coerce_prepared_manifest(prepared_manifest),
            task=run_manifest.task,
        )

    evaluation = EvaluationManifest(
        run_id=run_manifest.run_id,
        task=run_manifest.task,
        model=run_manifest.model,
        split_name=run_manifest.split_name,
        baselines=baselines,
        model_metrics=metrics,
        comparison=_build_comparison(
            run_manifest.task,
            metrics,
            baselines,
            run_manifest.supports_unseen_direct,
        ),
        created_at=utc_now(),
    )

    evaluation_path = Path(run_manifest.training_metrics_path).parent.parent / "evaluation.json"
    write_json(evaluation_path, evaluation.model_dump())
    run_manifest.evaluation_path = str(evaluation_path)
    write_json(
        run_manifest_path(project_root, config, run_manifest.run_id),
        run_manifest.model_dump(),
    )
    _log_evaluation_to_mlflow(config, project_root, run_manifest)
    return evaluation


def build_report(
    config: ProjectConfig,
    project_root: Path,
    *,
    run_id: str,
) -> Path:
    run_manifest = load_run_manifest(project_root, config, run_id)
    evaluation = (
        read_json(Path(run_manifest.evaluation_path))
        if run_manifest.evaluation_path and Path(run_manifest.evaluation_path).exists()
        else evaluate_run(config, project_root, run_id=run_id).model_dump()
    )

    report_path = Path(run_manifest.results_dir).parent / "benchmark_report.md"
    report_path.write_text(_render_report(evaluation))
    run_manifest.report_path = str(report_path)
    write_json(
        run_manifest_path(project_root, config, run_manifest.run_id),
        run_manifest.model_dump(),
    )
    return report_path


def _build_comparison(
    task: str,
    model_metrics: dict,
    baselines: dict[str, dict],
    supports_unseen_direct: bool,
) -> dict[str, object]:
    if task == "classification":
        return {
            "primary_metric": "accuracy",
            "accuracy": model_metrics.get("accuracy"),
            "top5_accuracy": model_metrics.get("top5_accuracy"),
        }

    naive = baselines.get("naive_control", {})
    nearest = baselines.get("nearest_seen", {})
    model_per_pert = float(model_metrics.get("mean_per_pert_pearson_r", 0.0))
    model_mse = float(model_metrics.get("test_mse", 0.0))
    naive_per_pert = float(naive.get("mean_per_pert_pearson_r", 0.0))
    nearest_per_pert = float(nearest.get("mean_per_pert_pearson_r", 0.0))
    naive_mse = float(naive.get("test_mse", 0.0))
    nearest_mse = float(nearest.get("test_mse", 0.0))

    direct_supported = supports_unseen_direct or task == "response_known"
    beats_nearest = model_per_pert > nearest_per_pert and model_mse <= nearest_mse
    preferred_ranking_mode = "direct" if direct_supported and beats_nearest else "proxy"

    return {
        "primary_metric": "mean_per_pert_pearson_r",
        "delta_vs_naive_mean_per_pert": round(model_per_pert - naive_per_pert, 6),
        "delta_vs_nearest_mean_per_pert": round(model_per_pert - nearest_per_pert, 6),
        "delta_vs_naive_test_mse": round(naive_mse - model_mse, 6),
        "delta_vs_nearest_test_mse": round(nearest_mse - model_mse, 6),
        "beats_naive": model_per_pert > naive_per_pert and model_mse <= naive_mse,
        "beats_nearest_seen": beats_nearest,
        "preferred_ranking_mode": preferred_ranking_mode,
    }


def _render_report(evaluation: dict) -> str:
    lines = [
        "# Benchmark Report",
        "",
        f"- Run ID: `{evaluation['run_id']}`",
        f"- Task: `{evaluation['task']}`",
        f"- Model: `{evaluation['model']}`",
        f"- Split: `{evaluation['split_name']}`",
        "",
        "## Model Metrics",
    ]
    metrics = evaluation["model_metrics"]
    for key in (
        "accuracy",
        "top5_accuracy",
        "test_mse",
        "mean_cell_pearson_r",
        "mean_gene_pearson_r",
        "mean_per_pert_pearson_r",
    ):
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")

    if evaluation["baselines"]:
        lines.extend(["", "## Baselines"])
        for name, baseline in evaluation["baselines"].items():
            lines.append(
                f"- {name}: per-pert r `{baseline.get('mean_per_pert_pearson_r')}`, "
                f"test MSE `{baseline.get('test_mse')}`"
            )
        lines.extend(["", "## Comparison"])
        for key, value in evaluation["comparison"].items():
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def _coerce_prepared_manifest(payload: dict):
    from perturbation_dd.types import PreparedDatasetManifest

    return PreparedDatasetManifest.model_validate(payload)


def _log_evaluation_to_mlflow(
    config: ProjectConfig,
    project_root: Path,
    run_manifest,
) -> None:
    if not run_manifest.mlflow_run_id or not run_manifest.evaluation_path:
        return
    tracking_uri = resolve_project_path(project_root, Path(config.paths.mlflow_tracking_uri))
    mlflow.set_tracking_uri(str(tracking_uri))
    evaluation = read_json(Path(run_manifest.evaluation_path))
    with mlflow.start_run(run_id=run_manifest.mlflow_run_id):
        mlflow.log_artifact(run_manifest.evaluation_path, artifact_path="evaluation")
        for key, value in evaluation["comparison"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(f"comparison.{key}", float(value))
