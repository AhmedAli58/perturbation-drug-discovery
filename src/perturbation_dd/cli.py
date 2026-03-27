"""Primary CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer

from perturbation_dd.config import load_project_config
from perturbation_dd.data.prepare import prepare_dataset
from perturbation_dd.evaluation.reporting import build_report, evaluate_run
from perturbation_dd.ranking.service import rank_candidates
from perturbation_dd.training.backends import train_backend_run

app = typer.Typer(help="Perturbation benchmark and prioritization CLI.")
DEFAULT_CONFIG_PATH = typer.Option(Path("configs/base.yaml"), "--config")
TASK_OPTION = typer.Option(..., "--task")
MODEL_OPTION = typer.Option(..., "--model")
SPLIT_OPTION = typer.Option(..., "--split")
RUN_ID_OPTION = typer.Option(..., "--run-id")
INPUT_OPTION = typer.Option(..., "--input")
OUTPUT_OPTION = typer.Option(..., "--output")
MODEL_FAMILY_OPTION = typer.Option("ensemble", "--model-family")


def _infer_project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return resolved.parent.parent if resolved.parent.name == "configs" else resolved.parent


@app.command("prepare-data")
def prepare_data_command(
    config_path: Path = DEFAULT_CONFIG_PATH,
    split: str = SPLIT_OPTION,
) -> None:
    """Prepare a dataset artifact and write a split manifest."""
    project_root = _infer_project_root(config_path)
    config = load_project_config(config_path)
    manifest = prepare_dataset(config=config, project_root=project_root, split_name=split)
    typer.echo(f"Prepared dataset: {manifest.prepared_dataset_path}")
    typer.echo(f"Manifest: {Path(manifest.prepared_dataset_path).parent / 'manifest.json'}")


@app.command("train")
def train_command(
    config_path: Path = DEFAULT_CONFIG_PATH,
    task: str = TASK_OPTION,
    model: str = MODEL_OPTION,
    split: str = SPLIT_OPTION,
) -> None:
    """Train a model through the shared filesystem-backed run contract."""
    project_root = _infer_project_root(config_path)
    config = load_project_config(config_path)
    run_manifest = train_backend_run(
        config,
        project_root,
        task=task,
        model=model,
        split_name=split,
    )
    typer.echo(f"Run ID: {run_manifest.run_id}")
    typer.echo(f"Training metrics: {run_manifest.training_metrics_path}")
    typer.echo(f"Model artifact: {run_manifest.model_artifact_path}")


@app.command("evaluate")
def evaluate_command(
    config_path: Path = DEFAULT_CONFIG_PATH,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Evaluate a trained run against shared baselines."""
    project_root = _infer_project_root(config_path)
    config = load_project_config(config_path)
    evaluation = evaluate_run(config, project_root, run_id=run_id)
    typer.echo(f"Evaluation manifest created for run: {evaluation.run_id}")
    typer.echo(f"Primary metric: {evaluation.comparison.get('primary_metric')}")


@app.command("rank-candidates")
def rank_candidates_command(
    config_path: Path = DEFAULT_CONFIG_PATH,
    input_path: Path = INPUT_OPTION,
    output_path: Path = OUTPUT_OPTION,
    split: str = SPLIT_OPTION,
    model_family: str = MODEL_FAMILY_OPTION,
) -> None:
    """Rank perturbation candidates for wet-lab follow-up."""
    project_root = _infer_project_root(config_path)
    config = load_project_config(config_path)
    ranking = rank_candidates(
        config,
        project_root,
        input_path=input_path,
        output_path=output_path,
        split_name=split,
        model_family=model_family,
    )
    typer.echo(f"Wrote {len(ranking)} ranked candidates to {output_path}")


@app.command("build-report")
def build_report_command(
    config_path: Path = DEFAULT_CONFIG_PATH,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Build a markdown benchmark report for an evaluated run."""
    project_root = _infer_project_root(config_path)
    config = load_project_config(config_path)
    report_path = build_report(config, project_root, run_id=run_id)
    typer.echo(f"Report: {report_path}")
