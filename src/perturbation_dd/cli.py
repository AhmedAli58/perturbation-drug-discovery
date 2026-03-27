"""Primary CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer

from perturbation_dd.config import load_project_config
from perturbation_dd.data.prepare import prepare_dataset

app = typer.Typer(help="Perturbation benchmark and prioritization CLI.")
DEFAULT_CONFIG_PATH = typer.Option(Path("configs/base.yaml"), "--config")
TASK_OPTION = typer.Option(..., "--task")
MODEL_OPTION = typer.Option(..., "--model")
SPLIT_OPTION = typer.Option(..., "--split")
RUN_ID_OPTION = typer.Option(..., "--run-id")
INPUT_OPTION = typer.Option(..., "--input")
OUTPUT_OPTION = typer.Option(..., "--output")


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
    task: str = TASK_OPTION,
    model: str = MODEL_OPTION,
    split: str = SPLIT_OPTION,
) -> None:
    """Training backend wiring is implemented in the next slice."""
    typer.echo(
        f"Training not yet wired for task={task}, model={model}, split={split}",
        err=True,
    )
    raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate_command(run_id: str = RUN_ID_OPTION) -> None:
    """Evaluation wiring is implemented in the next slice."""
    typer.echo(f"Evaluation not yet wired for run {run_id}", err=True)
    raise typer.Exit(code=1)


@app.command("rank-candidates")
def rank_candidates_command(
    input_path: Path = INPUT_OPTION,
    output_path: Path = OUTPUT_OPTION,
    split: str = SPLIT_OPTION,
) -> None:
    """Ranking wiring is implemented in the next slice."""
    typer.echo(
        f"Ranking not yet wired for {input_path} -> {output_path} on split {split}",
        err=True,
    )
    raise typer.Exit(code=1)


@app.command("build-report")
def build_report_command(run_id: str = RUN_ID_OPTION) -> None:
    """Report generation wiring is implemented in the next slice."""
    typer.echo(f"Report generation not yet wired for run {run_id}", err=True)
    raise typer.Exit(code=1)
