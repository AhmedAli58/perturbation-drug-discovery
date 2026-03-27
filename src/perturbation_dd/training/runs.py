"""Run registry helpers for filesystem-backed benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

from perturbation_dd.config import ProjectConfig
from perturbation_dd.data.access import resolve_project_path
from perturbation_dd.types import RunManifest
from perturbation_dd.utils.io import read_json


def runs_root(project_root: Path, config: ProjectConfig) -> Path:
    artifact_root = resolve_project_path(project_root, config.paths.artifact_root)
    return artifact_root / "runs"


def run_manifest_path(project_root: Path, config: ProjectConfig, run_id: str) -> Path:
    return runs_root(project_root, config) / run_id / "run.json"


def load_run_manifest(project_root: Path, config: ProjectConfig, run_id: str) -> RunManifest:
    path = run_manifest_path(project_root, config, run_id)
    if not path.exists():
        raise FileNotFoundError(f"Run manifest not found at {path}.")
    return RunManifest.model_validate(read_json(path))


def list_run_manifests(project_root: Path, config: ProjectConfig) -> list[RunManifest]:
    root = runs_root(project_root, config)
    if not root.exists():
        return []
    manifests: list[RunManifest] = []
    for path in sorted(root.glob("*/run.json")):
        manifests.append(RunManifest.model_validate(read_json(path)))
    return manifests


def latest_run_for(
    project_root: Path,
    config: ProjectConfig,
    *,
    split_name: str,
    task: str,
    model: str,
    status: str = "completed",
) -> RunManifest | None:
    candidates = [
        run
        for run in list_run_manifests(project_root, config)
        if (
            run.split_name == split_name
            and run.task == task
            and run.model == model
            and run.status == status
        )
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.created_at)[-1]
