"""Baseline evaluators for prepared perturbation-response datasets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from perturbation_dd.config import ProjectConfig
from perturbation_dd.data.access import load_prepared_adata, resolve_project_path
from perturbation_dd.evaluation.metrics import evaluate_response_predictions, to_dense
from perturbation_dd.types import PreparedDatasetManifest


def compute_response_baselines(
    config: ProjectConfig,
    project_root: Path,
    manifest: PreparedDatasetManifest,
    task: str,
) -> dict[str, dict]:
    adata = load_prepared_adata(manifest)
    matrix = to_dense(adata.X)
    perturbation_col = manifest.perturbation_key
    perturbations = adata.obs[perturbation_col].astype(str).to_numpy()
    splits = adata.obs["split"].astype(str).to_numpy()

    control_mask = perturbations == manifest.control_label
    train_mask = splits == "train"
    target_mask = _target_mask(
        perturbations=perturbations,
        splits=splits,
        control_label=manifest.control_label,
        manifest=manifest,
        task=task,
    )
    target_matrix = matrix[target_mask]
    target_perturbations = perturbations[target_mask]

    if target_matrix.shape[0] == 0:
        raise ValueError(f"No target cells available for baseline evaluation on task={task}.")

    train_control = matrix[train_mask & control_mask]
    if train_control.shape[0] == 0:
        raise ValueError("No train control cells available for baseline evaluation.")
    mean_control = train_control.mean(axis=0)

    train_mean_by_perturbation = {
        pert: matrix[(train_mask) & (perturbations == pert)].mean(axis=0)
        for pert in sorted(np.unique(perturbations[train_mask]))
        if pert != manifest.control_label and np.any((train_mask) & (perturbations == pert))
    }
    train_support = {
        pert: int(((train_mask) & (perturbations == pert)).sum())
        for pert in train_mean_by_perturbation
    }

    naive_predictions = {
        pert: mean_control.copy()
        for pert in sorted(np.unique(target_perturbations).tolist())
    }
    naive_metrics = evaluate_response_predictions(
        predictions_by_perturbation=naive_predictions,
        target_matrix=target_matrix,
        target_perturbations=target_perturbations,
    )
    naive_metrics.update(
        {
            "name": "naive_control",
            "target_perturbations": sorted(np.unique(target_perturbations).tolist()),
            "proxy_map": {},
            "train_support_cells": {},
        }
    )

    proxy_map: dict[str, str] = {}
    nearest_predictions: dict[str, np.ndarray] = {}
    seen_perts = sorted(train_mean_by_perturbation)
    string_neighbors = load_string_neighbors(
        resolve_project_path(project_root, config.paths.string_network)
    )

    for candidate in sorted(np.unique(target_perturbations).tolist()):
        if candidate in train_mean_by_perturbation:
            proxy = candidate
        else:
            proxy = choose_proxy_perturbation(
                candidate=candidate,
                seen_perturbations=seen_perts,
                string_neighbors=string_neighbors,
                support_counts=train_support,
            )
        proxy_map[candidate] = proxy
        nearest_predictions[candidate] = train_mean_by_perturbation.get(proxy, mean_control.copy())

    nearest_metrics = evaluate_response_predictions(
        predictions_by_perturbation=nearest_predictions,
        target_matrix=target_matrix,
        target_perturbations=target_perturbations,
    )
    nearest_metrics.update(
        {
            "name": "nearest_seen",
            "target_perturbations": sorted(np.unique(target_perturbations).tolist()),
            "proxy_map": proxy_map,
            "train_support_cells": {
                candidate: train_support.get(proxy, 0) for candidate, proxy in proxy_map.items()
            },
        }
    )
    return {"naive_control": naive_metrics, "nearest_seen": nearest_metrics}


def load_string_neighbors(path: Path) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return {}
    with path.open() as handle:
        next(handle, None)
        for line in handle:
            left, right, *_ = line.strip().split("\t")
            neighbors[left].add(right)
            neighbors[right].add(left)
    return dict(neighbors)


def choose_proxy_perturbation(
    candidate: str,
    seen_perturbations: list[str],
    string_neighbors: dict[str, set[str]],
    support_counts: dict[str, int],
) -> str:
    candidate_genes = set(candidate.split("_"))
    best_name = seen_perturbations[0] if seen_perturbations else candidate
    best_score = float("-inf")

    for seen in seen_perturbations:
        seen_genes = set(seen.split("_"))
        overlap = len(candidate_genes & seen_genes)
        direct_links = sum(
            1
            for left in candidate_genes
            for right in seen_genes
            if right in string_neighbors.get(left, set())
        )
        shared_neighbors = sum(
            len(string_neighbors.get(left, set()) & string_neighbors.get(right, set()))
            for left in candidate_genes
            for right in seen_genes
        )
        score = (
            overlap * 100
            + direct_links * 25
            + shared_neighbors
            + np.log1p(support_counts.get(seen, 0))
        )
        if score > best_score:
            best_score = score
            best_name = seen
    return best_name


def _target_mask(
    perturbations: np.ndarray,
    splits: np.ndarray,
    control_label: str,
    manifest: PreparedDatasetManifest,
    task: str,
) -> np.ndarray:
    non_control = perturbations != control_label
    if task == "response_known":
        return (splits == "test") & non_control
    if task == "response_heldout":
        heldout = set(manifest.heldout_perturbations)
        return (splits == "test") & non_control & np.isin(perturbations, list(heldout))
    raise ValueError(f"Unsupported response task for baselines: {task}")
