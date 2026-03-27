"""Shared metrics for response prediction benchmarks."""

from __future__ import annotations

from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr


def to_dense(matrix: np.ndarray | sp.spmatrix) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    try:
        score = pearsonr(left, right)[0]
        return 0.0 if np.isnan(score) else float(score)
    except Exception:
        return 0.0


def evaluate_response_predictions(
    predictions_by_perturbation: dict[str, np.ndarray],
    target_matrix: np.ndarray,
    target_perturbations: np.ndarray,
) -> dict[str, object]:
    pred_matrix = np.stack(
        [predictions_by_perturbation[str(pert)] for pert in target_perturbations],
        axis=0,
    )
    target_matrix = np.asarray(target_matrix, dtype=np.float32)

    cell_scores = np.array(
        [
            safe_pearson(pred_matrix[idx], target_matrix[idx])
            for idx in range(target_matrix.shape[0])
        ]
    )
    gene_scores = np.array(
        [
            safe_pearson(pred_matrix[:, gene_idx], target_matrix[:, gene_idx])
            for gene_idx in range(target_matrix.shape[1])
        ]
    )

    per_perturbation_eval: dict[str, dict[str, float | int]] = {}
    counts = Counter(target_perturbations.tolist())
    unique_perturbations = sorted(counts)

    for pert in unique_perturbations:
        mask = target_perturbations == pert
        mean_target = target_matrix[mask].mean(axis=0)
        pred = predictions_by_perturbation[str(pert)]
        per_perturbation_eval[str(pert)] = {
            "n_cells": int(counts[pert]),
            "mse": round(float(np.mean((pred - mean_target) ** 2)), 6),
            "pearson_r": round(safe_pearson(pred, mean_target), 6),
        }

    per_pert_scores = [float(item["pearson_r"]) for item in per_perturbation_eval.values()]
    return {
        "test_mse": round(float(np.mean((pred_matrix - target_matrix) ** 2)), 6),
        "mean_cell_pearson_r": round(float(np.mean(cell_scores)), 6),
        "mean_gene_pearson_r": round(float(np.mean(gene_scores)), 6),
        "mean_per_pert_pearson_r": round(float(np.mean(per_pert_scores)), 6),
        "per_perturbation_eval": per_perturbation_eval,
    }
