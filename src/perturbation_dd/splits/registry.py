"""Deterministic split builders for benchmark tasks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from perturbation_dd.config import SplitConfig


@dataclass
class SplitAssignment:
    labels: np.ndarray
    heldout_perturbations: list[str]
    train_perturbations: list[str]
    val_perturbations: list[str]
    test_perturbations: list[str]
    perturbation_counts: dict[str, int]


def build_split_assignment(
    perturbations: np.ndarray,
    split_name: str,
    control_label: str,
    split_config: SplitConfig,
) -> SplitAssignment:
    if split_name == "cell_split_v1":
        return _build_cell_split_v1(perturbations, control_label, split_config)
    if split_name == "pert_split_v1":
        return _build_pert_split_v1(perturbations, control_label, split_config)
    raise ValueError(f"Unsupported split: {split_name}")


def _build_cell_split_v1(
    perturbations: np.ndarray,
    control_label: str,
    split_config: SplitConfig,
) -> SplitAssignment:
    rng = np.random.default_rng(split_config.seed)
    labels = np.full(perturbations.shape[0], "test", dtype=object)
    unique_perts = sorted(np.unique(perturbations).tolist())

    for pert in unique_perts:
        idxs = np.where(perturbations == pert)[0]
        shuffled = rng.permutation(idxs)
        n_cells = len(shuffled)
        n_train = max(1, int(n_cells * split_config.train_frac))
        n_val = max(0, int(n_cells * split_config.val_frac))
        if n_train + n_val >= n_cells and n_cells > 1:
            n_val = max(0, n_cells - n_train - 1)
        labels[shuffled[:n_train]] = "train"
        labels[shuffled[n_train : n_train + n_val]] = "val"

    train_perts = sorted(np.unique(perturbations[labels == "train"]).tolist())
    val_perts = sorted(np.unique(perturbations[labels == "val"]).tolist())
    test_perts = sorted(np.unique(perturbations[labels == "test"]).tolist())
    return SplitAssignment(
        labels=labels,
        heldout_perturbations=[],
        train_perturbations=train_perts,
        val_perturbations=val_perts,
        test_perturbations=test_perts,
        perturbation_counts=dict(Counter(perturbations.tolist())),
    )


def _build_pert_split_v1(
    perturbations: np.ndarray,
    control_label: str,
    split_config: SplitConfig,
) -> SplitAssignment:
    rng = np.random.default_rng(split_config.seed)
    labels = np.full(perturbations.shape[0], "train", dtype=object)
    counts = Counter(perturbations.tolist())

    eligible = sorted(
        pert
        for pert, count in counts.items()
        if pert != control_label and count >= split_config.min_cells_per_pert
    )
    heldout_count = max(1, int(len(eligible) * split_config.heldout_frac)) if eligible else 0
    heldout = (
        sorted(rng.choice(np.array(eligible), size=heldout_count, replace=False).tolist())
        if heldout_count
        else []
    )
    heldout_set = set(heldout)

    seen_train_perts: list[str] = []
    seen_val_perts: list[str] = []

    seen_train_share = split_config.train_frac / (split_config.train_frac + split_config.val_frac)

    for pert in sorted(np.unique(perturbations).tolist()):
        idxs = np.where(perturbations == pert)[0]
        shuffled = rng.permutation(idxs)

        if pert == control_label:
            n_cells = len(shuffled)
            n_train = max(1, int(n_cells * split_config.train_frac))
            n_val = max(0, int(n_cells * split_config.val_frac))
            if n_train + n_val >= n_cells and n_cells > 1:
                n_val = max(0, n_cells - n_train - 1)
            labels[shuffled[:n_train]] = "train"
            labels[shuffled[n_train : n_train + n_val]] = "val"
            labels[shuffled[n_train + n_val :]] = "test"
            continue

        if pert in heldout_set:
            labels[shuffled] = "test"
            continue

        n_cells = len(shuffled)
        n_train = max(1, int(n_cells * seen_train_share))
        if n_train >= n_cells and n_cells > 1:
            n_train = n_cells - 1
        labels[shuffled[:n_train]] = "train"
        labels[shuffled[n_train:]] = "val"
        seen_train_perts.append(pert)
        seen_val_perts.append(pert)

    test_perts = sorted(np.unique(perturbations[labels == "test"]).tolist())
    train_perts = sorted(
        set(seen_train_perts + ([control_label] if np.any(labels == "train") else []))
    )
    val_perts = sorted(
        set(seen_val_perts + ([control_label] if np.any(labels == "val") else []))
    )
    return SplitAssignment(
        labels=labels,
        heldout_perturbations=heldout,
        train_perturbations=train_perts,
        val_perturbations=val_perts,
        test_perturbations=test_perts,
        perturbation_counts=dict(counts),
    )
