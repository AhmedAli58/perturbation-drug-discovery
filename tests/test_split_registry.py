from __future__ import annotations

import numpy as np

from perturbation_dd.config import SplitConfig
from perturbation_dd.splits.registry import build_split_assignment


def test_cell_split_registry_is_deterministic() -> None:
    perturbations = np.array(["control"] * 10 + ["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    config = SplitConfig(seed=42)

    first = build_split_assignment(perturbations, "cell_split_v1", "control", config)
    second = build_split_assignment(perturbations, "cell_split_v1", "control", config)

    assert first.labels.tolist() == second.labels.tolist()
    assert first.heldout_perturbations == []


def test_pert_split_registry_holds_out_full_perturbations() -> None:
    perturbations = np.array(
        ["control"] * 12 + ["A"] * 120 + ["B"] * 120 + ["C"] * 120 + ["D"] * 120
    )
    config = SplitConfig(seed=7, heldout_frac=0.25, min_cells_per_pert=100)

    assignment = build_split_assignment(perturbations, "pert_split_v1", "control", config)

    assert len(assignment.heldout_perturbations) == 1
    heldout = assignment.heldout_perturbations[0]
    heldout_labels = assignment.labels[perturbations == heldout]
    assert set(heldout_labels.tolist()) == {"test"}
