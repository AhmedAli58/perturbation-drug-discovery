"""
Shared constants for all training and data scripts.
All values are read from configs/default.yaml — edit there, not here.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text())

CONTROL_LABEL    : str   = _cfg["data"]["control_label"]
PERTURBATION_COL : str   = _cfg["data"]["perturbation_key"]
SEED             : int   = _cfg["split"]["seed"]
TRAIN_FRAC       : float = _cfg["split"]["train_frac"]
VAL_FRAC         : float = _cfg["split"]["val_frac"]
TEST_FRAC        : float = _cfg["split"]["test_frac"]
