"""Helpers for routing legacy scripts through configurable artifact paths."""

from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default
