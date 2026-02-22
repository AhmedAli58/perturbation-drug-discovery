"""
Unit tests for src/data/loader.py — runs without real datasets.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_dataset


def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        from src.data.loader import load_h5ad
        load_h5ad(tmp_path / "nonexistent.h5ad")


def test_load_dataset_wrong_extension(tmp_path):
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b,c")
    with pytest.raises(ValueError, match="Cannot determine dataset format"):
        load_dataset(bad_file)


def test_load_dataset_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="Cannot determine dataset format"):
        load_dataset(tmp_path)


def test_load_h5ad_wrong_suffix(tmp_path):
    from src.data.loader import load_h5ad
    bad = tmp_path / "data.loom"
    bad.write_text("")
    with pytest.raises(ValueError, match=".h5ad"):
        load_h5ad(bad)
