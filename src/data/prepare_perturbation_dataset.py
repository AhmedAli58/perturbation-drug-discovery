"""
Preprocessing and dataset preparation for perturbation modeling.

Steps:
  1. Load raw Norman 2019 AnnData
  2. Run QC → normalize → HVG preprocessing
  3. Validate results
  4. Build perturbation label encoding
  5. Stratified train/test split by perturbation
  6. Save processed AnnData + metadata JSON
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

# ---------------------------------------------------------------------------
# Resolve project root so the script works from any cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import preprocess  # noqa: E402
from src.constants import (  # noqa: E402
    CONTROL_LABEL, PERTURBATION_COL, SEED, TRAIN_FRAC, VAL_FRAC,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "norman2019.h5ad"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_PATH = PROCESSED_DIR / "norman2019_processed.h5ad"
METADATA_PATH = PROCESSED_DIR / "dataset_metadata.json"

RANDOM_SEED = SEED

# ---------------------------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------------------------
if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Raw dataset not found at {RAW_PATH}. "
        "Run: python src/data/download_norman2019.py"
    )
logger.info("Loading %s …", RAW_PATH)
adata = sc.read_h5ad(RAW_PATH)
logger.info("Raw AnnData: %d cells × %d genes", adata.n_obs, adata.n_vars)

# Preserve original perturbation labels before any subsetting
if PERTURBATION_COL not in adata.obs.columns:
    raise KeyError(
        f"Column '{PERTURBATION_COL}' not found. "
        f"Available columns: {list(adata.obs.columns)}"
    )

# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
logger.info("Running preprocessing pipeline …")
adata = preprocess(adata)

# ---------------------------------------------------------------------------
# 4. Validate preprocessing results
# ---------------------------------------------------------------------------
n_cells = adata.n_obs
n_genes = adata.n_vars
perturbation_series = adata.obs[PERTURBATION_COL].astype(str)
unique_perturbations = sorted(perturbation_series.unique())
n_perturbations = len(unique_perturbations)

print("\n── Preprocessing results ─────────────────────────────────")
print(f"  Cells after QC filtering : {n_cells:,}")
print(f"  Genes after HVG selection: {n_genes:,}")
print(f"  Unique perturbations     : {n_perturbations:,}")
print("──────────────────────────────────────────────────────────\n")

# ---------------------------------------------------------------------------
# 5. Build perturbation label encoding
# ---------------------------------------------------------------------------
pert_to_idx: dict[str, int] = {p: i for i, p in enumerate(unique_perturbations)}
idx_to_pert: dict[int, str] = {i: p for p, i in pert_to_idx.items()}

adata.obs["perturbation_idx"] = (
    perturbation_series.map(pert_to_idx).astype(int)
)

logger.info("Perturbation encoding built: %d classes", n_perturbations)

# Store encoding in uns for easy retrieval later
# h5py requires all dict keys to be strings, so idx_to_pert uses str keys
adata.uns["perturbation_encoding"] = {
    "pert_to_idx": pert_to_idx,
    "idx_to_pert": {str(i): p for i, p in idx_to_pert.items()},
}

# ---------------------------------------------------------------------------
# 6. Stratified train / val / test split by perturbation group  (80/10/10)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)

train_mask = np.zeros(n_cells, dtype=bool)
val_mask   = np.zeros(n_cells, dtype=bool)

for pert in unique_perturbations:
    cell_indices = np.where(perturbation_series.values == pert)[0]
    n_pert = len(cell_indices)
    n_train = max(1, int(n_pert * TRAIN_FRAC))
    n_val   = max(0, int(n_pert * VAL_FRAC))
    # guarantee at least 1 test cell per perturbation if possible
    if n_train + n_val >= n_pert:
        n_val = max(0, n_pert - n_train - 1)
    shuffled = rng.permutation(cell_indices)
    train_mask[shuffled[:n_train]]               = True
    val_mask[shuffled[n_train:n_train + n_val]]  = True

test_mask = ~train_mask & ~val_mask

adata.obs["split"] = "test"
adata.obs.loc[adata.obs.index[train_mask], "split"] = "train"
adata.obs.loc[adata.obs.index[val_mask],   "split"] = "val"

n_train = int(train_mask.sum())
n_val   = int(val_mask.sum())
n_test  = int(test_mask.sum())

print(f"  Train cells : {n_train:,}  ({n_train / n_cells * 100:.1f}%)")
print(f"  Val   cells : {n_val:,}  ({n_val   / n_cells * 100:.1f}%)")
print(f"  Test  cells : {n_test:,}  ({n_test  / n_cells * 100:.1f}%)")

# Sanity check: every perturbation appears in train and test
train_perts = set(adata.obs.loc[adata.obs["split"] == "train", PERTURBATION_COL])
test_perts  = set(adata.obs.loc[adata.obs["split"] == "test",  PERTURBATION_COL])
missing_in_test = set(unique_perturbations) - test_perts
if missing_in_test:
    logger.warning(
        "%d perturbation(s) have no test cells (too few cells per group): %s",
        len(missing_in_test), missing_in_test,
    )

# ---------------------------------------------------------------------------
# 7. Save processed AnnData
# ---------------------------------------------------------------------------
logger.info("Saving processed AnnData to %s …", PROCESSED_PATH)
adata.write_h5ad(PROCESSED_PATH, compression="gzip")
logger.info("Saved.")

# ---------------------------------------------------------------------------
# 8. Save metadata JSON
# ---------------------------------------------------------------------------
metadata = {
    "dataset": "Norman2019 Perturb-seq (scPerturb curated)",
    "source_zenodo": "10.5281/zenodo.7041849",
    "cells": n_cells,
    "genes": n_genes,
    "perturbations": n_perturbations,
    "train_cells": n_train,
    "perturbation_column": PERTURBATION_COL,
    "unique_perturbations": unique_perturbations,
    "preprocessing": {
        "min_genes": 200,
        "min_cells": 3,
        "max_pct_mito": 20.0,
        "target_sum": 10000,
        "n_top_genes": 2000,
        "normalization": "log1p(total_count / 10000 * 10000)",
        "hvg_flavor": "seurat_v3",
    },
    "train_cells": n_train,
    "val_cells":   n_val,
    "test_cells":  n_test,
    "split": {
        "train_frac": TRAIN_FRAC,
        "val_frac":   VAL_FRAC,
        "test_frac":  round(1 - TRAIN_FRAC - VAL_FRAC, 2),
        "seed": RANDOM_SEED,
        "stratified_by": PERTURBATION_COL,
    },
}

METADATA_PATH.write_text(json.dumps(metadata, indent=2))
logger.info("Metadata saved to %s", METADATA_PATH)

# ---------------------------------------------------------------------------
# 9. Confirmation
# ---------------------------------------------------------------------------
print("\nPreprocessing complete. Dataset ready for perturbation modeling.")
