"""
Download and validate the Norman et al. 2019 Perturb-seq dataset.

Dataset : https://doi.org/10.1126/science.aax9800
Source  : scPerturb collection on Zenodo (DOI: 10.5281/zenodo.7041849)
          NormanWeissman2019_filtered.h5ad  (~666 MB)
Target  : data/raw/norman2019.h5ad
"""

import json
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Ensure directories exist
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET = RAW_DIR / "norman2019.h5ad"

# Zenodo direct-download URL (scPerturb curated, ~666 MB, no WAF/bot challenge)
URL = "https://zenodo.org/api/records/7041849/files/NormanWeissman2019_filtered.h5ad/content"
MIN_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# 2. Download (skip if already present)
# ---------------------------------------------------------------------------
def _progress(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        print(f"\r  {pct:5.1f}%  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)
    else:
        print(f"\r  {downloaded / 1_048_576:.1f} MB downloaded", end="", flush=True)


if TARGET.exists():
    print(f"File already exists, skipping download: {TARGET}")
else:
    print(f"Downloading Norman 2019 Perturb-seq dataset …")
    print(f"  URL    : {URL}")
    print(f"  Target : {TARGET}")
    urllib.request.urlretrieve(URL, TARGET, reporthook=_progress)
    print()  # newline after progress bar
    print("Download complete.")

# ---------------------------------------------------------------------------
# 3. Validate file
# ---------------------------------------------------------------------------
if not TARGET.exists():
    raise FileNotFoundError(f"Expected file not found after download: {TARGET}")

file_size = TARGET.stat().st_size
print(f"File size: {file_size / 1_048_576:.1f} MB")

if file_size < MIN_SIZE_BYTES:
    raise ValueError(
        f"File is too small ({file_size / 1_048_576:.1f} MB). "
        "Expected > 100 MB. The download may have failed or returned an error page."
    )

# ---------------------------------------------------------------------------
# 4. Load dataset
# ---------------------------------------------------------------------------
import scanpy as sc  # noqa: E402  (heavy import deferred intentionally)

print("\nLoading AnnData …")
adata = sc.read_h5ad(TARGET)

# ---------------------------------------------------------------------------
# 5. Print dataset info
# ---------------------------------------------------------------------------
print(f"\nDataset shape : {adata.n_obs} cells × {adata.n_vars} genes")

# ---------------------------------------------------------------------------
# 6. Detect perturbation column
# ---------------------------------------------------------------------------
POSSIBLE_PERTURBATION_COLS = [
    "perturbation",
    "perturbation_name",
    "guide",
    "guide_ids",
    "target_gene",
    "gene",
    "condition",
]

pert_col = next(
    (col for col in POSSIBLE_PERTURBATION_COLS if col in adata.obs.columns),
    None,
)

if pert_col:
    n_unique = adata.obs[pert_col].nunique()
    print(f"Perturbation column : '{pert_col}'  ({n_unique} unique values)")
    print(f"  Sample values     : {list(adata.obs[pert_col].unique()[:8])}")
else:
    print("No known perturbation column found. First 10 obs columns:")
    print(f"  {list(adata.obs.columns[:10])}")

# ---------------------------------------------------------------------------
# 7. Save dataset summary
# ---------------------------------------------------------------------------
summary = {
    "dataset": "Norman2019 Perturb-seq",
    "cells": adata.n_obs,
    "genes": adata.n_vars,
    "obs_columns": list(adata.obs.columns),
}
summary_path = PROCESSED_DIR / "dataset_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"\nSummary saved to: {summary_path}")

# ---------------------------------------------------------------------------
# 8. Confirmation
# ---------------------------------------------------------------------------
print("\nNorman 2019 Perturb-seq dataset downloaded and validated successfully.")
