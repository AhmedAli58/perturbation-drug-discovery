# Perturbation-Based Drug Target Discovery

A machine learning pipeline for identifying drug targets using single-cell perturbation data (e.g., Perturb-seq, CRISPR screens).

## Project Structure

```
perturbation-drug-discovery/
├── data/
│   ├── raw/          # Original .h5ad / .mtx datasets (gitignored)
│   └── processed/    # Preprocessed AnnData objects (gitignored)
├── notebooks/        # Exploratory analysis and figures
├── src/
│   ├── data/         # Data loading, preprocessing, splitting
│   ├── models/       # Model architectures (VAE, GNN, etc.)
│   └── evaluation/   # Metrics and benchmarking
├── configs/          # YAML experiment configs
├── results/          # Model outputs, plots (gitignored)
├── tests/            # Unit tests
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Formats Supported

| Format | Description |
|--------|-------------|
| `.h5ad` | AnnData HDF5 (scanpy native) |
| `.mtx` | 10x Genomics sparse matrix directory |

## Pipeline Overview

1. **Data Loading** — `src/data/loader.py` reads `.h5ad` or 10x `.mtx` directories into `AnnData` objects.
2. **Preprocessing** — QC filtering, normalization, log1p, HVG selection.
3. **Perturbation Encoding** — Encode CRISPR guide assignments as covariates.
4. **Modeling** — Perturbation-aware latent space models.
5. **Evaluation** — Gene expression prediction, enrichment, target ranking.

## Datasets (not included)

- [Norman et al. 2019 (Perturb-seq)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344)
- [Replogle et al. 2022 (Genome-wide Perturb-seq)](https://doi.org/10.1016/j.cell.2022.05.013)
- [LINCS L1000](https://lincsproject.org/)
