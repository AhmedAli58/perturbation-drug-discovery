# Perturbation-Based Drug Target Discovery

> A deep learning pipeline for identifying drug targets from single-cell CRISPR perturbation data (Perturb-seq). Predicts how gene knockouts alter transcriptomic profiles to rank candidate therapeutic targets.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project builds a full ML pipeline on the [Norman et al. 2019](https://doi.org/10.1126/science.aax4438) Perturb-seq dataset — 111,391 K562 cells with 237 single and combinatorial CRISPR perturbations. The goal is to predict post-perturbation gene expression profiles from control cell states, enabling virtual drug target screening without requiring wet-lab experiments for every candidate gene.

**Best result:** Per-perturbation Pearson r = **0.9957** (Effect MLP), zero-shot generalization to unseen perturbations r = **0.9843** (scGen VAE).

---

## Features

- **Five model architectures** — from logistic regression baseline to graph-aware VAE
- **Zero-shot generalization** — predicts unseen perturbation effects via latent space interpolation
- **STRING PPI integration** — uses protein interaction network as biological prior for GCN
- **Interactive dashboard** — Streamlit app for exploring predictions, embeddings, and target rankings
- **Reproducible pipeline** — YAML configs, stratified splits, seeded experiments

---

## Results

| Model | Task | Metric | Score |
|---|---|---|---|
| Logistic Regression | Perturbation Classification | Top-1 Accuracy | 37.4% |
| MLP Classifier | Perturbation Classification | Top-1 Accuracy | 45.9% |
| Effect MLP | Expression Prediction | Per-Pert Pearson r | **0.9957** |
| Graph GCN | Expression Prediction | Per-Pert Pearson r | 0.9903 |
| scGen VAE | Expression Prediction | Per-Pert Pearson r | 0.9798 |
| VAE Zero-Shot | Unseen Perturbations (44) | Per-Pert Pearson r | 0.9843 |

> **Limitation:** Mean gene-level Pearson r < 0.12 across all models — predicting individual gene magnitudes remains the hardest sub-task.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Single-cell analysis | `scanpy`, `anndata` |
| Deep learning | `PyTorch` |
| Graph modeling | Manual GCN (no PyG required) |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| Data | `scipy`, `pandas`, `numpy`, `h5py` |
| Config | `pyyaml` |
| Tests | `pytest` |

---

## Project Structure

```
perturbation-drug-discovery/
├── app.py                          # Streamlit dashboard (run: streamlit run app.py)
├── requirements.txt
├── configs/
│   └── default.yaml                # Hyperparameters, paths, split config
├── data/
│   ├── raw/                        # Raw .h5ad (gitignored — download via script)
│   ├── processed/                  # Preprocessed AnnData + metadata (gitignored)
│   ├── external/                   # STRING PPI network (TSV)
│   ├── models/                     # Trained model checkpoints (gitignored)
│   └── results/                    # Per-model evaluation metrics (JSON)
├── reports/
│   ├── summary.txt                 # Biological interpretation narrative
│   └── figures/                    # Publication-quality plots
├── src/
│   ├── data/
│   │   ├── loader.py               # .h5ad and 10x .mtx loading
│   │   ├── preprocessor.py         # QC, normalization, HVG selection
│   │   ├── prepare_perturbation_dataset.py   # End-to-end preprocessing script
│   │   ├── download_norman2019.py  # Download dataset from Zenodo
│   │   └── download_string_ppi.py  # Fetch STRING PPI network via API
│   ├── models/
│   │   ├── train_baseline_classifier.py      # Logistic Regression
│   │   ├── train_mlp_classifier.py           # 3-layer MLP classifier
│   │   ├── train_perturbation_effect_model.py # MLP effect predictor
│   │   ├── train_graph_perturbation_model.py  # GCN + STRING PPI
│   │   └── train_scgen_style_model.py         # VAE (scGen-inspired)
│   ├── experiments/
│   │   └── unseen_perturbation_generalization.py  # Zero-shot evaluation
│   └── analysis/
│       ├── interpret_perturbation_results.py  # Biological interpretation
│       └── visualize_results.py               # Result figures
└── tests/
    └── test_loader.py
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/perturbation-drug-discovery.git
cd perturbation-drug-discovery

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quickstart

### 1. Download data

```bash
python src/data/download_norman2019.py
python src/data/download_string_ppi.py
```

### 2. Preprocess

```bash
python src/data/prepare_perturbation_dataset.py
```

### 3. Train models (in order)

```bash
# Baseline
python src/models/train_baseline_classifier.py

# MLP classifier
python src/models/train_mlp_classifier.py

# Effect prediction models
python src/models/train_perturbation_effect_model.py
python src/models/train_graph_perturbation_model.py
python src/models/train_scgen_style_model.py
```

### 4. Evaluate zero-shot generalization

```bash
python src/experiments/unseen_perturbation_generalization.py
```

### 5. Generate figures and interpretation

```bash
python src/analysis/visualize_results.py
python src/analysis/interpret_perturbation_results.py
```

### 6. Launch dashboard

```bash
streamlit run app.py
```

---

## Dataset

**Norman et al. 2019 Perturb-seq** (Science 365:786–793)

| Property | Value |
|---|---|
| Cells (post-QC) | 111,391 |
| Highly variable genes | 2,000 |
| Perturbations | 237 (single + double CRISPR knockouts) |
| Control cells | ~11,849 |
| Cell line | K562 (chronic myelogenous leukemia) |
| Source | Zenodo `10.5281/zenodo.7041849` |

QC thresholds: min 200 genes/cell, min 3 cells/gene, max 20% mitochondrial reads. Normalization: total count normalization to 10,000 reads + log1p. HVG selection: Seurat v3 flavor.

---

## Model Architectures

### Effect MLP (best performer)
```
Input: control_expr (2000) + pert_id → Embedding(237, 64)
       → concat [512 + 64] → Linear(512) → ReLU → Dropout(0.3)
       → Linear(512) → ReLU → Dropout(0.3) → Linear(2000)
Loss: MSELoss  |  Optimizer: Adam (lr=1e-3)  |  Epochs: 30
```

### scGen-style VAE
```
Encoder: Linear(2000→512→256) → μ(128), logvar(128)
Reparam: z = μ + ε·exp(½logvar)
Decoder: concat(z:128, pert_emb:64) → Linear(192→256→512→2000)
Loss: ELBO = MSE + β·KL  |  β annealed 0→1e-4 over 10 epochs
```

### Graph GCN
```
Gene embeddings (2000 × 64) → GCN(64→128) → GCN(128→64)
Control expr → Linear(2000→512) → cat [ctrl_feat, pert_graph] → Linear(576→2000)
Adjacency: Normalized STRING PPI  (D^{-½}(A+I)D^{-½})
```

---

## Configuration

Edit `configs/default.yaml` to change paths, QC thresholds, model hyperparameters, and training settings. All training scripts read from this config.

---

## Tests

```bash
pytest tests/ -v
```

---

## Key Biological Findings

- **KLF1 perturbation** drives strong erythroid differentiation signature (r > 0.999)
- **BAK1** shows near-perfect apoptotic pathway activation (r = 0.9988)
- **IRF1** activates interferon response genes — potential immunotherapy target
- **Double knockouts** (e.g., BCL2L11_BAK1) exhibit synergistic effects captured by the VAE latent space

---

## Limitations

- Gene-level Pearson r < 0.12 — models cannot reliably rank individual gene magnitudes
- Double perturbation effects are underfit relative to single knockouts
- KL regularization in VAE compresses rare perturbation variation
- No compound-level data — results are gene-centric, not small-molecule-ready

---

## References

- Norman et al. (2019). *Exploring genetic interaction manifolds constructed from rich single-cell phenotypes.* Science 365:786–793.
- Lotfollahi et al. (2019). *scGen predicts single-cell perturbation responses.* Nature Methods.
- Roohani et al. (2023). *GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations.* Nature Biotechnology.
- Szklarczyk et al. (2023). *The STRING database in 2023.* Nucleic Acids Research.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
