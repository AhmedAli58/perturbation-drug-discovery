# Perturbation-Based Drug Target Discovery

Deep learning pipeline for predicting how CRISPR gene knockouts reshape cell-wide gene expression. Built on the Norman 2019 Perturb-seq dataset — 111k single cells, 237 perturbations, K562 leukemia cell line.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Wet-lab CRISPR screens are expensive and slow — testing every candidate gene as a drug target is not feasible at scale. If you could accurately simulate what a perturbation does to a cell's expression profile, you could screen thousands of knockouts computationally before committing to a single experiment.

This project trains five models of increasing complexity to do exactly that, benchmarks them against each other and a naive baseline, and tests whether the best model can generalise to perturbations it never saw during training.

---

## Results

### Expression prediction (core task)

| Model | Per-pert Pearson r | Gene-level Pearson r | Test MSE |
|---|---|---|---|
| **Naive baseline** (predict control mean) | 0.9829 | — | — |
| Effect MLP | **0.9957** | 0.118 | 0.077 |
| Graph GCN | 0.9903 | 0.087 | 0.079 |
| scGen VAE | 0.9798 | 0.031 | 0.083 |

### Perturbation classification (secondary task)

| Model | Top-1 accuracy | Top-5 accuracy |
|---|---|---|
| Logistic Regression | 37.4% | 64.7% |
| MLP Classifier | 45.9% | 70.7% |
| Random chance | 0.4% | 2.1% |

### Unseen perturbation generalisation (zero-shot)

The scGen VAE is the only model that can generalise to perturbations it never trained on, by doing latent-space arithmetic with a nearest-seen perturbation's embedding.

| Condition | Per-pert Pearson r |
|---|---|
| VAE zero-shot (44 unseen perturbations) | 0.9843 |
| VAE oracle (true embedding, upper bound) | 0.9846 |
| Nearest-seen Δ expression baseline | 0.9831 |

**Reading the numbers:** The naive baseline (always predict control mean) already scores r=0.9829 because the vast majority of the 2,000 genes are unaffected by any single perturbation. What matters is the gap: the Effect MLP closes it to 0.9957, capturing the specific differential expression that distinguishes each knockout. The gene-level r (< 0.12 for all models) reflects a harder challenge — ranking individual genes by perturbation magnitude — and is a known limitation of the per-perturbation mean-prediction paradigm used here and in scGen/CPA/GEARS.

---

## Figures

### Predicted vs observed gene expression (scGen VAE)

![Six-panel scatter: predicted vs observed mean expression per perturbation](reports/figures/pred_vs_real.png)

*Each panel is one held-out perturbation. Each point is one of 2,000 genes. Panels are ordered from worst to best Pearson r to show the full range of model behaviour.*

### Model comparison across both tasks

![Bar chart comparing all models](reports/figures/model_comparison.png)

*Left: perturbation classification (top-1 and top-5 accuracy). Right: expression prediction (per-perturbation and gene-level Pearson r). The gene-level bars reveal how much harder it is to rank individual genes than to predict mean perturbation direction.*

---

## How It Works

Five models in increasing architectural complexity, each addressing a different angle of the drug discovery problem:

**1. Logistic Regression** (`train_baseline_classifier.py`)
Maps each cell's 2,000-gene expression profile to a perturbation label. Establishes a linear floor — anything below this isn't useful. Achieves 37.4% top-1 accuracy on 237 classes (random = 0.4%).

**2. MLP Classifier** (`train_mlp_classifier.py`)
Three-layer feedforward network on the same classification task. Demonstrates that a neural model can meaningfully improve on the linear baseline (+8.5 pp top-1). This is the diagnostic model: if MLP can't beat logistic regression, the expression signal isn't strong enough for deeper modelling.

**3. Effect MLP** (`train_perturbation_effect_model.py`)
Shifts to the generative task: given a control cell and a perturbation ID, predict the full post-perturbation expression profile. Architecture: a cell encoder (2000→512) concatenated with a perturbation embedding (64-d), decoded through a 512→2000 layer. Trained on random control–perturbed pairs; evaluated using the mean control cell as a fixed reference. Best overall performance (r=0.9957).

**4. Graph GCN** (`train_graph_perturbation_model.py`)
Augments the Effect MLP with a structural prior: a two-layer graph convolution over the 2,000-gene interaction network from STRING, using only high-confidence edges (score ≥ 700). The perturbation representation is the mean GCN embedding of the target gene's STRING neighbours. Slightly below the plain MLP (r=0.9903), suggesting that at this dataset scale the PPI prior doesn't add signal beyond what the data already contains.

**5. scGen-style VAE** (`train_scgen_style_model.py`)
Learns a disentangled latent space where perturbation effects are additive shifts. Encoder maps a cell to μ, σ; decoder conditions on z concatenated with a perturbation embedding. KL weight is annealed from 0 → 1×10⁻⁴ over the first 10 epochs to prevent posterior collapse. Weaker on seen perturbations (r=0.9798) but the only model that generalises zero-shot to new ones via nearest-seen embedding transfer — which is the property that makes it useful for computational screening.

---

## Dataset

**Norman et al. 2019 Perturb-seq** — [Science 365:786–793](https://doi.org/10.1126/science.aax4438)

| | |
|---|---|
| Cells (post-QC) | 111,391 |
| Genes (HVG) | 2,000 (Seurat v3 selection) |
| Perturbations | 237 (single + combinatorial CRISPR knockouts) |
| Cell line | K562 (chronic myelogenous leukaemia) |
| Split | 80 / 10 / 10 stratified by perturbation group |
| Source | Zenodo [`10.5281/zenodo.7041849`](https://zenodo.org/record/7041849) |

Preprocessing: library-size normalisation → log1p → 2,000 HVG selection. Raw counts are not retained after normalisation.

---

## Project Structure

```
perturbation-drug-discovery/
├── configs/
│   └── default.yaml              # all hyperparameters and data constants
├── src/
│   ├── constants.py              # single source of truth (reads default.yaml)
│   ├── data/
│   │   ├── download_norman2019.py
│   │   ├── download_string_ppi.py
│   │   ├── prepare_perturbation_dataset.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── train_baseline_classifier.py
│   │   ├── train_mlp_classifier.py
│   │   ├── train_perturbation_effect_model.py
│   │   ├── train_graph_perturbation_model.py
│   │   └── train_scgen_style_model.py
│   ├── experiments/
│   │   └── unseen_perturbation_generalization.py
│   └── analysis/
│       ├── visualize_results.py
│       └── interpret_perturbation_results.py
├── data/
│   ├── raw/                      # downloaded, gitignored
│   ├── processed/                # h5ad, gitignored
│   ├── models/                   # .pt checkpoints, gitignored
│   └── results/                  # metrics JSON, gitignored
├── reports/figures/              # publication-quality PNGs, committed
└── Makefile
```

---

## Quickstart

```bash
# 1. Create environment and install dependencies
make install

# 2. Download dataset (~2 GB) and STRING PPI network
make data

# 3. Preprocess: QC filtering, HVG selection, train/val/test split
make preprocess

# 4. Train all five models sequentially
make train

# 5. Run tests
make test
```

Individual model training:
```bash
source .venv/bin/activate
python src/models/train_perturbation_effect_model.py   # Effect MLP
python src/models/train_scgen_style_model.py           # VAE
python src/experiments/unseen_perturbation_generalization.py
python src/analysis/visualize_results.py               # regenerate figures
```

---

## Known Limitations

- **Gene-level Pearson r < 0.12** across all models — bulk perturbation direction is captured well, but ranking individual genes by response magnitude is not reliable. Drug target ranking from these outputs requires an additional differential expression step.
- **Double perturbations are underfit** — combinatorial knockouts have fewer cells per condition, producing noisier supervision.
- **Cell line specificity** — all results are on K562. Generalisation to primary cells or other lines is untested.
- **No compound-level data** — the pipeline is gene-centric; mapping from gene targets to small molecules requires external databases (e.g. ChEMBL, DGIdb).

---

## License

MIT — see [LICENSE](LICENSE).
