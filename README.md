# Perturbation-Based Drug Target Discovery

Deep learning pipeline for predicting how CRISPR gene knockouts alter cell-wide gene expression, built on the Norman 2019 Perturb-seq dataset (111k cells, 237 perturbations).

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

![Predicted vs actual mean gene expression across six perturbations](reports/figures/pred_vs_real.png)

*Predicted vs actual mean gene expression for six held-out perturbations. Each point is one of 2,000 genes.*

---

## The Problem

Wet-lab CRISPR screens are expensive and slow. Testing every candidate gene as a drug target is not feasible at scale. This project trains models to predict post-perturbation gene expression from a cell's baseline state, so you can screen thousands of knockouts computationally before running a single experiment.

---

## Results

| Model | Task | Metric | Score |
|---|---|---|---|
| **Naive baseline** (predict control mean) | Expression prediction | Per-pert Pearson r | 0.9829 |
| Effect MLP | Expression prediction | Per-pert Pearson r | **0.9957** |
| Graph GCN | Expression prediction | Per-pert Pearson r | 0.9903 |
| scGen VAE | Expression prediction | Per-pert Pearson r | 0.9798 |
| scGen VAE — zero-shot | Unseen perturbations (44) | Per-pert Pearson r | **0.9843** |
| Logistic Regression | Perturbation classification | Top-1 accuracy | 37.4% |
| MLP Classifier | Perturbation classification | Top-1 accuracy | 45.9% |

The naive baseline (predicting no change from control) already scores r=0.9829 because most genes are unaffected by any single perturbation. The Effect MLP closes the remaining gap to 0.9957 — capturing the specific changes that actually matter. The VAE generalises to 44 perturbations it never saw during training (r=0.9843), which is the core property needed for computational screening.

**Known limitation:** gene-level Pearson r stays below 0.12 across all models. The models predict the *direction* of perturbation effects well but cannot reliably rank individual gene magnitudes. Drug target ranking from these outputs requires caution.

---

## How It Works

Five models in increasing complexity:

1. **Logistic Regression** — classifies which perturbation a cell received from its expression profile. Establishes a floor.
2. **MLP Classifier** — same task, neural network. Confirms baseline is beatable.
3. **Effect MLP** — given a control cell and a perturbation ID, predicts the full post-perturbation expression profile. Best overall performance.
4. **Graph GCN** — same as Effect MLP but uses STRING protein interaction network as a structural prior over the 2,000 genes. Slightly lower than MLP, which suggests the PPI prior doesn't add much at this dataset scale.
5. **scGen-style VAE** — encodes cells into a perturbation-aware latent space. Weaker on seen perturbations but the only model that generalises zero-shot to new ones via latent arithmetic.

---

## Quickstart

```bash
# 1. Install
make install

# 2. Download data (~2 GB)
make data

# 3. Preprocess
make preprocess

# 4. Train all models
make train

# 5. Run tests
make test
```

---

## Dataset

**Norman et al. 2019 Perturb-seq** — [Science 365:786–793](https://doi.org/10.1126/science.aax4438)

| | |
|---|---|
| Cells (post-QC) | 111,391 |
| Genes (HVG) | 2,000 |
| Perturbations | 237 (single + combinatorial CRISPR knockouts) |
| Cell line | K562 (chronic myelogenous leukemia) |
| Source | Zenodo [`10.5281/zenodo.7041849`](https://zenodo.org/record/7041849) |

---

## Limitations

- Gene-level Pearson r < 0.12 — models capture bulk effect direction but not individual gene magnitudes
- Double perturbations are underfit — too few cells per combination for reliable training
- No compound-level data — findings are gene-centric; translating to small molecules requires additional steps

---

## License

MIT — see [LICENSE](LICENSE).
