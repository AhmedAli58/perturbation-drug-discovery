# Benchmark Specification

## Dataset

- Source dataset: Norman et al. 2019 Perturb-seq
- Dataset scope: single public dataset only for v1
- Entity of interest: perturbation labels and their mean expression responses

## Tasks

### 1. Classification

Input:
- single-cell expression vector

Output:
- perturbation label

Primary metrics:
- top-1 accuracy
- top-5 accuracy
- mean per-class accuracy

### 2. Response Known

Input:
- control template derived from train controls
- perturbation identity present in training support

Output:
- mean predicted response vector

Primary metrics:
- mean per-perturbation Pearson r
- mean gene-level Pearson r
- mean MSE

Required baselines:
- naive control baseline
- nearest-seen perturbation baseline

### 3. Response Held-Out

Input:
- control template derived from train controls
- perturbation identity excluded before training

Output:
- mean predicted response vector or proxy prediction

Primary metrics:
- mean per-perturbation Pearson r
- mean gene-level Pearson r
- mean MSE

Required baselines:
- naive control baseline
- nearest-seen perturbation baseline

## Split Definitions

### `cell_split_v1`

- same perturbations can appear in train, val, and test
- used for `classification` and `response_known`
- train, val, test are disjoint at the cell level

### `pert_split_v1`

- held-out perturbations are excluded from training
- used for `response_heldout`
- held-out perturbations must not appear in training artifacts or train labels

## Preparation Rules

- QC is applied globally after raw load
- HVGs are selected using train cells only
- the selected train-derived HVG list is applied to all splits
- train controls define the default control template for training and evaluation
- each prepared artifact writes a manifest containing hashes, split metadata, HVG list, and perturbation inclusion lists

## Ranking Rules

Each ranked perturbation record must include:

- `candidate`
- `status`
- `proxy_source`
- `effect_strength`
- `model_agreement`
- `support_cells`
- `confidence`
- `priority_score`
- `notes`

Default score components:

- `effect_strength`: normalized delta norm from train control
- `model_agreement`: inverse normalized variance across response models
- `support_cells`: normalized log training support
- `priority_score = 0.50 * effect_strength + 0.30 * model_agreement + 0.20 * support_cells`
- `confidence = 0.60 * model_agreement + 0.40 * support_cells`

## Acceptance Criteria

- splits are deterministic for fixed seed
- held-out perturbations are absent from training support in `pert_split_v1`
- report generation does not hardcode benchmark values
- unsupported candidates are surfaced as unsupported rather than fabricated
- proxy candidates explicitly state which source perturbation was used
