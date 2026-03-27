# Runbook

This repository is designed to be operated through the `perturbation_dd` CLI.

## 1. Install

```bash
python3 -m pip install -e ".[dev]"
```

## 2. Prepare Reproducible Benchmark Splits

```bash
python3 -m perturbation_dd prepare-data --config configs/base.yaml --split cell_split_v1
python3 -m perturbation_dd prepare-data --config configs/base.yaml --split pert_split_v1
```

Outputs:

- `artifacts/prepared/<dataset>/<split>/prepared.h5ad`
- `artifacts/prepared/<dataset>/<split>/manifest.json`

## 3. Train Benchmarks

```bash
python3 -m perturbation_dd train --config configs/base.yaml --task classification --model logreg --split cell_split_v1
python3 -m perturbation_dd train --config configs/base.yaml --task classification --model mlp --split cell_split_v1
python3 -m perturbation_dd train --config configs/base.yaml --task response_known --model effect_mlp --split cell_split_v1
python3 -m perturbation_dd train --config configs/base.yaml --task response_known --model graph_gcn --split cell_split_v1
python3 -m perturbation_dd train --config configs/base.yaml --task response_known --model scgen --split cell_split_v1
python3 -m perturbation_dd train --config configs/base.yaml --task response_heldout --model graph_gcn --split pert_split_v1
```

Each training run writes:

- `artifacts/runs/<run-id>/run.json`
- `artifacts/runs/<run-id>/results/*.json`
- `artifacts/runs/<run-id>/models/*`
- `artifacts/runs/<run-id>/logs/stdout.log`
- `artifacts/runs/<run-id>/logs/stderr.log`

## 4. Evaluate Against Shared Baselines

```bash
python3 -m perturbation_dd evaluate --config configs/base.yaml --run-id <run-id>
python3 -m perturbation_dd build-report --config configs/base.yaml --run-id <run-id>
```

Evaluation creates:

- `artifacts/runs/<run-id>/evaluation.json`
- `artifacts/runs/<run-id>/benchmark_report.md`

## 5. Rank Candidate Perturbations

Input file:

```json
{
  "candidates": ["KLF1", "CEBPA", "GATA1"]
}
```

Command:

```bash
python3 -m perturbation_dd rank-candidates --config configs/base.yaml --input candidates.json --output ranked.json --split pert_split_v1
```

The output schema contains:

- `candidate`
- `status`
- `proxy_source`
- `effect_strength`
- `model_agreement`
- `support_cells`
- `confidence`
- `priority_score`
- `notes`

## 6. Serve the Ranking API

```bash
uvicorn perturbation_dd.serving.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /runs/{run_id}`
- `POST /rank`

Example request:

```json
{
  "candidates": ["KLF1", "CEBPA"],
  "split": "pert_split_v1",
  "model_family": "ensemble"
}
```

## 7. CI Expectations

CI validates:

- `ruff check src tests`
- `pytest tests/ -q`

The unit suite includes smoke coverage for:

- split registry determinism
- `prepare-data`
- training backend run manifest generation
- baseline evaluation
- ranking behavior
- API run retrieval
