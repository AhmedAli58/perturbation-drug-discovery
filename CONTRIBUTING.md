# Contributing

Contributions are welcome. Please follow the steps below.

## Setup

```bash
git clone https://github.com/AhmedAli58/perturbation-drug-discovery.git
cd perturbation-drug-discovery
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Conventions

- One script per model in `src/models/`
- All hyperparameters at the top of each script as module-level constants
- Results saved as JSON to `data/results/`
- Fixed seed `42` everywhere for reproducibility
- Commit style: `feat:`, `fix:`, `docs:`, `chore:`

## Reporting Issues

Open an issue describing:
1. What you expected to happen
2. What actually happened
3. Steps to reproduce
