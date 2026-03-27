.PHONY: install lint test prepare-cell prepare-heldout train-logreg train-mlp train-effect train-graph train-scgen evaluate-graph report-graph serve

PYTHON ?= python3
PDD ?= $(PYTHON) -m perturbation_dd
CONFIG ?= configs/base.yaml

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check src tests

test:
	pytest tests/ -q

prepare-cell:
	$(PDD) prepare-data --config $(CONFIG) --split cell_split_v1

prepare-heldout:
	$(PDD) prepare-data --config $(CONFIG) --split pert_split_v1

train-logreg:
	$(PDD) train --config $(CONFIG) --task classification --model logreg --split cell_split_v1

train-mlp:
	$(PDD) train --config $(CONFIG) --task classification --model mlp --split cell_split_v1

train-effect:
	$(PDD) train --config $(CONFIG) --task response_known --model effect_mlp --split cell_split_v1

train-graph:
	$(PDD) train --config $(CONFIG) --task response_known --model graph_gcn --split cell_split_v1

train-scgen:
	$(PDD) train --config $(CONFIG) --task response_known --model scgen --split cell_split_v1

evaluate-graph:
	@echo "Pass RUN_ID=<run-id> to evaluate a run."
	$(PDD) evaluate --config $(CONFIG) --run-id $(RUN_ID)

report-graph:
	@echo "Pass RUN_ID=<run-id> to build a report."
	$(PDD) build-report --config $(CONFIG) --run-id $(RUN_ID)

serve:
	uvicorn perturbation_dd.serving.api:app --host 0.0.0.0 --port 8000
