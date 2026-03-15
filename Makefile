.PHONY: install data preprocess train test

install:
	pip install -r requirements.txt

data:
	python src/data/download_norman2019.py
	python src/data/download_string_ppi.py

preprocess:
	python src/data/prepare_perturbation_dataset.py

train:
	python src/models/train_baseline_classifier.py
	python src/models/train_mlp_classifier.py
	python src/models/train_perturbation_effect_model.py
	python src/models/train_graph_perturbation_model.py
	python src/models/train_scgen_style_model.py
	python src/experiments/unseen_perturbation_generalization.py

test:
	pytest tests/ -v
