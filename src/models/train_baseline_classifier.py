"""
Baseline perturbation classifier: Logistic Regression on HVG expression.

Goal: predict which CRISPR perturbation produced a cell's gene expression
profile. Establishes a reproducible baseline before deep learning models.

Inputs : data/processed/norman2019_processed.h5ad
Outputs: data/results/baseline_metrics.json
         data/models/baseline_logreg.pkl
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import SEED  # noqa: E402
from src.runtime_paths import env_path  # noqa: E402

PROCESSED_PATH = env_path(
    "PDD_PROCESSED_PATH",
    PROJECT_ROOT / "data" / "processed" / "norman2019_processed.h5ad",
)
RESULTS_DIR    = env_path("PDD_RESULTS_DIR", PROJECT_ROOT / "data" / "results")
MODELS_DIR     = env_path("PDD_MODELS_DIR", PROJECT_ROOT / "data" / "models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
MODEL_PATH   = MODELS_DIR  / "baseline_logreg.pkl"

# ---------------------------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------------------------
if not PROCESSED_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found at {PROCESSED_PATH}. "
        "Run: make preprocess"
    )
logger.info("Loading %s …", PROCESSED_PATH)
adata = sc.read_h5ad(PROCESSED_PATH)
logger.info("AnnData: %d cells × %d genes", adata.n_obs, adata.n_vars)

# ---------------------------------------------------------------------------
# 3. Extract inputs
# ---------------------------------------------------------------------------
import scipy.sparse as sp  # noqa: E402

# Convert sparse → dense float32 array (2000 HVGs × ~111k cells is ~840 MB)
X = adata.X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)

# Use stored perturbation encoding if available, otherwise fit a fresh one
if "perturbation_encoding" in adata.uns:
    pert_to_idx: dict[str, int] = adata.uns["perturbation_encoding"]["pert_to_idx"]
    classes = sorted(pert_to_idx, key=pert_to_idx.get)
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    y = np.array([pert_to_idx[p] for p in adata.obs["perturbation"].astype(str)])
    logger.info("Using stored perturbation encoding (%d classes)", len(classes))
else:
    le = LabelEncoder()
    y = le.fit_transform(adata.obs["perturbation"].astype(str))
    classes = list(le.classes_)
    logger.info("Fitted fresh LabelEncoder (%d classes)", len(classes))

n_classes = len(classes)

# ---------------------------------------------------------------------------
# 4. Train / test split (from stored obs["split"])
# ---------------------------------------------------------------------------
train_mask = (adata.obs["split"] == "train").values
test_mask  = (adata.obs["split"] == "test").values

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

n_train = int(train_mask.sum())
n_test  = int(test_mask.sum())
logger.info("Split — train: %d  test: %d", n_train, n_test)

# ---------------------------------------------------------------------------
# 5 & 6. Train baseline model
# ---------------------------------------------------------------------------
logger.info("Training LogisticRegression (multinomial, max_iter=200) …")
clf = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=SEED,
    verbose=1,
)
clf.fit(X_train, y_train)
logger.info("Training complete.")

# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------
logger.info("Evaluating on test set …")

# Top-1 accuracy
y_pred = clf.predict(X_test)
accuracy = float((y_pred == y_test).mean())

# Top-5 accuracy — predict_proba gives scores for all classes
y_proba = clf.predict_proba(X_test)          # (n_test, n_classes)
top5_indices = np.argsort(y_proba, axis=1)[:, -5:]   # top-5 class indices
top5_accuracy = float(
    np.array([y_test[i] in top5_indices[i] for i in range(len(y_test))]).mean()
)

# Per-perturbation accuracy (useful diagnostic)
per_pert_acc: dict[str, float] = {}
for idx, pert in enumerate(classes):
    mask = y_test == idx
    if mask.sum() > 0:
        per_pert_acc[pert] = float((y_pred[mask] == idx).mean())

mean_per_pert_acc = float(np.mean(list(per_pert_acc.values())))

print("\n── Baseline Classifier Results ───────────────────────────")
print(f"  Training samples      : {n_train:,}")
print(f"  Test samples          : {n_test:,}")
print(f"  Number of perturbations: {n_classes:,}")
print(f"  Top-1 accuracy        : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Top-5 accuracy        : {top5_accuracy:.4f}  ({top5_accuracy*100:.2f}%)")
print(f"  Mean per-class acc    : {mean_per_pert_acc:.4f}  ({mean_per_pert_acc*100:.2f}%)")
print("──────────────────────────────────────────────────────────\n")

# Bottom / top 5 perturbations by per-class accuracy
sorted_perts = sorted(per_pert_acc.items(), key=lambda x: x[1])
print("  Hardest perturbations (lowest accuracy):")
for p, a in sorted_perts[:5]:
    print(f"    {p:<30s}  {a*100:.1f}%")
print("  Easiest perturbations (highest accuracy):")
for p, a in sorted_perts[-5:]:
    print(f"    {p:<30s}  {a*100:.1f}%")
print()

# ---------------------------------------------------------------------------
# 8. Save metrics
# ---------------------------------------------------------------------------
metrics = {
    "model": "logistic_regression",
    "solver": "lbfgs",
    "max_iter": 1000,
    "n_perturbations": n_classes,
    "train_cells": n_train,
    "test_cells": n_test,
    "accuracy": round(accuracy, 6),
    "top5_accuracy": round(top5_accuracy, 6),
    "mean_per_class_accuracy": round(mean_per_pert_acc, 6),
    "per_perturbation_accuracy": {k: round(v, 6) for k, v in per_pert_acc.items()},
}
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
logger.info("Metrics saved to %s", METRICS_PATH)

# ---------------------------------------------------------------------------
# 9. Save trained model
# ---------------------------------------------------------------------------
payload = {"model": clf, "label_encoder_classes": classes}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(payload, f, protocol=5)
logger.info("Model saved to %s", MODEL_PATH)

# ---------------------------------------------------------------------------
# 10. Confirmation
# ---------------------------------------------------------------------------
print("Baseline model training complete. Results saved to data/results/baseline_metrics.json")
