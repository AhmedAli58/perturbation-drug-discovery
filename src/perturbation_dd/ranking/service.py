"""Candidate ranking pipeline for follow-up prioritization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from perturbation_dd.config import ProjectConfig
from perturbation_dd.data.access import (
    load_prepared_adata,
    load_prepared_manifest,
    resolve_project_path,
)
from perturbation_dd.evaluation.baselines import (
    choose_proxy_perturbation,
    load_string_neighbors,
)
from perturbation_dd.evaluation.reporting import evaluate_run
from perturbation_dd.models.inference import load_predictor
from perturbation_dd.training.runs import latest_run_for
from perturbation_dd.types import RankingRecord
from perturbation_dd.utils.io import read_json, write_json

RESPONSE_MODELS = ("effect_mlp", "graph_gcn", "scgen")


def rank_candidates(
    config: ProjectConfig,
    project_root: Path,
    *,
    input_path: Path,
    output_path: Path,
    split_name: str,
    model_family: str = "ensemble",
) -> list[RankingRecord]:
    candidates = load_candidates(input_path)
    if not candidates:
        raise ValueError(f"No candidates found in {input_path}.")

    prepared_manifest = load_prepared_manifest(project_root, config, split_name)
    adata = load_prepared_adata(prepared_manifest)
    perturbations = adata.obs[prepared_manifest.perturbation_key].astype(str).to_numpy()
    splits = adata.obs["split"].astype(str).to_numpy()
    train_mask = splits == "train"
    seen_train_perts = sorted(
        set(
            perturbations[train_mask].tolist()
        )
        - {prepared_manifest.control_label}
    )
    support_counts = {
        pert: int(((train_mask) & (perturbations == pert)).sum())
        for pert in seen_train_perts
    }
    mean_control = np.asarray(
        adata.X[(train_mask) & (perturbations == prepared_manifest.control_label)].mean(axis=0)
    ).reshape(-1)
    string_neighbors = load_string_neighbors(
        resolve_project_path(project_root, config.paths.string_network)
    )

    selected_models = _select_models(model_family)
    predictors = {}
    run_context = {}
    for model_name in selected_models:
        task = (
            "response_heldout"
            if split_name == "pert_split_v1" and model_name == "graph_gcn"
            else "response_known"
        )
        run_manifest = latest_run_for(
            project_root,
            config,
            split_name=split_name if task == "response_heldout" else "cell_split_v1",
            task=task,
            model=model_name,
        )
        if run_manifest is None and task == "response_known" and split_name != "cell_split_v1":
            run_manifest = latest_run_for(
                project_root,
                config,
                split_name="cell_split_v1",
                task="response_known",
                model=model_name,
            )
        if run_manifest is None:
            continue
        if not run_manifest.evaluation_path or not Path(run_manifest.evaluation_path).exists():
            evaluate_run(config, project_root, run_id=run_manifest.run_id)
            run_manifest = latest_run_for(
                project_root,
                config,
                split_name=run_manifest.split_name,
                task=run_manifest.task,
                model=run_manifest.model,
            ) or run_manifest
        predictors[model_name] = load_predictor(config, project_root, run_id=run_manifest.run_id)
        run_context[model_name] = {
            "run_manifest": run_manifest,
            "evaluation": (
                read_json(Path(run_manifest.evaluation_path))
                if run_manifest.evaluation_path
                else {}
            ),
        }

    if not predictors:
        raise FileNotFoundError(
            "No evaluated response-model runs available for ranking. Run train + evaluate first."
        )

    raw_records: list[dict] = []
    seen_dataset_perts = set(prepared_manifest.perturbation_counts)

    for candidate in candidates:
        notes: list[str] = []
        status = "seen" if candidate in seen_dataset_perts else "proxy"
        proxy_source: str | None = None
        model_predictions: list[np.ndarray] = []
        proxy = candidate

        if candidate not in seen_train_perts:
            proxy = choose_proxy_perturbation(
                candidate=candidate,
                seen_perturbations=seen_train_perts,
                string_neighbors=string_neighbors,
                support_counts=support_counts,
            )
            proxy_source = proxy
            notes.append(f"ranking uses proxy perturbation {proxy}")

        for model_name, predictor in predictors.items():
            evaluation = run_context[model_name]["evaluation"]
            comparison = evaluation.get("comparison", {})
            prefers_direct = comparison.get("preferred_ranking_mode") == "direct"
            target_name = candidate if candidate in seen_train_perts and prefers_direct else proxy
            prediction = predictor.predict(target_name)
            if not prediction.supported or prediction.vector is None:
                notes.append(f"{model_name} unavailable for {target_name}")
                continue
            model_predictions.append(prediction.vector)
            notes.extend(prediction.notes)

        if not model_predictions:
            raw_records.append(
                {
                    "candidate": candidate,
                    "status": "unsupported",
                    "proxy_source": proxy_source,
                    "notes": ["no compatible response-model predictions available"],
                }
            )
            continue

        if candidate not in seen_dataset_perts:
            status = "proxy"

        stacked = np.stack(model_predictions, axis=0)
        delta = stacked.mean(axis=0) - mean_control
        raw_records.append(
            {
                "candidate": candidate,
                "status": status,
                "proxy_source": proxy_source,
                "raw_effect_strength": float(np.linalg.norm(delta)),
                "raw_model_agreement": float(1.0 / (1.0 + np.var(stacked, axis=0).mean())),
                "raw_support_cells": float(np.log1p(support_counts.get(proxy, 0))),
                "notes": notes,
            }
        )

    _apply_scores(raw_records, config)
    ranking = [RankingRecord.model_validate(record) for record in raw_records]
    write_json(output_path, [record.model_dump() for record in ranking])
    return ranking


def load_candidates(input_path: Path) -> list[str]:
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("candidates", [])
        return [str(item).strip() for item in payload if str(item).strip()]
    if input_path.suffix.lower() == ".csv":
        with input_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                field = "candidate" if "candidate" in reader.fieldnames else reader.fieldnames[0]
                return [str(row[field]).strip() for row in reader if str(row[field]).strip()]
    return [line.strip() for line in input_path.read_text().splitlines() if line.strip()]


def _select_models(model_family: str) -> tuple[str, ...]:
    if model_family == "ensemble":
        return RESPONSE_MODELS
    if model_family not in RESPONSE_MODELS:
        raise ValueError(f"Unsupported ranking model family: {model_family}")
    return (model_family,)


def _apply_scores(records: list[dict], config: ProjectConfig) -> None:
    supported = [record for record in records if record.get("status") != "unsupported"]
    if not supported:
        return

    effect_values = [record["raw_effect_strength"] for record in supported]
    agreement_values = [record["raw_model_agreement"] for record in supported]
    support_values = [record["raw_support_cells"] for record in supported]

    effect_norm = _minmax(effect_values)
    agreement_norm = _minmax(agreement_values)
    support_norm = _minmax(support_values)

    for index, record in enumerate(supported):
        record["effect_strength"] = effect_norm[index]
        record["model_agreement"] = agreement_norm[index]
        record["support_cells"] = support_norm[index]
        record["confidence"] = round(
            config.ranking.confidence_agreement_weight * record["model_agreement"]
            + config.ranking.confidence_support_weight * record["support_cells"],
            6,
        )
        record["priority_score"] = round(
            config.ranking.effect_strength_weight * record["effect_strength"]
            + config.ranking.model_agreement_weight * record["model_agreement"]
            + config.ranking.support_cells_weight * record["support_cells"],
            6,
        )
        record.pop("raw_effect_strength", None)
        record.pop("raw_model_agreement", None)
        record.pop("raw_support_cells", None)

    for record in records:
        if record.get("status") == "unsupported":
            record.setdefault("effect_strength", None)
            record.setdefault("model_agreement", None)
            record.setdefault("support_cells", None)
            record.setdefault("confidence", None)
            record.setdefault("priority_score", None)


def _minmax(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [1.0]
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        return [1.0 for _ in values]
    return [round((value - low) / (high - low), 6) for value in values]
