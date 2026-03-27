"""FastAPI wrapper around the ranking and run-report surfaces."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from perturbation_dd.config import load_project_config
from perturbation_dd.ranking.service import rank_candidates
from perturbation_dd.training.runs import load_run_manifest


class RankRequest(BaseModel):
    candidates: list[str] = Field(default_factory=list)
    split: str = "pert_split_v1"
    model_family: str = "ensemble"


def create_app(
    *,
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> FastAPI:
    resolved_config = config_path or Path(os.environ.get("PDD_CONFIG_PATH", "configs/base.yaml"))
    resolved_root = project_root or (
        resolved_config.resolve().parent.parent
        if resolved_config.resolve().parent.name == "configs"
        else resolved_config.resolve().parent
    )
    config = load_project_config(resolved_config)

    app = FastAPI(title="Perturbation DD API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        try:
            return load_run_manifest(resolved_root, config, run_id).model_dump()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/rank")
    def rank(payload: RankRequest) -> list[dict]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "candidates.json"
            output_path = Path(tmp_dir) / "ranked.json"
            input_path.write_text(payload.model_dump_json(indent=2))
            try:
                ranking = rank_candidates(
                    config,
                    resolved_root,
                    input_path=input_path,
                    output_path=output_path,
                    split_name=payload.split,
                    model_family=payload.model_family,
                )
            except Exception as exc:  # pragma: no cover - surfaced as API error
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return [record.model_dump() for record in ranking]

    return app


app = create_app()
