"""Inference helpers for trained response models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from perturbation_dd.config import ProjectConfig
from perturbation_dd.data.access import resolve_project_path
from perturbation_dd.evaluation.baselines import load_string_neighbors
from perturbation_dd.training.runs import load_run_manifest

DEVICE = torch.device("cpu")


class EffectModel(nn.Module):
    def __init__(self, n_genes: int, n_perts: int, embed_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.pert_emb = nn.Embedding(n_perts, embed_dim)
        self.encoder = nn.Sequential(nn.Linear(n_genes, hidden), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden + embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, control_expr: torch.Tensor, pert_idx: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(control_expr)
        embedding = self.pert_emb(pert_idx)
        return self.decoder(torch.cat([hidden, embedding], dim=1))


class ManualGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.linear(adj @ x)


class GraphModel(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        adj_norm: torch.Tensor,
        gene_emb_dim: int,
        gcn_hidden: int,
        gcn_out: int,
        ctrl_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.register_buffer("adj_norm", adj_norm)
        self.gene_emb = nn.Embedding(n_genes, gene_emb_dim)
        self.gcn1 = ManualGCNLayer(gene_emb_dim, gcn_hidden)
        self.gcn2 = ManualGCNLayer(gcn_hidden, gcn_out)
        self.gcn_drop = nn.Dropout(dropout)
        self.ctrl_encoder = nn.Sequential(nn.Linear(n_genes, ctrl_hidden), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(ctrl_hidden + gcn_out, ctrl_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ctrl_hidden, n_genes),
        )

    def gene_features(self) -> torch.Tensor:
        features = self.gene_emb.weight
        features = F.relu(self.gcn1(features, self.adj_norm))
        features = self.gcn_drop(features)
        features = F.relu(self.gcn2(features, self.adj_norm))
        return features

    def predict_from_mask(self, control_expr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gene_features = self.gene_features()
        pert_graph = mask @ gene_features
        ctrl_features = self.ctrl_encoder(control_expr)
        return self.decoder(torch.cat([ctrl_features, pert_graph], dim=1))


class ScGenVAE(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        n_perts: int,
        latent_dim: int,
        pert_emb_dim: int,
        enc_h1: int,
        enc_h2: int,
        dec_h1: int,
        dec_h2: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, enc_h1),
            nn.ReLU(),
            nn.Linear(enc_h1, enc_h2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(enc_h2, latent_dim)
        self.fc_logvar = nn.Linear(enc_h2, latent_dim)
        self.pert_emb = nn.Embedding(n_perts, pert_emb_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + pert_emb_dim, dec_h1),
            nn.ReLU(),
            nn.Linear(dec_h1, dec_h2),
            nn.ReLU(),
            nn.Linear(dec_h2, n_genes),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def decode(self, z: torch.Tensor, pert_idx: torch.Tensor) -> torch.Tensor:
        embedding = self.pert_emb(pert_idx)
        return self.decoder(torch.cat([z, embedding], dim=1))


@dataclass
class PredictionResult:
    vector: np.ndarray | None
    supported: bool
    notes: list[str]


class ResponsePredictor:
    def predict(self, candidate: str) -> PredictionResult:
        raise NotImplementedError


class EffectPredictor(ResponsePredictor):
    def __init__(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        self.classes = checkpoint["classes"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.mean_control = (
            torch.from_numpy(np.asarray(checkpoint["mean_ctrl"], dtype=np.float32))
            .unsqueeze(0)
            .to(DEVICE)
        )
        self.model = EffectModel(
            n_genes=checkpoint["n_genes"],
            n_perts=checkpoint["n_classes"],
            embed_dim=checkpoint["embed_dim"],
            hidden=checkpoint["hidden1"],
            dropout=checkpoint["dropout"],
        ).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, candidate: str) -> PredictionResult:
        if candidate not in self.class_to_idx:
            return PredictionResult(
                vector=None,
                supported=False,
                notes=["candidate not encoded in effect model"],
            )
        with torch.no_grad():
            pred = self.model(
                self.mean_control,
                torch.tensor([self.class_to_idx[candidate]], dtype=torch.long, device=DEVICE),
            )
        return PredictionResult(vector=pred.squeeze(0).cpu().numpy(), supported=True, notes=[])


class ScGenPredictor(ResponsePredictor):
    def __init__(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        self.classes = checkpoint["classes"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.mean_control = (
            torch.from_numpy(np.asarray(checkpoint["mean_ctrl"], dtype=np.float32))
            .unsqueeze(0)
            .to(DEVICE)
        )
        self.model = ScGenVAE(
            n_genes=checkpoint["n_genes"],
            n_perts=checkpoint["n_classes"],
            latent_dim=checkpoint["latent_dim"],
            pert_emb_dim=checkpoint["pert_emb_dim"],
            enc_h1=checkpoint["enc_h1"],
            enc_h2=checkpoint["enc_h2"],
            dec_h1=checkpoint["dec_h1"],
            dec_h2=checkpoint["dec_h2"],
        ).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        with torch.no_grad():
            self.control_latent, _ = self.model.encode(self.mean_control)

    def predict(self, candidate: str) -> PredictionResult:
        if candidate not in self.class_to_idx:
            return PredictionResult(
                vector=None,
                supported=False,
                notes=["candidate not encoded in scGen model"],
            )
        with torch.no_grad():
            pred = self.model.decode(
                self.control_latent,
                torch.tensor([self.class_to_idx[candidate]], dtype=torch.long, device=DEVICE),
            )
        return PredictionResult(vector=pred.squeeze(0).cpu().numpy(), supported=True, notes=[])


class GraphPredictor(ResponsePredictor):
    def __init__(self, checkpoint_path: Path, *, prepared_dataset_path: Path, string_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        import anndata as ad

        adata = ad.read_h5ad(prepared_dataset_path)
        self.gene_names = list(adata.var_names)
        self.gene_to_idx = {name: idx for idx, name in enumerate(self.gene_names)}
        self.string_neighbors = load_string_neighbors(string_path)
        self.mean_control = (
            torch.from_numpy(np.asarray(checkpoint["mean_ctrl"], dtype=np.float32))
            .unsqueeze(0)
            .to(DEVICE)
        )
        adjacency = _build_adjacency(self.gene_to_idx, string_path)
        self.model = GraphModel(
            n_genes=checkpoint["n_genes"],
            adj_norm=adjacency,
            gene_emb_dim=checkpoint["gene_emb_dim"],
            gcn_hidden=checkpoint["gcn_hidden"],
            gcn_out=checkpoint["gcn_out"],
            ctrl_hidden=checkpoint["ctrl_hidden"],
            dropout=checkpoint["dropout"],
        ).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, candidate: str) -> PredictionResult:
        mask = _candidate_mask(candidate, self.gene_to_idx, self.string_neighbors)
        if mask is None:
            return PredictionResult(
                vector=None,
                supported=False,
                notes=["candidate not connected to supported graph genes"],
            )
        with torch.no_grad():
            pred = self.model.predict_from_mask(self.mean_control, mask.to(DEVICE))
        return PredictionResult(vector=pred.squeeze(0).cpu().numpy(), supported=True, notes=[])


def load_predictor(
    config: ProjectConfig,
    project_root: Path,
    *,
    run_id: str,
) -> ResponsePredictor:
    run_manifest = load_run_manifest(project_root, config, run_id)
    checkpoint_path = Path(run_manifest.model_artifact_path)
    if run_manifest.model == "effect_mlp":
        return EffectPredictor(checkpoint_path)
    if run_manifest.model == "graph_gcn":
        return GraphPredictor(
            checkpoint_path,
            prepared_dataset_path=Path(run_manifest.prepared_dataset_path),
            string_path=resolve_project_path(project_root, config.paths.string_network),
        )
    if run_manifest.model == "scgen":
        return ScGenPredictor(checkpoint_path)
    raise ValueError(f"Unsupported predictor model: {run_manifest.model}")


def _build_adjacency(gene_to_idx: dict[str, int], string_path: Path) -> torch.Tensor:
    n_genes = len(gene_to_idx)
    adjacency = torch.zeros(n_genes, n_genes)
    if string_path.exists():
        with string_path.open() as handle:
            next(handle, None)
            for line in handle:
                left, right, weight = line.strip().split("\t")
                if left in gene_to_idx and right in gene_to_idx:
                    left_idx = gene_to_idx[left]
                    right_idx = gene_to_idx[right]
                    score = float(weight)
                    adjacency[left_idx, right_idx] = score
                    adjacency[right_idx, left_idx] = score
    adjacency = adjacency + torch.eye(n_genes)
    degree = adjacency.sum(1)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[degree == 0] = 0.0
    diagonal = torch.diag(degree_inv_sqrt)
    return diagonal @ adjacency @ diagonal


def _candidate_mask(
    candidate: str,
    gene_to_idx: dict[str, int],
    string_neighbors: dict[str, set[str]],
) -> torch.Tensor | None:
    genes = set(candidate.split("_"))
    indices: set[int] = set()
    for gene in genes:
        if gene in gene_to_idx:
            indices.add(gene_to_idx[gene])
        for neighbor in string_neighbors.get(gene, set()):
            if neighbor in gene_to_idx:
                indices.add(gene_to_idx[neighbor])
    if not indices:
        return None
    mask = torch.zeros(1, len(gene_to_idx), dtype=torch.float32)
    index_list = sorted(indices)
    mask[0, index_list] = 1.0 / len(index_list)
    return mask
