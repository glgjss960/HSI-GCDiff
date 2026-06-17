import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from .graph_builder import GraphBundle, GraphView
from .utils import ensure_dir, save_json, to_torch_sparse


class SparseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj, self.linear(x))


class ViewGraphAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.conv1 = SparseGraphConv(in_dim, hidden_dim)
        self.conv2 = SparseGraphConv(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = F.normalize(self.conv2(h, adj), p=2, dim=1)
        recon = self.decoder(z)
        return z, recon


def _fuse_mean(z_views: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(z_views) == 1:
        return z_views[0]
    return F.normalize(torch.stack(list(z_views), dim=0).mean(dim=0), p=2, dim=1)


class TaskAuxGraphEncoder(nn.Module):
    def __init__(
        self,
        task_input_dims: List[int],
        aux_input_dims: List[int],
        task_hidden_dim: int,
        aux_hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.task_encoders = nn.ModuleList(
            [ViewGraphAutoEncoder(in_dim, task_hidden_dim, latent_dim, dropout) for in_dim in task_input_dims]
        )
        self.aux_encoders = nn.ModuleList(
            [ViewGraphAutoEncoder(in_dim, aux_hidden_dim, latent_dim, dropout) for in_dim in aux_input_dims]
        )

    def forward(
        self,
        task_features: List[torch.Tensor],
        task_adjs: List[torch.Tensor],
        aux_features: List[torch.Tensor],
        aux_adjs: List[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
        task_z, task_recon = [], []
        for encoder, x, adj in zip(self.task_encoders, task_features, task_adjs):
            z, recon = encoder(x, adj)
            task_z.append(z)
            task_recon.append(recon)

        aux_z, aux_recon = [], []
        for encoder, x, adj in zip(self.aux_encoders, aux_features, aux_adjs):
            z, recon = encoder(x, adj)
            aux_z.append(z)
            aux_recon.append(recon)

        return {
            "task_z_views": task_z,
            "aux_z_views": aux_z,
            "task_recons": task_recon,
            "aux_recons": aux_recon,
            "z_task": _fuse_mean(task_z),
            "z_aux": _fuse_mean(aux_z),
        }


@dataclass
class EncoderArtifacts:
    model_path: str
    embedding_path: str
    meta_path: str
    param_count: int


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _make_tensors(views: List[GraphView], device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features = [torch.from_numpy(view.features.astype(np.float32)).to(device) for view in views]
    adjs = [to_torch_sparse(view.adjacency, device=device) for view in views]
    return features, adjs


def _adjacency_recon_loss(z: torch.Tensor, adj: torch.Tensor, max_edges: int) -> torch.Tensor:
    coo = adj.coalesce()
    indices = coo.indices()
    src, dst = indices[0], indices[1]
    mask = src != dst
    src, dst = src[mask], dst[mask]
    if src.numel() == 0:
        return z.new_tensor(0.0)
    if src.numel() > max_edges:
        pick = torch.randperm(src.numel(), device=z.device)[:max_edges]
        src, dst = src[pick], dst[pick]

    n = z.shape[0]
    neg_src = torch.randint(0, n, (src.numel(),), device=z.device)
    neg_dst = torch.randint(0, n, (src.numel(),), device=z.device)
    pos_logits = (z[src] * z[dst]).sum(dim=1)
    neg_logits = (z[neg_src] * z[neg_dst]).sum(dim=1)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def train_encoder(graph: GraphBundle, cfg: Dict, output_dir: str, device: torch.device) -> EncoderArtifacts:
    ensure_dir(output_dir)
    enc_cfg = cfg.get("encoder", {})
    hidden_dim = int(enc_cfg.get("hidden_dim", 256))
    task_hidden_dim = int(enc_cfg.get("task_hidden_dim", hidden_dim))
    aux_hidden_dim = int(enc_cfg.get("aux_hidden_dim", hidden_dim))
    latent_dim = int(enc_cfg.get("latent_dim", 128))
    dropout = float(enc_cfg.get("dropout", 0.1))
    epochs = int(enc_cfg.get("epochs", 200))
    lr = float(enc_cfg.get("lr", 1e-3))
    weight_decay = float(enc_cfg.get("weight_decay", 1e-4))
    aux_align_weight = float(enc_cfg.get("aux_align_weight", 0.02))
    adj_recon_weight = float(enc_cfg.get("adj_recon_weight", 0.05))
    max_adj_edges = int(enc_cfg.get("max_adj_edges", 20000))
    log_every = int(enc_cfg.get("log_every", 25))

    task_views = graph.task_views
    aux_views = graph.aux_views
    if not task_views or not aux_views:
        raise ValueError("Both task and aux graph views are required.")

    model = TaskAuxGraphEncoder(
        task_input_dims=[view.features.shape[1] for view in task_views],
        aux_input_dims=[view.features.shape[1] for view in aux_views],
        task_hidden_dim=task_hidden_dim,
        aux_hidden_dim=aux_hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(device)
    task_features, task_adjs = _make_tensors(task_views, device)
    aux_features, aux_adjs = _make_tensors(aux_views, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    iterator = trange(1, epochs + 1, desc="encoders", leave=False) if epochs > 0 else []
    for epoch in iterator:
        model.train()
        out = model(task_features, task_adjs, aux_features, aux_adjs)
        task_recon = sum(F.mse_loss(recon, x) for recon, x in zip(out["task_recons"], task_features)) / len(task_features)
        aux_recon = sum(F.mse_loss(recon, x) for recon, x in zip(out["aux_recons"], aux_features)) / len(aux_features)
        adj_loss = z_align = out["z_task"].new_tensor(0.0)
        if adj_recon_weight > 0:
            adj_terms = []
            for z, adj in zip(out["task_z_views"], task_adjs):
                adj_terms.append(_adjacency_recon_loss(z, adj, max_adj_edges))
            for z, adj in zip(out["aux_z_views"], aux_adjs):
                adj_terms.append(_adjacency_recon_loss(z, adj, max_adj_edges))
            adj_loss = sum(adj_terms) / max(len(adj_terms), 1)
        if aux_align_weight > 0:
            z_align = 1.0 - (out["z_aux"] * out["z_task"].detach()).sum(dim=1).mean()
        loss = task_recon + aux_recon + adj_recon_weight * adj_loss + aux_align_weight * z_align
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "task_recon": float(task_recon.detach().cpu()),
                    "aux_recon": float(aux_recon.detach().cpu()),
                    "adj_loss": float(adj_loss.detach().cpu()),
                    "aux_align": float(z_align.detach().cpu()),
                }
            )

    model.eval()
    with torch.no_grad():
        out = model(task_features, task_adjs, aux_features, aux_adjs)
    embeddings = {
        "z_task": out["z_task"].detach().cpu().numpy().astype(np.float32),
        "z_aux": out["z_aux"].detach().cpu().numpy().astype(np.float32),
        "z_source": out["z_task"].detach().cpu().numpy().astype(np.float32),
        "z_fuse": out["z_task"].detach().cpu().numpy().astype(np.float32),
    }
    for idx, z in enumerate(out["task_z_views"]):
        embeddings[f"z_task_view_{idx}"] = z.detach().cpu().numpy().astype(np.float32)
    for idx, z in enumerate(out["aux_z_views"]):
        embeddings[f"z_aux_view_{idx}"] = z.detach().cpu().numpy().astype(np.float32)

    model_path = os.path.join(output_dir, "encoder.pt")
    embedding_path = os.path.join(output_dir, "embeddings.npz")
    meta_path = os.path.join(output_dir, "encoder_meta.json")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": enc_cfg,
            "task_input_dims": [view.features.shape[1] for view in task_views],
            "aux_input_dims": [view.features.shape[1] for view in aux_views],
            "task_view_names": [view.name for view in task_views],
            "aux_view_names": [view.name for view in aux_views],
        },
        model_path,
    )
    np.savez_compressed(embedding_path, **embeddings)
    save_json(
        {
            "param_count": count_parameters(model),
            "task_view_names": [view.name for view in task_views],
            "aux_view_names": [view.name for view in aux_views],
            "task_input_dims": [int(view.features.shape[1]) for view in task_views],
            "aux_input_dims": [int(view.features.shape[1]) for view in aux_views],
            "history": history,
            "latent_dim": latent_dim,
        },
        meta_path,
    )
    return EncoderArtifacts(model_path, embedding_path, meta_path, count_parameters(model))
