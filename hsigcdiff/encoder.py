import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from .graph_builder import GraphBundle
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


class MeanFusion(nn.Module):
    def forward(self, z_views: List[torch.Tensor]) -> torch.Tensor:
        return F.normalize(torch.stack(z_views, dim=0).mean(dim=0), p=2, dim=1)


class AttentionFusion(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.score = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, z_views: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(z_views, dim=1)
        weights = torch.softmax(self.score(stacked).squeeze(-1), dim=1)
        return F.normalize((stacked * weights.unsqueeze(-1)).sum(dim=1), p=2, dim=1)


class ConcatLinearFusion(nn.Module):
    def __init__(self, latent_dim: int, n_views: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim * n_views, latent_dim)

    def forward(self, z_views: List[torch.Tensor]) -> torch.Tensor:
        return F.normalize(self.proj(torch.cat(z_views, dim=1)), p=2, dim=1)


class MultiViewGraphEncoder(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int, latent_dim: int, dropout: float, fusion: str):
        super().__init__()
        self.encoders = nn.ModuleList(
            [ViewGraphAutoEncoder(in_dim, hidden_dim, latent_dim, dropout) for in_dim in input_dims]
        )
        if fusion == "attention":
            self.fusion = AttentionFusion(latent_dim)
        elif fusion == "concat_linear":
            self.fusion = ConcatLinearFusion(latent_dim, len(input_dims))
        elif fusion == "mean":
            self.fusion = MeanFusion()
        else:
            raise ValueError(f"Unknown fusion mode: {fusion}")

    def forward(self, features: List[torch.Tensor], adjs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_views, recons = [], []
        for encoder, x, adj in zip(self.encoders, features, adjs):
            z, recon = encoder(x, adj)
            z_views.append(z)
            recons.append(recon)
        z_source = F.normalize(torch.stack(z_views, dim=0).mean(dim=0), p=2, dim=1)
        z_fuse = self.fusion(z_views)
        return {"z_views": z_views, "recons": recons, "z_source": z_source, "z_fuse": z_fuse}


@dataclass
class EncoderArtifacts:
    model_path: str
    embedding_path: str
    meta_path: str
    param_count: int


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _make_tensors(graph: GraphBundle, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features = [torch.from_numpy(view.features.astype(np.float32)).to(device) for view in graph.views]
    adjs = [to_torch_sparse(view.adjacency, device=device) for view in graph.views]
    return features, adjs


def train_encoder(graph: GraphBundle, cfg: Dict, output_dir: str, device: torch.device) -> EncoderArtifacts:
    ensure_dir(output_dir)
    enc_cfg = cfg.get("encoder", {})
    hidden_dim = int(enc_cfg.get("hidden_dim", 256))
    latent_dim = int(enc_cfg.get("latent_dim", 128))
    dropout = float(enc_cfg.get("dropout", 0.1))
    fusion = enc_cfg.get("fusion", "mean")
    epochs = int(enc_cfg.get("epochs", 200))
    lr = float(enc_cfg.get("lr", 1e-3))
    weight_decay = float(enc_cfg.get("weight_decay", 1e-4))
    consistency_weight = float(enc_cfg.get("consistency_weight", 0.05))
    log_every = int(enc_cfg.get("log_every", 25))

    input_dims = [view.features.shape[1] for view in graph.views]
    model = MultiViewGraphEncoder(input_dims, hidden_dim, latent_dim, dropout, fusion).to(device)
    features, adjs = _make_tensors(graph, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    iterator = trange(1, epochs + 1, desc="encoder", leave=False) if epochs > 0 else []
    for epoch in iterator:
        model.train()
        out = model(features, adjs)
        recon_loss = sum(F.mse_loss(recon, x) for recon, x in zip(out["recons"], features)) / len(features)
        consistency = sum(1.0 - (z * out["z_source"].detach()).sum(dim=1).mean() for z in out["z_views"]) / len(features)
        loss = recon_loss + consistency_weight * consistency
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "recon_loss": float(recon_loss.detach().cpu()),
                    "consistency": float(consistency.detach().cpu()),
                }
            )

    model.eval()
    with torch.no_grad():
        out = model(features, adjs)
    embeddings = {
        "z_source": out["z_source"].detach().cpu().numpy().astype(np.float32),
        "z_fuse": out["z_fuse"].detach().cpu().numpy().astype(np.float32),
    }
    for idx, z in enumerate(out["z_views"]):
        embeddings[f"z_view_{idx}"] = z.detach().cpu().numpy().astype(np.float32)

    model_path = os.path.join(output_dir, "encoder.pt")
    embedding_path = os.path.join(output_dir, "embeddings.npz")
    meta_path = os.path.join(output_dir, "encoder_meta.json")
    torch.save({"state_dict": model.state_dict(), "config": enc_cfg, "input_dims": input_dims}, model_path)
    np.savez_compressed(embedding_path, **embeddings)
    save_json(
        {
            "param_count": count_parameters(model),
            "input_dims": input_dims,
            "history": history,
            "fusion": fusion,
            "latent_dim": latent_dim,
        },
        meta_path,
    )
    return EncoderArtifacts(
        model_path=model_path,
        embedding_path=embedding_path,
        meta_path=meta_path,
        param_count=count_parameters(model),
    )
