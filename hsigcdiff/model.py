from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .diffusion import ConditionalDenoiser, LatentDiffusion


class SparseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        if adj.is_sparse:
            out = torch.sparse.mm(adj, support)
        else:
            out = adj @ support
        return out + self.bias


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SparseGraphConv(in_dim, hidden_dim)
        self.conv2 = SparseGraphConv(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, adj)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, adj)
        h = self.norm2(h)
        return F.normalize(h, dim=-1)


class RelationAttention(nn.Module):
    def __init__(self, latent_dim: int, n_relations: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1, bias=False),
        )
        self.n_relations = n_relations

    def forward(self, relation_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(relation_embeddings).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(relation_embeddings * weights.unsqueeze(-1), dim=1)
        return F.normalize(fused, dim=-1), weights


class HSIGCDiffModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        context_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
        diffusion_timesteps: int,
        diffusion_schedule: str,
        diffusion_time_emb_dim: int,
        diffusion_hidden_dim: int,
        offset_noise: float,
    ):
        super().__init__()
        self.spectral_encoder = GCNEncoder(node_dim, hidden_dim, latent_dim, dropout=dropout)
        self.spatial_encoder = GCNEncoder(node_dim, hidden_dim, latent_dim, dropout=dropout)
        self.context_encoder = GCNEncoder(context_dim, hidden_dim, latent_dim, dropout=dropout)
        self.fusion = RelationAttention(latent_dim, n_relations=3)

        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.context_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )

        self.denoiser = ConditionalDenoiser(
            latent_dim=latent_dim,
            condition_dim=latent_dim,
            time_emb_dim=diffusion_time_emb_dim,
            hidden_dim=diffusion_hidden_dim,
            dropout=dropout,
        )
        self.diffusion = LatentDiffusion(
            timesteps=diffusion_timesteps,
            schedule=diffusion_schedule,
            offset_noise=offset_noise,
        )

    def encode(
        self,
        node_features: torch.Tensor,
        context_features: torch.Tensor,
        adjs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        z_spe = self.spectral_encoder(node_features, adjs["spectral"])
        z_spa = self.spatial_encoder(node_features, adjs["spatial"])
        z_ctx = self.context_encoder(context_features, adjs["context"])
        stack = torch.stack([z_spe, z_spa, z_ctx], dim=1)
        z_fuse, weights = self.fusion(stack)
        source = F.normalize((z_spe + z_spa + z_ctx) / 3.0, dim=-1)
        return {
            "z_spectral": z_spe,
            "z_spatial": z_spa,
            "z_context": z_ctx,
            "z_fuse": z_fuse,
            "z_source": source,
            "relation_weights": weights,
        }

    def forward(
        self,
        node_features: torch.Tensor,
        context_features: torch.Tensor,
        adjs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out = self.encode(node_features, context_features, adjs)
        out["node_recon"] = self.node_decoder(out["z_fuse"])
        out["context_recon"] = self.context_decoder(out["z_context"])
        return out

    @torch.no_grad()
    def denoise_embeddings(
        self,
        source: torch.Tensor,
        condition: torch.Tensor,
        steps: int = 0,
    ) -> torch.Tensor:
        return F.normalize(self.diffusion.sample(self.denoiser, source, condition, steps=steps), dim=-1)

