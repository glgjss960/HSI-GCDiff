import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from .utils import ensure_dir, save_json


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, time_dim: int, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        in_dim = latent_dim * 2 + hidden_dim
        blocks: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(max(int(layers) - 2, 0)):
            blocks.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
        blocks.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*blocks)
        self.time_dim = int(time_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        return F.normalize(self.net(torch.cat([z_t, cond, t_emb], dim=1)), p=2, dim=1)


class GraphLatentDiffusion:
    def __init__(self, timesteps: int = 100, schedule: str = "linear", device: torch.device = torch.device("cpu")):
        self.timesteps = int(timesteps)
        if schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, self.timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / self.timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(1e-4, 0.999)
        else:
            raise ValueError(f"Unknown diffusion schedule: {schedule}")
        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(device)
        self.device = device

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bar[t].unsqueeze(1)
        return torch.sqrt(alpha_bar) * x_start + torch.sqrt(1.0 - alpha_bar) * noise


@dataclass
class DenoiserArtifacts:
    model_path: str
    meta_path: str
    param_count: int


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    node_loss = ((pred - target) ** 2).mean(dim=1)
    denom = torch.clamp(weight.sum(), min=1e-6)
    return (node_loss * weight).sum() / denom


def train_denoiser(
    embeddings: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    cfg: Dict,
    output_dir: str,
    device: torch.device,
) -> DenoiserArtifacts:
    ensure_dir(output_dir)
    den_cfg = cfg.get("denoiser", {})
    z_source = torch.from_numpy(np.asarray(embeddings[den_cfg.get("source_key", "z_source")], dtype=np.float32)).to(device)
    cond = torch.from_numpy(np.asarray(embeddings[den_cfg.get("cond_key", "z_fuse")], dtype=np.float32)).to(device)
    z_target = torch.from_numpy(np.asarray(target["z_target"], dtype=np.float32)).to(device)
    confidence = torch.from_numpy(np.asarray(target["confidence"], dtype=np.float32)).to(device)
    confidence = torch.clamp(confidence, min=float(den_cfg.get("min_weight", 0.05)))
    if float(den_cfg.get("confidence_power", 1.0)) != 1.0:
        confidence = confidence.pow(float(den_cfg.get("confidence_power", 1.0)))

    latent_dim = z_source.shape[1]
    hidden_dim = int(den_cfg.get("hidden_dim", 384))
    time_dim = int(den_cfg.get("time_dim", 64))
    model = LatentDenoiser(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        layers=int(den_cfg.get("layers", 3)),
        dropout=float(den_cfg.get("dropout", 0.1)),
    ).to(device)
    diffusion = GraphLatentDiffusion(
        timesteps=int(den_cfg.get("timesteps", 100)),
        schedule=den_cfg.get("schedule", "linear"),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(den_cfg.get("lr", 1e-3)),
        weight_decay=float(den_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(den_cfg.get("epochs", 1000))
    batch_size = int(den_cfg.get("batch_size", 0))
    n = z_source.shape[0]
    history = []
    iterator = trange(1, epochs + 1, desc="denoiser", leave=False) if epochs > 0 else []
    for epoch in iterator:
        model.train()
        if batch_size and batch_size < n:
            indices = torch.randperm(n, device=device)[:batch_size]
        else:
            indices = torch.arange(n, device=device)
        t = torch.randint(0, diffusion.timesteps, (indices.numel(),), device=device)
        noise = torch.randn_like(z_source[indices])
        z_t = diffusion.q_sample(z_source[indices], t, noise)
        pred = model(z_t, t, cond[indices])
        loss = _weighted_mse(pred, z_target[indices], confidence[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_every = int(den_cfg.get("log_every", 100))
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})

    model_path = os.path.join(output_dir, "denoiser.pt")
    meta_path = os.path.join(output_dir, "denoiser_meta.json")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": den_cfg,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "time_dim": time_dim,
        },
        model_path,
    )
    save_json(
        {
            "param_count": count_parameters(model),
            "history": history,
            "latent_dim": int(latent_dim),
            "timesteps": int(diffusion.timesteps),
        },
        meta_path,
    )
    return DenoiserArtifacts(model_path=model_path, meta_path=meta_path, param_count=count_parameters(model))


def load_denoiser(path: str, device: torch.device) -> LatentDenoiser:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    model = LatentDenoiser(
        latent_dim=int(ckpt["latent_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        time_dim=int(ckpt["time_dim"]),
        layers=int(cfg.get("layers", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def denoise_embeddings(
    model: LatentDenoiser,
    embeddings: Dict[str, np.ndarray],
    cfg: Dict,
    t_values: Iterable[int],
    noise_seed: int,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    den_cfg = cfg.get("denoiser", {})
    z_source = torch.from_numpy(np.asarray(embeddings[den_cfg.get("source_key", "z_source")], dtype=np.float32)).to(device)
    cond = torch.from_numpy(np.asarray(embeddings[den_cfg.get("cond_key", "z_fuse")], dtype=np.float32)).to(device)
    diffusion = GraphLatentDiffusion(
        timesteps=int(den_cfg.get("timesteps", 100)),
        schedule=den_cfg.get("schedule", "linear"),
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_seed))
    out: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for t_value in t_values:
            t_value = max(0, min(int(t_value), diffusion.timesteps - 1))
            t = torch.full((z_source.shape[0],), t_value, dtype=torch.long, device=device)
            if t_value == 0:
                noise = torch.zeros_like(z_source)
            else:
                noise = torch.randn(z_source.shape, generator=generator, device=device)
            z_t = diffusion.q_sample(z_source, t, noise)
            pred = model(z_t, t, cond)
            out[t_value] = pred.detach().cpu().numpy().astype(np.float32)
    return out
