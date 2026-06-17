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
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        time_dim: int,
        layers: int = 3,
        dropout: float = 0.1,
        condition: str = "none",
        normalize_output: bool = True,
    ):
        super().__init__()
        if condition not in {"none", "task"}:
            raise ValueError("condition must be 'none' or 'task'.")
        self.condition = condition
        self.normalize_output = bool(normalize_output)
        self.time_dim = int(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        in_dim = latent_dim + hidden_dim + (latent_dim if condition == "task" else 0)
        blocks: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(max(int(layers) - 2, 0)):
            blocks.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
        blocks.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        parts = [x_t, t_emb]
        if self.condition == "task":
            if cond is None:
                raise ValueError("condition='task' requires cond tensor.")
            parts.append(cond)
        out = self.net(torch.cat(parts, dim=1))
        return F.normalize(out, p=2, dim=1) if self.normalize_output else out


class DiffGraphGaussianDiffusion:
    def __init__(self, timesteps: int = 100, noise_scale: float = 1.0, device: torch.device = torch.device("cpu")):
        self.timesteps = int(timesteps)
        self.noise_scale = float(noise_scale)
        betas = torch.linspace(1e-4, 2e-2, self.timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.posterior_variance = betas * (1.0 - alpha_bar_prev) / torch.clamp(1.0 - alpha_bar, min=1e-20)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alpha_bar_prev) / torch.clamp(1.0 - alpha_bar, min=1e-20)
        self.posterior_mean_coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alphas) / torch.clamp(1.0 - alpha_bar, min=1e-20)
        self.device = device

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = arr[t].float()
        while len(out.shape) < len(shape):
            out = out[..., None]
        return out.expand(shape)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if self.noise_scale == 0:
            return x_start
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract(self.sqrt_alpha_bar, t, x_start.shape) * x_start + self._extract(
            self.sqrt_one_minus_alpha_bar, t, x_start.shape
        ) * noise

    def snr_weight(self, t: torch.Tensor) -> torch.Tensor:
        snr = self.alpha_bar / torch.clamp(1.0 - self.alpha_bar, min=1e-12)
        prev = torch.where(t > 0, snr[torch.clamp(t - 1, min=0)], torch.ones_like(t, dtype=torch.float32))
        cur = snr[t]
        weight = torch.clamp(prev - cur, min=0.0)
        weight = torch.where(t == 0, torch.ones_like(weight), weight)
        return weight

    def training_losses2(
        self,
        model: LatentDenoiser,
        target_embeds: torch.Tensor,
        x_start: torch.Tensor,
        confidence: torch.Tensor,
        cond: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x_start.shape[0]
        if indices is None:
            indices = torch.arange(n, device=x_start.device)
        ts = torch.randint(0, self.timesteps, (indices.numel(),), device=x_start.device)
        noise = torch.randn_like(x_start[indices])
        x_t = self.q_sample(x_start[indices], ts, noise)
        cond_batch = cond[indices] if cond is not None else None
        model_output = model(x_t, ts, cond_batch)
        mse = ((target_embeds[indices] - model_output) ** 2).mean(dim=1)
        weight = self.snr_weight(ts) * confidence[indices]
        loss = (weight * mse).sum() / torch.clamp(weight.sum(), min=1e-6)
        return loss, model_output

    def p_mean_variance(
        self,
        model: LatentDenoiser,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_output = model(x_t, t, cond)
        return self._extract(self.posterior_mean_coef1, t, x_t.shape) * model_output + self._extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t

    def p_sample(
        self,
        model: LatentDenoiser,
        x_start: torch.Tensor,
        sampling_steps: int,
        cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        sampling_steps = max(0, min(int(sampling_steps), self.timesteps))
        if sampling_steps == 0:
            x_t = x_start
        else:
            t0 = torch.full((x_start.shape[0],), sampling_steps - 1, dtype=torch.long, device=x_start.device)
            noise = torch.randn(x_start.shape, generator=generator, device=x_start.device)
            x_t = self.q_sample(x_start, t0, noise)
        for i in range(sampling_steps - 1, -1, -1):
            t = torch.full((x_t.shape[0],), i, dtype=torch.long, device=x_t.device)
            x_t = self.p_mean_variance(model, x_t, t, cond)
        return F.normalize(x_t, p=2, dim=1)


@dataclass
class DenoiserArtifacts:
    model_path: str
    meta_path: str
    param_count: int


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_denoiser(
    embeddings: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    cfg: Dict,
    output_dir: str,
    device: torch.device,
) -> DenoiserArtifacts:
    ensure_dir(output_dir)
    den_cfg = cfg.get("denoiser", {})
    source_key = den_cfg.get("source_key", "z_aux")
    target_key = den_cfg.get("target_key", "z_target")
    cond_mode = den_cfg.get("condition", "none")
    x_start = torch.from_numpy(np.asarray(embeddings[source_key], dtype=np.float32)).to(device)
    target_embeds = torch.from_numpy(np.asarray(target[target_key], dtype=np.float32)).to(device)
    cond = torch.from_numpy(np.asarray(embeddings["z_task"], dtype=np.float32)).to(device) if cond_mode == "task" else None
    confidence = torch.from_numpy(np.asarray(target.get("confidence", np.ones(x_start.shape[0])), dtype=np.float32)).to(device)
    confidence = torch.clamp(confidence, min=float(den_cfg.get("min_weight", 0.05)))

    latent_dim = x_start.shape[1]
    model = LatentDenoiser(
        latent_dim=latent_dim,
        hidden_dim=int(den_cfg.get("hidden_dim", 384)),
        time_dim=int(den_cfg.get("time_dim", 64)),
        layers=int(den_cfg.get("layers", 3)),
        dropout=float(den_cfg.get("dropout", 0.1)),
        condition=cond_mode,
        normalize_output=bool(den_cfg.get("normalize_output", True)),
    ).to(device)
    diffusion = DiffGraphGaussianDiffusion(
        timesteps=int(den_cfg.get("timesteps", 100)),
        noise_scale=float(den_cfg.get("noise_scale", 1.0)),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(den_cfg.get("lr", 1e-3)),
        weight_decay=float(den_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(den_cfg.get("epochs", 1000))
    batch_size = int(den_cfg.get("batch_size", 0))
    n = x_start.shape[0]
    history = []
    iterator = trange(1, epochs + 1, desc="denoiser", leave=False) if epochs > 0 else []
    for epoch in iterator:
        model.train()
        indices = torch.randperm(n, device=device)[:batch_size] if batch_size and batch_size < n else None
        loss, _ = diffusion.training_losses2(model, target_embeds, x_start, confidence, cond=cond, indices=indices)
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
            "latent_dim": int(latent_dim),
            "hidden_dim": int(den_cfg.get("hidden_dim", 384)),
            "time_dim": int(den_cfg.get("time_dim", 64)),
        },
        model_path,
    )
    save_json(
        {
            "param_count": count_parameters(model),
            "history": history,
            "latent_dim": int(latent_dim),
            "timesteps": int(diffusion.timesteps),
            "source_key": source_key,
            "target_key": target_key,
            "condition": cond_mode,
        },
        meta_path,
    )
    return DenoiserArtifacts(model_path, meta_path, count_parameters(model))


def load_denoiser(path: str, device: torch.device) -> LatentDenoiser:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    model = LatentDenoiser(
        latent_dim=int(ckpt["latent_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        time_dim=int(ckpt["time_dim"]),
        layers=int(cfg.get("layers", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        condition=cfg.get("condition", "none"),
        normalize_output=bool(cfg.get("normalize_output", True)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def sample_denoised_aux(
    model: LatentDenoiser,
    embeddings: Dict[str, np.ndarray],
    cfg: Dict,
    sampling_steps: Iterable[int],
    noise_seed: int,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    den_cfg = cfg.get("denoiser", {})
    source_key = den_cfg.get("source_key", "z_aux")
    x_start = torch.from_numpy(np.asarray(embeddings[source_key], dtype=np.float32)).to(device)
    cond = torch.from_numpy(np.asarray(embeddings["z_task"], dtype=np.float32)).to(device) if den_cfg.get("condition", "none") == "task" else None
    diffusion = DiffGraphGaussianDiffusion(
        timesteps=int(den_cfg.get("timesteps", 100)),
        noise_scale=float(den_cfg.get("noise_scale", 1.0)),
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_seed))
    out: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for steps in sampling_steps:
            z = diffusion.p_sample(model, x_start, int(steps), cond=cond, generator=generator)
            out[int(steps)] = z.detach().cpu().numpy().astype(np.float32)
    return out
