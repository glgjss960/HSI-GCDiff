import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        time_emb_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim + time_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        return self.net(torch.cat([z_t, condition, t_emb], dim=-1))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999).float()


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class LatentDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 100,
        schedule: str = "cosine",
        offset_noise: float = 0.0,
    ):
        super().__init__()
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unsupported schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.timesteps = timesteps
        self.offset_noise = offset_noise
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _extract(self, values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, t)
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)

    def noise_like(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        if self.offset_noise != 0.0:
            noise = noise + self.offset_noise
        return noise

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = self.noise_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def training_step(
        self,
        denoiser: nn.Module,
        source: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.randint(0, self.timesteps, (source.size(0),), device=source.device).long()
        noise = self.noise_like(source)
        z_t = self.q_sample(source, t, noise)
        noise_pred = denoiser(z_t, t, condition)
        z0_hat = self.predict_x0_from_noise(z_t, t, noise_pred)
        return noise_pred, noise, z0_hat, t

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        source: torch.Tensor,
        condition: torch.Tensor,
        steps: int = 0,
    ) -> torch.Tensor:
        if steps <= 0:
            return source
        steps = min(steps, self.timesteps)
        t_start = torch.full((source.size(0),), steps - 1, device=source.device, dtype=torch.long)
        x_t = self.q_sample(source, t_start, noise=self.noise_like(source))
        for i in reversed(range(steps)):
            t = torch.full((source.size(0),), i, device=source.device, dtype=torch.long)
            noise_pred = denoiser(x_t, t, condition)
            x0 = self.predict_x0_from_noise(x_t, t, noise_pred)
            if i == 0:
                x_t = x0
            else:
                coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
                coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
                x_t = coef1 * x0 + coef2 * x_t
        return x_t

