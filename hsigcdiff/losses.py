from typing import Dict

import torch
import torch.nn.functional as F

from .clustering import student_t_distribution, target_distribution


def reconstruction_loss(
    node_features: torch.Tensor,
    context_features: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    return F.mse_loss(outputs["node_recon"], node_features) + F.mse_loss(outputs["context_recon"], context_features)


def clustering_kl_loss(z: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    q = student_t_distribution(F.normalize(z, dim=-1), F.normalize(centers, dim=-1))
    p = target_distribution(q)
    return F.kl_div(torch.log(q.clamp_min(1e-12)), p, reduction="batchmean")


def view_consistency_loss(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    z1 = outputs["z_spectral"]
    z2 = outputs["z_spatial"]
    z3 = outputs["z_context"]
    return (F.mse_loss(z1, z2) + F.mse_loss(z1, z3) + F.mse_loss(z2, z3)) / 3.0


def confidence_weighted_mse(pred: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    per_node = (pred - target).pow(2).mean(dim=1)
    weights = confidence / confidence.mean().clamp_min(1e-6)
    return (per_node * weights).mean()


def hard_sample_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    confidence: torch.Tensor,
    temperature: float = 0.5,
    beta: float = 3.0,
) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    labels = torch.cat([labels, labels], dim=0)
    conf = torch.cat([confidence, confidence], dim=0)
    n = z.size(0)

    sim = z @ z.t()
    logits = sim / max(temperature, 1e-6)
    self_mask = torch.eye(n, device=z.device, dtype=torch.bool)
    positive = (labels[:, None] == labels[None, :]) & (~self_mask)
    valid = positive.any(dim=1)
    if not torch.any(valid):
        return z.sum() * 0.0

    sim_norm = (sim - sim.min()) / (sim.max() - sim.min()).clamp_min(1e-6)
    q = positive.float()
    pair_conf = torch.sqrt((conf[:, None] * conf[None, :]).clamp_min(0.0))
    hard_weight = 1.0 + pair_conf * torch.abs(q - sim_norm).pow(beta)

    exp_logits = torch.exp(logits) * (~self_mask).float() * hard_weight.detach()
    pos_sum = (exp_logits * positive.float()).sum(dim=1)
    all_sum = exp_logits.sum(dim=1).clamp_min(1e-12)
    loss = -torch.log(pos_sum.clamp_min(1e-12) / all_sum)
    return loss[valid].mean()

