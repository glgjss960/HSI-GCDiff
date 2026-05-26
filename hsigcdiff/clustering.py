from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def run_kmeans(features: np.ndarray, n_clusters: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    labels = model.fit_predict(features)
    return labels.astype(np.int64), model.cluster_centers_.astype(np.float32)


def student_t_distribution(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    dist = torch.cdist(z, centers, p=2).pow(2)
    q = (1.0 + dist / alpha).pow(-(alpha + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return (weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)).detach()


def confidence_from_centers(z: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    q = student_t_distribution(F.normalize(z, dim=-1), F.normalize(centers, dim=-1))
    confidence = q.max(dim=1).values
    return confidence.detach()


def prototypes_from_labels(
    z: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
    confidence: Optional[torch.Tensor] = None,
    tau: float = 0.0,
    fallback_centers: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    protos = []
    for cls in range(n_clusters):
        mask = labels == cls
        if confidence is not None:
            mask = mask & (confidence >= tau)
        if torch.any(mask):
            proto = z[mask].mean(dim=0)
        elif fallback_centers is not None:
            proto = fallback_centers[cls]
        else:
            proto = z[labels == cls].mean(dim=0) if torch.any(labels == cls) else z.mean(dim=0)
        protos.append(proto)
    return F.normalize(torch.stack(protos, dim=0), dim=-1)


def prototype_corrected_target(
    teacher_z: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    momentum: float,
) -> torch.Tensor:
    target = (1.0 - momentum) * teacher_z + momentum * prototypes[labels]
    return F.normalize(target.detach(), dim=-1)

