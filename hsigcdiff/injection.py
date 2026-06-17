from typing import Dict, Iterable

import numpy as np

from .utils import l2_normalize_np


def residual_inject(z_task: np.ndarray, z_residual: np.ndarray, alpha: float) -> np.ndarray:
    z_task = np.asarray(z_task, dtype=np.float32)
    z_residual = np.asarray(z_residual, dtype=np.float32)
    return l2_normalize_np((z_task + float(alpha) * z_residual).astype(np.float32))


def build_injection_embeddings(
    z_task: np.ndarray,
    z_aux: np.ndarray,
    z_diff: np.ndarray,
    alphas: Iterable[float],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for alpha in alphas:
        label = f"{float(alpha):g}"
        out[f"z_task_plus_raw_aux_alpha={label}"] = residual_inject(z_task, z_aux, float(alpha))
        out[f"z_task_plus_denoised_aux_alpha={label}"] = residual_inject(z_task, z_diff, float(alpha))
    return out
