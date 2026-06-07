import os
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .etap_teacher import TeacherResult
from .utils import ensure_dir, l2_normalize_np, row_normalize_np


@dataclass
class TargetResult:
    z_target: np.ndarray
    prototypes: np.ndarray
    confidence: np.ndarray
    y_anchor: np.ndarray
    source_key: str


def build_target(embeddings: Dict[str, np.ndarray], teacher: TeacherResult, cfg: Dict) -> TargetResult:
    target_cfg = cfg.get("target", {})
    source_key = target_cfg.get("source_key", "z_fuse")
    if source_key not in embeddings:
        raise KeyError(f"Embedding key '{source_key}' not found. Available: {sorted(embeddings.keys())}")
    z_base = l2_normalize_np(np.asarray(embeddings[source_key], dtype=np.float32))
    y_anchor = np.asarray(teacher.y_anchor, dtype=np.float32)
    if target_cfg.get("assignment", "soft") == "hard":
        hard = y_anchor.argmax(axis=1)
        y_anchor = np.zeros_like(y_anchor, dtype=np.float32)
        y_anchor[np.arange(hard.size), hard] = 1.0
    else:
        y_anchor = row_normalize_np(y_anchor)

    confidence = np.asarray(teacher.confidence, dtype=np.float32).reshape(-1)
    min_conf = float(target_cfg.get("min_confidence", 0.0))
    weight = confidence.copy()
    weight[confidence < min_conf] = 0.0
    weighted_y = y_anchor * weight[:, None]
    denom = weighted_y.sum(axis=0, keepdims=True).T
    prototypes = np.zeros((y_anchor.shape[1], z_base.shape[1]), dtype=np.float32)
    valid = denom.reshape(-1) > 1e-12
    if np.any(valid):
        prototypes[valid] = (weighted_y[:, valid].T @ z_base) / denom[valid]
    if np.any(~valid):
        fallback = z_base.mean(axis=0, keepdims=True)
        prototypes[~valid] = fallback
    prototypes = l2_normalize_np(prototypes.astype(np.float32))
    z_target = l2_normalize_np((y_anchor @ prototypes).astype(np.float32))
    return TargetResult(
        z_target=z_target.astype(np.float32),
        prototypes=prototypes.astype(np.float32),
        confidence=confidence.astype(np.float32),
        y_anchor=y_anchor.astype(np.float32),
        source_key=source_key,
    )


def save_target(target: TargetResult, output_dir: str) -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "target.npz")
    np.savez_compressed(
        path,
        z_target=target.z_target,
        prototypes=target.prototypes,
        confidence=target.confidence,
        y_anchor=target.y_anchor,
        source_key=np.asarray([target.source_key]),
    )
    return path


def load_target(path: str) -> TargetResult:
    data = np.load(path)
    source_key = str(data["source_key"][0]) if "source_key" in data else "z_fuse"
    return TargetResult(
        z_target=np.asarray(data["z_target"], dtype=np.float32),
        prototypes=np.asarray(data["prototypes"], dtype=np.float32),
        confidence=np.asarray(data["confidence"], dtype=np.float32),
        y_anchor=np.asarray(data["y_anchor"], dtype=np.float32),
        source_key=source_key,
    )
