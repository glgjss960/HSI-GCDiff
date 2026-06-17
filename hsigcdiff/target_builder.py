import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .etap_teacher import TeacherResult
from .utils import ensure_dir, l2_normalize_np, row_normalize_np


@dataclass
class TargetResult:
    z_target: np.ndarray
    confidence: np.ndarray
    mode: str
    source_key: str
    prototypes: Optional[np.ndarray] = None
    y_anchor: Optional[np.ndarray] = None


def _anchor_proto_target(embeddings: Dict[str, np.ndarray], teacher: TeacherResult, cfg: Dict) -> TargetResult:
    target_cfg = cfg.get("target", {})
    source_key = target_cfg.get("source_key", "z_task")
    z_base = l2_normalize_np(np.asarray(embeddings[source_key], dtype=np.float32))
    y_anchor = row_normalize_np(np.asarray(teacher.y_anchor, dtype=np.float32))
    confidence = np.asarray(teacher.confidence, dtype=np.float32).reshape(-1)
    weight = confidence.copy()
    min_conf = float(target_cfg.get("min_confidence", 0.0))
    weight[confidence < min_conf] = 0.0
    weighted_y = y_anchor * weight[:, None]
    denom = weighted_y.sum(axis=0, keepdims=True).T
    prototypes = np.zeros((y_anchor.shape[1], z_base.shape[1]), dtype=np.float32)
    valid = denom.reshape(-1) > 1e-12
    if np.any(valid):
        prototypes[valid] = (weighted_y[:, valid].T @ z_base) / denom[valid]
    if np.any(~valid):
        prototypes[~valid] = z_base.mean(axis=0, keepdims=True)
    prototypes = l2_normalize_np(prototypes.astype(np.float32))
    z_proto = l2_normalize_np((y_anchor @ prototypes).astype(np.float32))
    beta = float(target_cfg.get("beta", 0.0))
    z_target = l2_normalize_np(((1.0 - beta) * z_base + beta * z_proto).astype(np.float32))
    return TargetResult(
        z_target=z_target.astype(np.float32),
        confidence=confidence.astype(np.float32),
        mode="anchor_proto",
        source_key=source_key,
        prototypes=prototypes.astype(np.float32),
        y_anchor=y_anchor.astype(np.float32),
    )


def build_target(embeddings: Dict[str, np.ndarray], cfg: Dict, teacher: Optional[TeacherResult] = None) -> TargetResult:
    target_cfg = cfg.get("target", {})
    mode = target_cfg.get("mode", "task")
    if mode == "task":
        source_key = target_cfg.get("source_key", "z_task")
        if source_key not in embeddings:
            raise KeyError(f"Embedding key '{source_key}' not found. Available: {sorted(embeddings.keys())}")
        z_target = l2_normalize_np(np.asarray(embeddings[source_key], dtype=np.float32))
        confidence = np.ones(z_target.shape[0], dtype=np.float32)
        return TargetResult(z_target=z_target.astype(np.float32), confidence=confidence, mode=mode, source_key=source_key)
    if mode == "anchor_proto":
        if teacher is None:
            raise ValueError("target.mode='anchor_proto' requires a teacher artifact.")
        return _anchor_proto_target(embeddings, teacher, cfg)
    raise ValueError(f"Unknown target mode: {mode}")


def save_target(target: TargetResult, output_dir: str) -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "target.npz")
    payload = {
        "z_target": target.z_target,
        "confidence": target.confidence,
        "mode": np.asarray([target.mode]),
        "source_key": np.asarray([target.source_key]),
    }
    if target.prototypes is not None:
        payload["prototypes"] = target.prototypes
    if target.y_anchor is not None:
        payload["y_anchor"] = target.y_anchor
    np.savez_compressed(path, **payload)
    return path


def load_target(path: str) -> TargetResult:
    data = np.load(path)
    mode = str(data["mode"][0]) if "mode" in data else "task"
    source_key = str(data["source_key"][0]) if "source_key" in data else "z_task"
    return TargetResult(
        z_target=np.asarray(data["z_target"], dtype=np.float32),
        confidence=np.asarray(data["confidence"], dtype=np.float32),
        mode=mode,
        source_key=source_key,
        prototypes=np.asarray(data["prototypes"], dtype=np.float32) if "prototypes" in data else None,
        y_anchor=np.asarray(data["y_anchor"], dtype=np.float32) if "y_anchor" in data else None,
    )
