import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from .graph_builder import GraphBundle
from .utils import ensure_dir, row_normalize_np


@dataclass
class TeacherResult:
    final_z: np.ndarray
    g_anchor: np.ndarray
    y_anchor: np.ndarray
    y_hard: np.ndarray
    confidence: np.ndarray
    anchor_labels: np.ndarray
    final_s: np.ndarray
    backend: str


def _standardize(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(x, dtype=np.float32)).astype(np.float32)


def _select_anchor_indices(features: np.ndarray, n_anchors: int, seed: int) -> np.ndarray:
    n = features.shape[0]
    if n_anchors >= n:
        return np.arange(n, dtype=np.int64)
    kmeans = KMeans(n_clusters=n_anchors, n_init=10, random_state=seed).fit(features)
    distances = pairwise_distances(kmeans.cluster_centers_, features)
    selected = np.unique(distances.argmin(axis=1)).astype(np.int64)
    if selected.size < n_anchors:
        min_to_selected = pairwise_distances(features, features[selected]).min(axis=1)
        fill = np.argsort(-min_to_selected)
        fill = [idx for idx in fill.tolist() if idx not in set(selected.tolist())]
        selected = np.concatenate([selected, np.asarray(fill[: n_anchors - selected.size], dtype=np.int64)])
    return np.sort(selected[:n_anchors]).astype(np.int64)


def _adaptive_anchor_affinity(features: np.ndarray, anchor_features: np.ndarray, neighbors: int) -> np.ndarray:
    n, m = features.shape[0], anchor_features.shape[0]
    if m == 1:
        return np.ones((n, 1), dtype=np.float32)
    dist = pairwise_distances(features, anchor_features, metric="euclidean") ** 2
    k = max(1, min(int(neighbors), m - 1))
    order = np.argpartition(dist, kth=k, axis=1)[:, : k + 1]
    row_sort = np.take_along_axis(dist, order, axis=1).argsort(axis=1)
    order = np.take_along_axis(order, row_sort, axis=1)
    sorted_dist = np.take_along_axis(dist, order, axis=1)
    idx = order[:, :k]
    dk1 = sorted_dist[:, k : k + 1]
    dsel = sorted_dist[:, :k]
    denom = k * dk1 - dsel.sum(axis=1, keepdims=True)
    weights = (dk1 - dsel) / np.maximum(denom, 1e-12)
    weights = np.maximum(weights, 0.0)
    z = np.zeros((n, m), dtype=np.float32)
    z[np.arange(n)[:, None], idx] = weights.astype(np.float32)
    return row_normalize_np(z.astype(np.float32))


def _anchor_graph(anchor_features: np.ndarray, neighbors: int) -> np.ndarray:
    m = anchor_features.shape[0]
    if m <= 1:
        return np.ones((m, m), dtype=np.float32)
    k = max(1, min(int(neighbors), m - 1))
    adj = kneighbors_graph(anchor_features, n_neighbors=k, mode="distance", include_self=False).tocsr()
    if adj.data.size:
        sigma = float(np.median(adj.data))
        if sigma <= 1e-8:
            sigma = 1.0
        adj.data = np.exp(-(adj.data ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    adj = adj.maximum(adj.T) + sp.eye(m, dtype=np.float32, format="csr")
    dense = adj.toarray().astype(np.float32)
    dense /= max(float(dense.max()), 1e-12)
    return dense


def _anchor_labels(final_s: np.ndarray, anchor_features: np.ndarray, n_classes: int, seed: int, strict_components: bool) -> np.ndarray:
    graph = sp.csr_matrix(final_s > 0)
    n_components, labels = connected_components(graph, directed=False)
    if strict_components and n_components == n_classes:
        return labels.astype(np.int64)
    try:
        labels = SpectralClustering(
            n_clusters=n_classes,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=seed,
        ).fit_predict(np.maximum(final_s, final_s.T))
        return labels.astype(np.int64)
    except Exception:
        return KMeans(n_clusters=n_classes, n_init=20, random_state=seed).fit_predict(anchor_features).astype(np.int64)


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((labels.size, n_classes), dtype=np.float32)
    out[np.arange(labels.size), labels.astype(np.int64)] = 1.0
    return out


def _load_teacher_npz(path: str, n_classes: int) -> TeacherResult:
    data = np.load(path)
    final_z = np.asarray(data["Final_Z"] if "Final_Z" in data else data["final_z"], dtype=np.float32)
    if "G_A" in data:
        g_anchor = np.asarray(data["G_A"], dtype=np.float32)
        anchor_labels = g_anchor.argmax(axis=1).astype(np.int64)
    elif "g_anchor" in data:
        g_anchor = np.asarray(data["g_anchor"], dtype=np.float32)
        anchor_labels = g_anchor.argmax(axis=1).astype(np.int64)
    elif "anchor_labels" in data:
        anchor_labels = np.asarray(data["anchor_labels"], dtype=np.int64).reshape(-1)
        g_anchor = _one_hot(anchor_labels, n_classes)
    else:
        raise KeyError("Teacher npz must contain G_A/g_anchor or anchor_labels.")
    y_anchor = row_normalize_np(final_z @ g_anchor)
    return TeacherResult(
        final_z=final_z,
        g_anchor=g_anchor,
        y_anchor=y_anchor,
        y_hard=y_anchor.argmax(axis=1).astype(np.int64),
        confidence=y_anchor.max(axis=1).astype(np.float32),
        anchor_labels=anchor_labels,
        final_s=np.asarray(
            data["Final_S"] if "Final_S" in data else (data["final_s"] if "final_s" in data else np.eye(g_anchor.shape[0])),
            dtype=np.float32,
        ),
        backend="load_npz",
    )


def build_etap_teacher(graph: GraphBundle, cfg: Dict, seed: int) -> TeacherResult:
    teacher_cfg = cfg.get("teacher", {})
    n_classes = int(teacher_cfg.get("n_clusters") or cfg.get("n_clusters") or graph.superpixels.n_classes or 0)
    if n_classes <= 1:
        raise ValueError("n_clusters must be provided when ground truth is unavailable.")
    backend = teacher_cfg.get("backend", "python_etap_lite")
    if backend == "load_npz":
        return _load_teacher_npz(teacher_cfg["path"], n_classes=n_classes)
    if backend != "python_etap_lite":
        raise ValueError(f"Unknown teacher backend: {backend}")

    k = int(teacher_cfg.get("k", 5))
    n_anchors = min(int(teacher_cfg.get("n_anchors", graph.n_nodes)), graph.n_nodes)
    strict_components = bool(teacher_cfg.get("strict_components", False))
    view_features = [_standardize(view.features) for view in graph.views]
    joint_features = _standardize(np.concatenate(view_features, axis=1))
    anchor_idx = _select_anchor_indices(joint_features, n_anchors=n_anchors, seed=seed)

    final_z = np.zeros((graph.n_nodes, anchor_idx.size), dtype=np.float32)
    final_s = np.zeros((anchor_idx.size, anchor_idx.size), dtype=np.float32)
    anchor_joint = joint_features[anchor_idx]
    for feat in view_features:
        anchor_feat = feat[anchor_idx]
        final_z += _adaptive_anchor_affinity(feat, anchor_feat, neighbors=k)
        final_s += _anchor_graph(anchor_feat, neighbors=k)
    final_z = row_normalize_np(final_z / max(len(view_features), 1))
    final_s = final_s / max(len(view_features), 1)
    anchor_labels = _anchor_labels(final_s, anchor_joint, n_classes=n_classes, seed=seed, strict_components=strict_components)
    g_anchor = _one_hot(anchor_labels, n_classes)
    y_anchor = row_normalize_np(final_z @ g_anchor)
    return TeacherResult(
        final_z=final_z.astype(np.float32),
        g_anchor=g_anchor.astype(np.float32),
        y_anchor=y_anchor.astype(np.float32),
        y_hard=y_anchor.argmax(axis=1).astype(np.int64),
        confidence=y_anchor.max(axis=1).astype(np.float32),
        anchor_labels=anchor_labels.astype(np.int64),
        final_s=final_s.astype(np.float32),
        backend=backend,
    )


def save_teacher(teacher: TeacherResult, output_dir: str) -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "teacher.npz")
    np.savez_compressed(
        path,
        final_z=teacher.final_z,
        g_anchor=teacher.g_anchor,
        y_anchor=teacher.y_anchor,
        y_hard=teacher.y_hard,
        confidence=teacher.confidence,
        anchor_labels=teacher.anchor_labels,
        final_s=teacher.final_s,
        backend=np.asarray([teacher.backend]),
    )
    return path


def load_teacher(path: str) -> TeacherResult:
    data = np.load(path)
    backend = str(data["backend"][0]) if "backend" in data else "unknown"
    return TeacherResult(
        final_z=np.asarray(data["final_z"], dtype=np.float32),
        g_anchor=np.asarray(data["g_anchor"], dtype=np.float32),
        y_anchor=np.asarray(data["y_anchor"], dtype=np.float32),
        y_hard=np.asarray(data["y_hard"], dtype=np.int64),
        confidence=np.asarray(data["confidence"], dtype=np.float32),
        anchor_labels=np.asarray(data["anchor_labels"], dtype=np.int64),
        final_s=np.asarray(data["final_s"], dtype=np.float32),
        backend=backend,
    )
