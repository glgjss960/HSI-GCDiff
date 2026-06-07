from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score

from .graph_builder import GraphBundle


@dataclass
class EvalResult:
    acc: float
    kappa: float
    nmi: float
    ari: float
    purity: float
    oa: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _empty_result() -> EvalResult:
    return EvalResult(acc=float("nan"), kappa=float("nan"), nmi=float("nan"), ari=float("nan"), purity=float("nan"), oa=float("nan"))


def _hungarian_map(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    pred_ids = np.unique(y_pred)
    confusion = np.zeros((len(pred_ids), n_classes), dtype=np.int64)
    pred_to_row = {pred: idx for idx, pred in enumerate(pred_ids.tolist())}
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < n_classes:
            confusion[pred_to_row[pred], truth] += 1
    rows, cols = linear_sum_assignment(confusion.max() - confusion)
    mapping = {pred_ids[row]: col for row, col in zip(rows, cols)}
    fallback = int(np.bincount(y_true, minlength=n_classes).argmax())
    return np.asarray([mapping.get(pred, fallback) for pred in y_pred], dtype=np.int64)


def _purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    for pred in np.unique(y_pred):
        members = y_true[y_pred == pred]
        if members.size:
            total += int(np.bincount(members).max())
    return float(total / max(len(y_true), 1))


def evaluate_pixel_labels(pixel_labels: np.ndarray, gt: Optional[np.ndarray], n_classes: Optional[int] = None) -> EvalResult:
    if gt is None:
        return _empty_result()
    gt = np.asarray(gt).reshape(-1)
    pred = np.asarray(pixel_labels).reshape(-1).astype(np.int64)
    mask = gt > 0
    if not np.any(mask):
        return _empty_result()
    y_true = gt[mask].astype(np.int64) - 1
    y_pred = pred[mask]
    n_classes = int(n_classes or (y_true.max() + 1))
    mapped = _hungarian_map(y_true, y_pred, n_classes)
    acc = float(np.mean(mapped == y_true))
    return EvalResult(
        acc=acc,
        oa=acc,
        kappa=float(cohen_kappa_score(y_true, mapped)),
        nmi=float(normalized_mutual_info_score(y_true, y_pred)),
        ari=float(adjusted_rand_score(y_true, y_pred)),
        purity=_purity(y_true, y_pred),
    )


def superpixel_to_pixel_labels(graph: GraphBundle, sp_labels: np.ndarray) -> np.ndarray:
    sp_labels = np.asarray(sp_labels).reshape(-1).astype(np.int64)
    return sp_labels[graph.superpixels.labels.reshape(-1)]


def evaluate_superpixel_labels(graph: GraphBundle, sp_labels: np.ndarray) -> EvalResult:
    pixel_labels = superpixel_to_pixel_labels(graph, sp_labels)
    return evaluate_pixel_labels(pixel_labels, graph.superpixels.gt, graph.superpixels.n_classes)


def cluster_embedding(embedding: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    return KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(embedding)


def evaluate_embedding(graph: GraphBundle, embedding: np.ndarray, n_clusters: int, seed: int) -> EvalResult:
    labels = cluster_embedding(embedding, n_clusters=n_clusters, seed=seed)
    return evaluate_superpixel_labels(graph, labels)


def evaluate_assignment(graph: GraphBundle, assignment: np.ndarray) -> EvalResult:
    labels = np.asarray(assignment).argmax(axis=1)
    return evaluate_superpixel_labels(graph, labels)
