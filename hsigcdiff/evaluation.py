from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


@dataclass
class EvalResult:
    acc: float
    kappa: float
    nmi: float
    ari: float
    purity: float
    class_acc: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return {
            "acc": float(self.acc),
            "kappa": float(self.kappa),
            "nmi": float(self.nmi),
            "ari": float(self.ari),
            "purity": float(self.purity),
        }


def best_map(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    n = max(len(true_labels), len(pred_labels))
    true_to_idx = {label: i for i, label in enumerate(true_labels)}
    pred_to_idx = {label: i for i, label in enumerate(pred_labels)}

    counts = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        counts[pred_to_idx[p], true_to_idx[t]] += 1
    row_ind, col_ind = linear_sum_assignment(-counts)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(pred_labels) and c < len(true_labels):
            mapping[pred_labels[r]] = true_labels[c]
    fallback = true_labels[0] if len(true_labels) else 0
    return np.asarray([mapping.get(p, fallback) for p in y_pred], dtype=np.int64)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    return float(np.sum(np.max(contingency, axis=0)) / np.sum(contingency))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = []
    for label in np.unique(y_true):
        mask = y_true == label
        out.append(metrics.accuracy_score(y_true[mask], y_pred[mask]))
    return np.asarray(out, dtype=np.float32)


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, EvalResult]:
    aligned = best_map(y_true, y_pred)
    result = EvalResult(
        acc=metrics.accuracy_score(y_true, aligned),
        kappa=metrics.cohen_kappa_score(y_true, aligned),
        nmi=metrics.normalized_mutual_info_score(y_true, y_pred),
        ari=metrics.adjusted_rand_score(y_true, aligned),
        purity=purity_score(y_true, aligned),
        class_acc=per_class_accuracy(y_true, aligned),
    )
    return aligned, result


def superpixel_to_pixel_labels(sp_labels: np.ndarray, association: np.ndarray) -> np.ndarray:
    return (association @ np.asarray(sp_labels).reshape(-1, 1)).reshape(-1).astype(np.int64)


def evaluate_pixel_clustering(
    superpixel_labels: np.ndarray,
    association: np.ndarray,
    gt: Optional[np.ndarray],
) -> Optional[EvalResult]:
    if gt is None:
        return None
    pixel_pred = superpixel_to_pixel_labels(superpixel_labels, association).reshape(gt.shape)
    mask = gt.reshape(-1) != 0
    if not np.any(mask):
        return None
    y_true = gt.reshape(-1)[mask]
    y_pred = pixel_pred.reshape(-1)[mask]
    _, result = cluster_accuracy(y_true, y_pred)
    return result

