from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from skimage.measure import regionprops
from skimage.segmentation import slic
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import minmax_scale

from .utils import add_self_loops, sparse_symmetric_normalize


@dataclass
class GraphData:
    image: np.ndarray
    gt: Optional[np.ndarray]
    superpixel_map: np.ndarray
    active_superpixels: np.ndarray
    association_full: np.ndarray
    association_active: np.ndarray
    spectral_features: np.ndarray
    context_features: np.ndarray
    node_features: np.ndarray
    graphs: Dict[str, sp.csr_matrix]
    n_clusters: Optional[int]

    def recover_full_superpixel_labels(self, active_labels: np.ndarray) -> np.ndarray:
        full = np.zeros(self.association_full.shape[1], dtype=np.int64)
        full[self.active_superpixels] = np.asarray(active_labels, dtype=np.int64) + 1
        return full


def create_superpixels(
    image: np.ndarray,
    n_segments: int,
    compactness: float = 10.0,
) -> np.ndarray:
    try:
        labels = slic(
            image,
            n_segments=n_segments,
            compactness=compactness,
            convert2lab=False,
            enforce_connectivity=True,
            min_size_factor=0.3,
            max_size_factor=2.0,
            start_label=0,
            channel_axis=-1,
        )
    except TypeError:
        labels = slic(
            image,
            n_segments=n_segments,
            compactness=compactness,
            convert2lab=False,
            enforce_connectivity=True,
            min_size_factor=0.3,
            max_size_factor=2.0,
            start_label=0,
        )
    return labels.astype(np.int64)


def create_association(labels: np.ndarray) -> np.ndarray:
    flat = labels.reshape(-1)
    n_pixels = flat.size
    n_sp = int(flat.max()) + 1
    assoc = np.zeros((n_pixels, n_sp), dtype=np.float32)
    assoc[np.arange(n_pixels), flat] = 1.0
    return assoc


def _safe_minmax(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[:, None]
    out = features.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        denom = col.max() - col.min()
        if denom > 0:
            out[:, j] = (col - col.min()) / denom
        else:
            out[:, j] = 0.0
    return out


def aggregate_superpixel_features(image: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    h, w, bands = image.shape
    flat = image.reshape(-1, bands)
    flat_labels = labels.reshape(-1)
    n_sp = int(flat_labels.max()) + 1

    counts = np.bincount(flat_labels, minlength=n_sp).astype(np.float32)
    counts[counts == 0] = 1.0

    sums = np.zeros((n_sp, bands), dtype=np.float32)
    np.add.at(sums, flat_labels, flat)
    means = sums / counts[:, None]

    sq_sums = np.zeros((n_sp, bands), dtype=np.float32)
    np.add.at(sq_sums, flat_labels, flat * flat)
    stds = np.sqrt(np.maximum(sq_sums / counts[:, None] - means * means, 0.0))

    props = regionprops(labels + 1)
    centroids = np.zeros((n_sp, 2), dtype=np.float32)
    area = np.zeros((n_sp, 1), dtype=np.float32)
    eccentricity = np.zeros((n_sp, 1), dtype=np.float32)
    for idx, prop in enumerate(props):
        centroids[idx] = prop.centroid
        area[idx, 0] = prop.area
        eccentricity[idx, 0] = prop.eccentricity

    centroids[:, 0] = centroids[:, 0] / max(h - 1, 1)
    centroids[:, 1] = centroids[:, 1] / max(w - 1, 1)
    geometry = np.concatenate([centroids, _safe_minmax(area), eccentricity], axis=1)

    spectral = np.concatenate([means, stds], axis=1).astype(np.float32)
    node = np.concatenate([spectral, geometry], axis=1).astype(np.float32)
    return {
        "mean": means.astype(np.float32),
        "std": stds.astype(np.float32),
        "spectral": spectral,
        "geometry": geometry.astype(np.float32),
        "node": node,
        "centroids": centroids,
    }


def extract_center_patches(image: np.ndarray, labels: np.ndarray, patch_size: int) -> np.ndarray:
    if patch_size % 2 != 1:
        raise ValueError("patch_size must be odd.")
    pad = patch_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    props = regionprops(labels + 1)
    patches = np.zeros((len(props), patch_size * patch_size * image.shape[-1]), dtype=np.float32)
    for idx, prop in enumerate(props):
        r, c = prop.centroid
        r = int(round(r)) + pad
        c = int(round(c)) + pad
        patch = padded[r - pad : r + pad + 1, c - pad : c + pad + 1, :]
        patches[idx] = patch.reshape(-1)
    return patches


def build_knn_graph(features: np.ndarray, n_neighbors: int) -> sp.csr_matrix:
    n = features.shape[0]
    if n <= 1:
        return sp.eye(n, dtype=np.float32, format="csr")
    k = max(1, min(int(n_neighbors), n - 1))
    feats = _safe_minmax(features)
    adj = kneighbors_graph(feats, n_neighbors=k, mode="distance", include_self=False).tocsr()
    data = adj.data
    if data.size:
        var = float(np.var(feats))
        gamma = 1.0 / (feats.shape[1] * var) if var > 0 else 1.0
        adj.data = np.exp(-(data ** 2) * gamma).astype(np.float32)
    adj = adj.maximum(adj.T)
    adj = add_self_loops(adj)
    return sparse_symmetric_normalize(adj)


def build_spatial_adjacency(labels: np.ndarray) -> sp.csr_matrix:
    n_sp = int(labels.max()) + 1
    rows = []
    cols = []
    right_a = labels[:, :-1].reshape(-1)
    right_b = labels[:, 1:].reshape(-1)
    down_a = labels[:-1, :].reshape(-1)
    down_b = labels[1:, :].reshape(-1)
    for a, b in zip(np.concatenate([right_a, down_a]), np.concatenate([right_b, down_b])):
        if a != b:
            rows.extend([int(a), int(b)])
            cols.extend([int(b), int(a)])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_sp, n_sp), dtype=np.float32).tocsr()
    adj.data[:] = 1.0
    adj = add_self_loops(adj)
    return sparse_symmetric_normalize(adj)


def build_centroid_graph(centroids: np.ndarray, n_neighbors: int) -> sp.csr_matrix:
    return build_knn_graph(centroids, n_neighbors=n_neighbors)


def majority_superpixel_labels(labels: np.ndarray, gt: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if gt is None:
        return None
    n_sp = int(labels.max()) + 1
    out = np.zeros(n_sp, dtype=np.int64)
    for sp_id in range(n_sp):
        values = gt[labels == sp_id]
        values = values[values != 0]
        if values.size == 0:
            out[sp_id] = 0
            continue
        counts = np.bincount(values.astype(np.int64))
        out[sp_id] = counts.argmax()
    return out


def prepare_graph_data(
    image: np.ndarray,
    gt: Optional[np.ndarray],
    n_segments: int,
    compactness: float,
    patch_size: int,
    spectral_neighbors: int,
    context_neighbors: int,
    remove_background: bool = True,
) -> GraphData:
    sp_map = create_superpixels(image, n_segments=n_segments, compactness=compactness)
    assoc_full = create_association(sp_map)
    features = aggregate_superpixel_features(image, sp_map)
    context = extract_center_patches(image, sp_map, patch_size=patch_size)
    spatial = build_spatial_adjacency(sp_map)
    centroid_graph = build_centroid_graph(features["centroids"], n_neighbors=context_neighbors)

    n_sp = assoc_full.shape[1]
    active = np.arange(n_sp)
    majority = majority_superpixel_labels(sp_map, gt)
    if remove_background and majority is not None:
        active = np.where(majority != 0)[0]
        if active.size == 0:
            active = np.arange(n_sp)

    spectral_features = features["spectral"][active]
    context_features = context[active]
    node_features = features["node"][active]

    spectral = build_knn_graph(features["mean"][active], n_neighbors=spectral_neighbors)
    context_graph = build_knn_graph(context_features, n_neighbors=context_neighbors)
    spatial_active = spatial[active, :][:, active].tocsr()
    centroid_active = centroid_graph[active, :][:, active].tocsr()
    if spatial_active.nnz == spatial_active.shape[0]:
        spatial_active = centroid_active
    spatial_active = sparse_symmetric_normalize(add_self_loops(spatial_active))

    assoc_active = assoc_full[:, active]
    n_clusters = None
    if gt is not None:
        n_clusters = len(np.unique(gt)) - (1 if np.any(gt == 0) else 0)

    return GraphData(
        image=image,
        gt=gt,
        superpixel_map=sp_map,
        active_superpixels=active,
        association_full=assoc_full,
        association_active=assoc_active,
        spectral_features=spectral_features.astype(np.float32),
        context_features=context_features.astype(np.float32),
        node_features=node_features.astype(np.float32),
        graphs={
            "spectral": spectral,
            "spatial": spatial_active,
            "context": context_graph,
        },
        n_clusters=n_clusters,
    )

