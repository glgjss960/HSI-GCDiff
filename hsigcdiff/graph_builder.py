from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from .data import MultiViewHSIData
from .superpixels import SuperpixelData, aggregate_mean_std, extract_center_patches
from .utils import add_self_loops, sparse_symmetric_normalize


@dataclass
class GraphView:
    name: str
    features: np.ndarray
    adjacency: sp.csr_matrix


@dataclass
class GraphBundle:
    views: List[GraphView]
    superpixels: SuperpixelData

    @property
    def n_nodes(self) -> int:
        return self.superpixels.n_superpixels

    @property
    def n_views(self) -> int:
        return len(self.views)


def _standardize(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(x, dtype=np.float32)).astype(np.float32)


def _knn_graph(features: np.ndarray, neighbors: int) -> sp.csr_matrix:
    n = features.shape[0]
    if n <= 1:
        return sp.eye(n, dtype=np.float32, format="csr")
    k = max(1, min(int(neighbors), n - 1))
    feats = _standardize(features)
    adj = kneighbors_graph(feats, n_neighbors=k, mode="distance", include_self=False).tocsr()
    if adj.data.size:
        sigma = float(np.median(adj.data))
        if sigma <= 1e-8:
            sigma = 1.0
        adj.data = np.exp(-(adj.data ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    adj = adj.maximum(adj.T)
    return sparse_symmetric_normalize(add_self_loops(adj))


def _spatial_adjacency(labels: np.ndarray) -> sp.csr_matrix:
    n_sp = int(labels.max()) + 1
    rows, cols = [], []
    pairs = [
        (labels[:, :-1].reshape(-1), labels[:, 1:].reshape(-1)),
        (labels[:-1, :].reshape(-1), labels[1:, :].reshape(-1)),
    ]
    for left, right in pairs:
        diff = left != right
        a = left[diff].astype(np.int64)
        b = right[diff].astype(np.int64)
        rows.extend(a.tolist())
        cols.extend(b.tolist())
        rows.extend(b.tolist())
        cols.extend(a.tolist())
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_sp, n_sp), dtype=np.float32).tocsr()
    adj.data[:] = 1.0
    return sparse_symmetric_normalize(add_self_loops(adj))


def build_graph_bundle(hsi: MultiViewHSIData, superpixels: SuperpixelData, config: Dict) -> GraphBundle:
    graph_cfgs = config.get("graph_views")
    if graph_cfgs is None:
        graph_cfgs = []
        for idx, view in enumerate(hsi.views):
            graph_cfgs.append({"name": view.name, "source_view": idx, "mode": "mean_std_geo"})

    views = []
    for idx, cfg in enumerate(graph_cfgs):
        source = int(cfg.get("source_view", 0))
        image = hsi.views[source].image
        mode = cfg.get("mode", "mean_std_geo")
        stats = aggregate_mean_std(image, superpixels.labels)
        if mode == "mean":
            feat = stats["mean"]
        elif mode == "mean_std":
            feat = np.concatenate([stats["mean"], stats["std"]], axis=1)
        elif mode == "center_patch":
            feat = extract_center_patches(image, superpixels.labels, patch_size=int(cfg.get("patch_size", 7)))
        elif mode == "geometry":
            feat = np.concatenate([superpixels.centroids, superpixels.area], axis=1)
        else:
            feat = np.concatenate([stats["mean"], stats["std"], superpixels.centroids, superpixels.area], axis=1)
        feat = _standardize(feat)

        if cfg.get("graph", "knn") == "spatial":
            adj = _spatial_adjacency(superpixels.labels)
        else:
            adj = _knn_graph(feat, int(cfg.get("neighbors", config.get("neighbors", 20))))
        views.append(GraphView(name=cfg.get("name", f"view{idx}"), features=feat, adjacency=adj))

    if config.get("include_spatial_graph", False):
        feat = _standardize(np.concatenate([superpixels.centroids, superpixels.area], axis=1))
        views.append(GraphView(name="spatial", features=feat, adjacency=_spatial_adjacency(superpixels.labels)))
    return GraphBundle(views=views, superpixels=superpixels)

