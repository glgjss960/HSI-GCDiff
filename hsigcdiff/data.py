from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ViewData:
    name: str
    image: np.ndarray
    raw: np.ndarray


@dataclass
class MultiViewHSIData:
    views: List[ViewData]
    gt: Optional[np.ndarray]
    n_classes: Optional[int]


def mat_keys(path: str) -> List[str]:
    mat = sio.loadmat(path)
    return [k for k in mat.keys() if not k.startswith("__")]


def _first_array_key(mat: Dict, ndim: Optional[int] = None) -> str:
    keys = [k for k in mat.keys() if not k.startswith("__")]
    for key in keys:
        arr = mat[key]
        if isinstance(arr, np.ndarray) and (ndim is None or arr.ndim == ndim):
            return key
    if not keys:
        raise ValueError("No ndarray key found in .mat file.")
    return keys[0]


def load_mat_array(path: str, key: Optional[str] = None, ndim: Optional[int] = None) -> np.ndarray:
    mat = sio.loadmat(path)
    selected = key or _first_array_key(mat, ndim=ndim)
    if selected not in mat:
        raise KeyError(f"Key '{selected}' not found in {path}. Available keys: {mat_keys(path)}")
    return np.asarray(mat[selected])


def standardize_labels(gt: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt).squeeze()
    labels = np.unique(gt)
    out = np.zeros_like(gt, dtype=np.int64)
    next_label = 1
    for label in labels:
        if label == 0:
            continue
        out[gt == label] = next_label
        next_label += 1
    return out


def _as_hwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D image array, got shape {arr.shape}")
    return arr


def preprocess_image(image: np.ndarray, standardize: bool = True, pca_dim: Optional[int] = None) -> np.ndarray:
    image = _as_hwc(image)
    h, w, b = image.shape
    flat = image.reshape(-1, b)
    if standardize:
        flat = StandardScaler().fit_transform(flat).astype(np.float32)
    if pca_dim is not None and pca_dim > 0 and pca_dim < b:
        flat = PCA(n_components=pca_dim, random_state=0).fit_transform(flat).astype(np.float32)
    return flat.reshape(h, w, -1).astype(np.float32)


def load_multiview_hsi(config: Dict) -> MultiViewHSIData:
    views = []
    for idx, view_cfg in enumerate(config["views"]):
        raw = load_mat_array(view_cfg["path"], key=view_cfg.get("key"), ndim=view_cfg.get("ndim"))
        image = preprocess_image(
            raw,
            standardize=view_cfg.get("standardize", config.get("standardize", True)),
            pca_dim=view_cfg.get("pca_dim"),
        )
        views.append(ViewData(name=view_cfg.get("name", f"view{idx}"), image=image, raw=_as_hwc(raw)))

    if not views:
        raise ValueError("At least one data view is required.")
    hw = views[0].image.shape[:2]
    for view in views:
        if view.image.shape[:2] != hw:
            raise ValueError(f"All views must share H,W. {view.name} has {view.image.shape[:2]}, expected {hw}.")

    gt = None
    n_classes = None
    if config.get("gt_path"):
        gt = standardize_labels(load_mat_array(config["gt_path"], key=config.get("gt_key"), ndim=2))
    elif config.get("gt_paths"):
        gt_sum = None
        for gt_cfg in config["gt_paths"]:
            arr = load_mat_array(gt_cfg["path"], key=gt_cfg.get("key"), ndim=2)
            gt_sum = arr if gt_sum is None else gt_sum + arr
        gt = standardize_labels(gt_sum)
    if gt is not None:
        if gt.shape != hw:
            raise ValueError(f"GT shape {gt.shape} does not match image shape {hw}.")
        n_classes = len(np.unique(gt)) - (1 if np.any(gt == 0) else 0)

    return MultiViewHSIData(views=views, gt=gt, n_classes=n_classes)


def make_synthetic_multiview(height: int = 32, width: int = 32, classes: int = 4, seed: int = 0) -> MultiViewHSIData:
    rng = np.random.default_rng(seed)
    gt = np.zeros((height, width), dtype=np.int64)
    regions = [
        (slice(0, height // 2), slice(0, width // 2)),
        (slice(0, height // 2), slice(width // 2, width)),
        (slice(height // 2, height), slice(0, width // 2)),
        (slice(height // 2, height), slice(width // 2, width)),
    ]
    bands = [12, 5]
    images = [np.zeros((height, width, b), dtype=np.float32) for b in bands]
    protos = [rng.normal(size=(classes, b)).astype(np.float32) for b in bands]
    for cls, region in enumerate(regions[:classes]):
        gt[region] = cls + 1
        for v, image in enumerate(images):
            image[region] = protos[v][cls] + 0.08 * rng.normal(size=image[region].shape)
    views = [
        ViewData(name="spectral", image=preprocess_image(images[0], pca_dim=8), raw=images[0]),
        ViewData(name="aux", image=preprocess_image(images[1], pca_dim=None), raw=images[1]),
    ]
    return MultiViewHSIData(views=views, gt=gt, n_classes=classes)
