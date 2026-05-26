from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class HSIData:
    image: np.ndarray
    gt: Optional[np.ndarray]
    image_raw: np.ndarray
    gt_raw: Optional[np.ndarray]
    n_classes: Optional[int]


def _first_array_key(mat: dict, ndim: Optional[int] = None) -> str:
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if ndim is None:
        return keys[0]
    for key in keys:
        arr = mat[key]
        if isinstance(arr, np.ndarray) and arr.ndim == ndim:
            return key
    if not keys:
        raise ValueError("No ndarray key found in .mat file.")
    return keys[0]


def load_mat_array(path: str, key: Optional[str] = None, ndim: Optional[int] = None) -> np.ndarray:
    mat = sio.loadmat(path)
    selected = key or _first_array_key(mat, ndim=ndim)
    if selected not in mat:
        raise KeyError(f"Key '{selected}' not found in {path}. Available keys: {[k for k in mat if not k.startswith('__')]}")
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


def preprocess_hsi(
    image: np.ndarray,
    standardize: bool = True,
    pca_dim: Optional[int] = None,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected HSI image with shape [H, W, B], got {image.shape}")

    h, w, b = image.shape
    flat = image.reshape(-1, b)
    if standardize:
        flat = StandardScaler().fit_transform(flat).astype(np.float32)
    if pca_dim is not None and pca_dim > 0 and pca_dim < b:
        flat = PCA(n_components=pca_dim, random_state=0).fit_transform(flat).astype(np.float32)
    return flat.reshape(h, w, -1)


def load_hsi_data(
    image_path: str,
    gt_path: Optional[str] = None,
    image_key: Optional[str] = None,
    gt_key: Optional[str] = None,
    standardize: bool = True,
    pca_dim: Optional[int] = None,
) -> HSIData:
    image_raw = load_mat_array(image_path, key=image_key, ndim=3).astype(np.float32)
    gt_raw = None
    gt = None
    n_classes = None
    if gt_path:
        gt_raw = load_mat_array(gt_path, key=gt_key, ndim=2)
        gt = standardize_labels(gt_raw)
        n_classes = len(np.unique(gt)) - (1 if np.any(gt == 0) else 0)

    image = preprocess_hsi(image_raw, standardize=standardize, pca_dim=pca_dim)
    return HSIData(image=image, gt=gt, image_raw=image_raw, gt_raw=gt_raw, n_classes=n_classes)


def make_synthetic_hsi(
    height: int = 36,
    width: int = 36,
    bands: int = 16,
    classes: int = 4,
    noise: float = 0.08,
    seed: int = 0,
) -> HSIData:
    rng = np.random.default_rng(seed)
    gt = np.zeros((height, width), dtype=np.int64)
    image = np.zeros((height, width, bands), dtype=np.float32)
    prototypes = rng.normal(size=(classes, bands)).astype(np.float32)

    block_h = height // 2
    block_w = width // 2
    regions = [
        (slice(0, block_h), slice(0, block_w)),
        (slice(0, block_h), slice(block_w, width)),
        (slice(block_h, height), slice(0, block_w)),
        (slice(block_h, height), slice(block_w, width)),
    ]
    for idx, region in enumerate(regions[:classes]):
        cls = idx + 1
        gt[region] = cls
        image[region] = prototypes[idx] + noise * rng.normal(size=image[region].shape)

    return HSIData(
        image=preprocess_hsi(image, standardize=True, pca_dim=min(8, bands)),
        gt=gt,
        image_raw=image,
        gt_raw=gt,
        n_classes=classes,
    )

