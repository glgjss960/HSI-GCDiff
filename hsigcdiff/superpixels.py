from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import slic

from .data import MultiViewHSIData


@dataclass
class SuperpixelData:
    labels: np.ndarray
    association: np.ndarray
    centroids: np.ndarray
    area: np.ndarray
    gt: Optional[np.ndarray]
    n_classes: Optional[int]

    @property
    def n_superpixels(self) -> int:
        return int(self.association.shape[1])


def create_superpixels(image: np.ndarray, n_segments: int, compactness: float = 10.0) -> np.ndarray:
    try:
        labels = slic(
            image,
            n_segments=n_segments,
            compactness=compactness,
            convert2lab=False,
            enforce_connectivity=True,
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


def build_superpixels(hsi: MultiViewHSIData, config: Dict) -> SuperpixelData:
    seg_cfg = config.get("superpixel", {})
    source = int(seg_cfg.get("source_view", 0))
    image = hsi.views[source].image
    labels = create_superpixels(
        image,
        n_segments=int(seg_cfg.get("n_segments", 1000)),
        compactness=float(seg_cfg.get("compactness", 10.0)),
    )
    assoc = create_association(labels)

    props = regionprops(labels + 1)
    centroids = np.zeros((len(props), 2), dtype=np.float32)
    area = np.zeros((len(props), 1), dtype=np.float32)
    h, w = labels.shape
    for idx, prop in enumerate(props):
        centroids[idx] = prop.centroid
        area[idx, 0] = prop.area
    centroids[:, 0] /= max(h - 1, 1)
    centroids[:, 1] /= max(w - 1, 1)
    area /= max(float(area.max()), 1.0)
    return SuperpixelData(labels=labels, association=assoc, centroids=centroids, area=area, gt=hsi.gt, n_classes=hsi.n_classes)


def aggregate_mean_std(image: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
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
    return {"mean": means.astype(np.float32), "std": stds.astype(np.float32)}


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

