import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch


def ensure_dir(path: str) -> str:
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def project_path(path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir or os.getcwd(), path))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def save_pickle(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def to_torch_sparse(matrix: sp.spmatrix, device: Optional[torch.device] = None) -> torch.Tensor:
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    tensor = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def row_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(x.sum(axis=1, keepdims=True), eps)


def sparse_row_normalize(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    rowsum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(rowsum, dtype=np.float32)
    np.divide(1.0, rowsum, out=inv, where=rowsum > 0)
    return sp.diags(inv).dot(matrix).tocsr()


def sparse_symmetric_normalize(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    rowsum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    np.divide(1.0, np.sqrt(rowsum), out=inv_sqrt, where=rowsum > 0)
    return sp.diags(inv_sqrt).dot(matrix).dot(sp.diags(inv_sqrt)).tocsr()


def add_self_loops(matrix: sp.spmatrix) -> sp.csr_matrix:
    return (matrix.tocsr() + sp.eye(matrix.shape[0], dtype=np.float32, format="csr")).tocsr()


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)

