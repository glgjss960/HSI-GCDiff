import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def to_torch_sparse(matrix: sp.spmatrix, device: Optional[torch.device] = None) -> torch.Tensor:
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data)
    tensor = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


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


def project_path(path: str, base_dir: Optional[str] = None) -> str:
    if os.path.isabs(path):
        return path
    base = base_dir or os.getcwd()
    return os.path.abspath(os.path.join(base, path))

