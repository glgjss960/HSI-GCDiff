import csv
import os
from typing import Dict, List

import numpy as np
import torch

from .data import load_multiview_hsi, make_synthetic_multiview
from .diffgraph_diffusion import load_denoiser, sample_denoised_aux, train_denoiser
from .encoder import train_encoder
from .etap_teacher import build_etap_teacher, load_teacher, save_teacher
from .evaluation import evaluate_embedding
from .graph_builder import GraphBundle, build_graph_bundle
from .injection import residual_inject
from .superpixels import build_superpixels
from .target_builder import build_target, load_target, save_target
from .utils import ensure_dir, load_json, load_pickle, project_path, save_json, save_pickle, seed_everything


STAGES = ["build_graph", "train_encoder", "build_target", "train_denoiser", "eval"]
OPTIONAL_STAGES = ["build_teacher"]


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def load_config(path: str, output_override: str = None, device_override: str = None) -> Dict:
    cfg = load_json(path)
    root = project_root()
    data_cfg = cfg.get("data", {})
    if not data_cfg.get("synthetic", False):
        for view in data_cfg.get("views", []):
            view["path"] = project_path(view["path"], root)
        if data_cfg.get("gt_path"):
            data_cfg["gt_path"] = project_path(data_cfg["gt_path"], root)
        for gt_cfg in data_cfg.get("gt_paths", []):
            gt_cfg["path"] = project_path(gt_cfg["path"], root)
    if cfg.get("teacher", {}).get("backend") == "load_npz":
        cfg["teacher"]["path"] = project_path(cfg["teacher"]["path"], root)
    run_cfg = cfg.setdefault("run", {})
    if output_override:
        run_cfg["output_dir"] = output_override
    output_dir = run_cfg.get("output_dir", os.path.join("runs", cfg.get("name", "proof")))
    run_cfg["output_dir"] = project_path(output_dir, root)
    if device_override:
        cfg["device"] = device_override
    return cfg


def paths(cfg: Dict) -> Dict[str, str]:
    out = cfg["run"]["output_dir"]
    return {
        "run": out,
        "graph": os.path.join(out, "graph.pkl"),
        "graph_meta": os.path.join(out, "graph_meta.json"),
        "encoder_dir": os.path.join(out, "encoder"),
        "embeddings": os.path.join(out, "encoder", "embeddings.npz"),
        "teacher_dir": os.path.join(out, "teacher"),
        "teacher": os.path.join(out, "teacher", "teacher.npz"),
        "target_dir": os.path.join(out, "target"),
        "target": os.path.join(out, "target", "target.npz"),
        "denoiser_dir": os.path.join(out, "denoiser"),
        "denoiser": os.path.join(out, "denoiser", "denoiser.pt"),
        "metrics": os.path.join(out, "metrics.csv"),
        "effective_config": os.path.join(out, "config.effective.json"),
    }


def _load_data(cfg: Dict):
    data_cfg = cfg.get("data", {})
    if data_cfg.get("synthetic", False):
        return make_synthetic_multiview(
            height=int(data_cfg.get("height", 32)),
            width=int(data_cfg.get("width", 32)),
            classes=int(data_cfg.get("classes", cfg.get("n_clusters", 4))),
            seed=int(cfg.get("seed", 0)),
        )
    return load_multiview_hsi(data_cfg)


def _load_npz_dict(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def stage_build_graph(cfg: Dict, force: bool = False) -> str:
    p = paths(cfg)
    if os.path.exists(p["graph"]) and not force:
        return p["graph"]
    hsi = _load_data(cfg)
    if not cfg.get("n_clusters") and hsi.n_classes:
        cfg["n_clusters"] = int(hsi.n_classes)
        save_json(cfg, p["effective_config"])
    superpixels = build_superpixels(hsi, cfg)
    graph = build_graph_bundle(hsi, superpixels, cfg.get("graph", {}))
    ensure_dir(p["run"])
    save_pickle(graph, p["graph"])
    save_json(
        {
            "n_nodes": graph.n_nodes,
            "n_views": graph.n_views,
            "view_names": [view.name for view in graph.views],
            "roles": [view.role for view in graph.views],
            "task_view_names": [view.name for view in graph.task_views],
            "aux_view_names": [view.name for view in graph.aux_views],
            "feature_dims": [int(view.features.shape[1]) for view in graph.views],
            "n_classes": graph.superpixels.n_classes,
            "actual_superpixels": graph.superpixels.n_superpixels,
        },
        p["graph_meta"],
    )
    return p["graph"]


def stage_train_encoder(cfg: Dict, device: torch.device, force: bool = False) -> str:
    p = paths(cfg)
    if os.path.exists(p["embeddings"]) and not force:
        return p["embeddings"]
    graph: GraphBundle = load_pickle(p["graph"])
    train_encoder(graph, cfg, p["encoder_dir"], device=device)
    return p["embeddings"]


def stage_build_teacher(cfg: Dict, force: bool = False) -> str:
    p = paths(cfg)
    if os.path.exists(p["teacher"]) and not force:
        return p["teacher"]
    graph: GraphBundle = load_pickle(p["graph"])
    teacher = build_etap_teacher(graph, cfg, seed=int(cfg.get("seed", 0)))
    save_teacher(teacher, p["teacher_dir"])
    return p["teacher"]


def stage_build_target(cfg: Dict, force: bool = False) -> str:
    p = paths(cfg)
    if os.path.exists(p["target"]) and not force:
        return p["target"]
    embeddings = _load_npz_dict(p["embeddings"])
    teacher = load_teacher(p["teacher"]) if cfg.get("target", {}).get("mode") == "anchor_proto" else None
    target = build_target(embeddings, cfg, teacher=teacher)
    save_target(target, p["target_dir"])
    return p["target"]


def stage_train_denoiser(cfg: Dict, device: torch.device, force: bool = False) -> str:
    p = paths(cfg)
    if os.path.exists(p["denoiser"]) and not force:
        return p["denoiser"]
    embeddings = _load_npz_dict(p["embeddings"])
    target = _load_npz_dict(p["target"])
    train_denoiser(embeddings, target, cfg, p["denoiser_dir"], device=device)
    return p["denoiser"]


def _eval_embedding_row(
    graph: GraphBundle,
    method: str,
    embedding: np.ndarray,
    n_clusters: int,
    seed: int,
    **extra,
) -> Dict:
    row = {"method": method, **extra}
    row.update(evaluate_embedding(graph, embedding, n_clusters=n_clusters, seed=seed).to_dict())
    return row


def stage_eval(cfg: Dict, device: torch.device) -> str:
    p = paths(cfg)
    graph: GraphBundle = load_pickle(p["graph"])
    embeddings = _load_npz_dict(p["embeddings"])
    target = load_target(p["target"])
    n_clusters = int(cfg.get("n_clusters") or graph.superpixels.n_classes)
    seed = int(cfg.get("seed", 0))
    eval_cfg = cfg.get("eval", {})
    alphas = [float(a) for a in cfg.get("injection", {}).get("alphas", eval_cfg.get("alphas", [0, 0.05, 0.1, 0.2, 0.5, 1.0]))]
    sampling_steps = [int(s) for s in eval_cfg.get("sampling_steps", [0, 10, 20])]
    noise_seeds = [int(s) for s in eval_cfg.get("noise_seeds", [0, 1, 2, 3, 4])]
    z_task = np.asarray(embeddings["z_task"], dtype=np.float32)
    z_aux = np.asarray(embeddings["z_aux"], dtype=np.float32)

    rows: List[Dict] = []
    rows.append(_eval_embedding_row(graph, "z_task", z_task, n_clusters, seed, alpha="", sampling_steps="", noise_seed=""))
    rows.append(_eval_embedding_row(graph, "z_aux", z_aux, n_clusters, seed, alpha="", sampling_steps="", noise_seed=""))
    rows.append(_eval_embedding_row(graph, "z_target", target.z_target, n_clusters, seed, alpha="", sampling_steps="", noise_seed=""))
    for key in sorted(k for k in embeddings if k.startswith("z_task_view_")):
        rows.append(_eval_embedding_row(graph, key, embeddings[key], n_clusters, seed, alpha="", sampling_steps="", noise_seed=""))
    for key in sorted(k for k in embeddings if k.startswith("z_aux_view_")):
        rows.append(_eval_embedding_row(graph, key, embeddings[key], n_clusters, seed, alpha="", sampling_steps="", noise_seed=""))

    raw_acc_by_alpha: Dict[float, float] = {}
    for alpha in alphas:
        z_raw = residual_inject(z_task, z_aux, alpha)
        row = _eval_embedding_row(graph, "z_task_plus_raw_aux", z_raw, n_clusters, seed, alpha=alpha, sampling_steps="", noise_seed="")
        raw_acc_by_alpha[alpha] = row["acc"]
        rows.append(row)

    if os.path.exists(p["denoiser"]):
        denoiser = load_denoiser(p["denoiser"], device=device)
        for noise_seed in noise_seeds:
            denoised_by_steps = sample_denoised_aux(denoiser, embeddings, cfg, sampling_steps, noise_seed, device)
            for steps, z_diff in denoised_by_steps.items():
                rows.append(
                    _eval_embedding_row(
                        graph,
                        "z_diff",
                        z_diff,
                        n_clusters,
                        seed,
                        alpha="",
                        sampling_steps=steps,
                        noise_seed=noise_seed,
                    )
                )
                for alpha in alphas:
                    z_final = residual_inject(z_task, z_diff, alpha)
                    rows.append(
                        _eval_embedding_row(
                            graph,
                            "z_task_plus_denoised_aux",
                            z_final,
                            n_clusters,
                            seed,
                            alpha=alpha,
                            sampling_steps=steps,
                            noise_seed=noise_seed,
                        )
                    )

    task_acc = next((row["acc"] for row in rows if row["method"] == "z_task"), float("nan"))
    aux_acc = next((row["acc"] for row in rows if row["method"] == "z_aux"), float("nan"))
    for row in rows:
        row["gain_vs_task"] = row["acc"] - task_acc if np.isfinite(row["acc"]) else float("nan")
        row["gain_vs_aux"] = row["acc"] - aux_acc if np.isfinite(row["acc"]) else float("nan")
        if row["method"] == "z_task_plus_denoised_aux" and row["alpha"] in raw_acc_by_alpha:
            row["gain_vs_raw_aux_injection"] = row["acc"] - raw_acc_by_alpha[float(row["alpha"])]
        else:
            row["gain_vs_raw_aux_injection"] = ""

    ensure_dir(os.path.dirname(p["metrics"]))
    fieldnames = [
        "method",
        "alpha",
        "sampling_steps",
        "noise_seed",
        "acc",
        "oa",
        "kappa",
        "nmi",
        "ari",
        "purity",
        "gain_vs_task",
        "gain_vs_aux",
        "gain_vs_raw_aux_injection",
    ]
    with open(p["metrics"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return p["metrics"]


def run_protocol(
    config_path: str,
    stage: str = "all",
    device: torch.device = torch.device("cpu"),
    force: bool = False,
    output_dir: str = None,
) -> Dict[str, str]:
    cfg = load_config(config_path, output_override=output_dir, device_override=str(device))
    seed_everything(int(cfg.get("seed", 0)))
    ensure_dir(paths(cfg)["run"])
    save_json(cfg, paths(cfg)["effective_config"])
    selected = STAGES if stage == "all" else [stage]
    for name in selected:
        if name == "build_graph":
            stage_build_graph(cfg, force=force)
        elif name == "train_encoder":
            stage_train_encoder(cfg, device=device, force=force)
        elif name == "build_teacher":
            stage_build_teacher(cfg, force=force)
        elif name == "build_target":
            stage_build_target(cfg, force=force)
        elif name == "train_denoiser":
            stage_train_denoiser(cfg, device=device, force=force)
        elif name == "eval":
            stage_eval(cfg, device=device)
        else:
            valid = ", ".join(["all"] + STAGES + OPTIONAL_STAGES)
            raise ValueError(f"Unknown stage '{name}'. Valid stages: {valid}")
    return paths(cfg)
