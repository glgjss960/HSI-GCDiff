import csv
import os
from typing import Dict, Iterable, List

import numpy as np
import torch

from .data import load_multiview_hsi, make_synthetic_multiview
from .diffgraph_diffusion import GraphLatentDiffusion, denoise_embeddings, load_denoiser, train_denoiser
from .encoder import train_encoder
from .etap_teacher import build_etap_teacher, load_teacher, save_teacher
from .evaluation import evaluate_assignment, evaluate_embedding, evaluate_superpixel_labels
from .graph_builder import GraphBundle, build_graph_bundle
from .superpixels import build_superpixels
from .target_builder import build_target, load_target, save_target
from .utils import ensure_dir, load_json, load_pickle, project_path, save_json, save_pickle, seed_everything


STAGES = ["build_graph", "train_encoder", "build_teacher", "build_target", "train_denoiser", "eval"]


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
    teacher = load_teacher(p["teacher"])
    target = build_target(embeddings, teacher, cfg)
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


def _eval_embedding_row(graph: GraphBundle, method: str, embedding: np.ndarray, n_clusters: int, seed: int, **extra) -> Dict:
    row = {"method": method, **extra}
    row.update(evaluate_embedding(graph, embedding, n_clusters=n_clusters, seed=seed).to_dict())
    return row


def _eval_noised_rows(
    graph: GraphBundle,
    embeddings: Dict[str, np.ndarray],
    cfg: Dict,
    t_values: Iterable[int],
    noise_seeds: Iterable[int],
    device: torch.device,
) -> List[Dict]:
    den_cfg = cfg.get("denoiser", {})
    source_key = den_cfg.get("source_key", "z_source")
    z_source = torch.from_numpy(np.asarray(embeddings[source_key], dtype=np.float32)).to(device)
    diffusion = GraphLatentDiffusion(
        timesteps=int(den_cfg.get("timesteps", 100)),
        schedule=den_cfg.get("schedule", "linear"),
        device=device,
    )
    n_clusters = int(cfg["n_clusters"])
    rows = []
    for noise_seed in noise_seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(noise_seed))
        for t_value in t_values:
            t_value = max(0, min(int(t_value), diffusion.timesteps - 1))
            t = torch.full((z_source.shape[0],), t_value, dtype=torch.long, device=device)
            if t_value == 0:
                noise = torch.zeros_like(z_source)
            else:
                noise = torch.randn(z_source.shape, generator=generator, device=device)
            with torch.no_grad():
                z_t = diffusion.q_sample(z_source, t, noise).detach().cpu().numpy().astype(np.float32)
            rows.append(
                _eval_embedding_row(
                    graph,
                    method="z_noised",
                    embedding=z_t,
                    n_clusters=n_clusters,
                    seed=int(cfg.get("seed", 0)),
                    t=t_value,
                    noise_seed=int(noise_seed),
                )
            )
    return rows


def stage_eval(cfg: Dict, device: torch.device) -> str:
    p = paths(cfg)
    graph: GraphBundle = load_pickle(p["graph"])
    embeddings = _load_npz_dict(p["embeddings"])
    teacher = load_teacher(p["teacher"])
    target = load_target(p["target"])
    n_clusters = int(cfg.get("n_clusters") or graph.superpixels.n_classes)
    seed = int(cfg.get("seed", 0))
    rows: List[Dict] = []

    row = {"method": "ETAP_hard", "t": "", "noise_seed": ""}
    row.update(evaluate_superpixel_labels(graph, teacher.y_hard).to_dict())
    rows.append(row)
    row = {"method": "Y_anchor", "t": "", "noise_seed": ""}
    row.update(evaluate_assignment(graph, teacher.y_anchor).to_dict())
    rows.append(row)
    for key in sorted([k for k in embeddings if k.startswith("z_view_")]):
        rows.append(_eval_embedding_row(graph, key, embeddings[key], n_clusters, seed, t="", noise_seed=""))
    rows.append(_eval_embedding_row(graph, "z_source", embeddings["z_source"], n_clusters, seed, t="", noise_seed=""))
    rows.append(_eval_embedding_row(graph, "z_fuse", embeddings["z_fuse"], n_clusters, seed, t="", noise_seed=""))
    rows.append(_eval_embedding_row(graph, "z_target", target.z_target, n_clusters, seed, t="", noise_seed=""))

    eval_cfg = cfg.get("eval", {})
    t_values = eval_cfg.get("timesteps", [0, 10, 25, 50])
    noise_seeds = eval_cfg.get("noise_seeds", [0, 1, 2, 3, 4])
    rows.extend(_eval_noised_rows(graph, embeddings, cfg, t_values, noise_seeds, device))
    if os.path.exists(p["denoiser"]):
        denoiser = load_denoiser(p["denoiser"], device=device)
        for noise_seed in noise_seeds:
            denoised = denoise_embeddings(denoiser, embeddings, cfg, t_values, int(noise_seed), device)
            for t_value, z in denoised.items():
                rows.append(
                    _eval_embedding_row(
                        graph,
                        "z_denoised",
                        z,
                        n_clusters,
                        seed,
                        t=t_value,
                        noise_seed=int(noise_seed),
                    )
                )

    source_acc = next((row["acc"] for row in rows if row["method"] == "z_source"), float("nan"))
    fuse_acc = next((row["acc"] for row in rows if row["method"] == "z_fuse"), float("nan"))
    for row in rows:
        row["gain_vs_source"] = row["acc"] - source_acc if np.isfinite(row["acc"]) else float("nan")
        row["gain_vs_fuse"] = row["acc"] - fuse_acc if np.isfinite(row["acc"]) else float("nan")

    ensure_dir(os.path.dirname(p["metrics"]))
    fieldnames = ["method", "t", "noise_seed", "acc", "oa", "kappa", "nmi", "ari", "purity", "gain_vs_source", "gain_vs_fuse"]
    with open(p["metrics"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return p["metrics"]


def run_protocol(config_path: str, stage: str = "all", device: torch.device = torch.device("cpu"), force: bool = False, output_dir: str = None) -> Dict[str, str]:
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
            raise ValueError(f"Unknown stage '{name}'. Valid stages: all, {', '.join(STAGES)}")
    return paths(cfg)
