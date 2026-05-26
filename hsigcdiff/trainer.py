import argparse
import csv
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from .clustering import (
    confidence_from_centers,
    prototype_corrected_target,
    prototypes_from_labels,
    run_kmeans,
)
from .data import HSIData, load_hsi_data
from .evaluation import EvalResult, evaluate_pixel_clustering
from .losses import (
    clustering_kl_loss,
    confidence_weighted_mse,
    hard_sample_contrastive_loss,
    reconstruction_loss,
    view_consistency_loss,
)
from .model import HSIGCDiffModel
from .superpixels import GraphData, prepare_graph_data
from .utils import ensure_dir, load_json, project_path, resolve_device, save_json, seed_everything, to_torch_sparse


def _standardize_features(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(x).astype(np.float32)


def _metrics_row(epoch: int, result: Optional[EvalResult], losses: Dict[str, float]) -> Dict[str, float]:
    row: Dict[str, float] = {"epoch": epoch}
    row.update(losses)
    if result is not None:
        row.update(result.as_dict())
    return row


def _print_metrics(epoch: int, losses: Dict[str, float], result: Optional[EvalResult]) -> None:
    loss_text = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
    if result is None:
        print(f"Epoch {epoch:03d}: {loss_text}")
    else:
        print(
            f"Epoch {epoch:03d}: {loss_text}, "
            f"ACC={result.acc:.4f}, Kappa={result.kappa:.4f}, NMI={result.nmi:.4f}, "
            f"ARI={result.ari:.4f}, Purity={result.purity:.4f}"
        )


def _write_history(path: str, rows: list) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if "epoch" in fieldnames:
        fieldnames.remove("epoch")
        fieldnames = ["epoch"] + fieldnames
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_graph_data(config: Dict[str, Any], config_dir: str) -> GraphData:
    data_cfg = config["data"]
    sp_cfg = config["superpixel"]
    image_path = project_path(data_cfg["image_path"], base_dir=config_dir)
    gt_path = data_cfg.get("gt_path")
    gt_path = project_path(gt_path, base_dir=config_dir) if gt_path else None

    hsi = load_hsi_data(
        image_path=image_path,
        gt_path=gt_path,
        image_key=data_cfg.get("image_key"),
        gt_key=data_cfg.get("gt_key"),
        standardize=data_cfg.get("standardize", True),
        pca_dim=data_cfg.get("pca_dim"),
    )
    return graph_data_from_hsi(hsi, config)


def graph_data_from_hsi(hsi: HSIData, config: Dict[str, Any]) -> GraphData:
    sp_cfg = config["superpixel"]
    data_cfg = config.get("data", {})
    return prepare_graph_data(
        image=hsi.image,
        gt=hsi.gt,
        n_segments=sp_cfg["n_segments"],
        compactness=sp_cfg.get("compactness", 10.0),
        patch_size=sp_cfg.get("patch_size", 7),
        spectral_neighbors=sp_cfg.get("spectral_neighbors", 50),
        context_neighbors=sp_cfg.get("context_neighbors", 50),
        remove_background=data_cfg.get("remove_background", True),
    )


def train_graph_data(
    graph_data: GraphData,
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Tuple[Optional[EvalResult], Dict[str, Any]]:
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(config.get("device", "cuda:0"))
    output_dir = output_dir or config.get("output", {}).get("dir", "runs/default")
    ensure_dir(output_dir)

    train_cfg = config["train"]
    model_cfg = config["model"]
    diff_cfg = config["diffusion"]

    n_clusters = config.get("n_clusters") or graph_data.n_clusters
    if n_clusters is None:
        raise ValueError("n_clusters is required when no ground-truth label file is provided.")
    n_clusters = int(n_clusters)

    node_np = _standardize_features(graph_data.node_features)
    context_np = _standardize_features(graph_data.context_features)

    node_features = torch.from_numpy(node_np).to(device)
    context_features = torch.from_numpy(context_np).to(device)
    adjs = {name: to_torch_sparse(adj, device=device) for name, adj in graph_data.graphs.items()}

    model = HSIGCDiffModel(
        node_dim=node_features.shape[1],
        context_dim=context_features.shape[1],
        hidden_dim=model_cfg.get("hidden_dim", 256),
        latent_dim=model_cfg.get("latent_dim", 128),
        dropout=model_cfg.get("dropout", 0.2),
        diffusion_timesteps=diff_cfg.get("timesteps", 100),
        diffusion_schedule=diff_cfg.get("schedule", "cosine"),
        diffusion_time_emb_dim=diff_cfg.get("time_emb_dim", 64),
        diffusion_hidden_dim=diff_cfg.get("hidden_dim", 512),
        offset_noise=diff_cfg.get("offset_noise", 0.0),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 5e-4),
    )

    teacher_z = None
    labels_t = None
    centers_t = None
    confidence_t = None
    best_result: Optional[EvalResult] = None
    best_score = -1.0
    history = []
    ema_decay = train_cfg.get("ema_decay", 0.99)
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    cluster_interval = train_cfg.get("cluster_interval", 1)
    sample_steps = diff_cfg.get("sample_steps", 0)

    iterator = trange(1, train_cfg.get("epochs", 100) + 1, desc="Training", leave=True)
    for epoch in iterator:
        model.train()
        outputs = model(node_features, context_features, adjs)

        with torch.no_grad():
            current = outputs["z_fuse"].detach()
            teacher_z = current if teacher_z is None else F.normalize(ema_decay * teacher_z + (1.0 - ema_decay) * current, dim=-1)
            if labels_t is None or epoch % cluster_interval == 0:
                labels_np, centers_np = run_kmeans(teacher_z.detach().cpu().numpy(), n_clusters, seed=seed)
                labels_t = torch.from_numpy(labels_np).long().to(device)
                centers_t = torch.from_numpy(centers_np).float().to(device)
                centers_t = F.normalize(centers_t, dim=-1)
                confidence_t = confidence_from_centers(teacher_z, centers_t)

        prototypes = prototypes_from_labels(
            teacher_z,
            labels_t,
            n_clusters=n_clusters,
            confidence=confidence_t,
            tau=train_cfg.get("tau_confidence", 0.7),
            fallback_centers=centers_t,
        )
        target_z = prototype_corrected_target(
            teacher_z,
            labels_t,
            prototypes,
            momentum=diff_cfg.get("prototype_momentum", 0.5),
        )

        noise_pred, noise, z0_hat, t = model.diffusion.training_step(
            model.denoiser,
            outputs["z_source"],
            outputs["z_fuse"],
        )
        diff_loss = F.mse_loss(noise_pred, noise)
        align_loss = confidence_weighted_mse(F.normalize(z0_hat, dim=-1), target_z, confidence_t)

        if train_cfg.get("lambda_consistency", 0.0) > 0:
            t_prev = torch.clamp(t - 1, min=0)
            z_t_prev = model.diffusion.q_sample(outputs["z_source"], t_prev, noise=noise)
            noise_pred_prev = model.denoiser(z_t_prev, t_prev, outputs["z_fuse"])
            consistency_loss = F.mse_loss(noise_pred, noise_pred_prev)
        else:
            consistency_loss = node_features.sum() * 0.0

        z_clean = F.normalize(0.5 * outputs["z_fuse"] + 0.5 * z0_hat, dim=-1)

        recon_loss = reconstruction_loss(node_features, context_features, outputs)
        cluster_loss = clustering_kl_loss(z_clean, centers_t)
        hard_loss = hard_sample_contrastive_loss(
            outputs["z_spectral"],
            outputs["z_context"],
            labels_t,
            confidence_t,
            temperature=train_cfg.get("hard_temperature", 0.5),
            beta=train_cfg.get("hard_beta", 3.0),
        )
        view_loss = view_consistency_loss(outputs)

        diffusion_weight = 0.0 if epoch <= warmup_epochs else 1.0
        total = (
            train_cfg.get("lambda_recon", 1e-4) * recon_loss
            + train_cfg.get("lambda_cluster", 1.0) * cluster_loss
            + train_cfg.get("lambda_hard", 1.0) * hard_loss
            + diffusion_weight * train_cfg.get("lambda_diff", 1.0) * diff_loss
            + diffusion_weight * train_cfg.get("lambda_align", 1.0) * align_loss
            + diffusion_weight * train_cfg.get("lambda_consistency", 0.0) * consistency_loss
            + train_cfg.get("lambda_view", 0.0) * view_loss
        )

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        losses = {
            "loss": float(total.detach().cpu()),
            "recon": float(recon_loss.detach().cpu()),
            "cluster": float(cluster_loss.detach().cpu()),
            "hard": float(hard_loss.detach().cpu()),
            "diff": float(diff_loss.detach().cpu()),
            "align": float(align_loss.detach().cpu()),
            "cons": float(consistency_loss.detach().cpu()),
            "view": float(view_loss.detach().cpu()),
        }

        result = None
        if epoch % train_cfg.get("eval_interval", 1) == 0:
            result = evaluate_model(model, graph_data, node_features, context_features, adjs, n_clusters, sample_steps, seed)
            if result is not None and result.acc > best_score:
                best_score = result.acc
                best_result = result
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "metrics": result.as_dict(),
                    },
                    os.path.join(output_dir, "best_model.pt"),
                )

        history.append(_metrics_row(epoch, result, losses))
        iterator.set_postfix(loss=f"{losses['loss']:.4f}", acc=f"{result.acc:.4f}" if result else "NA")
        _print_metrics(epoch, losses, result)

    _write_history(os.path.join(output_dir, "history.csv"), history)
    save_json(config, os.path.join(output_dir, "config.effective.json"))
    return best_result, {"history": history, "output_dir": output_dir}


@torch.no_grad()
def evaluate_model(
    model: HSIGCDiffModel,
    graph_data: GraphData,
    node_features: torch.Tensor,
    context_features: torch.Tensor,
    adjs: Dict[str, torch.Tensor],
    n_clusters: int,
    sample_steps: int,
    seed: int,
) -> Optional[EvalResult]:
    model.eval()
    outputs = model(node_features, context_features, adjs)
    z_eval = model.denoise_embeddings(outputs["z_source"], outputs["z_fuse"], steps=sample_steps)
    labels_np, _ = run_kmeans(z_eval.detach().cpu().numpy(), n_clusters, seed=seed)
    full_sp_labels = graph_data.recover_full_superpixel_labels(labels_np)
    return evaluate_pixel_clustering(full_sp_labels, graph_data.association_full, graph_data.gt)


def train_from_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Tuple[Optional[EvalResult], Dict[str, Any]]:
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config = load_json(config_path)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
    graph_data = build_graph_data(config, config_dir=config_dir)
    output_dir = project_path(config.get("output", {}).get("dir", "runs/default"), base_dir=os.path.dirname(config_path))
    return train_graph_data(graph_data, config, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HSI-GCDiff for unsupervised HSI pixel clustering.")
    parser.add_argument("--config", default="configs/xuzhou.json", help="Path to a JSON config file.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda:0.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--n-clusters", type=int, default=None, help="Required if no GT file is supplied.")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_json(config_path)
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.n_clusters is not None:
        config["n_clusters"] = args.n_clusters

    graph_data = build_graph_data(config, config_dir=os.path.dirname(config_path))
    output_dir = project_path(config.get("output", {}).get("dir", "runs/default"), base_dir=os.path.dirname(config_path))
    best, info = train_graph_data(graph_data, config, output_dir=output_dir)
    if best is None:
        print(f"Training finished. Outputs: {info['output_dir']}")
    else:
        print(f"Best ACC={best.acc:.4f}, Kappa={best.kappa:.4f}, NMI={best.nmi:.4f}. Outputs: {info['output_dir']}")

