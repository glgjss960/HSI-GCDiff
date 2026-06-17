import csv
import os

from .proof_protocol import project_root, run_protocol
from .utils import resolve_device, save_json


def main():
    root = project_root()
    config_path = os.path.join(root, "codex_runs", "smoke_v2_config.json")
    cfg = {
        "name": "smoke_v2",
        "seed": 7,
        "device": "cpu",
        "n_clusters": 4,
        "run": {"output_dir": "codex_runs/smoke_v2"},
        "data": {"synthetic": True, "height": 28, "width": 28, "classes": 4},
        "superpixel": {"n_segments": 120, "compactness": 8.0, "source_view": 0},
        "graph": {
            "neighbors": 8,
            "graph_views": [
                {"name": "spectral_stats", "role": "task", "source_view": 0, "mode": "mean_std", "neighbors": 8},
                {"name": "aux_stats", "role": "aux", "source_view": 1, "mode": "mean_std", "neighbors": 8},
                {"name": "spatial", "role": "aux", "source_view": 0, "mode": "geometry", "graph": "spatial"},
            ],
        },
        "encoder": {
            "task_hidden_dim": 64,
            "aux_hidden_dim": 64,
            "latent_dim": 32,
            "epochs": 4,
            "lr": 0.002,
            "adj_recon_weight": 0.02,
            "aux_align_weight": 0.01,
            "log_every": 2,
        },
        "target": {"mode": "task", "source_key": "z_task"},
        "denoiser": {
            "source_key": "z_aux",
            "target_key": "z_target",
            "condition": "none",
            "hidden_dim": 96,
            "time_dim": 32,
            "layers": 3,
            "timesteps": 20,
            "epochs": 6,
            "lr": 0.002,
            "log_every": 3,
        },
        "injection": {"alphas": [0, 0.1, 0.2]},
        "eval": {"sampling_steps": [0, 5], "noise_seeds": [0, 1]},
    }
    save_json(cfg, config_path)
    paths = run_protocol(config_path, stage="all", device=resolve_device("cpu"), force=True)
    required = ["graph", "embeddings", "target", "denoiser", "metrics"]
    missing = [name for name in required if not os.path.exists(paths[name])]
    if missing:
        raise RuntimeError(f"Missing smoke artifacts: {missing}")
    with open(paths["metrics"], newline="", encoding="utf-8") as f:
        methods = {row["method"] for row in csv.DictReader(f)}
    needed_methods = {"z_task", "z_aux", "z_diff", "z_task_plus_raw_aux", "z_task_plus_denoised_aux"}
    if not needed_methods.issubset(methods):
        raise RuntimeError(f"Missing smoke metric methods: {sorted(needed_methods - methods)}")
    print(f"Smoke test passed: {paths['metrics']}")


if __name__ == "__main__":
    main()
