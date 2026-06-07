import os

from .proof_protocol import project_root, run_protocol
from .utils import resolve_device, save_json


def main():
    root = project_root()
    config_path = os.path.join(root, "codex_runs", "smoke_proof_config.json")
    cfg = {
        "name": "smoke_proof",
        "seed": 7,
        "device": "cpu",
        "n_clusters": 4,
        "run": {"output_dir": "codex_runs/smoke_proof"},
        "data": {"synthetic": True, "height": 28, "width": 28, "classes": 4},
        "superpixel": {"n_segments": 120, "compactness": 8.0, "source_view": 0},
        "graph": {
            "neighbors": 8,
            "include_spatial_graph": True,
            "graph_views": [
                {"name": "spectral_stats", "source_view": 0, "mode": "mean_std", "neighbors": 8},
                {"name": "aux_stats", "source_view": 1, "mode": "mean_std", "neighbors": 8},
            ],
        },
        "encoder": {
            "hidden_dim": 64,
            "latent_dim": 32,
            "fusion": "mean",
            "epochs": 4,
            "lr": 0.002,
            "log_every": 2,
        },
        "teacher": {"backend": "python_etap_lite", "n_anchors": 80, "k": 5},
        "target": {"source_key": "z_fuse", "assignment": "soft", "min_confidence": 0.0},
        "denoiser": {
            "source_key": "z_source",
            "cond_key": "z_fuse",
            "hidden_dim": 96,
            "time_dim": 32,
            "layers": 3,
            "timesteps": 20,
            "epochs": 6,
            "lr": 0.002,
            "log_every": 3,
        },
        "eval": {"timesteps": [0, 5, 10], "noise_seeds": [0, 1]},
    }
    save_json(cfg, config_path)
    paths = run_protocol(config_path, stage="all", device=resolve_device("cpu"), force=True)
    required = ["graph", "embeddings", "teacher", "target", "denoiser", "metrics"]
    missing = [name for name in required if not os.path.exists(paths[name])]
    if missing:
        raise RuntimeError(f"Missing smoke artifacts: {missing}")
    print(f"Smoke test passed: {paths['metrics']}")


if __name__ == "__main__":
    main()
