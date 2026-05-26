import copy

from .data import make_synthetic_hsi
from .trainer import graph_data_from_hsi, train_graph_data


def main() -> None:
    config = {
        "seed": 1,
        "device": "cpu",
        "data": {
            "remove_background": False,
            "standardize": True,
            "pca_dim": 8,
        },
        "superpixel": {
            "n_segments": 64,
            "compactness": 5.0,
            "patch_size": 3,
            "spectral_neighbors": 6,
            "context_neighbors": 6,
        },
        "model": {
            "hidden_dim": 64,
            "latent_dim": 32,
            "dropout": 0.1,
        },
        "diffusion": {
            "timesteps": 20,
            "schedule": "cosine",
            "time_emb_dim": 32,
            "hidden_dim": 128,
            "sample_steps": 0,
            "offset_noise": 0.0,
            "prototype_momentum": 0.5,
        },
        "train": {
            "epochs": 3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
            "cluster_interval": 1,
            "eval_interval": 1,
            "tau_confidence": 0.5,
            "lambda_recon": 1e-3,
            "lambda_cluster": 1.0,
            "lambda_hard": 0.2,
            "lambda_diff": 0.5,
            "lambda_align": 0.5,
            "lambda_consistency": 0.1,
            "lambda_view": 0.1,
            "hard_temperature": 0.5,
            "hard_beta": 2.0,
        },
        "output": {
            "dir": "runs/smoke",
        },
    }
    hsi = make_synthetic_hsi(height=32, width=32, bands=12, classes=4, seed=1)
    graph_data = graph_data_from_hsi(hsi, copy.deepcopy(config))
    best, info = train_graph_data(graph_data, config, output_dir=config["output"]["dir"])
    if best is None:
        raise RuntimeError("Smoke test did not produce evaluation metrics.")
    print(f"Smoke test passed. Best ACC={best.acc:.4f}. Outputs: {info['output_dir']}")


if __name__ == "__main__":
    main()

