import argparse

from hsigcdiff.proof_protocol import OPTIONAL_STAGES, STAGES, run_protocol
from hsigcdiff.utils import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="HSI-GCDiff proof-of-denoising protocol")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--stage", default="all", choices=["all"] + STAGES + OPTIONAL_STAGES, help="Pipeline stage to run.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--output", default=None, help="Override run.output_dir.")
    parser.add_argument("--force", action="store_true", help="Recompute a stage even if its artifact exists.")
    parser.add_argument("--epochs", type=int, default=None, help="Override encoder.epochs.")
    parser.add_argument("--denoise-epochs", type=int, default=None, help="Override denoiser.epochs.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if args.epochs is not None or args.denoise_epochs is not None:
        from hsigcdiff.proof_protocol import load_config, paths
        from hsigcdiff.utils import save_json

        cfg = load_config(args.config, output_override=args.output, device_override=str(device))
        if args.epochs is not None:
            cfg.setdefault("encoder", {})["epochs"] = args.epochs
        if args.denoise_epochs is not None:
            cfg.setdefault("denoiser", {})["epochs"] = args.denoise_epochs
        save_json(cfg, paths(cfg)["effective_config"])
        temp_config = paths(cfg)["effective_config"]
        result_paths = run_protocol(temp_config, stage=args.stage, device=device, force=args.force, output_dir=args.output)
    else:
        result_paths = run_protocol(args.config, stage=args.stage, device=device, force=args.force, output_dir=args.output)
    print(f"Run directory: {result_paths['run']}")
    if args.stage in ("all", "eval"):
        print(f"Metrics: {result_paths['metrics']}")


if __name__ == "__main__":
    main()
