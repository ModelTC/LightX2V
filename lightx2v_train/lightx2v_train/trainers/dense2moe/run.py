"""Explicit experimental entry point, reusing native H3 loaders and packing."""

import argparse
import faulthandler
import hashlib
import json
import signal
from pathlib import Path

from omegaconf import OmegaConf


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("calibrate", "shared", "moe", "benchmark"), required=True)
    parser.add_argument("--smoke", action="store_true", help="Real weights, short engineering validation; not a converged model")
    parser.add_argument("--benchmark-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--benchmark-frames", type=int)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    config["moe"]["layers"] = list(range(*config["moe"].pop("layer_range")))
    config["prompts_sha256"] = hashlib.sha256(Path(config["prompts"]).read_bytes()).hexdigest()
    if args.smoke:
        config.update(iterations={"calibrate": 2, "shared": 2, "moe": 2})
        config["output_dir"] += "_smoke"
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    if args.stage == "benchmark":
        if args.benchmark_size:
            config["benchmark"]["height"], config["benchmark"]["width"] = args.benchmark_size
        if args.benchmark_frames:
            config["benchmark"]["num_frames"] = args.benchmark_frames
        from .benchmark import benchmark

        benchmark(config)
    else:
        from lightx2v_train.runtime.distributed import cleanup_distributed, get_rank, init_distributed

        from .trainer import Dense2MoETrainer

        init_distributed({"distributed": {"fsdp2": {"enabled": True}}})
        complete = Path(config["output_dir"], args.stage + ".complete.json")
        if complete.exists():
            if json.loads(complete.read_text()) != config:
                raise ValueError("Completed stage uses a different recipe; choose a new output directory")
            if get_rank() == 0:
                print(f"{args.stage}: already complete", flush=True)
            cleanup_distributed()
            return
        if get_rank() == 0:
            Path(config["output_dir"], "recipe.json").write_text(json.dumps(config, indent=2))
        Dense2MoETrainer(config, args.stage).train()
        if get_rank() == 0:
            complete.write_text(json.dumps(config, indent=2))
        cleanup_distributed()


if __name__ == "__main__":
    main()
