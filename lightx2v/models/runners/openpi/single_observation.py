"""Run one OpenPI image/state observation without importing the shared CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightx2v.utils.input_info import init_empty_input_info
from lightx2v.utils.lockable_dict import LockableDict

from .openpi_runner import OpenPIRunner


def _load_config(args: argparse.Namespace) -> LockableDict:
    config_path = args.config_json.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"OpenPI config JSON does not exist: {config_path}")

    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"OpenPI PyTorch checkpoint is incomplete: {model_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(
        {
            "model_cls": "openpi",
            "task": "i2va",
            "model_path": str(model_path),
            "config_json": str(config_path),
            "seed": args.seed,
            "warmup": False,
        }
    )
    return LockableDict(config)


def run_single_observation(args: argparse.Namespace) -> Path:
    runner = OpenPIRunner(_load_config(args))
    try:
        runner.init_modules()
        input_info = init_empty_input_info("i2va")
        input_info.seed = args.seed
        input_info.prompt = args.task_description
        input_info.image_path = str(args.image_path)
        input_info.state_path = str(args.state_path)
        input_info.save_action_path = str(args.save_action_path)
        runner.run_pipeline(input_info)
    finally:
        runner.end_run()
    return args.save_action_path.expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local pi05-LIBERO image/state observation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--save-action-path", type=Path, required=True)
    return parser


def main() -> None:
    output_path = run_single_observation(build_parser().parse_args())
    print(f"Saved OpenPI action chunk: {output_path}")


if __name__ == "__main__":
    main()
