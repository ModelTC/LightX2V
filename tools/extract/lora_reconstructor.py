"""Reconstruct a Qwen-Image 2.1 LoRA, optionally using torchrun DDP."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.extract.reconstruction.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", choices=("qdiff", "tfmq"))
    parser.add_argument("--calibration")
    parser.add_argument("--output")
    parser.add_argument("--work-dir")
    parser.add_argument("--device")
    workflow = parser.add_mutually_exclusive_group()
    workflow.add_argument("--collect-calibration", action="store_true", help="Collect native target trajectories using the config's per-task request counts (single GPU or torchrun workers)")
    workflow.add_argument("--collect-and-reconstruct", action="store_true", help="Collect target trajectories, then reconstruct the LoRA in the same launch")
    parser.add_argument("--dry-run", action="store_true", help="Validate calibration selection without inference or reconstruction; requires a collection option")
    args = vars(parser.parse_args())
    collect, combined, dry_run = args.pop("collect_calibration"), args.pop("collect_and_reconstruct"), args.pop("dry_run")
    if dry_run and not (collect or combined):
        parser.error("--dry-run requires --collect-calibration or --collect-and-reconstruct")
    config = load_config(args.pop("config"), args)
    if collect or combined:
        from tools.extract.reconstruction.calibration import collect_calibration

        collect_calibration(config, dry_run=dry_run)
    if not dry_run and not collect:
        from tools.extract.reconstruction.engine import reconstruct

        reconstruct(config)


if __name__ == "__main__":
    main()
