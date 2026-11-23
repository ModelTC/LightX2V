"""Prepare the existing selfgen GAN view and check the DCM EMA initialization."""

import argparse
import json
from pathlib import Path

from minimax_h3_action_pdmd_selfgen_gan_preflight import (
    load_config, own_stage_resume, prepare_selfgen_view, validate_model, validate_view,
)
from minimax_h3_pdmd_from_turbo_preflight import validate_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_model(config)
    training = config["training"]
    view_path = Path(config["data"]["train"]["data_path"])
    if view_path.exists():
        view = validate_view(view_path, "selfgen")
    else:
        view = prepare_selfgen_view(args.source, view_path, dry_run=args.dry_run)
    resume = own_stage_resume(config)
    if resume is None:
        weights, tensors = validate_source(Path(training["student"]["checkpoint_path"]), training["student"]["lora"])
        print(f"DCM EMA student initialization: {weights}; tensors={tensors}; fake/teacher=H3 base; GAN head=fresh")
    else:
        print(f"Full PDMD + GAN resume: {resume}")
    print(json.dumps({"selfgen_rows": view["rows"], "selfgen_manifest": view["manifest"],
                      "steps": training["dmd"]["num_inference_steps"],
                      "schedule": training["dmd"]["rollout_shift_schedule"],
                      "optimizer_updates": training["max_train_iters"],
                      "student_updates": 500, "fake_updates": 2500,
                      "output": training["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
