"""Compare upstream OpenPI PyTorch and LightX2V with identical input/noise.

Run this with OpenPI's patched conversion environment. It is a validation tool;
the deployed LightX2V runtime itself does not import OpenPI or JAX.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OPENPI_DATA_ROOT = WORKSPACE_ROOT / "openpi_data"


def load_sample(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        image = np.asarray(demo["obs/agentview_rgb"][0], dtype=np.uint8)
        wrist = np.asarray(demo["obs/eye_in_hand_rgb"][0], dtype=np.uint8)
        state = np.concatenate([demo["obs/ee_pos"][0], demo["obs/ee_ori"][0], demo["obs/gripper_states"][0]]).astype(np.float32)
    return image, wrist, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OPENPI_DATA_ROOT / "openpi-assets/checkpoints/pi05_libero_pytorch",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/openpi/pi05_libero.json",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=OPENPI_DATA_ROOT / "raw/huggingface/yifengzhu-hf/LIBERO-datasets/libero_spatial" / "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OPENPI_DATA_ROOT / "results/pi05_libero_pytorch_parity.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    sample_path = args.sample.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    image, wrist, state = load_sample(sample_path)
    prompt = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    noise = np.random.default_rng(0).standard_normal((10, 32)).astype(np.float32)

    from openpi.policies import policy_config
    from openpi.training import config as training_config

    upstream_config = training_config.get_config("pi05_libero")
    upstream_config = dataclasses.replace(
        upstream_config,
        model=dataclasses.replace(upstream_config.model, pytorch_compile_mode=None),
    )
    upstream_policy = policy_config.create_trained_policy(
        upstream_config,
        checkpoint,
        pytorch_device=args.device,
    )
    raw_input = {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }
    upstream_actions = np.asarray(upstream_policy.infer(raw_input, noise=noise)["actions"])
    del upstream_policy
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Import only the LightX2V network family; bypass LightX2V's top-level
    # platform initialization because this tool already owns the torch device.
    package = types.ModuleType("lightx2v")
    package.__path__ = [str(PROJECT_ROOT / "lightx2v")]
    sys.modules["lightx2v"] = package
    from lightx2v.models.networks.openpi import OpenPIModel

    with config_path.open("r", encoding="utf-8") as handle:
        local_config = json.load(handle)
    # The model JSON intentionally contains architecture/runtime values only;
    # checkpoint-local weights, norm stats, and tokenizer assets are resolved
    # from this explicit directory by OpenPIModel.
    local_config["model_path"] = str(checkpoint)
    local_config["device"] = args.device
    local_config["seed"] = 0
    model = OpenPIModel.from_config(local_config)
    normalized = model.predict_normalized_action_chunk(
        {"agentview": image, "wrist": wrist},
        state,
        prompt,
        noise=noise,
    )
    local_actions = model.post_infer.infer(normalized)

    difference = np.asarray(local_actions, dtype=np.float64) - np.asarray(upstream_actions, dtype=np.float64)
    report = {
        "upstream_shape": list(upstream_actions.shape),
        "lightx2v_shape": list(local_actions.shape),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "allclose_atol_1e-6": bool(np.allclose(local_actions, upstream_actions, rtol=0.0, atol=1e-6)),
        "upstream_actions": upstream_actions.tolist(),
        "lightx2v_actions": local_actions.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("actions")}, indent=2))
    if not report["allclose_atol_1e-6"]:
        raise SystemExit("OpenPI PyTorch parity check failed")


if __name__ == "__main__":
    main()
