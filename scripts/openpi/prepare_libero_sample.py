"""Extract one reproducible offline OpenPI input from a downloaded LIBERO HDF5.

This is a data-preparation utility, not part of model runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

DEFAULT_SOURCE = Path(
    "/data/liuhongda/openpi_data/raw/huggingface/yifengzhu-hf/LIBERO-datasets/"
    "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
DEFAULT_OUTPUT = Path("/data/liuhongda/openpi_data/examples/pi05_libero/libero_spatial_task0_demo0_step0")
DEFAULT_PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    with h5py.File(source, "r") as handle:
        demo = handle[f"data/{args.demo}"]
        step = int(args.step)
        agentview = np.asarray(demo["obs/agentview_rgb"][step], dtype=np.uint8)
        wrist = np.asarray(demo["obs/eye_in_hand_rgb"][step], dtype=np.uint8)
        state = np.concatenate(
            [demo["obs/ee_pos"][step], demo["obs/ee_ori"][step], demo["obs/gripper_states"][step]]
        ).astype(np.float32)
        reference_actions = np.asarray(demo["actions"][step : step + 10], dtype=np.float32)

    if state.shape != (8,):
        raise ValueError(f"Expected 8-D LIBERO state, got {state.shape}")
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(agentview, mode="RGB").save(output / "agentview_image.png")
    Image.fromarray(wrist, mode="RGB").save(output / "wrist_image.png")
    np.save(output / "state.npy", state)
    np.save(output / "reference_actions.npy", reference_actions)
    metadata = {
        "source": str(source),
        "demo": args.demo,
        "step": int(args.step),
        "task_description": DEFAULT_PROMPT,
        "state_layout": ["eef_pos[3]", "eef_axis_angle[3]", "gripper_qpos[2]"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
