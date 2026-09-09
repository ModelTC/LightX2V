"""Run the native PyTorch OpenPI policy in a local LIBERO simulator.

This is the non-ROS closed-loop evaluation path. OpenPI predicts robot actions,
not pixels, so an actual simulator rollout is required to produce a video of
the robot executing those actions.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .openpi_runner import OpenPIPolicy

LOGGER = logging.getLogger(__name__)

LIBERO_BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
MAX_STEPS_BY_BENCHMARK = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def _configure_libero(libero_root: Path, config_dir: Path):
    """Load the local LIBERO checkout without importing OpenPI runtime code."""
    libero_root = libero_root.expanduser().resolve()
    benchmark_root = libero_root / "libero" / "libero"
    required = (
        benchmark_root / "bddl_files",
        benchmark_root / "init_files",
        benchmark_root / "assets",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"LIBERO checkout is incomplete under {libero_root}: {missing}")

    config_dir = config_dir.expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        "\n".join(
            (
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                f"datasets: {libero_root / 'libero' / 'datasets'}",
                f"assets: {benchmark_root / 'assets'}",
                "",
            )
        ),
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    root_text = str(libero_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv


def _quat_to_axis_angle(quaternion) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32).copy()
    if quat.shape != (4,):
        raise ValueError(f"Expected LIBERO quaternion shape (4,), got {quat.shape}")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quat[3]) ** 2))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3])) / denominator)).astype(np.float32)


def _rotate_rgb(observation: dict, key: str) -> np.ndarray:
    if key not in observation:
        raise KeyError(f"LIBERO observation is missing {key!r}")
    image = np.asarray(observation[key])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"LIBERO image {key!r} must be HWC RGB, got {image.shape}")
    # LIBERO renders both policy cameras rotated by 180 degrees relative to the
    # released pi05_libero training observations.
    return np.ascontiguousarray(image[::-1, ::-1], dtype=np.uint8)


def _state_from_observation(observation: dict) -> np.ndarray:
    state = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            _quat_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite 8-D LIBERO state, got {state.shape}")
    return state


def _load_policy_config(args: argparse.Namespace) -> dict:
    config_path = args.config_json.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(
        {
            "model_cls": "openpi",
            "task": "i2va",
            "model_path": str(args.model_path.expanduser().resolve()),
            "config_json": str(config_path),
            "seed": args.seed,
            "actions_per_plan": args.actions_per_plan,
        }
    )
    return config


def _validate_output_path(path: Path, suffix: str, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.suffix.lower() != suffix:
        raise ValueError(f"{label} must end in {suffix}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise RuntimeError("LIBERO rollout produced no video frames")
    temporary = path.with_name(f".{path.stem}.tmp.mp4")
    imageio.mimwrite(temporary, frames, fps=fps)
    temporary.replace(path)


def run_rollout(args: argparse.Namespace) -> dict:
    if args.actions_per_plan < 1:
        raise ValueError("--actions-per-plan must be positive")
    if args.num_steps_wait < 0:
        raise ValueError("--num-steps-wait must be non-negative")
    if args.fps < 1:
        raise ValueError("--fps must be positive")
    if args.render_size < 1:
        raise ValueError("--render-size must be positive")
    max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS_BY_BENCHMARK[args.benchmark]
    if max_steps < 1:
        raise ValueError("--max-steps must be positive")

    save_video_path = _validate_output_path(args.save_video_path, ".mp4", "save_video_path")
    save_action_path = _validate_output_path(args.save_action_path, ".npy", "save_action_path")
    save_metrics_path = _validate_output_path(args.save_metrics_path, ".json", "save_metrics_path")

    benchmark_module, get_libero_path, env_type = _configure_libero(args.libero_root, args.libero_config_dir)
    factories = benchmark_module.get_benchmark_dict()
    task_suite = factories[args.benchmark]()
    task_count = task_suite.get_num_tasks()
    if not 0 <= args.task_id < task_count:
        raise ValueError(f"task_id must be in [0, {task_count}), got {args.task_id}")
    task = task_suite.get_task(args.task_id)
    # Torch >= 2.6 defaults torch.load to weights_only=True. LIBERO init-state
    # files are trusted local tensors rather than model weights, so load them
    # explicitly with the legacy behavior used by the existing ROS simulator.
    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    initial_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    if not 0 <= args.init_state_id < len(initial_states):
        raise ValueError(f"init_state_id must be in [0, {len(initial_states)}), got {args.init_state_id}")

    task_description = args.task_description.strip() or str(task.language)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = env_type(
        bddl_file_name=str(bddl_path),
        camera_heights=args.render_size,
        camera_widths=args.render_size,
    )
    env.seed(args.seed)

    policy = None
    frames: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    success = False
    started = time.perf_counter()

    try:
        LOGGER.info("Loading local PyTorch OpenPI policy")
        policy = OpenPIPolicy.from_config(_load_policy_config(args))
        LOGGER.info("Starting %s task %d, init state %d: %s", args.benchmark, args.task_id, args.init_state_id, task_description)
        env.reset()
        observation = env.set_init_state(initial_states[args.init_state_id])
        policy.reset()

        for _ in range(args.num_steps_wait):
            observation, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION.tolist())
            if done:
                success = True
                break

        frames.append(_rotate_rgb(observation, "agentview_image"))
        for step in range(0 if success else max_steps):
            images = {
                "agentview": _rotate_rgb(observation, "agentview_image"),
                "wrist": _rotate_rgb(observation, "robot0_eye_in_hand_image"),
            }
            action = np.asarray(
                policy.next_action(images=images, state=_state_from_observation(observation), task_description=task_description),
                dtype=np.float32,
            ).reshape(-1)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError(f"OpenPI returned an invalid LIBERO action: shape={action.shape}")
            executed_actions.append(action.copy())
            observation, _reward, done, _info = env.step(action.tolist())
            frames.append(_rotate_rgb(observation, "agentview_image"))
            success = bool(done)
            if success or (step + 1) % 10 == 0:
                LOGGER.info("Rollout step %d/%d, success=%s", step + 1, max_steps, success)
            if success:
                break
    finally:
        env.close()
        if policy is not None:
            policy.close()

    action_array = np.asarray(executed_actions, dtype=np.float32).reshape(-1, 7)
    np.save(save_action_path, action_array)
    _write_video(save_video_path, frames, args.fps)

    metrics = {
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "task_name": str(task.name),
        "task_description": task_description,
        "bddl_file": str(bddl_path),
        "init_states_file": str(init_states_path),
        "init_state_id": args.init_state_id,
        "init_state_count": len(initial_states),
        "seed": args.seed,
        "success": success,
        "policy_steps": int(action_array.shape[0]),
        "warmup_steps": args.num_steps_wait,
        "actions_per_plan": args.actions_per_plan,
        "video_frames": len(frames),
        "video_fps": args.fps,
        "elapsed_seconds": time.perf_counter() - started,
        "video_path": str(save_video_path),
        "action_path": str(save_action_path),
    }
    save_metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved rollout video: %s", save_video_path)
    LOGGER.info("Saved executed actions %s: %s", action_array.shape, save_action_path)
    LOGGER.info("Saved rollout metrics (success=%s): %s", success, save_metrics_path)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local pi05_libero PyTorch rollout and record MP4 video")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-config-dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=LIBERO_BENCHMARKS, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-description", default="")
    parser.add_argument("--actions-per-plan", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--save-video-path", type=Path, required=True)
    parser.add_argument("--save-action-path", type=Path, required=True)
    parser.add_argument("--save-metrics-path", type=Path, required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()
    metrics = run_rollout(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
