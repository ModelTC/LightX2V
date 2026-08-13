"""Adapter between RoboDojo observations and LightX2V's FastWAM policy."""

from pathlib import Path

import numpy as np

from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
from lightx2v.utils.set_config import get_default_config


def _rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected an HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


class FastWAMRoboDojoAdapter:
    """Small policy facade for XPolicyLab's RoboDojo server.

    RoboDojo state packing is intentionally supplied by the caller because its
    schema depends on ``env_cfg_type`` and ``action_type``. The adapter owns the
    camera mapping and the FastWAM action-chunk/replan behavior.
    """

    camera_keys = {
        "head_camera": "cam_head",
        "left_camera": "cam_left_wrist",
        "right_camera": "cam_right_wrist",
    }

    def __init__(self, config):
        config = dict(config)
        checkpoint = config.get("adapter_model_path") or config.get("checkpoint_path")
        if not checkpoint:
            raise ValueError("checkpoint_path (or adapter_model_path) is required")
        stats = config.get("dataset_stats_path")
        if not stats:
            raise ValueError("dataset_stats_path is required")

        runtime_config = get_default_config()
        runtime_config.update(config)
        runtime_config.update(
            {
                "model_cls": "fastwam",
                "task": "i2va",
                "adapter_model_path": str(Path(checkpoint).expanduser().resolve()),
                "dataset_stats_path": str(Path(stats).expanduser().resolve()),
                "policy_profile": "robotwin",
                "normalize_mode": "z-score",
                "action_dim": int(config.get("action_dim", 14)),
                "robot_state_dim": int(config.get("robot_state_dim", 14)),
                "action_chunk_size": int(config.get("action_chunk_size", 32)),
                "actions_per_plan": int(config.get("actions_per_plan", 24)),
                "action_infer_steps": int(config.get("action_infer_steps", 10)),
                "action_sample_shift": float(config.get("action_sample_shift", 5.0)),
                "default_prompt": config.get(
                    "default_prompt",
                    "A video recorded from a robot's point of view executing "
                    "the following instruction: {task_prompt}",
                ),
            }
        )
        self.policy = FastWAMPolicy.from_config(runtime_config)

    def predict(self, observation, packed_state, instruction):
        vision = observation["vision"]
        images = {
            policy_key: _rgb(vision[robodojo_key]["color"])
            for policy_key, robodojo_key in self.camera_keys.items()
        }
        return self.policy.predict_action_chunk(
            images=images,
            state=np.asarray(packed_state, dtype=np.float32),
            task_description=str(instruction),
        )[: self.policy.actions_per_plan]

    def reset(self):
        self.policy.reset()

    def close(self):
        self.policy.close()
