import importlib
import time

from scripts.bench.robotics.common.interfaces import ActionChunk


class NativeWAMAdapter:
    def __init__(self, cfg):
        from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
        from lightx2v.models.runners.wan.realtimewam_policy import RealtimeWAMPolicy
        from lightx2v.utils.set_config import get_default_config

        robotwin = cfg["benchmark"] == "robotwin"
        model, evaluation = cfg["model"], cfg["EVALUATION"]
        native = get_default_config()
        native.update(
            {
                "model_cls": "fastwam",
                "task": "i2va",
                "device": "cuda:0",
                "dim": 3072,
                "num_heads": 24,
                "num_layers": 30,
                "freq_dim": 256,
                "eps": 1e-6,
                "action_dim_hidden": 1024,
                "model_path": model["model_path"],
                "adapter_model_path": cfg["base_ckpt"] or cfg["ckpt"],
                "lora_path": cfg["lora_path"],
                "lora_weights": model["lora_weights"],
                "dataset_stats_path": evaluation["dataset_stats_path"],
                "action_dim": 14 if robotwin else 7,
                "robot_state_dim": 14 if robotwin else 8,
                "action_chunk_size": model["action_horizon"],
                "actions_per_plan": evaluation["replan_steps"],
                "action_infer_steps": evaluation["num_inference_steps"],
                "action_sample_shift": evaluation["sigma_shift"],
                "action_infer_mode": evaluation["action_infer_mode"],
                "num_video_frames": model["num_video_frames"],
                "policy_profile": "robotwin" if robotwin else "libero",
                "camera_size": 384 if robotwin else 224,
                "normalize_mode": "z-score" if robotwin else "min-max",
                "gripper_postprocess": not robotwin,
                "binarize_gripper": not robotwin,
                "seed": cfg["seed"],
                "t5_cpu_offload": model["t5_cpu_offload"],
                "vae_cpu_offload": model["vae_cpu_offload"],
                "default_prompt": "A video recorded from a robot's point of view executing the following instruction: {task_prompt}",
            }
        )
        if model["backbone"] == "fasterwam":
            native.update({"condition_layers": [0, 4, 8, 12, 16, 20, 24, 28], "video_kv_fusion": "interval_weighted_sum"})
        else:
            native.update({"condition_layers": None, "video_kv_fusion": None})
        cls = RealtimeWAMPolicy if model["name"] == "realtimewam" else FastWAMPolicy
        self.policy = cls.from_config(native)
        self.seed = cfg["seed"]
        self.action_space = "robotwin_joint_position" if robotwin else "libero_delta_eef"

    def reset_episode(self, metadata):
        self.policy.reset()

    def predict_action_chunk(self, observation):
        import torch

        started = time.perf_counter()
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(observation.images, observation.state, observation.prompt, seed=self.seed)
        return ActionChunk(actions, self.action_space, {"inference_seconds": time.perf_counter() - started})

    def close(self):
        close = getattr(self.policy, "close", None)
        if close:
            close()
        del self.policy


def build_policy(cfg):
    if cfg["model"]["factory"]:
        module, name = cfg["model"]["factory"].split(":", 1)
        return getattr(importlib.import_module(module), name)(cfg)
    if cfg["model"]["name"] not in ("realtimewam", "fastwam"):
        raise ValueError("Unknown model; register a Policy via model.factory=module:factory")
    return NativeWAMAdapter(cfg)
