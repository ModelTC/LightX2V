import importlib
import time

from simulator.sim.bench.interfaces import ActionChunk


class NativeWAMAdapter:
    def __init__(self, cfg):
        from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
        from lightx2v.models.runners.wan.realtimewam_policy import RealtimeWAMPolicy
        from lightx2v.utils.set_config import get_default_config

        model = cfg["model"]
        native = get_default_config()
        native.update(model["native_config"])
        cls = RealtimeWAMPolicy if model["name"] == "realtimewam" else FastWAMPolicy
        self.policy = cls.from_config(native)
        self.seed = cfg["seed"]
        self.action_space = {"robotwin": "robotwin_joint_position", "libero": "libero_delta_eef"}[native["policy_profile"]]

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
