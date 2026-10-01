"""Worker process: load the existing LV policy and run benchmark episodes."""

import importlib
import json
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simulator.sim.benchmark import atomic_json, get_benchmark_backend, result_path


@dataclass
class Observation:
    images: dict[str, np.ndarray]  # HWC uint8 RGB, semantic camera names
    state: np.ndarray  # physical units; model adapter normalizes
    prompt: str
    step: int = 0


@dataclass
class ActionChunk:
    actions: np.ndarray  # physical units, [horizon, action_dim]
    action_space: str  # libero_delta_eef or robotwin_joint_position
    timing: dict[str, float] = field(default_factory=dict)

    def validate(self, action_space, action_dim):
        array = np.asarray(self.actions, dtype=np.float32)
        if self.action_space != action_space:
            raise ValueError(f"Action space mismatch: {self.action_space} != {action_space}")
        if array.ndim != 2 or array.shape[1] != action_dim or len(array) < 1 or not np.isfinite(array).all():
            raise ValueError(f"Invalid action chunk: shape={array.shape}, expected [N,{action_dim}] finite values")
        return array


class NativeWAMAdapter:
    def __init__(self, cfg):
        from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
        from lightx2v.models.runners.wan.realtimewam_runner import RealtimeWAMPolicy
        from lightx2v.utils.set_config import build_startup_config

        model = cfg["model"]
        native = build_startup_config(model["native_config"])
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


def make_environment(cfg, task):
    return get_benchmark_backend(cfg["benchmark"]).create_adapter(cfg, task)


def run_task(cfg, task, policy):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    import torch

    torch.manual_seed(cfg["seed"])
    evaluation = cfg["EVALUATION"]
    skip_observation = evaluation.get("skip_get_obs_within_replan", False)
    trials = get_benchmark_backend(cfg["benchmark"]).num_trials(cfg)
    result = {"task": task, "status": "running", "episodes": [], "error": None}
    path = result_path(evaluation["output_dir"], task)
    environment = None
    try:
        environment = make_environment(cfg, task)
        for index in range(trials):
            obs = environment.reset(index)
            metadata = environment.episode_metadata()
            policy.reset_episode({**task, **metadata, "episode_index": index})
            pending = deque()
            started = time.perf_counter()
            inference_seconds, calls, success = 0.0, 0, False
            for step in range(environment.max_steps):
                if not pending:
                    if obs is None:
                        raise RuntimeError("Missing fresh observation at action replanning boundary")
                    chunk = policy.predict_action_chunk(obs)
                    actions = chunk.validate(environment.action_space, environment.action_dim)
                    pending.extend(actions[: evaluation["replan_steps"]])
                    inference_seconds += chunk.timing.get("inference_seconds", 0.0)
                    calls += 1
                action = pending.popleft()
                if skip_observation:
                    # Intermediate actions need success checks, not new images.
                    # Do not acquire an unused observation at the rollout cap.
                    observe = not pending and step + 1 < environment.max_steps
                    obs, success, done = environment.step(action, observe=observe)
                else:
                    obs, success, done = environment.step(action)
                if success or done:
                    break
            result["episodes"].append(
                {
                    "episode_index": index,
                    "success": bool(success),
                    "steps": step + 1,
                    "elapsed_seconds": time.perf_counter() - started,
                    "inference_seconds": inference_seconds,
                    "inference_calls": calls,
                    **metadata,
                }
            )
            atomic_json(path, result)
            print(f"task={task['key']} episode={index + 1}/{trials} success={success}", flush=True)
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(path, result)
        if environment:
            environment.close()


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    policy = build_policy(payload["config"])
    try:
        for task in payload["tasks"]:
            run_task(payload["config"], task, policy)
    finally:
        policy.close()


if __name__ == "__main__":
    main()
