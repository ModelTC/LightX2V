import random
import time
from collections import deque

import numpy as np

from experiments.common.results import atomic_json, result_path


def make_environment(cfg, task):
    if cfg["benchmark"] == "robotwin":
        from experiments.robotwin.adapter import RoboTwinAdapter as Adapter
    elif cfg["benchmark"] == "libero_plus":
        from experiments.libero_plus.adapter import LiberoPlusAdapter as Adapter
    else:
        from experiments.libero.adapter import LiberoAdapter as Adapter
    return Adapter(cfg, task)


def run_task(cfg, task, policy):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    import torch

    torch.manual_seed(cfg["seed"])
    evaluation = cfg["EVALUATION"]
    # Older queued jobs have no flag: preserve their original evaluation path.
    skip_observation = cfg["benchmark"] == "robotwin" and evaluation.get("skip_get_obs_within_replan", False)
    trials = evaluation["eval_num_episodes"] if cfg["benchmark"] == "robotwin" else evaluation["num_trials"]
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
