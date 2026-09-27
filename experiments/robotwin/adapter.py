"""Strict evaluation adapter over the repo-local, ROS-free RoboTwin environment.

Unlike interactive simulation, benchmarking never uses a dry-run expert or a
generic fallback instruction when a dependency is missing.
"""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.common.config import ROBOTWIN_ROOT, ROOT
from experiments.common.interfaces import Observation
from experiments.robotwin.seed_cache import SeedCache, find_validated_seed


def discover_tasks(cfg):
    root = Path(cfg["EVALUATION"]["robotwin_root"])
    if not (root / "envs").is_dir() or not (root / "description/task_instruction").is_dir():
        hint = "Check EVALUATION.robotwin_root."
        if root.resolve() == ROBOTWIN_ROOT.resolve():
            hint = f"Run git submodule update --init --recursive {ROBOTWIN_ROOT.relative_to(ROOT)} from the repository root."
        raise FileNotFoundError(f"Incomplete RoboTwin source: {root}. {hint}")
    names = cfg["MULTIRUN"]["task_names"]
    if names is None:
        names = sorted(p.stem for p in (root / "description/task_instruction").glob("*.json"))
    if not names:
        raise ValueError("No RoboTwin tasks; set MULTIRUN.task_names=[task,...]")
    phases = cfg["MULTIRUN"]["phases"]
    if set(phases) - {"clean", "random"}:
        raise ValueError("RoboTwin phases must be clean/random")
    return [{"key": f"{phase}/{name}", "task_name": name, "phase": phase, "category": None, "suite": None} for phase in phases for name in names]


class RoboTwinAdapter:
    action_dim = 14
    action_space = "robotwin_joint_position"

    def __init__(self, cfg, task):
        for relative in ("lightx2v_ros/src/common", "lightx2v_ros/src/simulator"):
            sys.path.insert(0, str(ROOT / relative))
        from common.contract import ROBOTWIN_CONTRACT
        from simulator.robotwin_node.env import RoboTwinEnv

        evaluation = cfg["EVALUATION"]
        self.seed_cache = cache = SeedCache(cfg, task)

        class StrictEnvironment(RoboTwinEnv):
            def _prepare_planner_runtime(self):
                if importlib.util.find_spec("curobo") is None:
                    raise ImportError("RoboTwin benchmark requires curobo expert checks; no dry-run fallback")
                from experiments.robotwin.planner_adapter import install_planner_adapter

                install_planner_adapter(self.robotwin_root, evaluation["output_dir"])

            @property
            def _expert_planner_available(self):
                return True

            def _setup_episode(self, max_seed_attempts=None):
                try:
                    super()._setup_episode(evaluation["max_seed_attempts"])
                except Exception:
                    cache.reject(self.seed)
                    raise
                cache.record_validated(self.seed)

            def _find_solvable_seed(self, max_attempts):
                return find_validated_seed(self, cache, max_attempts)

            def _resolve_instruction(self, episode_info):
                from generate_episode_instructions import generate_episode_descriptions

                descriptions = generate_episode_descriptions(self.task_name, [episode_info["info"]], evaluation["eval_num_episodes"])
                choices = descriptions[0][self.instruction_type]
                if not choices:
                    raise ValueError("No instructions for selected instruction_type")
                return str(np.random.choice(choices))

        self.cls = StrictEnvironment
        self.contract = replace(ROBOTWIN_CONTRACT, cameras=ROBOTWIN_CONTRACT.policy_input_cameras)
        self.cfg, self.task = cfg, task
        self.env = None
        self.next_seed = 100000 * (1 + cfg["seed"])
        self.max_steps = evaluation["max_steps"] or 1000

    def reset(self, episode_index):
        if self.env is None:
            self.env = self.cls(
                self.contract,
                task_name=self.task["task_name"],
                task_config="demo_clean" if self.task["phase"] == "clean" else "demo_randomized",
                embodiment=self.cfg["EVALUATION"]["embodiment"],
                instruction_type=self.cfg["EVALUATION"]["instruction_type"],
                seed=self.next_seed,
                render_publish_every=0,
                robotwin_root=self.cfg["EVALUATION"]["robotwin_root"],
            )
            raw = self.env.reset()
        else:
            raw = self.env.new_episode()
        self.max_steps = self.cfg["EVALUATION"]["max_steps"] or self.env.max_steps
        if not self.max_steps:
            raise ValueError("RoboTwin did not provide a step limit")
        self.step_index = 0
        return Observation(raw.images, raw.state, self.env.task_description)

    def step(self, action, *, observe=True):
        raw, done, success = self.env.step(action) if observe else self.env.step(action, observe=False)
        self.step_index += 1
        observation = Observation(raw.images, raw.state, self.env.task_description, self.step_index) if raw is not None else None
        return observation, success, done or self.step_index >= self.max_steps

    def episode_metadata(self):
        return {
            "environment_seed": self.env.seed,
            "instruction": self.env.task_description,
            "expert_check": True,
            "seed_cache_hit": self.seed_cache.current_hit,
            "seed_cache_path": str(self.seed_cache.path) if self.seed_cache.path else None,
        }

    def close(self):
        if self.env:
            self.env.close()
