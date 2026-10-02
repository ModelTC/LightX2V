"""RoboTwin evaluation, validated seeds and planner asset relocation."""

import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from simulator.sim.benchmark import Benchmark, atomic_json
from simulator.sim.evaluator import Observation

ROBOTWIN_ROOT = Path(__file__).resolve().parent / "RoboTwin"


def load_seeds(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else []


class SeedCache:
    def __init__(self, cfg, task):
        evaluation = cfg["EVALUATION"]
        self.path = None
        self.replay = []
        if evaluation["reuse_seed_cache"]:
            directory = evaluation["seed_cache_dir"] or os.environ.get("ROBOTWIN_SEED_DIR") or Path(evaluation["robotwin_root"]) / "my_seeds"
            self.path = Path(directory).expanduser().resolve() / task["phase"] / f"{task['task_name']}_seed.json"
            self.replay = load_seeds(self.path)
            print(f"Seed cache: {self.path} ({len(self.replay)} seeds); cached seeds override the initial seed but are expert-revalidated", flush=True)
        self.position = 0
        self.next_fresh = 100000 * (1 + cfg["seed"])
        self.current_hit = False
        self.rejected = set()
        self.accepted = set()

    def next_candidate(self, fallback):
        if self.path is None:
            self.current_hit = False
            return fallback
        self.current_hit = self.position < len(self.replay)
        if self.current_hit:
            seed = self.replay[self.position]
            self.position += 1
        else:
            seed = self.next_fresh
        self.next_fresh = seed + 1
        return seed

    def reject(self, seed):
        self.rejected.add(seed)

    def record_validated(self, seed):
        """Call after expert success AND successful policy-environment setup."""
        if self.path is None:
            return
        self.accepted.add(seed)
        self.rejected.discard(seed)
        seeds = sorted((set(self.replay) - self.rejected) | self.accepted)
        atomic_json(self.path, seeds)
        print(f"Saved {len(seeds)} validated seeds to {self.path}", flush=True)


def find_validated_seed(environment, cache, max_attempts):
    """Replay candidates using the same real expert checks as fresh seeds."""
    last_error = None
    for _ in range(max(1, max_attempts)):
        environment.seed = cache.next_candidate(environment.seed)
        try:
            environment._setup_demo()
            info = environment.env.play_once()
            solvable = bool(environment.env.plan_success) and bool(environment.env.check_success())
            environment._close_task_env()
            if solvable:
                return info
            environment._log(f"seed {environment.seed}: expert cannot solve this layout; trying next seed")
        except Exception as exc:  # noqa: BLE001 - legacy expert retries; bounded and logged
            last_error = exc
            environment._close_task_env()
            environment._log(f"seed {environment.seed}: expert check raised {exc!r}; trying next seed")
        cache.reject(environment.seed)
        environment.seed += 1
    raise RuntimeError(f"no expert-solvable seed found after {max_attempts} attempts; last error: {last_error}")


def relocate_asset_paths(value, root, config_dir, key=None):
    root, config_dir = Path(root).resolve(), Path(config_dir).resolve()
    if isinstance(value, dict):
        return {k: relocate_asset_paths(v, root, config_dir, k) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate_asset_paths(v, root, config_dir, key) for v in value]
    if not isinstance(value, str) or not value:
        return value
    path_keys = {"urdf_path", "collision_spheres", "asset_root_path", "usd_path", "isaac_usd_path"}
    if "/assets/" in value:
        target = root / "assets" / value.split("/assets/", 1)[1]
    elif value.startswith("assets/"):
        target = root / value
    elif key in path_keys:
        target = Path(value)
        if not target.is_absolute():
            target = config_dir / target
    else:
        return value
    return str(target.resolve())


def install_planner_adapter(root, output):
    """Patch only the benchmark's planner constructor in this isolated worker."""
    planner = importlib.import_module("envs.robot.planner")
    original = planner.CuroboPlanner
    if getattr(original, "_lightx2v_relocated", False):
        return

    class RepositoryCuroboPlanner(original):
        _lightx2v_relocated = True

        def __init__(self, robot_origion_pose, active_joints_name, all_joints, yml_path=None):
            import yaml

            source = Path(yml_path).resolve()
            config = relocate_asset_paths(yaml.safe_load(source.read_text()), root, source.parent)
            # Preserve the embodiment name: upstream tests 'aloha-agilex' in this path.
            target = Path(output) / "runtime/robotwin_planner" / str(os.getpid()) / source.parent.name / source.name
            atomic_json(target, config)  # JSON is valid YAML for both planner loaders.
            super().__init__(robot_origion_pose, active_joints_name, all_joints, yml_path=str(target))

    planner.CuroboPlanner = RepositoryCuroboPlanner
    # envs.robot may already import the class by value while loading its package.
    robot = sys.modules.get("envs.robot.robot")
    if robot and getattr(robot, "CuroboPlanner", None) is original:
        robot.CuroboPlanner = RepositoryCuroboPlanner


class RoboTwinAdapter:
    action_dim = 14
    action_space = "robotwin_joint_position"

    def __init__(self, cfg, task):
        from common.contract import ROBOTWIN_CONTRACT

        from simulator.robotwin_node.env import RoboTwinEnv

        evaluation = cfg["EVALUATION"]
        self.seed_cache = cache = SeedCache(cfg, task)

        class StrictEnvironment(RoboTwinEnv):
            def _prepare_planner_runtime(self):
                install_planner_adapter(self.robotwin_root, evaluation["output_dir"])

            @property
            def _expert_planner_available(self):
                return True

            def _setup_episode(self, max_seed_attempts=None):
                try:
                    episode_info = self._find_solvable_seed(evaluation["max_seed_attempts"])
                    self._setup_demo()
                    self._task_description = self._resolve_instruction(episode_info)
                    self.env.set_instruction(instruction=self._task_description)
                except Exception:
                    cache.reject(self.seed)
                    raise
                cache.record_validated(self.seed)

            def new_episode(self, max_setup_retries=5):
                self._close_task_env(clear_cache=((self._episode_index + 2) % self.args["clear_cache_freq"] == 0))
                self._episode_index += 1
                for _ in range(max_setup_retries):
                    self.seed += 1
                    try:
                        self._setup_episode()
                        return self._observation()
                    except Exception:
                        self._close_task_env()
                        if _ + 1 == max_setup_retries:
                            raise

            def _find_solvable_seed(self, max_attempts):
                return find_validated_seed(self, cache, max_attempts)

            def _resolve_instruction(self, episode_info):
                from generate_episode_instructions import generate_episode_descriptions

                descriptions = generate_episode_descriptions(self.task_name, [episode_info["info"]], evaluation["eval_num_episodes"])
                choices = descriptions[0][self.instruction_type]
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
        self.step_index = 0
        return Observation(raw.images, raw.state, self.env.task_description)

    def step(self, action, *, observe=True):
        # Official RoboTwin evaluates eval_success after take_action. An extra
        # check_success/get_obs can mutate task state or random lighting.
        self.env.env.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")
        success = bool(self.env.env.eval_success)
        done = success or self.env.env.take_action_cnt >= self.max_steps
        raw = self.env._observation() if observe and not done else None
        self.step_index += 1
        observation = Observation(raw.images, raw.state, self.env.task_description, self.step_index) if raw is not None else None
        return observation, success, done

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


class RoboTwinBenchmark(Benchmark):
    name = "robotwin"
    policy_profile = "robotwin"
    trials_field = "eval_num_episodes"
    path_fields = ("EVALUATION.robotwin_root", "EVALUATION.seed_cache_dir")
    adapter_cls = RoboTwinAdapter

    @classmethod
    def defaults(cls):
        return {
            "EVALUATION": {
                "robotwin_root": os.environ.get("ROBOTWIN_ROOT") or str(ROBOTWIN_ROOT),
                "eval_num_episodes": 100,
                "skip_get_obs_within_replan": True,
                "instruction_type": "unseen",
                "embodiment": "aloha-agilex",
                "max_seed_attempts": 1000,
                "reuse_seed_cache": False,
                "seed_cache_dir": None,
            },
            "MULTIRUN": {"chunk_size": 1, "task_names": None, "phases": ["clean", "random"]},
        }

    @classmethod
    def worker_environment(cls, env):
        super().worker_environment(env)
        if not env.get("ROBOTWIN_NVIDIA_GL_ROOT"):
            return
        root = Path(env["ROBOTWIN_NVIDIA_GL_ROOT"])
        libgl = next((p for p in (root / "libGL.so.1", root / "libGL.so.1.7.0") if p.is_file()), None)
        if libgl:
            env["LD_PRELOAD"] = ":".join(filter(None, (str(libgl), env.get("LD_PRELOAD"))))
        icd = root / "nvidia_icd_abs.json"
        if icd.is_file():
            env["VK_ICD_FILENAMES"] = str(icd)

    @classmethod
    def discover_tasks(cls, cfg):
        root = Path(cfg["EVALUATION"]["robotwin_root"])
        names = cfg["MULTIRUN"]["task_names"]
        if names is None:
            names = sorted(p.stem for p in (root / "description/task_instruction").glob("*.json"))
        phases = cfg["MULTIRUN"]["phases"]
        return [{"key": f"{phase}/{name}", "task_name": name, "phase": phase, "category": None, "suite": None} for phase in phases for name in names]
