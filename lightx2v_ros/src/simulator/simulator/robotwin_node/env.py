"""RoboTwin (SAPIEN, dual-arm) implementation of the generic `BaseSimEnv`.

This adapter wraps a vendored RoboTwin task so the generic `SimulatorNode` can
drive it exactly like LIBERO. It mirrors the official evaluation flow in
``RoboTwin/script/eval_policy.py``:

1. **Expert check**: for each candidate seed, run ``setup_demo`` + ``play_once``
   (the scripted expert). Seeds where the expert plan fails or the task cannot
   be completed are skipped -- without this filter the policy is sometimes asked
   to solve unsolvable layouts, which looks like endless arm jitter + timeout.
2. **Instruction generation**: the expert run populates ``env.info["info"]``
   (e.g. which pot model was spawned); those parameters fill the task's
   instruction templates via ``generate_episode_descriptions``. Skipping the
   expert run is why instructions used to fall back to a generic sentence.
3. Re-run ``setup_demo`` with the accepted seed and expose ``reset``/``step``
   (``get_obs`` + ``take_action(qpos)``) to the ROS node.

RoboTwin heavy dependencies (sapien, mplib, curobo, ...) and assets are imported
lazily inside ``reset``/construction, so the ROS package builds and imports even
on machines where the RoboTwin runtime is not installed yet.
"""

import contextlib
import importlib
import io
import os
import sys
from pathlib import Path

import numpy as np
from common.contract import EnvContract

from ..sim.base_env import BaseSimEnv, Observation

FALLBACK_INSTRUCTION = "Complete the {task} task."


def default_robotwin_root() -> Path:
    return Path(__file__).resolve().parent / "RoboTwin"


def resolve_robotwin_root(path=None) -> Path:
    """Return a normalized RoboTwin root supplied by ROS or the default."""
    raw_path = str(path).strip() if path is not None else ""
    if not raw_path:
        return default_robotwin_root().resolve()
    return Path(os.path.expandvars(raw_path)).expanduser().resolve()


def _add_python_path(path) -> None:
    path = str(Path(path))
    if path not in sys.path:
        sys.path.insert(0, path)


class RoboTwinEnv(BaseSimEnv):
    """Single-episode RoboTwin environment exposed through the BaseSimEnv contract."""

    def __init__(
        self,
        contract: EnvContract,
        *,
        task_name: str = "click_alarmclock",
        task_config: str = "demo_clean",
        embodiment: str = "aloha-agilex",
        instruction_type: str = "unseen",
        instruction: str = "",
        seed: int = 0,
        expert_check: bool = True,
        render_publish_every: int = 15,
        intermediate_cameras: str = "head_camera",
        robotwin_root=None,
        logger=None,
    ):
        super().__init__(contract)
        self.robotwin_root = resolve_robotwin_root(robotwin_root)
        self.task_name = str(task_name)
        self.task_config = str(task_config)
        self.embodiment = str(embodiment).strip()
        self.instruction_type = str(instruction_type)
        self._fixed_instruction = str(instruction).strip()
        self.seed = int(seed)
        self.expert_check = bool(expert_check)
        self._render_every = int(render_publish_every)
        # Cameras rendered for intermediate (mid-action) viewer frames. Each one
        # costs a full extra scene render per frame (~50-100ms), so keep this list
        # short; the remaining cameras still refresh at every action boundary.
        self._intermediate_cams = tuple(c.strip() for c in str(intermediate_cameras).split(",") if c.strip() and c.strip() in contract.cameras)
        self._logger = logger
        self._episode_index = 0

        self._task_description = ""
        self._configs_path = self.robotwin_root / "task_config"
        self._node_frame_cb = None
        self._physics_step_count = 0

        self._prepare_runtime()
        self.args = self._build_task_args()
        self.env = self._instantiate_task()
        self._setup_episode()

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger.info(message)
        else:
            print(f"[robotwin_env] {message}")

    # ------------------------------------------------------------------ setup
    def _prepare_runtime(self) -> None:
        root = self.robotwin_root
        required_paths = {
            "envs/": (root / "envs").is_dir(),
            "task_config/": (root / "task_config").is_dir(),
            "assets/": (root / "assets").is_dir(),
            "assets/objects/objaverse/list.json": (
                root / "assets" / "objects" / "objaverse" / "list.json"
            ).is_file(),
            "assets/objects/same.json": (root / "assets" / "objects" / "same.json").is_file(),
        }
        missing = [name for name, exists in required_paths.items() if not exists]
        if missing:
            raise FileNotFoundError(
                f"Invalid or incomplete RoboTwin root '{root}': missing {', '.join(missing)}. "
                "Pass the directory containing envs/, task_config/, and assets/ "
                "with '--ros-args -p robotwin_root:=/path/to/RoboTwin'."
            )

        # RoboTwin source uses root-relative imports such as `from envs import ...`
        # and `from generate_episode_instructions import *`. It also resolves many
        # resources from the process working directory (for example
        # `./assets/objects/objaverse/list.json`), so the dedicated simulator node
        # must run from the selected RoboTwin root for its entire lifetime.
        os.chdir(root)
        _add_python_path(root)
        _add_python_path(root / "description" / "utils")
        self._log(f"using RoboTwin root: {root}")

    def _require_config(self, *parts) -> Path:
        path = self._configs_path.joinpath(*parts)
        if not path.exists():
            raise FileNotFoundError(f"Missing RoboTwin config: {path}. Populate `task_config/` (and `assets/`) from the official RoboTwin repo (see robotwin_node/RoboTwin/script).")
        return path

    def _prepare_planner_runtime(self) -> None:
        """Make RoboTwin importable when its optional Curobo planner is absent.

        RoboTwin catches the Curobo import error in ``planner.py``, but then
        ``robot.py`` unconditionally imports ``CuroboPlanner`` from that module.
        The ROS adapter only executes qpos actions, so it can safely use a small
        compatibility planner for scene initialization and gripper interpolation.
        The scripted expert remains disabled through ``CUROBO_AVAILABLE=False``.
        """
        planner_module = sys.modules.get("envs.robot.planner")
        if planner_module is not None and hasattr(planner_module, "CuroboPlanner"):
            if not hasattr(planner_module, "CUROBO_AVAILABLE"):
                planner_module.CUROBO_AVAILABLE = True
            return

        # Importing this submodule first executes envs.robot.__init__, whose
        # unconditional CuroboPlanner import is precisely the upstream bug. Keep
        # its expected warning/traceback out of the ROS log and inspect the
        # successfully loaded planner submodule after that import fails.
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        import_error = None
        try:
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                importlib.import_module("envs.robot.planner")
        except ImportError as exc:
            import_error = exc

        planner_module = sys.modules.get("envs.robot.planner")
        if planner_module is None:
            captured = captured_stdout.getvalue() + captured_stderr.getvalue()
            if captured:
                print(captured, file=sys.stderr, end="")
            if import_error is not None:
                raise import_error
            raise ImportError("RoboTwin did not load envs.robot.planner")

        if hasattr(planner_module, "CuroboPlanner"):
            planner_module.CUROBO_AVAILABLE = True
            return

        class CuroboPlanner:
            """No-Curobo compatibility planner for qpos-only simulation."""

            def __init__(self, *args, **kwargs):
                pass

            @staticmethod
            def plan_grippers(now_val, target_val):
                num_step = 200
                return {
                    "num_step": num_step,
                    "per_step": (target_val - now_val) / num_step,
                    "result": np.linspace(now_val, target_val, num_step),
                }

            @staticmethod
            def plan_path(*args, **kwargs):
                return {"status": "Fail"}

            @staticmethod
            def plan_batch(*args, **kwargs):
                return {"status": np.array(["Failure"], dtype=object)}

            @staticmethod
            def update_point_cloud(*args, **kwargs):
                return None

        CuroboPlanner.__module__ = planner_module.__name__
        planner_module.CuroboPlanner = CuroboPlanner
        planner_module.CUROBO_AVAILABLE = False

        # Python removes the failed parent imports automatically. Clear any
        # remaining partial modules so the next task import sees the patched
        # planner and constructs envs.robot normally.
        sys.modules.pop("envs.robot.robot", None)
        sys.modules.pop("envs.robot", None)
        envs_module = sys.modules.get("envs")
        if envs_module is not None and hasattr(envs_module, "robot"):
            delattr(envs_module, "robot")
        self._log("curobo is unavailable: using qpos-only planner compatibility mode")

    def _build_task_args(self) -> dict:
        """Replicates third_party/RoboTwin/script/eval_policy.py:main() arg assembly."""
        import yaml

        with open(self._require_config(f"{self.task_config}.yml"), "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        args["task_name"] = self.task_name
        args["task_config"] = self.task_config

        # Allow the launch parameter to pin the embodiment (e.g. "aloha-agilex").
        if self.embodiment:
            args["embodiment"] = [self.embodiment]
        embodiment_type = args.get("embodiment")
        if not isinstance(embodiment_type, list):
            raise ValueError(f"task_config embodiment must be a list, got {embodiment_type!r}")

        with open(self._require_config("_embodiment_config.yml"), "r", encoding="utf-8") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def embodiment_file(key):
            robot_file = embodiment_types[key]["file_path"]
            if robot_file is None:
                raise ValueError(f"No embodiment file for '{key}'")
            return os.path.join(str(self.robotwin_root), robot_file) if not os.path.isabs(robot_file) else robot_file

        def embodiment_config(robot_file):
            with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(self._require_config("_camera_config.yml"), "r", encoding="utf-8") as f:
            camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = embodiment_file(embodiment_type[0])
            args["right_robot_file"] = embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = embodiment_file(embodiment_type[0])
            args["right_robot_file"] = embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")

        args["left_embodiment_config"] = embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = embodiment_config(args["right_robot_file"])
        args["eval_mode"] = True
        # Headless: never spawn the on-screen SAPIEN viewer.
        args["render_freq"] = 0
        return args

    def _instantiate_task(self):
        self._prepare_planner_runtime()
        module = importlib.import_module(f"envs.{self.task_name}")
        task_cls = getattr(module, self.task_name)
        task = task_cls()
        # Hook consumed by the vendored `Base_Task.take_action` control loop so we
        # can stream intermediate frames to the viewer while an action executes.
        task._frame_callback = self._on_physics_step
        return task

    def _setup_demo(self) -> None:
        self.env.setup_demo(now_ep_num=self._episode_index, seed=self.seed, is_test=True, **self.args)

    def _close_task_env(self, clear_cache: bool = True) -> None:
        try:
            self.env.close_env(clear_cache=clear_cache)
        except Exception:
            pass

    def _setup_episode(self, max_seed_attempts: int = 50) -> None:
        """Find a solvable seed (expert check), then build the rollout episode."""
        episode_info = self._find_solvable_seed(max_seed_attempts)
        self._setup_demo()
        instruction = self._resolve_instruction(episode_info)
        self.env.set_instruction(instruction=instruction)
        self._task_description = instruction
        self._log(f"episode ready: task={self.task_name} config={self.task_config} seed={self.seed} instruction={instruction!r}")

    @property
    def _expert_planner_available(self) -> bool:
        try:
            from envs.robot.planner import CUROBO_AVAILABLE

            return bool(CUROBO_AVAILABLE)
        except Exception:
            return False

    def _find_solvable_seed(self, max_attempts: int):
        """Advance `self.seed` to the first seed the scripted expert can solve.

        Mirrors the official eval flow (`eval_policy.py`): setup + `play_once`
        (scripted expert) + close, skipping seeds the expert cannot solve. The
        returned expert episode info carries the scene parameters (which object
        model was spawned, which arm is used, ...) that fill the instruction
        templates.

        When curobo (the expert's motion planner) is unavailable we cannot run
        the real expert, but we still need the scene parameters. In that case we
        "dry-run" `play_once` with the motion primitive stubbed out: the info
        dict is derived from `load_actors` state, so it is still correct, we
        just lose the solvability filtering.
        """
        if not self.expert_check:
            return None

        full_check = self._expert_planner_available
        if not full_check:
            self._log("curobo not installed: extracting instruction params via dry-run expert (no seed solvability filtering)")

        if not full_check:
            # Dry-run mode: the info params are deterministic per seed, so a
            # failure here would just repeat; do a single attempt and fall back
            # to the generic instruction if it raises.
            try:
                self._setup_demo()
                episode_info = self._dry_run_play_once()
                self._close_task_env()
                return episode_info
            except Exception as exc:
                self._close_task_env()
                self._log(f"dry-run expert failed ({exc!r}); falling back to generic instruction")
                return None

        last_err = None
        for _ in range(max(1, max_attempts)):
            try:
                self._setup_demo()
                episode_info = self.env.play_once()
                solvable = bool(self.env.plan_success) and bool(self.env.check_success())
                self._close_task_env()
                if solvable:
                    return episode_info
                self._log(f"seed {self.seed}: expert cannot solve this layout; trying next seed")
            except Exception as exc:  # UnStableError, planner errors, ...
                last_err = exc
                self._close_task_env()
                self._log(f"seed {self.seed}: expert check raised {exc!r}; trying next seed")
            self.seed += 1
        raise RuntimeError(f"no expert-solvable seed found after {max_attempts} attempts; last error: {last_err}")

    def _dry_run_play_once(self):
        """Run `play_once` with motion primitives stubbed to no-ops.

        Task scripts compute their instruction parameters (`info["info"]`) from
        scene state chosen in `load_actors` (object model ids, target arm, ...),
        not from the motion itself, so skipping the actual arm movement still
        yields the correct parameters -- without needing the curobo planner.
        `need_plan=False` makes grasp/place action builders return dummy poses
        instead of invoking the motion planner.
        """
        task = self.env
        stubbed = {}
        for name in ("move", "delay"):
            if hasattr(task, name):
                stubbed[name] = getattr(task, name)
                setattr(task, name, lambda *args, **kwargs: True)
        old_need_plan = getattr(task, "need_plan", True)
        task.need_plan = False
        try:
            return task.play_once()
        finally:
            task.need_plan = old_need_plan
            for name, fn in stubbed.items():
                setattr(task, name, fn)

    def _resolve_instruction(self, episode_info) -> str:
        if self._fixed_instruction:
            return self._fixed_instruction
        if episode_info:
            try:
                from generate_episode_instructions import generate_episode_descriptions

                results = generate_episode_descriptions(self.task_name, [episode_info["info"]], 1)
                if results:
                    options = results[0].get(self.instruction_type) or results[0].get("seen") or results[0].get("unseen")
                    if options:
                        return str(np.random.choice(options))
            except Exception as exc:
                self._log(f"instruction generation failed: {exc!r}; using fallback")
        return FALLBACK_INSTRUCTION.format(task=self.task_name.replace("_", " "))

    # ------------------------------------------------------------- contract API
    @property
    def task_description(self) -> str:
        return self._task_description

    def reset(self) -> Observation:
        return self._observation()

    @property
    def max_steps(self):
        # RoboTwin sets a per-task rollout cap (`step_lim`) during setup_demo.
        return getattr(self.env, "step_lim", None)

    def new_episode(self, max_setup_retries: int = 5) -> Observation:
        """Tear down the current episode and set up a fresh one (new layout).

        Advances the seed so each episode gets a different object placement; the
        expert check inside `_setup_episode` additionally skips unsolvable seeds.
        """
        last_err = None
        for _ in range(max(1, max_setup_retries)):
            self.seed += 1
            try:
                self._close_task_env()
                self._episode_index += 1
                self._setup_episode()
                return self._observation()
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"RoboTwin failed to set up a new episode after {max_setup_retries} tries; last error: {last_err}")

    # ------------------------------------------------------------- task switch
    @property
    def supports_task_switch(self) -> bool:
        return True

    def list_tasks(self):
        envs_dir = self.robotwin_root / "envs"
        instr_dir = self.robotwin_root / "description" / "task_instruction"
        tasks = []
        for path in sorted(envs_dir.glob("*.py")):
            name = path.stem
            if name.startswith("_"):
                continue
            if (instr_dir / f"{name}.json").exists():
                tasks.append(name)
        return tasks

    def list_task_configs(self):
        return [p.stem for p in sorted(self._configs_path.glob("*.yml")) if not p.stem.startswith("_")]

    def set_task(self, task_name: str, task_config: str = "", seed=None) -> Observation:
        old = (self.task_name, self.task_config, self.seed, self.args, self.env)
        self._close_task_env()
        try:
            self.task_name = str(task_name)
            if task_config:
                self.task_config = str(task_config)
            if seed is not None and str(seed).strip() != "":
                self.seed = int(seed)
            else:
                self.seed += 1
            self._episode_index += 1
            self.args = self._build_task_args()
            self.env = self._instantiate_task()
            self._setup_episode()
            return self._observation()
        except Exception:
            # Roll back so the node can keep serving the previous task.
            self.task_name, self.task_config, self.seed, self.args, self.env = old
            try:
                self._setup_demo()
                self.env.set_instruction(instruction=self._task_description)
            except Exception:
                pass
            raise

    # ------------------------------------------------------------ frame stream
    def set_frame_callback(self, callback) -> None:
        self._node_frame_cb = callback

    def _on_physics_step(self):
        """Called by the vendored take_action control loop after every physics step."""
        if self._node_frame_cb is None or self._render_every <= 0 or not self._intermediate_cams:
            return
        self._physics_step_count += 1
        if self._physics_step_count % self._render_every:
            return
        try:
            self._node_frame_cb(self._render_images(self._intermediate_cams))
        except Exception:
            # A dropped viewer frame must never break the physics loop.
            pass

    def _camera_object(self, name):
        cams = self.env.cameras
        if name == "left_camera":
            return getattr(cams, "left_camera", None)
        if name == "right_camera":
            return getattr(cams, "right_camera", None)
        if name == "observer_camera":
            return getattr(cams, "observer_camera", None)
        for cam_obj, cam_name in zip(cams.static_camera_list, cams.static_camera_name):
            if cam_name == name:
                return cam_obj
        return None

    def _render_images(self, names) -> dict:
        # The take_action loop already calls `_update_render()` each physics step,
        # so scene state is current; we only pay for the per-camera renders here.
        images = {}
        for name in names:
            cam = self._camera_object(name)
            if cam is None:
                continue
            cam.take_picture()
            rgba = np.asarray(cam.get_picture("Color"))
            images[name] = np.ascontiguousarray((rgba * 255).clip(0, 255).astype(np.uint8)[..., :3])
        return images

    # ------------------------------------------------------------------- step
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # RoboTwin policies output absolute joint targets (qpos), matching FastWAM.
        self.env.take_action(action, action_type="qpos")
        obs = self._observation()
        success = bool(getattr(self.env, "eval_success", False)) or bool(self.env.check_success())
        return obs, success

    def _observation(self) -> Observation:
        raw = self.env.get_obs()
        cameras = raw["observation"]
        images = {}
        for cam in self.contract.cameras:
            if cam == "observer_camera":
                images[cam] = np.ascontiguousarray(np.asarray(self.env.cameras.get_observer_rgb())[..., :3].astype(np.uint8))
                continue
            if cam not in cameras or "rgb" not in cameras[cam]:
                raise KeyError(f"RoboTwin observation missing camera '{cam}' rgb; got {list(cameras)}")
            rgb = np.asarray(cameras[cam]["rgb"])[..., :3]
            images[cam] = np.ascontiguousarray(rgb.astype(np.uint8))
        state = np.asarray(raw["joint_action"]["vector"], dtype=np.float32).reshape(-1)
        return Observation(images=images, state=state)

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            try:
                env.close_env()
            except Exception:
                pass


def build_robotwin_env(node) -> RoboTwinEnv:
    contract = node.contract
    node.declare_parameter("robotwin_root", str(default_robotwin_root()))
    node.declare_parameter("task_name", "click_alarmclock")
    node.declare_parameter("task_config", "demo_clean")
    node.declare_parameter("embodiment", "aloha-agilex")
    node.declare_parameter("instruction_type", "unseen")
    node.declare_parameter("instruction", "")
    node.declare_parameter("seed", 0)
    # Skip seeds the scripted expert cannot solve (official eval behaviour).
    node.declare_parameter("expert_check", True)
    # Publish an intermediate viewer frame every N physics steps during
    # take_action (0 disables). Smaller = smoother video but slower simulation.
    node.declare_parameter("render_publish_every", 15)
    # Comma-separated cameras rendered for intermediate frames (each costs an
    # extra scene render per frame; keep short for simulation speed).
    node.declare_parameter("intermediate_cameras", "head_camera")

    return RoboTwinEnv(
        contract,
        task_name=node.get_parameter("task_name").value,
        task_config=node.get_parameter("task_config").value,
        embodiment=node.get_parameter("embodiment").value,
        instruction_type=node.get_parameter("instruction_type").value,
        instruction=node.get_parameter("instruction").value,
        seed=int(node.get_parameter("seed").value),
        expert_check=bool(node.get_parameter("expert_check").value),
        render_publish_every=int(node.get_parameter("render_publish_every").value),
        intermediate_cameras=str(node.get_parameter("intermediate_cameras").value),
        robotwin_root=node.get_parameter("robotwin_root").value,
        logger=node.get_logger(),
    )
