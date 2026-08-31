import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
from loguru import logger

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER


class OpenPIPolicy:
    """Stateful action-queue wrapper around the native OpenPI model."""

    def __init__(self, config):
        self.action_horizon = config["action_horizon"]
        self.output_action_dim = config["output_action_dim"]
        self.actions_per_plan = config.get("actions_per_plan", 5)
        if not 1 <= self.actions_per_plan <= self.action_horizon:
            raise ValueError(f"OpenPI actions_per_plan must be in [1, {self.action_horizon}], got {self.actions_per_plan}.")

        # Delay this import until the process has the patched Transformers runtime.
        from lightx2v.models.networks.openpi import OpenPIModel

        self.model = OpenPIModel.from_config(config)
        self.pending_actions = deque()

    def predict_action_chunk(self, images, state, task_description, seed=None):
        actions = self.model.predict_action_chunk(
            images=images,
            state=state,
            task_description=task_description,
            seed=seed,
        )
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.action_horizon, self.output_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"OpenPI expected action chunk shape {expected_shape}, got {actions.shape}.")
        if not np.isfinite(actions).all():
            raise ValueError("OpenPI produced non-finite actions.")
        return np.ascontiguousarray(actions)

    def next_action(self, images, state, task_description):
        if not self.pending_actions:
            # seed=None advances the model-local generator between replans.
            action_chunk = self.predict_action_chunk(images, state, task_description)
            self.pending_actions.extend(action.copy() for action in action_chunk[: self.actions_per_plan])
        return self.pending_actions.popleft()

    def clear_action_queue(self):
        self.pending_actions.clear()

    def reset_rng(self):
        self.model.reset()

    def reset(self):
        self.clear_action_queue()
        self.reset_rng()

    def close(self):
        self.clear_action_queue()


@RUNNER_REGISTER("openpi")
class OpenPIRunner(BaseRunner):
    """Dispatch local pi0.5-LIBERO rollout and evaluation workers."""

    def init_modules(self):
        if self.config["task"] != "i2va":
            raise ValueError(f"OpenPI currently supports only task='i2va', got {self.config['task']!r}.")

        self.run_mode = os.environ.get("OPENPI_RUN_MODE", "rollout")
        if self.run_mode not in {"rollout", "evaluate"}:
            raise ValueError(f"Unsupported OpenPI run mode {self.run_mode!r}; expected 'rollout' or 'evaluate'.")
        logger.info("OpenPI selected by lightx2v.infer; dispatch mode={}", self.run_mode)
        self.config.lock()

    @staticmethod
    def _require_path(value, label, suffix=None):
        if value is None or value == "":
            raise ValueError(f"OpenPI requires {label}.")
        path = Path(value).expanduser().resolve()
        if suffix is not None and path.suffix.lower() != suffix:
            raise ValueError(f"OpenPI {label} must end in {suffix}: {path}")
        return path

    @classmethod
    def _environment_path(cls, name, suffix=None):
        return cls._require_path(os.environ.get(name), name, suffix)

    def _model_path(self):
        return self._require_path(self.config["model_path"], "model_path")

    def _model_config_path(self):
        return self._require_path(self.config["config_json"], "config_json", ".json")

    def _worker_environment(self):
        runtime_path = self._environment_path("OPENPI_TRANSFORMERS_RUNTIME_PATH")
        if not (runtime_path / "transformers").is_dir():
            raise FileNotFoundError(f"OpenPI patched Transformers package is missing: {runtime_path / 'transformers'}")

        project_root = Path(__file__).resolve().parents[4]
        child_env = os.environ.copy()
        child_env["USE_FLAX"] = "0"
        python_paths = (str(runtime_path), str(project_root), child_env.get("PYTHONPATH", ""))
        child_env["PYTHONPATH"] = os.pathsep.join(path for path in python_paths if path)
        return child_env

    def _rollout_command(self, input_info):
        video_path = self._require_path(input_info.save_result_path, "save_result_path", ".mp4")
        action_path = self._require_path(input_info.save_action_path, "save_action_path", ".npy")
        metrics_value = os.environ.get("OPENPI_SAVE_METRICS_PATH", "").strip()
        metrics_path = self._require_path(metrics_value, "save_metrics_path", ".json") if metrics_value else video_path.with_suffix(".metrics.json")
        command = [
            "-m",
            "lightx2v.models.runners.openpi.libero_rollout",
            "--model-path",
            str(self._model_path()),
            "--config-json",
            str(self._model_config_path()),
            "--libero-root",
            str(self._environment_path("OPENPI_LIBERO_ROOT")),
            "--libero-config-dir",
            str(self._environment_path("OPENPI_LIBERO_CONFIG_DIR")),
            "--benchmark",
            os.environ.get("LIBERO_BENCHMARK", "libero_spatial"),
            "--task-id",
            os.environ.get("LIBERO_TASK_ID", "0"),
            "--init-state-id",
            os.environ.get("LIBERO_INIT_STATE_ID", "0"),
            "--seed",
            str(input_info.seed),
            "--actions-per-plan",
            os.environ.get("OPENPI_ACTIONS_PER_PLAN", str(self.config.get("actions_per_plan", 5))),
            "--num-steps-wait",
            os.environ.get("OPENPI_NUM_STEPS_WAIT", str(self.config.get("num_steps_wait", 10))),
            "--render-size",
            os.environ.get("OPENPI_RENDER_SIZE", "256"),
            "--fps",
            os.environ.get("OPENPI_VIDEO_FPS", "10"),
            "--save-video-path",
            str(video_path),
            "--save-action-path",
            str(action_path),
            "--save-metrics-path",
            str(metrics_path),
        ]
        task_description = str(input_info.prompt or "").strip()
        if task_description:
            command.extend(("--task-description", task_description))
        max_steps = os.environ.get("OPENPI_MAX_STEPS", "").strip()
        if max_steps:
            command.extend(("--max-steps", max_steps))
        return command

    @staticmethod
    def _environment_flag(name):
        value = os.environ.get(name)
        if value is None:
            return None
        if value not in {"0", "1"}:
            raise ValueError(f"{name} must be 0 or 1, got {value!r}.")
        return value == "1"

    def _evaluate_command(self, input_info):
        eval_config = self._environment_path("OPENPI_EVAL_CONFIG", ".json")
        if not eval_config.is_file():
            raise FileNotFoundError(f"OpenPI evaluation config does not exist: {eval_config}")

        output_dir = self._require_path(input_info.save_result_path, "evaluation output_dir")
        if output_dir.exists() and not output_dir.is_dir():
            raise NotADirectoryError(f"OpenPI evaluation output_dir is not a directory: {output_dir}")

        command = [
            "-m",
            "lightx2v.models.runners.openpi.libero_evaluate",
            "--model-path",
            str(self._model_path()),
            "--config-json",
            str(self._model_config_path()),
            "--eval-config",
            str(eval_config),
            "--output-dir",
            str(output_dir),
            "--libero-root",
            str(self._environment_path("OPENPI_LIBERO_ROOT")),
            "--libero-config-dir",
            str(self._environment_path("OPENPI_LIBERO_CONFIG_DIR")),
        ]

        benchmarks = os.environ.get("OPENPI_EVAL_BENCHMARKS", "").strip()
        if benchmarks:
            command.extend(("--benchmarks", benchmarks))

        override_arguments = (
            ("OPENPI_EVAL_TASK_IDS", "--task-ids"),
            ("OPENPI_EVAL_NUM_TRIALS_PER_TASK", "--num-trials-per-task"),
            ("OPENPI_EVAL_MAX_STEPS", "--max-steps"),
            ("OPENPI_EVAL_VIDEO_POLICY", "--video-policy"),
        )
        for environment_name, argument_name in override_arguments:
            value = os.environ.get(environment_name, "").strip()
            if value:
                command.extend((argument_name, value))

        boolean_overrides = (
            ("OPENPI_EVAL_RESUME", "--resume", "--no-resume"),
            ("OPENPI_EVAL_FAIL_FAST", "--fail-fast", "--no-fail-fast"),
            ("OPENPI_EVAL_SAVE_ACTIONS", "--save-actions", "--no-save-actions"),
        )
        for environment_name, enabled_argument, disabled_argument in boolean_overrides:
            enabled = self._environment_flag(environment_name)
            if enabled is not None:
                command.append(enabled_argument if enabled else disabled_argument)
        return command

    def _run_isolated_worker(self, input_info):
        arguments = self._rollout_command(input_info) if self.run_mode == "rollout" else self._evaluate_command(input_info)
        python_bin = os.environ.get("OPENPI_PYTHON", sys.executable)
        command = [python_bin, *arguments]
        logger.info("Starting synchronous local OpenPI {} worker", self.run_mode)
        subprocess.run(command, env=self._worker_environment(), check=True)
        logger.info("OpenPI {} worker completed", self.run_mode)

        if input_info.return_result_tensor and self.run_mode == "rollout":
            action_path = self._require_path(input_info.save_action_path, "save_action_path", ".npy")
            return {"actions": np.load(action_path)}
        return {"actions": None}

    def run_pipeline(self, input_info):
        return self._run_isolated_worker(input_info)
