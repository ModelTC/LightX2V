from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class OpenPIPolicy:
    def __init__(self, config: Any):
        self.action_horizon = int(config["action_horizon"])
        self.output_action_dim = int(config["output_action_dim"])

        # Import only after the worker activates the patched Transformers path.
        from lightx2v.models.networks.openpi import OpenPIModel

        self.model = OpenPIModel.from_config(config)

    def predict_action_chunk(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        task_description: str,
        seed: int | None = None,
    ) -> np.ndarray:
        actions = self.model.predict_action_chunk(
            images=images,
            state=state,
            task_description=task_description,
            seed=seed,
        )
        # Quantile unnormalization returns float64; casting here changes simulator inputs.
        actions = np.asarray(actions)
        expected_shape = (self.action_horizon, self.output_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"OpenPI expected action chunk shape {expected_shape}, got {actions.shape}.")
        if not np.isfinite(actions).all():
            raise ValueError("OpenPI produced non-finite actions.")
        return np.ascontiguousarray(actions)

    def reset(self) -> None:
        self.model.reset()

    def export_rng_state(self) -> str:
        encoded = self.model.get_rng_state().cpu().numpy().tobytes()
        return base64.b64encode(encoded).decode("ascii")

    def import_rng_state(self, encoded: str) -> None:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        state_array = np.frombuffer(raw, dtype=np.uint8).copy()
        import torch

        state_tensor = torch.from_numpy(state_array)
        self.model.set_rng_state(state_tensor)


@RUNNER_REGISTER("openpi")
class OpenPIRunner(BaseRunner):
    def init_modules(self) -> None:
        if self.config["task"] != "i2va":
            raise ValueError(f"OpenPI currently supports only task='i2va', got {self.config['task']!r}.")
        self.run_mode = str(self._value("run_mode", environment_name="OPENPI_RUN_MODE", default="rollout"))
        if self.run_mode not in {"rollout", "evaluate"}:
            raise ValueError(f"Unsupported OpenPI run mode {self.run_mode!r}; expected 'rollout' or 'evaluate'.")
        self.config.lock()

    def warmup(self) -> None:
        # Model initialization belongs to the isolated worker.
        pass

    def _value(self, config_name: str, *, environment_name: str | None = None, default: Any = None) -> Any:
        if environment_name:
            environment_value = os.environ.get(environment_name)
            if environment_value is not None and environment_value.strip() != "":
                return environment_value
        value = self.config.get(config_name)
        return default if value is None or value == "" else value

    @staticmethod
    def _path(value: Any, label: str, *, suffix: str | None = None, must_exist: bool = False) -> Path:
        if value is None or value == "":
            raise ValueError(f"OpenPI requires {label}.")
        path = Path(str(value)).expanduser().resolve()
        if suffix is not None and path.suffix.lower() != suffix:
            raise ValueError(f"OpenPI {label} must end in {suffix}: {path}")
        if must_exist and not path.exists():
            raise FileNotFoundError(f"OpenPI {label} does not exist: {path}")
        return path

    def _configured_path(
        self,
        config_name: str,
        label: str,
        environment_name: str | None = None,
        suffix: str | None = None,
        must_exist: bool = False,
    ) -> Path:
        return self._path(
            self._value(config_name, environment_name=environment_name),
            label,
            suffix=suffix,
            must_exist=must_exist,
        )

    def _model_path(self) -> Path:
        return self._configured_path("model_path", label="model_path", must_exist=True)

    def _model_config_path(self) -> Path:
        return self._configured_path("config_json", label="config_json", suffix=".json", must_exist=True)

    def _libero_root(self) -> Path:
        return self._configured_path(
            "libero_root",
            label="LIBERO root",
            environment_name="OPENPI_LIBERO_ROOT",
            must_exist=True,
        )

    def _libero_config_dir(self) -> Path:
        return self._configured_path(
            "libero_config_dir",
            label="LIBERO config directory",
            environment_name="OPENPI_LIBERO_CONFIG_DIR",
        )

    def _worker_environment(self) -> dict[str, str]:
        runtime_path = self._configured_path(
            "transformers_runtime_path",
            label="patched Transformers runtime",
            environment_name="OPENPI_TRANSFORMERS_RUNTIME_PATH",
            must_exist=True,
        )
        if not (runtime_path / "transformers").is_dir():
            raise FileNotFoundError(f"OpenPI patched Transformers package is missing: {runtime_path / 'transformers'}")

        child_env = os.environ.copy()
        child_env["USE_FLAX"] = "0"
        child_env.pop("MUJOCO_EGL_DEVICE_ID", None)
        child_env["PYTHONPATH"] = os.pathsep.join((str(runtime_path), str(PROJECT_ROOT)))
        return child_env

    def _option(self, environment_name: str, config_name: str, default: Any) -> str:
        return str(self._value(config_name, environment_name=environment_name, default=default))

    def _worker_command(self, module: str) -> list[str]:
        return [
            "-m",
            module,
            "--model-path",
            str(self._model_path()),
            "--config-json",
            str(self._model_config_path()),
            "--libero-root",
            str(self._libero_root()),
            "--libero-config-dir",
            str(self._libero_config_dir()),
        ]

    def _rollout_output_paths(self, input_info: Any) -> tuple[Path, Path, Path]:
        video_path = self._path(input_info.save_result_path, "save_result_path", suffix=".mp4")
        action_value = input_info.save_action_path or self._value("save_action_path", environment_name="OPENPI_SAVE_ACTION_PATH")
        action_path = self._path(action_value or video_path.with_suffix(".actions.npy"), "save_action_path", suffix=".npy")
        metrics_value = self._value("save_metrics_path", environment_name="OPENPI_SAVE_METRICS_PATH")
        metrics_path = self._path(metrics_value or video_path.with_suffix(".metrics.json"), "save_metrics_path", suffix=".json")
        return video_path, action_path, metrics_path

    def _rollout_command(self, input_info: Any) -> list[str]:
        video_path, action_path, metrics_path = self._rollout_output_paths(input_info)
        command = [
            *self._worker_command("lightx2v.models.runners.openpi.libero_rollout"),
            "--benchmark",
            self._option("LIBERO_BENCHMARK", "libero_benchmark", "libero_spatial"),
            "--task-id",
            self._option("LIBERO_TASK_ID", "libero_task_id", 0),
            "--init-state-id",
            self._option("LIBERO_INIT_STATE_ID", "libero_init_state_id", 0),
            "--env-seed",
            self._option("OPENPI_ENV_SEED", "env_seed", 7),
            "--policy-seed",
            str(self._value("policy_seed", environment_name="OPENPI_POLICY_SEED", default=input_info.seed)),
            "--actions-per-plan",
            self._option("OPENPI_ACTIONS_PER_PLAN", "actions_per_plan", 5),
            "--num-steps-wait",
            self._option("OPENPI_NUM_STEPS_WAIT", "num_steps_wait", 10),
            "--render-size",
            self._option("OPENPI_RENDER_SIZE", "render_size", 256),
            "--fps",
            self._option("OPENPI_VIDEO_FPS", "video_fps", 10),
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
        max_steps = self._value("max_steps", environment_name="OPENPI_MAX_STEPS")
        if max_steps is not None:
            command.extend(("--max-steps", str(max_steps)))
        return command

    @staticmethod
    def _boolean_argument(value: Any, enabled: str, disabled: str) -> str | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            is_enabled = value
        else:
            normalized = str(value).strip()
            if normalized not in {"0", "1"}:
                raise ValueError(f"Boolean OpenPI option must be 0 or 1, got {value!r}")
            is_enabled = normalized == "1"
        return enabled if is_enabled else disabled

    def _evaluate_command(self, input_info: Any) -> list[str]:
        eval_config = self._configured_path(
            "eval_config",
            label="evaluation config",
            environment_name="OPENPI_EVAL_CONFIG",
            suffix=".json",
            must_exist=True,
        )
        output_dir = self._path(input_info.save_result_path, "evaluation output directory")
        if output_dir.exists() and not output_dir.is_dir():
            raise NotADirectoryError(f"OpenPI evaluation output is not a directory: {output_dir}")
        command = [
            *self._worker_command("lightx2v.models.runners.openpi.libero_evaluate"),
            "--eval-config",
            str(eval_config),
            "--output-dir",
            str(output_dir),
        ]
        for environment_name, config_name, argument in (
            ("OPENPI_EVAL_BENCHMARKS", "eval_benchmarks", "--benchmarks"),
            ("OPENPI_EVAL_TASK_IDS", "eval_task_ids", "--task-ids"),
            ("OPENPI_EVAL_NUM_TRIALS_PER_TASK", "eval_num_trials_per_task", "--num-trials-per-task"),
            ("OPENPI_EVAL_MAX_STEPS", "eval_max_steps", "--max-steps"),
            ("OPENPI_EVAL_VIDEO_POLICY", "eval_video_policy", "--video-policy"),
        ):
            value = self._value(config_name, environment_name=environment_name)
            if value is not None and value != "":
                command.extend((argument, str(value)))
        for environment_name, config_name, enabled, disabled in (
            ("OPENPI_EVAL_RESUME", "eval_resume", "--resume", "--no-resume"),
            ("OPENPI_EVAL_FAIL_FAST", "eval_fail_fast", "--fail-fast", "--no-fail-fast"),
            ("OPENPI_EVAL_SAVE_ACTIONS", "eval_save_actions", "--save-actions", "--no-save-actions"),
        ):
            argument = self._boolean_argument(self._value(config_name, environment_name=environment_name), enabled, disabled)
            if argument is not None:
                command.append(argument)
        return command

    def _run_local_worker(self, input_info: Any) -> dict[str, Any]:
        arguments = self._rollout_command(input_info) if self.run_mode == "rollout" else self._evaluate_command(input_info)
        command = [sys.executable, *arguments]
        subprocess.run(command, env=self._worker_environment(), check=True)

        if input_info.return_result_tensor and self.run_mode == "rollout":
            _, action_path, _ = self._rollout_output_paths(input_info)
            return {"actions": np.load(action_path)}
        return {"actions": None}

    def run_pipeline(self, input_info: Any) -> dict[str, Any]:
        return self._run_local_worker(input_info)
