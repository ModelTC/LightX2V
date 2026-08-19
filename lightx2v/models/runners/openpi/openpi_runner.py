import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from loguru import logger

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER

AGENTVIEW_IMAGE_NAME = "agentview_image.png"
WRIST_IMAGE_NAME = "wrist_image.png"


class OpenPIPolicy:
    """Thin stateful policy wrapper around the native LightX2V OpenPI model."""

    def __init__(self, config):
        self.config = config
        self.action_horizon = int(config.get("action_horizon", 10))
        self.output_action_dim = int(config.get("output_action_dim", 7))
        self.actions_per_plan = int(config.get("actions_per_plan", 5))
        if self.action_horizon <= 0:
            raise ValueError(f"OpenPI requires a positive action_horizon, got {self.action_horizon}.")
        if self.output_action_dim <= 0:
            raise ValueError(f"OpenPI requires a positive output_action_dim, got {self.output_action_dim}.")
        if not 1 <= self.actions_per_plan <= self.action_horizon:
            raise ValueError(f"OpenPI actions_per_plan must be in [1, {self.action_horizon}], got {self.actions_per_plan}.")

        # Keep OpenPI/Transformers imports out of LightX2V's process startup.
        # The launcher prepends the isolated, patched Transformers runtime before
        # this constructor imports the native network implementation.
        from lightx2v.models.networks.openpi import OpenPIModel

        self.model = OpenPIModel.from_config(config)
        self.pending_actions = deque()

    @classmethod
    def from_config(cls, config):
        return cls(config)

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
            # With seed=None the model advances its config-seeded generator
            # between replans instead of replaying identical flow noise.
            action_chunk = self.predict_action_chunk(images, state, task_description)
            self.pending_actions.extend(action.copy() for action in action_chunk[: self.actions_per_plan])
        if not self.pending_actions:
            raise RuntimeError("OpenPI produced an empty action chunk.")
        return self.pending_actions.popleft()

    def reset(self):
        self.pending_actions.clear()
        self.model.reset()

    def close(self):
        self.pending_actions.clear()
        self.model.close()


@RUNNER_REGISTER("openpi")
class OpenPIRunner(BaseRunner):
    """Offline π0.5-LIBERO image/state-to-action runner."""

    def init_modules(self):
        if self.config.get("task") != "i2va":
            raise ValueError(f"OpenPI currently supports only task='i2va', got {self.config.get('task')!r}.")
        logger.info("Loading native PyTorch OpenPI policy...")
        self.policy = OpenPIPolicy.from_config(self.config)
        self.config.lock()
        logger.info("Native PyTorch OpenPI policy loaded.")

    @staticmethod
    def _load_rgb(path):
        image_path = Path(path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"OpenPI image does not exist: {image_path}")
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"OpenPI expected an HWC RGB image, got {image.shape} from {image_path}.")
        # The shared LIBERO simulator already performs the official 180-degree
        # rotation. Neither the runner nor the network flips these inputs again.
        return np.ascontiguousarray(image)

    def _load_image_pair(self):
        policy_image = getattr(self.input_info, "policy_image", None)
        if policy_image is not None:
            if not isinstance(policy_image, dict):
                raise TypeError("OpenPI policy_image must be a dict with logical keys 'agentview' and 'wrist'.")
            missing = [key for key in ("agentview", "wrist") if key not in policy_image]
            if missing:
                raise KeyError(f"OpenPI policy_image is missing camera keys: {missing}")
            images = {key: np.asarray(policy_image[key], dtype=np.uint8) for key in ("agentview", "wrist")}
            for key, image in images.items():
                if image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"OpenPI camera '{key}' must be HWC RGB, got {image.shape}.")
                images[key] = np.ascontiguousarray(image)
            return images

        image_path = str(getattr(self.input_info, "image_path", "") or "").strip()
        if not image_path:
            raise ValueError("OpenPI requires image_path.")
        expanded = os.path.expanduser(image_path)
        if os.path.isdir(expanded):
            agentview_path = os.path.join(expanded, AGENTVIEW_IMAGE_NAME)
            wrist_path = os.path.join(expanded, WRIST_IMAGE_NAME)
        else:
            paths = [item.strip() for item in expanded.split(",") if item.strip()]
            if len(paths) != 2:
                raise ValueError("OpenPI image_path must be a directory containing agentview_image.png and wrist_image.png, or two comma-separated image paths in that order.")
            agentview_path, wrist_path = paths
        return {
            "agentview": self._load_rgb(agentview_path),
            "wrist": self._load_rgb(wrist_path),
        }

    @staticmethod
    def _unwrap_state_payload(payload: Any):
        if isinstance(payload, dict):
            for key in ("state", "qpos", "robot_state", "observation.state", "observation/state"):
                if key in payload:
                    return payload[key]
            raise KeyError("OpenPI state mapping must contain one of: state, qpos, robot_state, observation.state, observation/state.")
        return payload

    @staticmethod
    def _parse_text_state(text):
        normalized = str(text).strip()
        if not normalized:
            return np.empty((0,), dtype=np.float32)
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            return OpenPIRunner._unwrap_state_payload(payload)
        normalized = normalized.translate(str.maketrans({",": " ", "[": " ", "]": " ", "(": " ", ")": " ", ";": " "}))
        return np.fromstring(normalized, sep=" ", dtype=np.float32)

    def _load_state(self):
        policy_state = getattr(self.input_info, "policy_state", None)
        if policy_state is not None:
            payload = policy_state
        else:
            state_source = str(getattr(self.input_info, "state_path", "") or "").strip()
            if not state_source:
                raise ValueError("OpenPI requires state_path containing the 8-dimensional LIBERO state.")
            state_path = Path(state_source).expanduser()
            if state_path.is_file():
                suffix = state_path.suffix.lower()
                if suffix == ".npy":
                    payload = np.load(state_path, allow_pickle=True)
                    if isinstance(payload, np.ndarray) and payload.shape == () and isinstance(payload.item(), dict):
                        payload = payload.item()
                elif suffix == ".npz":
                    with np.load(state_path, allow_pickle=True) as archive:
                        payload = {key: archive[key] for key in archive.files}
                else:
                    payload = self._parse_text_state(state_path.read_text(encoding="utf-8"))
            else:
                # Inline comma/space-separated state text is convenient for
                # smoke tests while retaining state_path CLI compatibility.
                payload = self._parse_text_state(state_source)

        payload = self._unwrap_state_payload(payload)
        state = np.asarray(payload, dtype=np.float32).reshape(-1)
        expected_dim = int(self.config.get("state_dim", 8))
        if state.size != expected_dim:
            raise ValueError(f"OpenPI LIBERO state must contain {expected_dim} floats, got {state.size}.")
        if not np.isfinite(state).all():
            raise ValueError("OpenPI state contains non-finite values.")
        return np.ascontiguousarray(state)

    def _resolve_action_output_path(self):
        save_action_path = str(getattr(self.input_info, "save_action_path", "") or "").strip()
        if not save_action_path:
            raise ValueError("OpenPI offline inference requires save_action_path.")
        output_path = Path(save_action_path).expanduser().resolve()
        if output_path.suffix.lower() != ".npy":
            raise ValueError(f"OpenPI save_action_path must end in .npy, got: {output_path}")
        return output_path

    def run_pipeline(self, input_info):
        self.input_info = input_info
        task_description = str(self.input_info.prompt or "").strip()
        if not task_description:
            raise ValueError("OpenPI requires prompt as the LIBERO task_description.")

        images = self._load_image_pair()
        state = self._load_state()
        actions = self.policy.predict_action_chunk(
            images=images,
            state=state,
            task_description=task_description,
            seed=self.input_info.seed,
        )

        output_path = self._resolve_action_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, actions)
        logger.info("Saved OpenPI action chunk {} to {}", actions.shape, output_path)

        if self.input_info.return_result_tensor:
            return {"actions": actions}
        return {"actions": None}

    def end_run(self):
        if hasattr(self, "policy"):
            self.policy.close()
