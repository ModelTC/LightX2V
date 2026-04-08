import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from lightx2v.models.networks.motus.model import MotusModel
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import save_to_video
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.profiler import *
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("motus")
class MotusRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(AI_DEVICE)
        self.adapter = None

    @ProfilingContext4DebugL2("Load models")
    def init_modules(self):
        self.adapter = MotusModel(self.config, self.device)

    def _load_state_value(self, state_path: str):
        state_path = str(Path(state_path).expanduser().resolve())
        suffix = Path(state_path).suffix.lower()
        if suffix == ".npy":
            return np.load(state_path)
        if suffix in [".pt", ".pth"]:
            value = torch.load(state_path, map_location="cpu")
            if isinstance(value, dict):
                for key in ["state", "qpos", "joint_state", "initial_state"]:
                    if key in value:
                        return value[key]
            return value
        if suffix == ".json":
            with open(state_path, "r") as f:
                value = json.load(f)
            if isinstance(value, dict):
                for key in ["state", "qpos", "joint_state", "initial_state"]:
                    if key in value:
                        return value[key]
            return value
        if suffix in [".txt", ".csv"]:
            text = Path(state_path).read_text().strip().replace("\n", ",")
            return [float(item) for item in text.split(",") if item.strip()]
        raise ValueError(f"Unsupported state file format: {state_path}")

    def _resolve_action_output_path(self):
        if self.input_info.save_action_path:
            return str(Path(self.input_info.save_action_path).expanduser().resolve())
        return str(Path(self.input_info.save_result_path).expanduser().resolve().with_suffix(".actions.json"))

    def _save_outputs(self, pred_frames: torch.Tensor, pred_actions: torch.Tensor):
        video_path = str(Path(self.input_info.save_result_path).expanduser().resolve())
        action_path = self._resolve_action_output_path()

        video = pred_frames.clamp(0, 1).permute(0, 2, 3, 1).contiguous()
        save_to_video(video, video_path, fps=float(self.config.get("fps", 4)), method="ffmpeg")

        Path(action_path).parent.mkdir(parents=True, exist_ok=True)
        with open(action_path, "w") as f:
            json.dump(pred_actions.detach().cpu().float().tolist(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved Motus video to {video_path}")
        logger.info(f"Saved Motus actions to {action_path}")

    @ProfilingContext4DebugL1("RUN pipeline", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_worker_request_duration, metrics_labels=["MotusRunner"])
    def run_pipeline(self, input_info):
        self.input_info = input_info
        if self.adapter is None:
            self.init_modules()

        if not self.input_info.image_path:
            raise ValueError("Motus requires `image_path`.")
        if not self.input_info.state_path:
            raise ValueError("Motus requires `state_path`.")
        if not self.input_info.prompt:
            raise ValueError("Motus requires `prompt`.")
        if not self.input_info.save_result_path:
            raise ValueError("Motus requires `save_result_path`.")

        state_value = self._load_state_value(self.input_info.state_path)
        pred_frames, pred_actions = self.adapter.infer(
            image_path=self.input_info.image_path,
            prompt=self.input_info.prompt,
            state_value=state_value,
            num_inference_steps=int(self.config.get("num_inference_steps", 10)),
        )
        self._save_outputs(pred_frames, pred_actions)

