import json
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel
from lightx2v.models.networks.motus.model import MotusModel
from lightx2v.models.runners.wan.wan_runner import Wan22DenseRunner
from lightx2v.models.schedulers.motus.scheduler import MotusScheduler
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.models.video_encoders.hf.wan.vae_tiny import Wan2_2_VAE_tiny
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import find_torch_model_path, save_to_video, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE


def _merge_wan_dense_defaults(config):
    defaults = {
        "feature_caching": "NoCaching",
        "use_image_encoder": False,
        "enable_cfg": False,
        "sample_guide_scale": 1.0,
        "sample_shift": 5.0,
        "text_len": 512,
        "num_channels_latents": 48,
        "patch_size": [1, 2, 2],
        "vae_stride": [4, 16, 16],
    }
    wan_config_path = Path(config["wan_path"]).expanduser().resolve() / "config.json"
    with config.temporarily_unlocked():
        for key, value in defaults.items():
            config.setdefault(key, value)
        if wan_config_path.exists():
            with open(wan_config_path, "r") as f:
                wan_config = json.load(f)
            for key, value in wan_config.items():
                config.setdefault(key, value)


@RUNNER_REGISTER("motus")
class MotusRunner(Wan22DenseRunner):
    def __init__(self, config):
        _merge_wan_dense_defaults(config)
        super().__init__(config)

    def set_init_device(self):
        self.init_device = torch.device(AI_DEVICE)

    def _wan_lookup_config(self):
        lookup_config = self.config.copy()
        with lookup_config.temporarily_unlocked():
            lookup_config["model_path"] = self.config["wan_path"]
        return lookup_config

    def init_scheduler(self):
        self.scheduler = MotusScheduler(self.config)

    def load_transformer(self):
        return MotusModel(self.config, self.init_device)

    def load_text_encoder(self):
        wan_lookup_config = self._wan_lookup_config()
        tokenizer_path = os.path.join(self.config["wan_path"], "google", "umt5-xxl")
        t5_original_ckpt = find_torch_model_path(wan_lookup_config, "t5_original_ckpt", "models_t5_umt5-xxl-enc-bf16.pth")
        text_encoder = T5EncoderModel(
            text_len=self.config["text_len"],
            dtype=torch.bfloat16,
            device=torch.device(AI_DEVICE),
            checkpoint_path=t5_original_ckpt,
            tokenizer_path=tokenizer_path,
            shard_fn=None,
            t5_quantized=False,
            t5_quantized_ckpt=None,
            quant_scheme=None,
            load_from_rank0=self.config.get("load_from_rank0", False),
            lazy_load=False,
        )
        return [text_encoder]

    def load_vae_encoder(self):
        wan_lookup_config = self._wan_lookup_config()
        vae_config = {
            "vae_path": find_torch_model_path(wan_lookup_config, "vae_path", self.vae_name),
            "device": torch.device(AI_DEVICE),
            "parallel": self.get_vae_parallel(),
            "use_tiling": self.config.get("use_tiling_vae", False),
            "dtype": GET_DTYPE(),
            "load_from_rank0": self.config.get("load_from_rank0", False),
            "use_lightvae": self.config.get("use_lightvae", False),
        }
        return Wan2_2_VAE(**vae_config)

    def load_vae_decoder(self):
        wan_lookup_config = self._wan_lookup_config()
        vae_config = {
            "vae_path": find_torch_model_path(wan_lookup_config, "vae_path", self.vae_name),
            "device": torch.device(AI_DEVICE),
            "parallel": self.get_vae_parallel(),
            "use_tiling": self.config.get("use_tiling_vae", False),
            "use_lightvae": self.config.get("use_lightvae", False),
            "dtype": GET_DTYPE(),
            "load_from_rank0": self.config.get("load_from_rank0", False),
        }
        if self.config.get("use_tae", False):
            tae_path = find_torch_model_path(wan_lookup_config, "tae_path", self.tiny_vae_name)
            return Wan2_2_VAE_tiny(vae_path=tae_path, device=self.init_device, need_scaled=self.config.get("need_scaled", False)).to(AI_DEVICE)
        return Wan2_2_VAE(**vae_config)

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

    def _save_outputs(self, decoded_video: torch.Tensor, pred_actions: torch.Tensor):
        video = wan_vae_to_comfy(decoded_video)
        video_path = str(Path(self.input_info.save_result_path).expanduser().resolve())
        action_path = self._resolve_action_output_path()

        save_to_video(video, video_path, fps=float(self.config.get("fps", 4)), method="ffmpeg")

        Path(action_path).parent.mkdir(parents=True, exist_ok=True)
        with open(action_path, "w") as f:
            json.dump(pred_actions.detach().cpu().float().tolist(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved Motus video to {video_path}")
        logger.info(f"Saved Motus actions to {action_path}")

    @ProfilingContext4DebugL1("RUN pipeline", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_worker_request_duration, metrics_labels=["MotusRunner"])
    def run_pipeline(self, input_info):
        self.input_info = input_info

        if not self.input_info.image_path:
            raise ValueError("Motus requires `image_path`.")
        if not self.input_info.state_path:
            raise ValueError("Motus requires `state_path`.")
        if not self.input_info.prompt:
            raise ValueError("Motus requires `prompt`.")
        if not self.input_info.save_result_path:
            raise ValueError("Motus requires `save_result_path`.")

        state_value = self._load_state_value(self.input_info.state_path)
        self.inputs = self.run_input_encoder()
        self.inputs = self.model.prepare_runtime_inputs(
            self.inputs,
            image_path=self.input_info.image_path,
            prompt=self.input_info.prompt,
            state_value=state_value,
        )
        prepared_state = self.inputs["motus_state"]
        self.scheduler.prepare(
            seed=self.input_info.seed,
            latent_shape=self.input_info.latent_shape,
            image_encoder_output=self.inputs["image_encoder_output"],
            action_shape=(prepared_state.shape[0], self.model.action_chunk_size, self.model.action_dim),
        )

        with ProfilingContext4DebugL1("Run Motus DiT"):
            for step_index in range(self.scheduler.infer_steps):
                self.scheduler.step_pre(step_index)
                self.model.infer(self.inputs)
                self.scheduler.step_post()

        video_latents = self.scheduler.video_latents.squeeze(0)
        pred_actions = self.model.postprocess_actions()
        pred_actions = pred_actions.detach().cpu()
        decoded_video = self.run_vae_decoder(video_latents)
        self._save_outputs(decoded_video, pred_actions)
        self.end_run()
        return {"video": None, "actions": pred_actions}
