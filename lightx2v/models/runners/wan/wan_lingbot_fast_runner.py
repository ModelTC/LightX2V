import os

import numpy as np
import torch

from lightx2v.models.input_encoders.lingbot_fast import pack_camera_embedding
from lightx2v.models.networks.wan.lingbot_fast_model import WanLingbotFastModel
from lightx2v.models.runners.wan.autoregressive_runner import WanAutoregressiveRunnerMixin
from lightx2v.models.runners.wan.wan_runner import LingbotRunner, WanRunner, build_wan_model_with_lora
from lightx2v.models.schedulers.wan.lingbot_fast.scheduler import WanLingbotFastScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.input_info import ActionI2VInputInfo
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("lingbot_world_fast")
class LingbotFastRunner(WanAutoregressiveRunnerMixin, LingbotRunner):
    """Share Wan AR scheduling while retaining Lingbot camera input handling."""

    _SUPPORTS_AUTOREGRESSIVE_WARMUP = True
    _WARMUP_TASKS = ("i2v",)
    supported_request_fields_by_task = {
        "i2v": LingbotRunner.supported_request_fields_by_task["i2v"] - {"num_frames"},
    }

    def __init__(self, config):
        WanRunner.__init__(self, config)
        self.control_type = config.get("control_type", "cam")

    def load_transformer(self):
        wan_model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = WanLingbotFastModel(**wan_model_kwargs)
        else:
            model = build_wan_model_with_lora(WanLingbotFastModel, self.config, wan_model_kwargs, lora_configs, model_type="wan2.1")
        return model

    def init_scheduler(self):
        self.scheduler = WanLingbotFastScheduler(self.config)

    def run_text_encoder(self, input_info):
        # Released LingBot Fast retains embeddings for repeated prompts.
        # Request-specific image/camera/VAE inputs are still prepared anew.
        cache = getattr(self, "_fast_text_cache", None)
        if cache is None:
            cache = self._fast_text_cache = {}
        key = (input_info.prompt, getattr(input_info, "negative_prompt", None))
        if key not in cache:
            cache[key] = super().run_text_encoder(input_info)
        return cache[key]

    def decode_segment_latents(self, segment_idx, segment_latents):
        return self.vae_decoder.cached_decode_withflag(
            segment_latents.to(self.vae_decoder.dtype),
            segment_idx == 0,
            segment_idx == self.video_segment_num - 1,
        )

    def validate_warmup(self):
        super().validate_warmup()
        if self.control_type != "cam":
            raise NotImplementedError("Lingbot Fast warmup supports camera-conditioned i2v")

    def get_target_latent_size(self, target_height, target_width):
        _, stride_h, stride_w = self.config["vae_stride"]
        _, patch_h, patch_w = self.config.get("patch_size", (1, 2, 2))
        latent_h = target_height // stride_h // patch_h * patch_h
        latent_w = target_width // stride_w // patch_w * patch_w
        if self.get_vae_parallel() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            latent_h, latent_w, _, _ = self._adjust_latent_for_grid_splitting(latent_h, latent_w, torch.distributed.get_world_size())
        return latent_h, latent_w

    def get_warmup_latent_shape(self, target_height, target_width):
        return self.get_latent_shape_with_lat_hw(*self.get_target_latent_size(target_height, target_width))

    def run_vae_encoder(self, first_frame, last_frame=None):
        # Use the camera-calibrated output grid for both real requests and warmup.
        latent_h, latent_w = self.get_target_latent_size(*self.get_target_size())
        encoded = self.get_vae_encoder_output(first_frame, latent_h, latent_w, last_frame)
        if torch.distributed.is_initialized() and (self.config.get("tensor_parallel", False) or self.config.get("seq_parallel", False)):
            # TP/SP replicas must consume identical conditioning even when
            # independently compiled FP32 convolutions choose different plans.
            encoded = encoded.contiguous()
            torch.distributed.broadcast(encoded, src=0)
        return encoded, self.get_latent_shape_with_lat_hw(latent_h, latent_w)

    def get_latent_shape_with_lat_hw(self, latent_h, latent_w):
        shape = super().get_latent_shape_with_lat_hw(latent_h, latent_w)
        chunk = self.scheduler.num_frame_per_chunk
        shape[1] -= shape[1] % chunk
        return shape

    def get_vae_encoder_output(self, first_frame, lat_h, lat_w, last_frame=None):
        # The released pipeline trims to complete latent chunks before it
        # constructs the RGB condition. Camera interpolation still uses the
        # complete input trajectory, including its otherwise unused tail.
        previous_frames = self.input_info.num_frames
        latent_frames = self.get_latent_shape_with_lat_hw(lat_h, lat_w)[1]
        self.input_info.num_frames = (latent_frames - 1) * self.config["vae_stride"][0] + 1
        try:
            return super().get_vae_encoder_output(first_frame, lat_h, lat_w, last_frame)
        finally:
            self.input_info.num_frames = previous_frames

    def _build_lingbot_dit_cond_dict(self, action_path):
        if self.control_type != "cam":
            return super()._build_lingbot_dit_cond_dict(action_path)
        latent_frames, latent_h, latent_w = self.input_info.latent_shape[1:]
        height, width = latent_h * self.config["vae_stride"][1], latent_w * self.config["vae_stride"][2]
        poses = np.load(os.path.join(action_path, "poses.npy"))
        frame_count = min(self.get_num_frames(), ((len(poses) - 1) // 4) * 4 + 1)
        # Match official CPU interpolation and relative-pose arithmetic before
        # moving these small matrices to the GPU.
        camera = torch.from_numpy(self._interp_c2ws_to_latf(poses[:frame_count], latent_frames))
        camera = self._compute_relative_poses(camera).to(AI_DEVICE)
        intrinsics = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
        intrinsics = self._get_Ks_transformed(intrinsics, 480, 832, height, width, height, width)[0].to(AI_DEVICE)
        gy, gx = torch.meshgrid(
            torch.arange(height, device=AI_DEVICE, dtype=torch.float32) + 0.5,
            torch.arange(width, device=AI_DEVICE, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        directions = torch.stack(
            [(gx.flatten() - intrinsics[2]) / intrinsics[0], (gy.flatten() - intrinsics[3]) / intrinsics[1], torch.ones(height * width, device=AI_DEVICE)],
            dim=-1,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True)
        packed = pack_camera_embedding(camera, directions, latent_h, latent_w, height=height, width=width, dtype=GET_DTYPE())
        return {"c2ws_plucker_emb": packed}

    def end_run(self):
        pre = getattr(self.model, "pre_infer", None)
        try:
            return super().end_run()
        finally:
            if pre is not None:
                pre.clear_condition_cache()

    def prepare_warmup_inputs(self, height, width, latent_shape, text_output):
        frame_count = (latent_shape[1] - 1) * self.config["vae_stride"][0] + 1
        self.input_info = ActionI2VInputInfo(task="i2v", prompt="warmup", size=[height, width], num_frames=frame_count)
        self.input_info.latent_shape = latent_shape
        image_output = self.get_warmup_image_encoder_output(latent_shape)
        camera_channels = 6 * self.config["vae_stride"][1] * self.config["vae_stride"][2]
        image_output["dit_cond_dict"] = {"c2ws_plucker_emb": torch.zeros(camera_channels, *latent_shape[1:], device=self.init_device, dtype=GET_DTYPE())}
        self.inputs = {"text_encoder_output": text_output, "image_encoder_output": image_output}
