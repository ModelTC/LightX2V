import gc
import os

import torch
from loguru import logger

from lightx2v.models.networks.worldplay.model import WorldPlayModel
from lightx2v.models.networks.worldplay.pose_utils import pose_to_input
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner
from lightx2v.models.schedulers.worldplay.scheduler import WorldPlayDistillScheduler
from lightx2v.utils.profiler import ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("worldplay_distill")
class WorldPlayDistillRunner(HunyuanVideo15Runner):
    """
    Runner for HY-WorldPlay distill model with action conditioning.

    Extends HunyuanVideo15Runner with:
    - Action conditioning support
    - ProPE (camera pose) conditioning
    - Autoregressive chunk-based generation
    - Few-step inference (4 steps by default)
    """

    def __init__(self, config):
        # Set default distill parameters
        if "denoising_step_list" not in config:
            config["denoising_step_list"] = [0, 250, 500, 750]
        if "infer_steps" not in config:
            config["infer_steps"] = len(config["denoising_step_list"])

        # WorldPlay-specific parameters
        self.chunk_latent_frames = config.get("chunk_latent_frames", 4)
        self.model_type = config.get("model_type", "ar")
        self.action_ckpt = config.get("action_ckpt", None)
        self.use_prope = config.get("use_prope", True)

        super().__init__(config)

    def init_scheduler(self):
        """Initialize WorldPlay distill scheduler."""
        self.scheduler = WorldPlayDistillScheduler(self.config)

        if self.sr_version is not None:
            from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15SRScheduler

            self.scheduler_sr = HunyuanVideo15SRScheduler(self.config_sr)
        else:
            self.scheduler_sr = None

    def load_transformer(self):
        """Load WorldPlay transformer with action conditioning."""
        model = WorldPlayModel(
            self.config["model_path"],
            self.config,
            self.init_device,
            action_ckpt=self.action_ckpt,
        )

        if self.sr_version is not None:
            from lightx2v.models.networks.hunyuan_video.model import HunyuanVideo15Model

            self.config_sr["transformer_model_path"] = os.path.join(os.path.dirname(self.config.transformer_model_path), self.sr_version)
            self.config_sr["is_sr_running"] = True
            model_sr = HunyuanVideo15Model(self.config_sr["model_path"], self.config_sr, self.init_device)
            self.config_sr["is_sr_running"] = False
        else:
            model_sr = None

        self.model_sr = model_sr
        return model

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        """Run encoders with pose processing for i2v task."""
        # Get base encoder outputs
        img_ori = self.read_image_input(self.input_info.image_path)
        if self.sr_version and self.config_sr["is_sr_running"]:
            self.latent_sr_shape = self.get_sr_latent_shape_with_target_hw()
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw(origin_size=img_ori.size)

        siglip_output, siglip_mask = self.run_image_encoder(img_ori) if self.config.get("use_image_encoder", True) else (None, None)
        cond_latents = self.run_vae_encoder(img_ori)
        text_encoder_output = self.run_text_encoder(self.input_info)

        # Process pose input if available
        pose_output = None
        if hasattr(self.input_info, "pose") and self.input_info.pose is not None:
            pose_output = self._process_pose_input(
                self.input_info.pose,
                self.input_info.latent_shape[1],  # num latent frames
            )

        torch_device_module.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "siglip_output": siglip_output,
                "siglip_mask": siglip_mask,
                "cond_latents": cond_latents,
            },
            "pose_output": pose_output,
        }

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2v(self):
        """Run encoders with pose processing for t2v task."""
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()
        text_encoder_output = self.run_text_encoder(self.input_info)

        # vision_states is all zero for t2v
        siglip_output = torch.zeros(1, self.vision_num_semantic_tokens, self.config["hidden_size"], dtype=torch.bfloat16).to(AI_DEVICE)
        siglip_mask = torch.zeros(1, self.vision_num_semantic_tokens, dtype=torch.bfloat16, device=torch.device(AI_DEVICE))

        # Process pose input if available
        pose_output = None
        if hasattr(self.input_info, "pose") and self.input_info.pose is not None:
            pose_output = self._process_pose_input(
                self.input_info.pose,
                self.input_info.latent_shape[1],  # num latent frames
            )

        torch_device_module.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "siglip_output": siglip_output,
                "siglip_mask": siglip_mask,
                "cond_latents": None,
            },
            "pose_output": pose_output,
        }

    def _process_pose_input(self, pose_data, latent_num):
        """
        Convert pose string/JSON to action tensors.

        Args:
            pose_data: Pose string (e.g., "w-3, right-0.5") or JSON path or dict
            latent_num: Number of latent frames

        Returns:
            Dict with viewmats, Ks, action tensors
        """
        try:
            viewmats, Ks, action = pose_to_input(pose_data, latent_num)

            # Move to device and add batch dimension
            viewmats = viewmats.unsqueeze(0).to(device=AI_DEVICE, dtype=torch.float32)
            Ks = Ks.unsqueeze(0).to(device=AI_DEVICE, dtype=torch.float32)
            action = action.unsqueeze(0).to(device=AI_DEVICE, dtype=torch.long)

            return {
                "viewmats": viewmats,
                "Ks": Ks,
                "action": action,
            }
        except Exception as e:
            logger.warning(f"Failed to process pose input: {e}. Continuing without pose conditioning.")
            return None

    def init_run(self):
        """Initialize run with pose conditioning support."""
        self.gen_video_final = None
        self.get_video_segment_num()

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

        pose_output = self.inputs.get("pose_output")
        self.scheduler.prepare(
            seed=self.input_info.seed,
            latent_shape=self.input_info.latent_shape,
            image_encoder_output=self.inputs["image_encoder_output"],
            pose_output=pose_output,
        )
        self.model.set_scheduler(self.scheduler)
