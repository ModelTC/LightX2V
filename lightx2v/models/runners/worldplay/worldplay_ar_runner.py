import gc
import os

import torch
from loguru import logger

from lightx2v.models.networks.worldplay.ar_model import WorldPlayARModel
from lightx2v.models.networks.worldplay.pose_utils import pose_to_input
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner
from lightx2v.models.schedulers.worldplay.ar_scheduler import WorldPlayARScheduler
from lightx2v.utils.profiler import ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("worldplay_ar")
class WorldPlayARRunner(HunyuanVideo15Runner):
    """
    Runner for HY-WorldPlay AR (Autoregressive) model with action conditioning.

    Key differences from WorldPlayDistillRunner:
    - Uses causal attention for autoregressive generation
    - Implements KV cache for efficient frame-by-frame generation
    - Supports chunk-based training and inference
    - No guidance embedding required

    Extends HunyuanVideo15Runner with:
    - Action conditioning support
    - ProPE (camera pose) conditioning
    - Autoregressive chunk-based generation
    - Memory window selection for long videos
    """

    def __init__(self, config):
        # AR-specific parameters
        self.chunk_latent_frames = config.get("chunk_latent_frames", 4)
        self.model_type = config.get("model_type", "ar")
        self.action_ckpt = config.get("action_ckpt", None)
        self.use_prope = config.get("use_prope", True)

        # Validate action_ckpt if provided
        if self.action_ckpt is not None and not os.path.exists(self.action_ckpt):
            raise FileNotFoundError(f"Action checkpoint not found: {self.action_ckpt}. Please provide a valid path to the action model checkpoint.")

        # Validate chunk_latent_frames
        if not isinstance(self.chunk_latent_frames, int) or self.chunk_latent_frames <= 0:
            raise ValueError(f"chunk_latent_frames must be a positive integer, got {self.chunk_latent_frames}")

        # Memory window for AR generation
        self.memory_window_size = config.get("memory_window_size", 8)
        self.select_window_out_flag = config.get("select_window_out_flag", True)

        super().__init__(config)

    def init_scheduler(self):
        """Initialize WorldPlay AR scheduler."""
        self.scheduler = WorldPlayARScheduler(self.config)

        if self.sr_version is not None:
            from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15SRScheduler

            self.scheduler_sr = HunyuanVideo15SRScheduler(self.config_sr)
        else:
            self.scheduler_sr = None

    def load_transformer(self):
        """Load WorldPlay AR transformer with action conditioning."""
        model = WorldPlayARModel(
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
        img_ori = self.read_image_input(self.input_info.image_path)
        if hasattr(self.input_info, "pose") and self.input_info.pose is not None:
            from lightx2v.models.networks.worldplay.pose_utils import get_latent_num_from_pose

            latent_num = get_latent_num_from_pose(self.input_info.pose)
            vae_stride_t = self.config["vae_stride"][0]
            with self.config.temporarily_unlocked():
                self.config["target_video_length"] = latent_num * vae_stride_t - (vae_stride_t - 1)
            logger.info(f"Auto-set target_video_length={self.config['target_video_length']} from pose ({latent_num} latent frames)")
        if self.sr_version and self.config_sr["is_sr_running"]:
            self.latent_sr_shape = self.get_sr_latent_shape_with_target_hw()
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw(origin_size=img_ori.size)

        siglip_output, siglip_mask = self.run_image_encoder(img_ori) if self.config.get("use_image_encoder", True) else (None, None)
        cond_latents = self.run_vae_encoder(img_ori)
        text_encoder_output = self.run_text_encoder(self.input_info)

        # Process pose input if available
        pose_output = None
        if hasattr(self.input_info, "pose") and self.input_info.pose is not None:
            pose_output = self._process_pose_input(self.input_info.pose, self.input_info.latent_shape[1])

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
        if hasattr(self.input_info, "pose") and self.input_info.pose is not None:
            from lightx2v.models.networks.worldplay.pose_utils import get_latent_num_from_pose

            latent_num = get_latent_num_from_pose(self.input_info.pose)
            vae_stride_t = self.config["vae_stride"][0]
            with self.config.temporarily_unlocked():
                self.config["target_video_length"] = latent_num * vae_stride_t - (vae_stride_t - 1)
            logger.info(f"Auto-set target_video_length={self.config['target_video_length']} from pose ({latent_num} latent frames)")
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()
        text_encoder_output = self.run_text_encoder(self.input_info)

        siglip_output = torch.zeros(1, self.vision_num_semantic_tokens, self.config["hidden_size"], dtype=torch.bfloat16).to(AI_DEVICE)
        siglip_mask = torch.zeros(1, self.vision_num_semantic_tokens, dtype=torch.bfloat16, device=torch.device(AI_DEVICE))

        # Process pose input if available
        pose_output = None
        if hasattr(self.input_info, "pose") and self.input_info.pose is not None:
            pose_output = self._process_pose_input(self.input_info.pose, self.input_info.latent_shape[1])

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
            pose_data: Pose string or JSON path or dict
            latent_num: Number of latent frames

        Returns:
            Dict with viewmats, Ks, action tensors
        """
        try:
            viewmats, Ks, action = pose_to_input(pose_data, latent_num)

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

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        """Override to use chunk-based AR generation instead of run_segment()."""
        self.init_run()
        self.run_denoising_loop()
        latents = self.scheduler.latents
        if self.config.get("use_stream_vae", False):
            frames = []
            for frame_segment in self.run_vae_decoder_stream(latents):
                frames.append(frame_segment)
            self.gen_video = torch.cat(frames, dim=2)
        else:
            self.gen_video = self.run_vae_decoder(latents)
        self.gen_video_final = self.gen_video
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    def run_denoising_loop(self):
        """
        Run autoregressive denoising loop with chunk-based generation.

        Preserves full pose data and slices per-chunk from it to avoid
        overwriting scheduler state with chunk-sized data.

        AR rollout flow (matching HY-WorldPlay ar_rollout):
        1. Cache text KV once at the beginning
        2. For each chunk:
           a. Cache context frame KV (if not first chunk)
           b. For each denoising step, run vision inference using cached KV
        """
        total_chunks = self.scheduler.total_chunks
        total_frames = self.scheduler.latents.shape[2]
        _, _, _, H, W = self.scheduler.latents.shape
        logger.info(f"Starting AR generation with {total_chunks} chunks")

        # Preserve full pose/cos_sin/cond data before chunk iteration
        full_viewmats = self.scheduler.viewmats
        full_Ks = self.scheduler.Ks
        full_action = self.scheduler.action
        full_cos_sin = self.scheduler.cos_sin
        full_cond_latents_concat = self.scheduler.cond_latents_concat
        full_mask_concat = self.scheduler.mask_concat

        # Remove pose_output from inputs so infer_vision() doesn't overwrite
        # chunk-sliced data on the scheduler. Runner manages pose directly.
        saved_pose_output = self.inputs.get("pose_output")
        self.inputs["pose_output"] = None

        # Step 1: Cache text KV (called once at generation start)
        logger.info("Caching text KV...")
        self.model.infer_txt(self.inputs, cache_txt=True)

        for chunk_idx in range(total_chunks):
            logger.info(f"Generating chunk {chunk_idx + 1}/{total_chunks}")

            # Calculate frame range for this chunk
            start_frame = chunk_idx * self.chunk_latent_frames
            end_frame = min(start_frame + self.chunk_latent_frames, total_frames)

            # Set chunk pose from FULL data
            if full_viewmats is not None:
                self.scheduler.viewmats = full_viewmats[:, start_frame:end_frame]
            if full_Ks is not None:
                self.scheduler.Ks = full_Ks[:, start_frame:end_frame]
            if full_action is not None:
                self.scheduler.action = full_action[:, start_frame:end_frame]
            # Set chunk cos_sin from full
            self._set_chunk_cos_sin(full_cos_sin, start_frame, end_frame, H, W)
            # Slice cond_latents_concat and mask_concat for this chunk
            if full_cond_latents_concat is not None:
                self.scheduler.cond_latents_concat = full_cond_latents_concat[:, :, start_frame:end_frame, :, :]
            if full_mask_concat is not None:
                self.scheduler.mask_concat = full_mask_concat[:, :, start_frame:end_frame, :, :]

            # Step 2: For non-first chunks, cache context frame KV
            if chunk_idx > 0:
                self.model.clear_vision_cache()
                self._cache_context_kv(chunk_idx, full_viewmats, full_Ks, full_action, full_cos_sin, full_cond_latents_concat, full_mask_concat, H, W)

            # Step 3: Run denoising for this chunk
            chunk_latents = self.scheduler.latents[:, :, start_frame:end_frame, :, :]
            chunk_output = self._denoise_chunk(chunk_idx, chunk_latents)

            # Update latents with generated chunk
            self.scheduler.update_chunk_latents(chunk_idx, chunk_output)

        # Restore full data
        self.scheduler.viewmats = full_viewmats
        self.scheduler.Ks = full_Ks
        self.scheduler.action = full_action
        self.scheduler.cos_sin = full_cos_sin
        self.scheduler.cond_latents_concat = full_cond_latents_concat
        self.scheduler.mask_concat = full_mask_concat
        self.inputs["pose_output"] = saved_pose_output

        # Clear KV cache after generation
        self.model.clear_kv_cache()

    def _set_chunk_cos_sin(self, full_cos_sin, start_frame, end_frame, H, W):
        """Slice cos_sin by frame range (token = frame * H * W)."""
        if full_cos_sin is None:
            return
        start_token = start_frame * H * W
        end_token = end_frame * H * W
        self.scheduler.cos_sin = full_cos_sin[start_token:end_token]

    def _cache_context_kv(self, chunk_idx, full_viewmats, full_Ks, full_action, full_cos_sin, full_cond_latents_concat, full_mask_concat, H, W):
        """
        Set scheduler state for context frames and cache their KV.

        Slices from full pose data to avoid index-out-of-bounds issues.
        """
        start_chunk, end_chunk = self.scheduler.get_memory_window(chunk_idx)
        if end_chunk <= 0:
            return

        context_start_frame = start_chunk * self.chunk_latent_frames
        context_end_frame = chunk_idx * self.chunk_latent_frames

        if context_end_frame <= context_start_frame:
            return

        # Save current scheduler state
        saved_latents = self.scheduler.latents
        saved_viewmats = self.scheduler.viewmats
        saved_Ks = self.scheduler.Ks
        saved_action = self.scheduler.action
        saved_cos_sin = self.scheduler.cos_sin
        saved_cond = self.scheduler.cond_latents_concat
        saved_mask = self.scheduler.mask_concat

        # Set context data from FULL arrays
        self.scheduler.latents = saved_latents[:, :, context_start_frame:context_end_frame, :, :]
        if full_viewmats is not None:
            self.scheduler.viewmats = full_viewmats[:, context_start_frame:context_end_frame]
        if full_Ks is not None:
            self.scheduler.Ks = full_Ks[:, context_start_frame:context_end_frame]
        if full_action is not None:
            self.scheduler.action = full_action[:, context_start_frame:context_end_frame]
        if full_cond_latents_concat is not None:
            self.scheduler.cond_latents_concat = full_cond_latents_concat[:, :, context_start_frame:context_end_frame, :, :]
        if full_mask_concat is not None:
            self.scheduler.mask_concat = full_mask_concat[:, :, context_start_frame:context_end_frame, :, :]
        self._set_chunk_cos_sin(full_cos_sin, context_start_frame, context_end_frame, H, W)

        # Cache context KV
        self.model.infer_vision(self.inputs, cache_vision=True)

        # Restore scheduler state for the current chunk
        self.scheduler.latents = saved_latents
        self.scheduler.viewmats = saved_viewmats
        self.scheduler.Ks = saved_Ks
        self.scheduler.action = saved_action
        self.scheduler.cos_sin = saved_cos_sin
        self.scheduler.cond_latents_concat = saved_cond
        self.scheduler.mask_concat = saved_mask

    def _denoise_chunk(self, chunk_idx, chunk_latents):
        """
        Run denoising loop for a single chunk using AR inference.

        Args:
            chunk_idx: Current chunk index
            chunk_latents: Initial latent tensor for this chunk

        Returns:
            Denoised latent tensor for this chunk
        """
        # Store original latents and replace with chunk
        original_latents = self.scheduler.latents
        self.scheduler.latents = chunk_latents

        # Reset step index for this chunk
        self.scheduler.step_index = 0

        # Run denoising steps
        for step_idx in range(self.scheduler.infer_steps):
            self.scheduler.step_index = step_idx

            # Run vision inference (using cached text KV and context KV)
            noise_pred = self.model.infer_vision(self.inputs, cache_vision=False)

            # Update noise prediction and step
            self.scheduler.noise_pred = noise_pred
            self.scheduler.step_post()

        # Get denoised chunk
        denoised_chunk = self.scheduler.latents.clone()

        # Restore original latents
        self.scheduler.latents = original_latents

        return denoised_chunk

    def cleanup(self):
        """Cleanup after generation."""
        super().cleanup()
        if hasattr(self, "model") and self.model is not None:
            self.model.clear_kv_cache()
