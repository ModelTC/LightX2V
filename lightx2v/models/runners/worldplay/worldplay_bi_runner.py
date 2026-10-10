import gc
import os

import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.hunyuan15.byt5.model import (
    ByT5TextEncoderForBI,
)
from lightx2v.models.networks.worldplay.bi_model import WorldPlayBIModel
from lightx2v.models.networks.worldplay.pose_utils import pose_to_input
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner
from lightx2v.models.schedulers.worldplay.bi_scheduler import WorldPlayBIScheduler
from lightx2v.utils.input_info import WorldPlayI2VInputInfo, WorldPlayT2VInputInfo
from lightx2v.utils.profiler import ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def generate_points_in_sphere(n_points: int, radius: float) -> torch.Tensor:
    """
    Uniformly sample points within a sphere of a specified radius.
    Used for FOV overlap calculation in memory frame selection.
    """
    import math

    samples_r = torch.rand(n_points)
    samples_phi = torch.rand(n_points)
    samples_u = torch.rand(n_points)

    r = radius * torch.pow(samples_r, 1 / 3)
    phi = 2 * math.pi * samples_phi
    theta = torch.acos(1 - 2 * samples_u)

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points


def select_aligned_memory_frames(
    w2c_list,
    current_frame_idx: int,
    memory_frames: int = 20,
    temporal_context_size: int = 12,
    pred_latent_size: int = 4,
    points_local=None,
    device=None,
):
    """
    Select memory frames based on FOV overlap similarity.
    Simplified version from HY-WorldPlay.
    """

    if current_frame_idx <= memory_frames:
        return list(range(0, current_frame_idx))

    num_total_frames = len(w2c_list)
    if current_frame_idx >= num_total_frames or current_frame_idx < 3:
        return list(range(0, min(current_frame_idx, memory_frames)))

    start_context_idx = max(0, current_frame_idx - temporal_context_size)
    context_frames_indices = list(range(start_context_idx, current_frame_idx))

    # Add first chunk as context
    memory_frames_indices = [0, 1, 2, 3] if current_frame_idx > 4 else []

    # For simplicity, use temporal proximity for additional frames
    # Full FOV overlap calculation can be added if needed
    historical_clip_indices = list(range(4, current_frame_idx - temporal_context_size, 4))

    remaining_slots = memory_frames - temporal_context_size - len(memory_frames_indices)
    if remaining_slots > 0 and historical_clip_indices:
        # Select evenly spaced historical frames
        step = max(1, len(historical_clip_indices) // (remaining_slots // 4 + 1))
        for i in range(0, len(historical_clip_indices), step):
            if len(memory_frames_indices) >= memory_frames - temporal_context_size:
                break
            start_idx = historical_clip_indices[i]
            if start_idx not in memory_frames_indices:
                memory_frames_indices.extend(range(start_idx, min(start_idx + 4, current_frame_idx)))

    # Combine and deduplicate
    selected_frames_set = set(context_frames_indices)
    selected_frames_set.update(memory_frames_indices)

    return sorted(list(selected_frames_set))


@RUNNER_REGISTER("worldplay_bi")
class WorldPlayBIRunner(HunyuanVideo15Runner):
    """
    Runner for HY-WorldPlay BI (Bidirectional) model with action conditioning.

    Key differences from AR runner:
    - Uses bidirectional attention (not causal)
    - No KV cache required
    - Standard 50-step inference
    - Supports classifier-free guidance

    Key differences from Distill runner:
    - Standard 50-step inference (not 4-step)
    - Uses guidance_scale for CFG

    Extends HunyuanVideo15Runner with:
    - Action conditioning support
    - ProPE (camera pose) conditioning
    - Chunk-based generation with context frame selection
    """

    input_info_cls_by_task = {"t2v": WorldPlayT2VInputInfo, "i2v": WorldPlayI2VInputInfo}
    supported_request_fields_by_task = {task: request_fields | {"pose"} for task, request_fields in HunyuanVideo15Runner.supported_request_fields_by_task.items()}

    def __init__(self, config):
        # BI-specific parameters
        self.chunk_latent_frames = config.get("chunk_latent_frames", 16)  # BI uses 16 by default
        self.action_ckpt = config.get("action_ckpt", None)
        self.use_prope = config.get("use_prope", True)

        # Validate action_ckpt if provided
        if self.action_ckpt is not None and not os.path.exists(self.action_ckpt):
            raise FileNotFoundError(f"Action checkpoint not found: {self.action_ckpt}. Please provide a valid path to the action model checkpoint.")

        # Validate chunk_latent_frames
        if not isinstance(self.chunk_latent_frames, int) or self.chunk_latent_frames <= 0:
            raise ValueError(f"chunk_latent_frames must be a positive integer, got {self.chunk_latent_frames}")

        # Memory frame selection parameters
        self.memory_frames = config.get("memory_frames", 20)
        self.temporal_context_size = config.get("temporal_context_size", 12)

        # Points for FOV overlap calculation
        self.points_local = None

        super().__init__(config)

    def init_scheduler(self):
        """Initialize WorldPlay BI scheduler."""
        self.scheduler = WorldPlayBIScheduler(self.config)

        if self.sr_version is not None:
            from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15SRScheduler

            self.scheduler_sr = HunyuanVideo15SRScheduler(self.config_sr)
        else:
            self.scheduler_sr = None

    def load_text_encoder(self):
        """Load text encoders with BI model support.

        For BI model with use_bi_model_as_main=True, load byt5_in weights
        from the action_ckpt instead of the base model path.
        """
        from lightx2v.models.input_encoders.hf.hunyuan15.byt5.model import ByT5TextEncoder
        from lightx2v.models.input_encoders.hf.hunyuan15.qwen25.model import Qwen25VL_TextEncoder

        qwen25vl_offload = self.config.get("qwen25vl_cpu_offload", self.config.get("cpu_offload"))
        if qwen25vl_offload:
            qwen25vl_device = torch.device("cpu")
        else:
            qwen25vl_device = torch.device(AI_DEVICE)

        qwen25vl_quantized = self.config.get("qwen25vl_quantized", False)
        qwen25vl_quant_scheme = self.config.get("qwen25vl_quant_scheme", None)
        qwen25vl_quantized_ckpt = self.config.get("qwen25vl_quantized_ckpt", None)

        text_encoder_path = os.path.join(self.config["model_path"], "text_encoder/llm")
        logger.info(f"Loading text encoder from {text_encoder_path}")
        text_encoder = Qwen25VL_TextEncoder(
            device=qwen25vl_device,
            checkpoint_path=text_encoder_path,
            cpu_offload=qwen25vl_offload,
            qwen25vl_quantized=qwen25vl_quantized,
            qwen25vl_quant_scheme=qwen25vl_quant_scheme,
            qwen25vl_quant_ckpt=qwen25vl_quantized_ckpt,
        )

        byt5_offload = self.config.get("byt5_cpu_offload", self.config.get("cpu_offload"))
        if byt5_offload:
            byt5_device = torch.device("cpu")
        else:
            byt5_device = torch.device(AI_DEVICE)

        # For BI model, we need to load byt5_in weights from action_ckpt
        use_bi_model_as_main = self.config.get("use_bi_model_as_main", False)
        if use_bi_model_as_main and self.action_ckpt is not None:
            # Create ByT5TextEncoder with custom byt5_in weights from action_ckpt
            byt5 = ByT5TextEncoderForBI(
                config=self.config,
                device=byt5_device,
                checkpoint_path=self.config["model_path"],
                action_ckpt=self.action_ckpt,
                cpu_offload=byt5_offload,
            )
        else:
            byt5 = ByT5TextEncoder(
                config=self.config,
                device=byt5_device,
                checkpoint_path=self.config["model_path"],
                cpu_offload=byt5_offload,
            )

        text_encoders = [text_encoder, byt5]
        return text_encoders

    def load_transformer(self):
        """Load WorldPlay BI transformer with action conditioning."""
        model = WorldPlayBIModel(
            self.config["model_path"],
            self.config,
            self.init_device,
            action_ckpt=self.action_ckpt,
        )

        self.model_sr = self.load_sr_transformer()
        return model

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        """Run encoders with pose processing for i2v task."""
        img_ori = self.read_image_input(self.input_info.image_path)
        if self.sr_version and self.config_sr["is_sr_running"]:
            self.latent_sr_shape = self.get_sr_latent_shape_with_target_hw()
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw(origin_size=img_ori.size)

        siglip_output, siglip_mask = self.run_image_encoder(img_ori) if self.config.get("use_image_encoder", True) else (None, None)
        cond_latents = self.run_vae_encoder(img_ori)
        text_encoder_output = self.run_text_encoder(self.input_info)

        # Process pose input if available
        pose_output = None
        if self.input_info.pose is not None:
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
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()
        text_encoder_output = self.run_text_encoder(self.input_info)

        siglip_output = torch.zeros(1, self.vision_num_semantic_tokens, self.config["hidden_size"], dtype=torch.bfloat16).to(AI_DEVICE)
        siglip_mask = torch.zeros(1, self.vision_num_semantic_tokens, dtype=torch.bfloat16, device=torch.device(AI_DEVICE))

        # Process pose input if available
        pose_output = None
        if self.input_info.pose is not None:
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
        viewmats, Ks, action = pose_to_input(pose_data, latent_num)
        return {
            "viewmats": viewmats.unsqueeze(0).to(device=AI_DEVICE, dtype=torch.float32),
            "Ks": Ks.unsqueeze(0).to(device=AI_DEVICE, dtype=torch.float32),
            "action": action.unsqueeze(0).to(device=AI_DEVICE, dtype=torch.long),
        }

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

        # Initialize points for FOV overlap calculation
        if self.points_local is None:
            self.points_local = generate_points_in_sphere(50000, 8.0).to(AI_DEVICE)

    def run_segment(self, segment_idx=0):
        self.run_denoising_loop(segment_idx)
        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
        return self.scheduler.latents

    def run_denoising_loop(self, segment_idx=0):
        """
        Run bidirectional denoising loop with chunk-based generation.

        BI rollout flow (matching HY-WorldPlay bi_rollout):
        1. For each chunk:
           a. Select context frames based on FOV overlap
           b. Concatenate context frames with current chunk
           c. Run denoising with bidirectional attention
           d. Extract and store generated chunk
        """
        total_chunks = self.scheduler.total_chunks
        logger.info(f"Starting BI generation with {total_chunks} chunks")

        for chunk_idx in range(total_chunks):
            logger.info(f"Generating chunk {chunk_idx + 1}/{total_chunks}")

            # Prepare chunk data
            chunk_latents, _, _, _ = self.scheduler.prepare_chunk(chunk_idx)

            # Select context frames for non-first chunks
            context_frame_indices = []
            if chunk_idx > 0 and self.scheduler.viewmats is not None:
                # Get full viewmats for frame selection
                full_viewmats = self.inputs.get("pose_output", {}).get("viewmats")
                if full_viewmats is not None:
                    current_frame_idx = chunk_idx * self.chunk_latent_frames
                    context_frame_indices = select_aligned_memory_frames(
                        full_viewmats[0].cpu().detach().numpy(),
                        current_frame_idx,
                        memory_frames=self.memory_frames,
                        temporal_context_size=self.temporal_context_size,
                        pred_latent_size=self.chunk_latent_frames,
                        points_local=self.points_local,
                        device=AI_DEVICE,
                    )
                    # Remove current chunk frames from context
                    to_remove = list(range(current_frame_idx, current_frame_idx + self.chunk_latent_frames))
                    context_frame_indices = [x for x in context_frame_indices if x not in to_remove]

            self.scheduler.set_context_frame_indices(context_frame_indices)

            # Run denoising for this chunk
            chunk_output = self._denoise_chunk_bi(chunk_idx, chunk_latents, context_frame_indices, segment_idx)

            # Update latents with generated chunk
            self.scheduler.update_chunk_latents(chunk_idx, chunk_output)

    def _denoise_chunk_bi(self, chunk_idx, chunk_latents, context_frame_indices, segment_idx=0):
        """Denoise a chunk with aligned context, conditions, pose and RoPE."""
        full_latents = self.scheduler.latents
        full_cond = self.scheduler.cond_latents_concat
        full_mask = self.scheduler.mask_concat
        full_viewmats = self.scheduler.viewmats
        full_Ks = self.scheduler.Ks
        full_action = self.scheduler.action
        full_cos_sin = self.scheduler.cos_sin
        full_timestep_input = self.scheduler.timestep_input

        num_chunk_frames = chunk_latents.shape[2]
        start_frame = chunk_idx * self.chunk_latent_frames
        frame_indices = context_frame_indices + list(range(start_frame, start_frame + num_chunk_frames))
        context_latents = full_latents[:, :, context_frame_indices] if context_frame_indices else None
        _, _, total_frames, height, width = full_latents.shape

        try:
            self.scheduler.latents = chunk_latents
            self.scheduler.cond_latents_concat = full_cond[:, :, frame_indices]
            self.scheduler.mask_concat = full_mask[:, :, frame_indices]
            self.scheduler.cos_sin = full_cos_sin.reshape(total_frames, height * width, -1)[frame_indices].flatten(0, 1)
            if full_viewmats is not None:
                self.scheduler.viewmats = full_viewmats[:, frame_indices]
            if full_Ks is not None:
                self.scheduler.Ks = full_Ks[:, frame_indices]
            if full_action is not None:
                self.scheduler.action = full_action[:, frame_indices]

            for step_idx in range(self.scheduler.infer_steps):
                if self.video_segment_num == 1:
                    self.check_stop()
                self.scheduler.step_pre(step_idx)
                t = self.scheduler.timesteps[step_idx]
                timestep_input = t.expand(num_chunk_frames)
                chunk_latents = self.scheduler.latents

                if context_frame_indices:
                    t_ctx = self.scheduler.get_context_timestep()
                    timestep_input = torch.cat([t_ctx, timestep_input])
                    self.scheduler.latents = torch.cat([context_latents, chunk_latents], dim=2)

                self.scheduler.timestep_input = timestep_input
                noise_pred = self.model.infer_bi(self.inputs)
                self.scheduler.latents = chunk_latents
                self.scheduler.noise_pred = noise_pred[:, :, -num_chunk_frames:]
                self.scheduler.step_post()

                if self.progress_callback:
                    segment_steps = self.scheduler.total_chunks * self.scheduler.infer_steps
                    current_step = segment_idx * segment_steps + chunk_idx * self.scheduler.infer_steps + step_idx + 1
                    self.progress_callback(current_step / (self.video_segment_num * segment_steps) * 100, 100)

            return self.scheduler.latents.clone()
        finally:
            self.scheduler.latents = full_latents
            self.scheduler.cond_latents_concat = full_cond
            self.scheduler.mask_concat = full_mask
            self.scheduler.viewmats = full_viewmats
            self.scheduler.Ks = full_Ks
            self.scheduler.action = full_action
            self.scheduler.cos_sin = full_cos_sin
            self.scheduler.timestep_input = full_timestep_input

    def cleanup(self):
        """Cleanup after generation."""
        super().cleanup()
        self.points_local = None
