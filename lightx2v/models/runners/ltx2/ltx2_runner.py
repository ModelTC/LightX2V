import gc
import os

import torch
import torch.distributed as dist

from lightx2v.models.input_encoders.hf.ltx2.model import LTX2TextEncoder
from lightx2v.models.networks.lora_adapter import LoraAdapter
from lightx2v.models.networks.ltx2.model import LTX2Model
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.ltx2.scheduler import LTX2Scheduler
from lightx2v.models.video_encoders.hf.ltx2.model import LTX2AudioVAE, LTX2Upsampler, LTX2VideoVAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.ltx2_media_io import encode_video as save_video
from lightx2v.utils.ltx2_media_io import load_image_conditioning
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def _ltx2_parse_image_paths(image_path: str) -> list[str]:
    return [p.strip() for p in image_path.split(",") if p.strip()]


def _ltx2_normalize_image_strengths(image_strength, n: int) -> list[float]:
    if not isinstance(image_strength, list):
        return [float(image_strength)] * n
    if len(image_strength) == 1:
        return [float(image_strength[0])] * n
    if len(image_strength) != n:
        raise ValueError(
            f"i2av image_strength: expected 1 or {n} values (scalar or list), got length {len(image_strength)}"
        )
    return [float(x) for x in image_strength]


def _ltx2_resolve_pixel_frame_indices(image_frame_idx, n: int, num_frames: int) -> list[int]:
    if not image_frame_idx:
        if n == 1:
            return [0]
        if num_frames <= 1:
            return [0] * n
        return [round(i * (num_frames - 1) / (n - 1)) for i in range(n)]
    if len(image_frame_idx) != n:
        raise ValueError(f"i2av image_frame_idx: expected {n} indices (one per image), got {len(image_frame_idx)}")
    hi = num_frames - 1
    return [max(0, min(hi, int(x))) for x in image_frame_idx]


def _ltx2_pixel_to_latent_frame_idx(pixel_frame_idx: int, temporal_scale: int) -> int:
    if pixel_frame_idx == 0:
        return 0
    return (pixel_frame_idx - 1) // temporal_scale + 1


def _ltx2_resize_video_denoise_mask_for_stage2(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize stage-1 unpatchified video denoise mask to stage-2 latent spatial size."""
    # mask shape: [1, F, H, W] -> [F, 1, H, W] for 2D interpolation
    m = mask.to(dtype=torch.float32)
    m = m.permute(1, 0, 2, 3)
    m = torch.nn.functional.interpolate(m, size=(target_h, target_w), mode="nearest")
    # back to [1, F, H, W]
    return m.permute(1, 0, 2, 3).contiguous()


@RUNNER_REGISTER("ltx2")
class LTX2Runner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)

    def init_scheduler(self):
        self.scheduler = LTX2Scheduler(self.config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.video_vae, self.audio_vae = self.load_vae()
        if self.config.get("use_upsampler", False):
            self.upsampler = self.load_upsampler()

    def load_transformer(self, use_distilled_lora=False):
        ltx2_model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = LTX2Model(**ltx2_model_kwargs)
        else:
            model = LTX2Model(**ltx2_model_kwargs)
            lora_adapter = LoraAdapter(model, model_prefix="model.diffusion_model.")
            lora_adapter.apply_lora(lora_configs)
        return model

    def load_upsampler(self):
        if self.config.get("upsampler_original_ckpt", None) is not None:
            ckpt_path = self.config["upsampler_original_ckpt"]
        else:
            ckpt_path = os.path.join(self.config["model_path"], "latent_upsampler")

        upsampler = LTX2Upsampler(
            checkpoint_path=ckpt_path,
            device=self.init_device,
            dtype=GET_DTYPE(),
            cpu_offload=self.config.get("cpu_offload", False),
        )
        return upsampler

    def load_text_encoder(self):
        # offload config
        text_encoder_offload = self.config.get("gemma_cpu_offload", self.config.get("cpu_offload", False))
        if text_encoder_offload:
            text_encoder_device = torch.device("cpu")
        else:
            text_encoder_device = torch.device(AI_DEVICE)

        if self.config.get("dit_original_ckpt", None) is not None:
            ckpt_path = self.config["dit_original_ckpt"]
        elif self.config.get("dit_quantized_ckpt", None) is not None:
            ckpt_path = self.config["dit_quantized_ckpt"]
        else:
            ckpt_path = os.path.join(self.config["model_path"], "transformer")

        if "gemma_original_ckpt" in self.config:
            gemma_ckpt = self.config["gemma_original_ckpt"]
        else:
            gemma_ckpt = self.config["model_path"]

        text_encoder = LTX2TextEncoder(
            checkpoint_path=ckpt_path,
            gemma_root=gemma_ckpt,
            device=text_encoder_device,
            dtype=GET_DTYPE(),
            cpu_offload=text_encoder_offload,
        )

        # Apply LoRA to text encoder if configured
        lora_configs = self.config.get("lora_configs")
        if lora_configs:
            text_encoder.apply_lora(lora_configs)

        text_encoders = [text_encoder]
        return text_encoders

    def load_vae(self):
        """Load video and audio VAE decoders."""
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device(AI_DEVICE)

        if self.config.get("dit_original_ckpt", None) is not None:
            ckpt_path = self.config["dit_original_ckpt"]
        elif self.config.get("dit_quantized_ckpt", None) is not None:
            ckpt_path = self.config["dit_quantized_ckpt"]
        else:
            ckpt_path = os.path.join(self.config["model_path"], "transformer")

        # Video VAE
        video_vae = LTX2VideoVAE(
            checkpoint_path=ckpt_path,
            device=vae_device,
            dtype=GET_DTYPE(),
            load_encoder=self.config["task"] == "i2av" or self.config.get("use_upsampler", False),
            use_tiling=self.config.get("use_tiling_vae", False),
            cpu_offload=vae_offload,
        )

        # Audio VAE
        audio_vae = LTX2AudioVAE(checkpoint_path=ckpt_path, device=vae_device, dtype=GET_DTYPE(), cpu_offload=vae_offload)

        return video_vae, audio_vae

    def get_latent_shape_with_target_hw(self):
        if self.input_info.target_shape:
            target_height = self.input_info.target_shape[0]
            target_width = self.input_info.target_shape[1]
        else:
            if self.config.get("use_upsampler", False):
                target_height = self.config["target_height"] // 2
                target_width = self.config["target_width"] // 2
            else:
                target_height = self.config["target_height"]
                target_width = self.config["target_width"]
            self.input_info.target_shape = [target_height, target_width]

        target_video_length = self.input_info.target_video_length or self.config["target_video_length"]
        video_latent_shape = (
            self.config.get("num_channels_latents", 128),
            (target_video_length - 1) // self.config["vae_scale_factors"][0] + 1,
            int(target_height) // self.config["vae_scale_factors"][1],
            int(target_width) // self.config["vae_scale_factors"][2],
        )

        duration = float(target_video_length) / float(self.config["fps"])
        latents_per_second = float(self.config["audio_sampling_rate"]) / float(self.config["audio_hop_length"]) / float(self.config["audio_scale_factor"])
        audio_frames = round(duration * latents_per_second)

        audio_latent_shape = (
            8,
            audio_frames,
            self.config["audio_mel_bins"],
        )

        return video_latent_shape, audio_latent_shape

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2av(self):
        self.video_denoise_mask = None
        self.initial_video_latent = None
        self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()  # Important: set latent_shape in input_info
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    def _normalize_i2av_input_fields(self) -> None:
        info = self.input_info
        if isinstance(info.image_strength, str):
            p = [float(x.strip()) for x in info.image_strength.split(",") if x.strip()]
            info.image_strength = 1.0 if not p else (p[0] if len(p) == 1 else p)
        if isinstance(info.image_frame_idx, str):
            p = [int(x.strip()) for x in info.image_frame_idx.split(",") if x.strip()]
            info.image_frame_idx = p or None
        n = len(_ltx2_parse_image_paths(info.image_path or ""))
        if n == 0:
            return
        st, fi = info.image_strength, info.image_frame_idx
        if isinstance(st, list) and len(st) not in (1, n):
            raise ValueError(f"i2av image_strength: need 1 or {n} values, got {len(st)}")
        if fi is not None and len(fi) != n:
            raise ValueError(f"i2av image_frame_idx: need {n} indices, got {len(fi)}")

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2av(self):
        self._normalize_i2av_input_fields()
        self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()
        text_encoder_output = self.run_text_encoder(self.input_info)
        self.video_denoise_mask, self.initial_video_latent = self.run_vae_encoder()
        torch_device_module.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
        }

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
        metrics_labels=["LTX2Runner"],
    )
    def run_vae_encoder(self):
        """
        Prepare image conditioning by loading images and encoding them to latents.

        Returns:
            tuple: (video_denoise_mask, initial_video_latent)
                - video_denoise_mask: Mask indicating which frames to denoise (unpatchified, shape [1, F, H, W])
                - initial_video_latent: Initial latent with conditioned frames (unpatchified, shape [C, F, H, W])
        """
        # Get latent shape
        C, F, H, W = self.input_info.video_latent_shape

        target_height = self.input_info.target_shape[0]
        target_width = self.input_info.target_shape[1]
        # Initialize denoise mask (1 = denoise, 0 = keep original)
        # Shape: [1, F, H, W]
        video_denoise_mask = torch.ones(
            1,
            F,
            H,
            W,
            dtype=torch.float32,
            device=AI_DEVICE,
        )

        # Initialize initial latent as zeros
        initial_video_latent = torch.zeros(
            C,
            F,
            H,
            W,
            dtype=GET_DTYPE(),
            device=AI_DEVICE,
        )

        image_paths = _ltx2_parse_image_paths(self.input_info.image_path)
        n = len(image_paths)
        if n == 0:
            logger.warning("i2av: image_path is empty, skipping image conditioning")
            self._i2av_guiding_keyframe_meta = None
            torch_device_module.empty_cache()
            gc.collect()
            return video_denoise_mask, initial_video_latent

        num_frames = self.input_info.target_video_length or self.config.get("target_video_length", 1)
        strengths = _ltx2_normalize_image_strengths(self.input_info.image_strength, n)
        raw_frame_idx = getattr(self.input_info, "image_frame_idx", None)
        pixel_frame_indices = _ltx2_resolve_pixel_frame_indices(raw_frame_idx, n, num_frames)
        temporal_scale = int(self.config["vae_scale_factors"][0])

        guiding_keyframe_meta: list[tuple[str, int, float]] = []

        for i, image_path in enumerate(image_paths):
            strength = strengths[i]
            pixel_frame_idx = pixel_frame_indices[i]
            logger.info(
                f"  📷 Loading image: {image_path} pixel_frame={pixel_frame_idx} strength={strength} ({i + 1}/{n})"
            )

            # Load and preprocess image
            image = load_image_conditioning(
                image_path=image_path,
                height=target_height,
                width=target_width,
                dtype=GET_DTYPE(),
                device=AI_DEVICE,
            )

            with torch.no_grad():
                encoded_latent = self.video_vae.encode(image)

            encoded_latent = encoded_latent.squeeze(0)

            # Pixel frame 0 → write into the latent time slot; other frames → guiding tokens appended in the scheduler.
            if pixel_frame_idx != 0:
                guiding_keyframe_meta.append((image_path, pixel_frame_idx, strength))
                continue

            # Get the latent frame index by converting pixel frame to latent frame
            # For LTX2, temporal compression is 8x, so latent_frame_idx = (frame_idx - 1) // 8 + 1 for frame_idx > 0
            # or 0 for frame_idx == 0
            latent_frame_idx = _ltx2_pixel_to_latent_frame_idx(pixel_frame_idx, temporal_scale)

            if latent_frame_idx >= F:
                logger.warning(f"⚠️  Latent frame index {latent_frame_idx} out of range [0, {F - 1}], skipping")
                continue

            # Set the latent at the specified frame
            # encoded_latent shape: [C, 1, H_latent, W_latent]
            initial_video_latent[:, latent_frame_idx : latent_frame_idx + 1, :, :] = encoded_latent

            # Update denoise mask based on strength
            # strength = 1.0 means keep original (don't denoise)
            # strength = 0.0 means fully denoise
            video_denoise_mask[:, latent_frame_idx, :, :] = 1.0 - strength

            logger.info(f"  ✓ Encoded image to latent frame {latent_frame_idx}")
        self._i2av_guiding_keyframe_meta = guiding_keyframe_meta

        torch_device_module.empty_cache()
        gc.collect()

        logger.info(f"✓ Image conditioning prepared successfully")

        return video_denoise_mask, initial_video_latent

    def _build_i2av_video_guiding_latents(self):
        """Encode guiding keyframe images at current target_shape for scheduler.append (stage 1 / 2)."""
        meta = getattr(self, "_i2av_guiding_keyframe_meta", None)
        if not meta:
            return None
        th, tw = self.input_info.target_shape[0], self.input_info.target_shape[1]
        out = []
        for path, pixel_idx, strength in meta:
            image = load_image_conditioning(
                image_path=path,
                height=th,
                width=tw,
                dtype=GET_DTYPE(),
                device=AI_DEVICE,
            )
            with torch.no_grad():
                enc = self.video_vae.encode(image).squeeze(0)
            out.append((enc, pixel_idx, strength))
        return out

    @ProfilingContext4DebugL1(
        "Run Text Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_text_encode_duration,
        metrics_labels=["WanRunner"],
    )
    def run_text_encoder(self, input_info):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()

        prompt = input_info.prompt
        neg_prompt = input_info.negative_prompt

        v_context_p, a_context_p, v_context_n, a_context_n = self.text_encoders[0].infer(
            prompt=prompt,
            negative_prompt=neg_prompt,
        )
        text_encoder_output = {
            "v_context_p": v_context_p,
            "a_context_p": a_context_p,
            "v_context_n": v_context_n,
            "a_context_n": a_context_n,
        }

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
            torch_device_module.empty_cache()
            gc.collect()

        return text_encoder_output

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_decode_duration,
        metrics_labels=["LTX2Runner"],
    )
    def run_vae_decoder(self, v_latent, a_latent):
        """Decode video and audio latents to frames and waveform."""
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.video_vae, self.audio_vae = self.load_vae()

        # Decode video latents (returns iterator)
        video = self.video_vae.decode(v_latent.unsqueeze(0).to(GET_DTYPE()))
        # Decode audio latents
        audio = self.audio_vae.decode(a_latent.unsqueeze(0).to(GET_DTYPE()))

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.video_vae
            del self.audio_vae
            torch_device_module.empty_cache()
            gc.collect()

        return video, audio

    def run_upsampler(self, v_latent, a_latent):
        """Run Stage 2: Upsampling and high-resolution refinement.

        This method handles the upsampling and scheduler preparation, then delegates
        the denoising loop to run_segment to reduce code duplication.
        """
        logger.info("🚀 Starting Stage 2: Upsampling and high-resolution refinement")

        upsample_distilled_sigmas = torch.tensor(self.config.get("distilled_sigma_values_upsample"), dtype=torch.float32, device=AI_DEVICE)
        self.model.scheduler.reset_sigmas(upsample_distilled_sigmas)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.upsampler = self.load_upsampler()

        upsampled_v_latent = self.upsampler.upsample(v_latent, self.video_vae.encoder).squeeze(0)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.upsampler
            torch_device_module.empty_cache()
            gc.collect()

        self.input_info.target_shape = [self.input_info.target_shape[0] * 2, self.input_info.target_shape[1] * 2]
        self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()
        _, _, stage2_h, stage2_w = self.input_info.video_latent_shape
        stage2_video_denoise_mask = None
        if hasattr(self, "video_denoise_mask") and self.video_denoise_mask is not None:
            stage2_video_denoise_mask = _ltx2_resize_video_denoise_mask_for_stage2(self.video_denoise_mask, stage2_h, stage2_w)

        # Prepare scheduler using the shared method
        self._prepare_scheduler(
            initial_video_latent=upsampled_v_latent,  # Use upsampled video latent
            initial_audio_latent=a_latent,  # Keep audio from stage 1 (aligned with distilled.py:183)
            video_denoise_mask=stage2_video_denoise_mask,  # Keep keyframe constraints in stage 2
            noise_scale=upsample_distilled_sigmas[0].item(),  # Use first sigma as noise_scale (aligned with distilled.py:181)
        )

        # Delegate denoising loop to run_segment with stage_name for logging
        logger.info(f"🔄 Stage 2 - Running {self.model.scheduler.infer_steps} denoising steps")
        v_latent, a_latent = self.run_segment(segment_idx=None, stage_name="Stage 2", cleanup_inputs=True)

        logger.info("✅ Stage 2 completed")
        return v_latent, a_latent

    def _prepare_scheduler(
        self,
        initial_video_latent=None,
        initial_audio_latent=None,
        video_denoise_mask=None,
        noise_scale=None,
    ):
        """
        Prepare scheduler with given latents and masks.

        Args:
            initial_video_latent: Initial video latent. If None, uses self.initial_video_latent.
            initial_audio_latent: Initial audio latent. If None, not passed to scheduler.
            video_denoise_mask: Video denoise mask. If None, uses self.video_denoise_mask.
            noise_scale: Noise scale for scheduler. If None, not passed to scheduler.
        """
        prepare_kwargs = {
            "seed": self.input_info.seed,
            "video_latent_shape": self.input_info.video_latent_shape,
            "audio_latent_shape": self.input_info.audio_latent_shape,
            "initial_video_latent": initial_video_latent if initial_video_latent is not None else self.initial_video_latent,
        }

        if initial_audio_latent is not None:
            prepare_kwargs["initial_audio_latent"] = initial_audio_latent

        if video_denoise_mask is not None:
            # Explicitly provided mask (not None)
            prepare_kwargs["video_denoise_mask"] = video_denoise_mask
        elif hasattr(self, "video_denoise_mask") and self.video_denoise_mask is not None:
            # video_denoise_mask was not explicitly provided, check if we should use self.video_denoise_mask
            # Only use self.video_denoise_mask if we're in Stage 1 (not Stage 2 upsampler)
            # Stage 2 passes explicit initial_video_latent (high-res), so mask should match high-res
            # Stage 1 uses self.initial_video_latent (low-res), so mask matches low-res
            if initial_video_latent is None or initial_video_latent is self.initial_video_latent:
                # Stage 1: use the mask (low-res matches low-res latent)
                prepare_kwargs["video_denoise_mask"] = self.video_denoise_mask
            # Stage 2: don't pass mask, let scheduler create a full mask (all 1s) matching the high-res latent
        # If video_denoise_mask is explicitly None and no self.video_denoise_mask exists,
        # scheduler will create a full mask (all 1s) matching the latent shape

        if noise_scale is not None:
            prepare_kwargs["noise_scale"] = noise_scale

        vg = self._build_i2av_video_guiding_latents()
        if vg:
            prepare_kwargs["video_guiding_latents"] = vg

        self.model.scheduler.prepare(**prepare_kwargs)

    def init_run(self):
        self.gen_video_final = None
        self.get_video_segment_num()

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

        if self.config.get("distilled_sigma_values") is not None:
            stage1_sigmas = torch.tensor(self.config["distilled_sigma_values"], dtype=torch.float32, device=AI_DEVICE)
            self.model.scheduler.reset_sigmas(stage1_sigmas)

        # Image conditioning (if any) is already prepared in run_input_encoder
        # and stored in self.video_denoise_mask and self.initial_video_latent
        self._prepare_scheduler()

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.init_run()
        if self.config.get("compile", False) and hasattr(self.model, "comple"):
            self.model.select_graph_for_compile(self.input_info)
        for segment_idx in range(self.video_segment_num):
            logger.info(f"🔄 start segment {segment_idx + 1}/{self.video_segment_num}")
            with ProfilingContext4DebugL1(
                f"segment end2end {segment_idx + 1}/{self.video_segment_num}",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                metrics_labels=["DefaultRunner"],
            ):
                self.check_stop()
                # 1. default do nothing
                self.init_run_segment(segment_idx)
                # 2. main inference loop
                v_latent, a_latent = self.run_segment(segment_idx)

                ## upsample latent
                if self.config.get("use_upsampler", False):
                    v_latent, a_latent = self.run_upsampler(v_latent, a_latent)
                # 3. vae decoder
                self.gen_video, self.gen_audio = self.run_vae_decoder(v_latent, a_latent)

                # 4. default do nothing
                self.end_run_segment(segment_idx)
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    def end_run_segment(self, segment_idx=None):
        self.gen_video_final = self.gen_video
        self.gen_audio_final = self.gen_audio

    def process_images_after_vae_decoder(self):
        if self.input_info.return_result_tensor:
            return {"video": self.gen_video_final, "audio": self.gen_audio_final}
        elif self.input_info.save_result_path is not None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(f"🎬 Start to save video 🎬")
                save_video(
                    video=self.gen_video_final,
                    fps=self.config.get("fps", 24),
                    audio=self.gen_audio_final,
                    output_path=self.input_info.save_result_path,
                    video_chunks_number=1,
                )

                logger.info(f"✅ Video saved successfully to: {self.input_info.save_result_path} ✅")
            return {"video": None}

    def run_segment(self, segment_idx=0, stage_name=None, cleanup_inputs=None):
        """
        Run denoising loop for a segment.

        Args:
            segment_idx: Segment index (0-based). Use None for upsampler stage.
            stage_name: Optional stage name for logging (e.g., "Stage 2"). If None, uses default logging.
            cleanup_inputs: Whether to cleanup inputs after completion. If None, uses default logic:
                - For upsampler (segment_idx=None): always cleanup
                - For regular segments: cleanup only if last segment and not using upsampler
        """
        infer_steps = self.model.scheduler.infer_steps

        # Determine cleanup behavior
        if cleanup_inputs is None:
            # Default logic: cleanup only for last segment when not using upsampler
            cleanup_inputs = not self.config.get("use_upsampler", False) and segment_idx is not None and segment_idx == self.video_segment_num - 1
        elif cleanup_inputs is True and segment_idx is None:
            # Explicit cleanup for upsampler stage
            cleanup_inputs = True

        for step_index in range(infer_steps):
            # only for single segment, check stop signal every step
            with ProfilingContext4DebugL1(
                f"Run Dit every step",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_per_step_dit_duration,
                metrics_labels=[step_index + 1, infer_steps],
            ):
                if self.video_segment_num == 1:
                    self.check_stop()

                # Use stage_name for logging if provided, otherwise use default
                if stage_name:
                    logger.info(f"==> {stage_name} step_index: {step_index + 1} / {infer_steps}")
                else:
                    logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")

                with ProfilingContext4DebugL1("step_pre"):
                    self.model.scheduler.step_pre(step_index=step_index)

                with ProfilingContext4DebugL1("🚀 infer_main"):
                    self.model.infer(self.inputs)

                with ProfilingContext4DebugL1("step_post"):
                    self.model.scheduler.step_post()

                # Progress callback only for regular segments (not upsampler)
                if self.progress_callback and segment_idx is not None:
                    current_step = segment_idx * infer_steps + step_index + 1
                    total_all_steps = self.video_segment_num * infer_steps
                    self.progress_callback((current_step / total_all_steps) * 100, 100)

        # Cleanup inputs if needed
        if cleanup_inputs:
            del self.inputs
            torch_device_module.empty_cache()

        return self.model.scheduler.video_latent_state.latent, self.model.scheduler.audio_latent_state.latent
