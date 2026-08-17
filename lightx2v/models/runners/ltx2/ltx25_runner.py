"""LTX-2.5 runner built on the native LTX-2.3 execution path."""

from __future__ import annotations

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.input_encoders.hf.ltx2.duration_head import LTX25DurationPredictor
from lightx2v.models.input_encoders.hf.ltx2.model import LTX25TextEncoder
from lightx2v.models.networks.ltx2.ltx25_model import LTX25Model
from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner
from lightx2v.models.schedulers.ltx2.ltx25_scheduler import LTX25Scheduler
from lightx2v.models.video_encoders.hf.ltx2.model import LTX25AudioVAE, LTX25VideoVAE
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.ltx2_media_io import encode_video_ltx25
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("ltx2_5")
class LTX25Runner(LTX2Runner):
    """Split-checkpoint LTX-2.5 runner.

    LTX-2.5 keeps the 22B audio/video transformer and two-stage latent
    lifecycle from LTX-2.3.  This subclass only owns the generation-specific
    pieces: split component paths, Gemma 4, DurationHead, DiffVAE and the
    stage-1 ancestral sampler.
    """

    transformer_model_class = LTX25Model
    text_encoder_class = LTX25TextEncoder
    video_vae_class = LTX25VideoVAE
    audio_vae_class = LTX25AudioVAE
    text_encoder_checkpoint_key = "dit_original_ckpt"
    text_encoder_root_key = "text_encoder_original_ckpt"
    video_vae_checkpoint_key = "video_vae_original_ckpt"
    audio_vae_checkpoint_key = "audio_vae_original_ckpt"

    def __init__(self, config):
        task = config.get("task")
        if task not in ("t2av", "i2av"):
            raise NotImplementedError(f"LTX-2.5 currently supports t2av and i2av, got {task!r}")
        if config.get("enable_cfg", False) or float(config.get("sample_guide_scale", 1.0)) != 1.0:
            raise ValueError("The LTX-2.5 distilled pipeline requires CFG=1 (enable_cfg=false)")
        if config.get("disagg_mode"):
            raise NotImplementedError("LTX-2.5 does not yet support disaggregated inference")
        if config.get("lazy_load", False) or config.get("unload_modules", False):
            raise NotImplementedError("LTX-2.5 does not yet support lazy_load or unload_modules; use the supplied CPU/block-offload profiles")
        super().__init__(config)
        self.duration_predictor = None

    def run_warmup(self):
        raise NotImplementedError("LTX-2.5 warmup is not yet supported")

    def init_scheduler(self):
        self.scheduler = LTX25Scheduler(self.config)

    def load_model(self):
        super().load_model()
        self.duration_predictor = self.load_duration_predictor()

    def _video_vae_extra_kwargs(self):
        return {"optimization": "chunked_eager"}

    def load_duration_predictor(self):
        path = self.config.get("duration_head_original_ckpt")
        if not path:
            return None
        return LTX25DurationPredictor(
            checkpoint_path=path,
            device=torch.device(AI_DEVICE),
            dtype=GET_DTYPE(),
            cpu_offload=self.config.get("gemma_cpu_offload", self.config.get("cpu_offload", False)),
        )

    def run_text_encoder(self, input_info):
        """Encode the one positive caption consumed by the distilled source path.

        The inherited LTX-2 runner always encodes a positive/negative pair for
        CFG.  LTX-2.5 distilled runs with CFG disabled and upstream calls
        ``PromptEncoder([prompt])`` exactly once.  Keeping a batch of one avoids
        doubling Gemma 4 memory/compute and preserves the same kernel shapes.
        The negative slots remain populated only to satisfy the shared input
        contract; the model never selects them while CFG is disabled.
        """
        ((video_context, audio_context),) = self.text_encoders[0].encode_text([input_info.prompt])
        return {
            "v_context_p": video_context,
            "a_context_p": audio_context,
            "v_context_n": video_context,
            "a_context_n": audio_context,
        }

    def _resolve_target_video_length(self, text_encoder_output) -> int:
        requested = int(self.input_info.target_video_length or 0)
        if requested > 0:
            num_frames = requested
            source = "request"
        elif self.config.get("auto_duration", True):
            if self.duration_predictor is None:
                self.duration_predictor = self.load_duration_predictor()
            if self.duration_predictor is None:
                raise ValueError("LTX-2.5 auto duration was requested but duration_head_original_ckpt is unavailable; pass --num_frames explicitly")
            num_frames = self.duration_predictor.predict(
                video_context=text_encoder_output["v_context_p"],
                audio_context=text_encoder_output["a_context_p"],
                frame_rate=float(self.config["fps"]),
                min_seconds=float(self.config.get("auto_duration_min_seconds", 1.0)),
                max_seconds=float(self.config.get("auto_duration_max_seconds", 20.0)),
            )
            source = "DurationHead"
        else:
            configured = self.config.get("target_video_length")
            if configured is None:
                raise ValueError("LTX-2.5 auto_duration is disabled and no frame count was provided; pass num_frames/--num_frames or set target_video_length in the profile")
            num_frames = int(configured)
            source = "config"

        if num_frames < 1 or (num_frames - 1) % 8 != 0:
            raise ValueError(f"LTX-2.5 output length must satisfy num_frames=8k+1, got {num_frames} from {source}")
        self.input_info.target_video_length = num_frames
        logger.info(f"LTX-2.5 target video length: {num_frames} frames ({source})")
        return num_frames

    def _prepare_stage1_target_shape(self) -> None:
        """Interpret request/config dimensions as final two-stage dimensions."""
        if self.input_info.target_shape:
            if len(self.input_info.target_shape) != 2:
                raise ValueError(f"LTX-2.5 target_shape must be [height, width], got {self.input_info.target_shape}")
            final_height, final_width = map(int, self.input_info.target_shape)
        else:
            final_height = int(self.config["target_height"])
            final_width = int(self.config["target_width"])
        if final_height % 64 != 0 or final_width % 64 != 0:
            raise ValueError(f"LTX-2.5 distilled two-stage output height and width must be divisible by 64, got {final_height}x{final_width}")
        self.input_info.target_shape = [final_height // 2, final_width // 2]

    def _validate_sequence_parallel_shape(self, num_frames: int, guiding_keyframes: int = 0) -> None:
        """Reject SP layouts that would introduce unmasked video tokens.

        The inherited LTX-2 sequence-parallel path pads the flattened video
        sequence to the world size.  Those padding tokens do not have an
        attention mask, so accepting a non-divisible custom canvas would
        silently change the real-token result.  The supplied 1024x1536
        profile is divisible at both stages; custom profiles must be too.
        """
        parallel = self.config.get("parallel")
        if not parallel:
            return
        seq_p_size = int(parallel.get("seq_p_size", 1))
        if seq_p_size <= 1:
            return

        latent_frames = (num_frames - 1) // int(self.config["vae_scale_factors"][0]) + 1
        stage1_height, stage1_width = map(int, self.input_info.target_shape)
        spatial_stride = int(self.config["vae_scale_factors"][1])
        stage_token_counts = {
            "stage 1": (latent_frames + guiding_keyframes) * (stage1_height // spatial_stride) * (stage1_width // spatial_stride),
            "stage 2": (latent_frames + guiding_keyframes) * (stage1_height * 2 // spatial_stride) * (stage1_width * 2 // spatial_stride),
        }
        invalid = {name: count for name, count in stage_token_counts.items() if count % seq_p_size}
        if invalid:
            details = ", ".join(f"{name}={count}" for name, count in invalid.items())
            raise ValueError(
                f"LTX-2.5 sequence-parallel video token counts must be divisible by seq_p_size={seq_p_size}; got {details}. Choose another final canvas/frame count or use the single-GPU profile."
            )

    def _run_input_encoder_local_t2av(self):
        self._clear_ltx2_reference_audio_state()
        self._clear_ltx2_reference_video_state()
        self.video_denoise_mask = None
        self.initial_video_latent = None
        self._prepare_stage1_target_shape()
        text_encoder_output = self.run_text_encoder(self.input_info)
        num_frames = self._resolve_target_video_length(text_encoder_output)
        self._validate_sequence_parallel_shape(num_frames)
        self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()
        self.maybe_empty_cache()
        return {"text_encoder_output": text_encoder_output, "image_encoder_output": None}

    def _run_input_encoder_local_i2av(self):
        self._clear_ltx2_reference_audio_state()
        self._clear_ltx2_reference_video_state()
        self._normalize_i2av_input_fields()
        self._prepare_stage1_target_shape()
        text_encoder_output = self.run_text_encoder(self.input_info)
        num_frames = self._resolve_target_video_length(text_encoder_output)
        image_paths = [path.strip() for path in (self.input_info.image_path or "").split(",") if path.strip()]
        frame_indices = self.input_info.image_frame_idx
        if not frame_indices:
            if len(image_paths) <= 1:
                frame_indices = [0] * len(image_paths)
            else:
                frame_indices = [round(i * (num_frames - 1) / (len(image_paths) - 1)) for i in range(len(image_paths))]
        guiding_keyframes = sum(int(frame_idx) != 0 for frame_idx in frame_indices)
        self._validate_sequence_parallel_shape(num_frames, guiding_keyframes)
        self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()
        self.video_denoise_mask, self.initial_video_latent = self.run_vae_encoder()
        self.maybe_empty_cache()
        return {"text_encoder_output": text_encoder_output}

    def init_run(self):
        self.scheduler.set_stage(1)
        return super().init_run()

    def run_upsampler(self, v_latent, a_latent, prepare_only=False):
        self.scheduler.set_stage(2)
        return super().run_upsampler(v_latent, a_latent, prepare_only=prepare_only)

    def process_images_after_vae_decoder(self):
        """Return/save the source-compatible float DiffVAE stream."""
        if self.input_info.return_result_tensor:
            return {"video": self.gen_video_final, "audio": self.gen_audio_final}
        if self.input_info.save_result_path is None:
            return {"video": None}
        if not dist.is_initialized() or dist.get_rank() == 0:
            encode_video_ltx25(
                video=self.gen_video_final,
                fps=int(self.config.get("fps", 24)),
                audio=self.gen_audio_final,
                output_path=self.input_info.save_result_path,
                video_chunks_number=1,
                crf=int(self.config.get("output_video_crf", 19)),
                preset=self.config.get("output_video_preset", "veryfast"),
            )
            logger.info(f"LTX-2.5 video saved to: {self.input_info.save_result_path}")
        return {"video": None}

    def run_pipeline(self, input_info):
        """Ensure a failed/cancelled LTX-2.5 request cannot retain RNG state."""
        try:
            return super().run_pipeline(input_info)
        except BaseException:
            # The inherited runner only calls end_run() on the success path.
            # A resident runner must not reuse a partially consumed main or
            # ancestral generator after an encoder/DiT/VAE/media failure.
            try:
                self.end_run()
            finally:
                self.__dict__.pop("inputs", None)
                self.input_info = None
            raise
