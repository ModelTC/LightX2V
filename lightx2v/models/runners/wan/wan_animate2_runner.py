import math
import os
import subprocess
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

try:
    from decord import VideoReader
except ImportError:
    VideoReader = None

from lightx2v.common.kvcache import KVCacheManager
from lightx2v.models.networks.wan.animate2_model import WanAnimate2Model
from lightx2v.models.runners.wan.wan_runner import WanRunner, build_wan_model_with_lora
from lightx2v.models.schedulers.wan.animate2 import WanAnimate2Scheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("wan2.2_animate2")
class WanAnimate2Runner(WanRunner):
    """Native LightX2V runner for Wan-Animate-2.

    Wan-Animate-2 is autoregressive across video clips, while each clip uses a
    static, layer-wise KV cache populated from its driving-video branch.  It is
    intentionally registered separately from the pose/face-adapter based
    ``wan2.2_animate`` runner.
    """

    _MODEL_PATH_KEYS = (
        "dit_original_ckpt",
        "dit_quantized_ckpt",
        "t5_original_ckpt",
        "t5_quantized_ckpt",
        "t5_tokenizer_path",
        "clip_original_ckpt",
        "clip_quantized_ckpt",
        "vae_path",
    )

    def __init__(self, config):
        self._resolve_model_relative_paths(config)
        super().__init__(config)
        if self.config.get("disagg_mode"):
            raise NotImplementedError("Wan-Animate-2 does not support disaggregated inference yet.")
        if self.config.get("lazy_load", False):
            raise NotImplementedError("Wan-Animate-2 does not support lazy loading yet.")
        if self.config.get("cpu_offload", False) and self.config.get("offload_granularity", "block") == "phase":
            raise NotImplementedError("Wan-Animate-2 supports model/block offload, not phase offload.")
        if self.config.get("enable_reuse", False):
            raise NotImplementedError("Wan-Animate-2 request reuse is not implemented for autoregressive inputs.")
        if self.config["task"] != "animate":
            raise ValueError("wan2.2_animate2 requires task='animate'.")
        if self.config.get("use_stream_vae", False):
            raise NotImplementedError("Wan-Animate-2 does not support stream VAE decode because the source drops its first latent.")
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("Wan-Animate-2 currently requires feature_caching='NoCaching'.")
        if self.config.get("use_tae", False):
            raise NotImplementedError("Wan-Animate-2 reference conditioning requires the full Wan VAE encoder.")
        if not self.config.get("use_image_encoder", True) or not self.config.get("use_img_emb", True):
            raise ValueError("Wan-Animate-2 requires both use_image_encoder=true and use_img_emb=true.")
        if not self.config.get("use_31_block", True):
            raise ValueError("Wan-Animate-2 source parity requires use_31_block=true for its CLIP image features.")
        source_uses_cfg = float(self.config["sample_guide_scale"]) > 1.0
        if bool(self.config.get("enable_cfg", False)) != source_uses_cfg:
            raise ValueError(
                "Wan-Animate-2 source inference enables CFG exactly when sample_guide_scale > 1; "
                f"got enable_cfg={self.config.get('enable_cfg', False)!r}, "
                f"sample_guide_scale={self.config['sample_guide_scale']!r}."
            )

    @classmethod
    def _resolve_model_relative_paths(cls, config):
        model_path = os.path.abspath(os.path.expanduser(str(config["model_path"])))
        config["model_path"] = model_path
        for key in cls._MODEL_PATH_KEYS:
            value = config.get(key)
            if not isinstance(value, str) or not value or os.path.isabs(value):
                continue
            expanded = os.path.expanduser(value)
            cwd_candidate = os.path.abspath(expanded)
            # ``auto_calc_config`` may already have prefixed a relative
            # model_path. Keep that path instead of adding the model root twice;
            # service callers that pass release-relative component paths still
            # take the normal join below.
            try:
                already_under_root = os.path.commonpath((cwd_candidate, model_path)) == model_path
            except ValueError:
                already_under_root = False
            config[key] = cwd_candidate if already_under_root else os.path.join(model_path, expanded)

    def init_scheduler(self):
        # DefaultRunner.__init__ dispatches here before WanRunner.__init__ has
        # finished, so keep this initializer independent of runner attributes.
        if self.config.get("disagg_mode") == "decode":
            return super().init_scheduler()
        self.scheduler = WanAnimate2Scheduler(self.config)
        logger.info("Using WanAnimate2Scheduler")

    def load_transformer(self):
        model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            return WanAnimate2Model(**model_kwargs)
        return build_wan_model_with_lora(
            WanAnimate2Model,
            self.config,
            model_kwargs,
            lora_configs,
            model_type="wan2.1",
        )

    def get_vae_parallel(self):
        # The released pipeline keeps a complete VAE replica on every context-
        # parallel rank.  Its dynamically selected area-preserving canvas is not
        # guaranteed to be splittable by LightX2V's spatial VAE grid either.
        return False

    def set_inputs(self, inputs):
        """Keep the reference prompt request-scoped in service mode."""
        super().set_inputs(inputs)
        self.input_info.prompt_ref = inputs.get("prompt_ref", "人物动作的参考视频")

    def run_image_encoder(self, first_frame, last_frame=None):
        # The released pipeline calls CLIP from inside its outer BF16 autocast
        # region.  CLIP's own nested FP16 autocast covers the vision model, but
        # its bicubic resize/normalization still inherits this outer context.
        # It also reaches Conv2d in channels-last layout after rearranging its
        # CTHW input; preserve that layout because cuDNN's result differs by a
        # few FP16 ULPs and the error compounds through all 31 vision blocks.
        # Force a real NCHW allocation before converting to channels-last.  A
        # size-one batch may otherwise already satisfy the channels-last
        # predicate with a non-standard batch stride, which makes interpolate
        # select a different CUDA layout than the native Animate-2 pipeline.
        first_frame = first_frame.contiguous().contiguous(memory_format=torch.channels_last)
        if last_frame is not None:
            last_frame = last_frame.contiguous().contiguous(memory_format=torch.channels_last)
        device_type = str(AI_DEVICE).split(":", 1)[0]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return super().run_image_encoder(first_frame, last_frame)

    @staticmethod
    def _padding_resize(image, height, width, return_padding_info=False):
        """Match the upstream black-padding resize, including integer rounding."""
        orig_h, orig_w = image.shape[:2]
        interpolation = cv2.INTER_AREA if height * width < orig_h * orig_w else cv2.INTER_LINEAR
        output = np.zeros((height, width, image.shape[2]), dtype=np.uint8)

        if orig_h / orig_w > height / width:
            side_long = int(height / orig_h * orig_w)
            resized = cv2.resize(image, (side_long, height), interpolation=interpolation)
            padding = int((width - side_long) / 2)
            output[:, padding : padding + side_long] = resized
            padding_type = "width"
        else:
            side_long = int(width / orig_w * orig_h)
            resized = cv2.resize(image, (width, side_long), interpolation=interpolation)
            padding = int((height - side_long) / 2)
            output[padding : padding + side_long] = resized
            padding_type = "height"

        if not return_padding_info:
            return output
        return output, {
            "padding_type": padding_type,
            "padding": padding,
            "side_long": side_long,
        }

    @classmethod
    def _resize_by_area(cls, image, target_area, divisor=16, return_padding_info=False):
        """Reproduce the effective resize path in the released upstream code.

        Its current ``calculate_new_size`` path raises on every candidate due to
        an argument mismatch, after which it falls back to these area-preserving,
        divisor-floored dimensions.
        """
        height, width = image.shape[:2]
        aspect = width / height
        new_h = math.sqrt(target_area / aspect)
        new_w = target_area / new_h
        new_w = max(divisor, int(new_w // divisor) * divisor)
        new_h = max(divisor, int(new_h // divisor) * divisor)
        return cls._padding_resize(image, new_h, new_w, return_padding_info=return_padding_info)

    @staticmethod
    def _frame_indices(frame_num, video_fps, clip_length, target_fps):
        times = np.arange(clip_length) / target_fps
        indices = np.round(times * video_fps).astype(int)
        return np.clip(indices, 0, frame_num - 1).tolist()

    @staticmethod
    def _padding_length(input_len, clip_len, overlap=1):
        remaining = (input_len - overlap) % (clip_len - overlap)
        if remaining < 28:
            padding = 28 - remaining
        else:
            padding = 4 - remaining % 4
        return input_len + padding

    @staticmethod
    def _zigzag_padding(frames, target_len):
        if not frames:
            raise ValueError("The driving video did not yield any frames.")
        if len(frames) == 1:
            return [deepcopy(frames[0]) for _ in range(target_len)]

        period = 2 * (len(frames) - 1)
        output = []
        for index in range(target_len):
            position = index % period
            source_index = position if position < len(frames) else period - position
            output.append(deepcopy(frames[source_index]))
        return output

    @staticmethod
    def _plan_segments(total_len, clip_len, overlap=1):
        segments = []
        start = 0
        while start + overlap < total_len:
            end = min(start + clip_len, total_len)
            segments.append((start, end))
            start += (end - start) - overlap
        if not segments:
            raise ValueError(f"Unable to plan clips for total_len={total_len}, clip_len={clip_len}.")
        return segments

    def _read_and_resample_video(self, video_path):
        if VideoReader is None:
            raise ImportError("Wan-Animate-2 video input requires decord. Install it with `pip install decord`.")

        reader = VideoReader(video_path)
        actual_frame_num = len(reader)
        if actual_frame_num <= 0:
            raise ValueError(f"Driving video contains no frames: {video_path}")
        video_fps = float(reader.get_avg_fps())
        if not np.isfinite(video_fps) or video_fps <= 0:
            raise ValueError(f"Invalid driving-video FPS {video_fps}: {video_path}")

        effective_frame_num = actual_frame_num
        try:
            duration = float(reader.get_frame_timestamp(-1)[-1])
        except Exception:
            duration = float("nan")
        if not (np.isfinite(duration) and 0 < duration < actual_frame_num / video_fps * 10):
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if probe.returncode != 0:
                raise RuntimeError(f"ffprobe failed to read driving-video duration: {probe.stderr.strip()}")
            duration = float(probe.stdout.strip())

        expected_frame_num = int(duration * video_fps + 0.5)
        if abs(actual_frame_num - expected_frame_num) / actual_frame_num > 0.1:
            logger.warning(
                "Driving-video metadata differs from decoded frame count: decoded={}, expected={}",
                actual_frame_num,
                expected_frame_num,
            )
            effective_frame_num = max(1, expected_frame_num)

        # Upstream uses one ``fps`` value for both resampling and output muxing.
        target_fps = float(self.config.get("fps", 24))
        target_num = int(effective_frame_num / video_fps * target_fps)
        if target_num <= 0:
            raise ValueError(f"Driving video is too short after FPS conversion: frames={effective_frame_num}, source_fps={video_fps}, target_fps={target_fps}.")
        indices = self._frame_indices(actual_frame_num, video_fps, target_num, target_fps)
        logger.info(
            "Wan-Animate-2 samples {} decoded frames at {:.3f} FPS to {} frames at {:.3f} FPS",
            actual_frame_num,
            video_fps,
            len(indices),
            target_fps,
        )
        return reader.get_batch(indices).asnumpy()

    def prepare_input(self):
        reference_path = (self.input_info.image_path or self.input_info.src_ref_images or "").split(",")[0].strip()
        video_path = (self.input_info.video_path or self.input_info.src_pose_path or "").strip()
        if not reference_path:
            raise ValueError("Wan-Animate-2 requires --image_path (or --src_ref_images).")
        if not video_path:
            raise ValueError("Wan-Animate-2 requires --video_path.")
        if not os.path.isfile(reference_path):
            raise FileNotFoundError(f"Reference image not found: {reference_path}")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Driving video not found: {video_path}")
        # DefaultRunner's audio mux reads video_path. Keep legacy src_pose_path
        # fallback inputs source-compatible by recording the resolved driver.
        self.input_info.video_path = video_path
        if self.input_info.seed == -1:
            # Match the source pipeline's request-time sentinel semantics while
            # keeping all distributed ranks on the same randomly chosen seed.
            # Do not sample with torch here: choose OS entropy on rank 0 so the
            # request is random even when the caller used the -1 sentinel.
            seed_tensor = torch.zeros((), dtype=torch.int64, device=AI_DEVICE)
            if not dist.is_initialized() or dist.get_rank() == 0:
                seed_tensor.fill_(int.from_bytes(os.urandom(4), "little"))
            if dist.is_initialized():
                dist.broadcast(seed_tensor, src=0)
            self.input_info.seed = int(seed_tensor.cpu())
            logger.info("Wan-Animate-2 selected random seed {}", self.input_info.seed)

        reference_bgr = cv2.imread(reference_path, cv2.IMREAD_COLOR)
        if reference_bgr is None:
            raise ValueError(f"Failed to decode reference image: {reference_path}")
        reference_rgb = reference_bgr[:, :, ::-1]

        target_area = int(self.config["target_width"]) * int(self.config["target_height"])
        self.reference_image, self.output_crop = self._resize_by_area(
            reference_rgb,
            target_area,
            divisor=16,
            return_padding_info=True,
        )

        frames = self._read_and_resample_video(video_path)
        driving_frames = [self._resize_by_area(frame, target_area, divisor=16) for frame in frames]
        reference_shape = self.reference_image.shape[:2]
        driving_shape = driving_frames[0].shape[:2]

        self.real_frame_len = len(driving_frames)
        clip_len = int(self.config["target_video_length"])
        if clip_len <= 1 or (clip_len - 1) % 4:
            raise ValueError(f"target_video_length must be 4k+1 and greater than 1, got {clip_len}.")
        padded_len = self._padding_length(self.real_frame_len, clip_len, overlap=1)
        self.driving_frames = self._zigzag_padding(driving_frames, padded_len)
        self.segment_plan = self._plan_segments(padded_len, clip_len, overlap=1)

        height, width = reference_shape
        first_clip_len = self.segment_plan[0][1] - self.segment_plan[0][0]
        self.input_info.target_shape = [height, width]
        self.input_info.latent_shape = self._generation_latent_shape(first_clip_len, height, width)
        logger.info(
            "Wan-Animate-2 input: real_frames={}, padded_frames={}, clips={}, generation_canvas={}x{}, driving_canvas={}x{}",
            self.real_frame_len,
            padded_len,
            len(self.segment_plan),
            height,
            width,
            driving_shape[0],
            driving_shape[1],
        )

    def _encode_text(self, prompt):
        # Upstream invokes UMT5 inside the same outer BF16 autocast region as
        # the rest of its conditioning stack.  Although the parameters and
        # residual stream are already BF16, autocast changes the einsum/GEMM
        # dispatch used inside each T5 block and is numerically observable.
        device_type = str(AI_DEVICE).split(":", 1)[0]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            encoded = self.text_encoders[0].infer([prompt])
        text_len = int(self.config["text_len"])
        return torch.stack([torch.cat([value, value.new_zeros(text_len - value.size(0), value.size(1))]) for value in encoded])

    def run_text_encoder(self, input_info):
        transient = self.config.get("lazy_load", False) or self.config.get("unload_modules", False)
        if transient:
            self.text_encoders = self.load_text_encoder()

        prompt = input_info.prompt_enhanced if self.config["use_prompt_enhancer"] else input_info.prompt
        prompt_ref = input_info.prompt_ref or prompt
        negative_prompt = input_info.negative_prompt or self.config.get("sample_neg_prompt", "")
        context_ref = self._encode_text(prompt_ref)

        if self.config.get("enable_cfg", False) and self.config.get("cfg_parallel", False):
            cfg_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            if dist.get_rank(cfg_group) == 0:
                output = {"context": self._encode_text(prompt), "context_ref": context_ref}
            else:
                output = {"context_null": self._encode_text(negative_prompt), "context_ref": context_ref}
        else:
            output = {
                "context": self._encode_text(prompt),
                "context_null": self._encode_text(negative_prompt) if self.config.get("enable_cfg", False) else None,
                "context_ref": context_ref,
            }

        if transient:
            del self.text_encoders
            self.maybe_empty_cache()
        return output

    def _vae_encode(self, pixels):
        transient = self.config.get("lazy_load", False) or self.config.get("unload_modules", False)
        if transient:
            self.vae_encoder = self.load_vae_encoder()
        try:
            dtype = self.vae_encoder.dtype
            device_type = str(AI_DEVICE).split(":", 1)[0]
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                return self.vae_encoder.encode(pixels.to(device=AI_DEVICE, dtype=dtype)).to(GET_DTYPE())
        finally:
            if transient:
                del self.vae_encoder
                self.maybe_empty_cache()

    @staticmethod
    def _i2v_mask(latent_t, latent_h, latent_w, mask_len):
        pixel_mask = torch.zeros(
            1,
            (latent_t - 1) * 4 + 1,
            latent_h,
            latent_w,
            dtype=GET_DTYPE(),
            device=AI_DEVICE,
        )
        pixel_mask[:, :mask_len] = 1
        pixel_mask = torch.cat([pixel_mask[:, :1].repeat_interleave(4, dim=1), pixel_mask[:, 1:]], dim=1)
        return pixel_mask.view(1, latent_t, 4, latent_h, latent_w).transpose(1, 2)[0]

    def _run_input_encoder_local_animate(self):
        self.prepare_input()
        text_encoder_output = self.run_text_encoder(self.input_info)

        # Upstream normalizes uint8 through NumPy float64 and casts once to
        # BF16; doing the arithmetic after a BF16 cast changes conditioning.
        reference = torch.from_numpy(self.reference_image / 127.5 - 1.0)
        reference = reference.to(device=AI_DEVICE, dtype=GET_DTYPE()).permute(2, 0, 1).unsqueeze(0)
        self.reference_pixels = reference
        self.generation_clip = self.run_image_encoder(reference)
        self.generation_reference_latents = self._vae_encode(reference.unsqueeze(2))
        self.maybe_empty_cache()

        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
            "animate2": {
                "generation_clip": self.generation_clip,
            },
        }

    @staticmethod
    def _generation_latent_shape(clip_len, height, width):
        # Upstream uses T=clip_len+1 and lat_t=T//4+2, then decodes latents[:, 1:].
        latent_t = (clip_len + 1) // 4 + 2
        return [16, latent_t, height // 8, width // 8]

    def get_video_segment_num(self):
        self.video_segment_num = len(self.segment_plan)

    def init_run(self):
        self.gen_video_final = None
        self.get_video_segment_num()
        self.all_out_frames = []
        self.previous_frame = None

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

    def _build_reference_cache(self, reference_latents):
        ref_t, ref_h, ref_w = reference_latents.shape[1:]
        ref_tokens = ref_t * (ref_h // 2) * (ref_w // 2)
        seq_group = getattr(self.model, "seq_p_group", None)
        sp_size = dist.get_world_size(seq_group) if seq_group is not None else 1
        cache_size = ((ref_tokens + sp_size - 1) // sp_size) * sp_size

        tp_size = int(getattr(self.model, "tp_size", 1))
        model_heads = int(self.config["num_heads"]) // tp_size
        if model_heads % sp_size:
            raise ValueError(f"Wan-Animate-2 attention heads after TP ({model_heads}) must be divisible by SP size ({sp_size}).")
        cache_heads = model_heads // sp_size

        # KVCacheManager temporarily overrides ar_config while constructing a
        # named cache.  Runtime config is locked after module initialization, so
        # give the manager an isolated mutable view.
        cache_config = dict(self.config)
        cache_config["ar_config"] = dict(self.config.get("ar_config", {}))
        manager = KVCacheManager(cache_config, device=torch.device(AI_DEVICE), sp_group=seq_group)
        cache = manager.create_self_attn_kv_cache(
            "animate2_reference",
            cache_size,
            kv_cache_scheme="static",
            step_kv_cache=False,
            dtype=GET_DTYPE(),
            num_heads=cache_heads,
        )
        self.reference_kv_cache_manager = manager
        return cache

    def _build_segment_inputs(self, segment_idx):
        start, end = self.segment_plan[segment_idx]
        clip_len = end - start
        height, width = self.reference_image.shape[:2]

        driving = np.stack(self.driving_frames[start:end]) / 127.5 - 1.0
        driving = torch.from_numpy(driving).to(device=AI_DEVICE, dtype=GET_DTYPE())
        driving = driving.permute(3, 0, 1, 2).unsqueeze(0)
        reference_latents = self._vae_encode(driving)
        ref_t, ref_h, ref_w = reference_latents.shape[1:]
        reference_mask = self._i2v_mask(ref_t, ref_h, ref_w, clip_len)
        reference_y = torch.cat([reference_mask, reference_latents], dim=0)
        reference_clip = self.run_image_encoder(driving[:, :, 0])

        reft_pixels = torch.zeros(
            1,
            3,
            clip_len,
            height,
            width,
            device=AI_DEVICE,
            dtype=GET_DTYPE(),
        )
        mask_len = 0
        if segment_idx > 0:
            reft_pixels[:, :, :1] = self.previous_frame.unsqueeze(0).to(device=AI_DEVICE, dtype=GET_DTYPE())
            mask_len = 1
        reft_latents = self._vae_encode(reft_pixels)
        reft_t, latent_h, latent_w = reft_latents.shape[1:]
        reft_mask = self._i2v_mask(reft_t, latent_h, latent_w, mask_len)
        generation_ref_mask = self._i2v_mask(1, latent_h, latent_w, 1)
        generation_ref_y = torch.cat([generation_ref_mask, self.generation_reference_latents], dim=0)
        generation_reft_y = torch.cat([reft_mask, reft_latents], dim=0)
        generation_y = torch.cat([generation_ref_y, generation_reft_y], dim=1)

        latent_shape = self._generation_latent_shape(clip_len, height, width)
        if list(generation_y.shape[1:]) != latent_shape[1:]:
            raise RuntimeError(f"Generation conditioning shape {tuple(generation_y.shape)} does not match latent shape {latent_shape}.")

        animate2 = {
            "reference_latents": reference_latents,
            "reference_y": reference_y,
            "reference_clip": reference_clip,
            "generation_y": generation_y,
            "generation_clip": self.generation_clip,
            "reference_kv_cache": self._build_reference_cache(reference_latents),
            "origin_len": int(self.config["target_video_length"]),
            "origin_area": [int(self.config["target_width"]), int(self.config["target_height"])],
            "clip_len": clip_len,
        }
        self.input_info.latent_shape = latent_shape
        self.inputs["animate2"] = animate2

    def init_run_segment(self, segment_idx):
        try:
            self._build_segment_inputs(segment_idx)
            self.model.prepare_reference(self.inputs)

            if segment_idx == 0:
                self.model.scheduler.prepare(self.input_info.seed, self.input_info.latent_shape)
            else:
                self.model.scheduler.reset(self.input_info.seed, self.input_info.latent_shape)
        except BaseException:
            # ``DefaultRunner.run_main`` calls this hook before it enters
            # ``run_segment``.  Clean up here as well so a failed/cancelled
            # reference prefill cannot strand the large static cache on a
            # resident service runner.
            self._release_segment_conditioning()
            raise

    def _release_segment_conditioning(self):
        if hasattr(self, "inputs"):
            self.inputs.pop("animate2", None)
        transformer_infer = getattr(self.model, "transformer_infer", None)
        if transformer_infer is not None:
            transformer_infer.reference_kv_cache = None
        if hasattr(self, "reference_kv_cache_manager"):
            del self.reference_kv_cache_manager
        self.maybe_empty_cache()

    def run_segment(self, segment_idx=0):
        try:
            return super().run_segment(segment_idx)
        finally:
            # Denoising is the final consumer of driving latents and static K/V.
            # Release them before a CPU-offloaded VAE moves onto the accelerator,
            # and do the same if a cancelled/failed service request unwinds here.
            self._release_segment_conditioning()

    def run_vae_decoder(self, latents):
        transient = self.config.get("lazy_load", False) or self.config.get("unload_modules", False)
        if transient:
            self.vae_decoder = self.load_vae_decoder()
        try:
            # Source passes FP32 x0 through a BF16-autocast VAE and materializes
            # FP32 decoded frames for the next segment's overlap condition.
            device_type = str(AI_DEVICE).split(":", 1)[0]
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                return self.vae_decoder.decode(latents[:, 1:].float()).float()
        finally:
            if transient:
                del self.vae_decoder
                self.maybe_empty_cache()

    def end_run_segment(self, segment_idx):
        self.previous_frame = self.gen_video[0, :, -1:].detach().clone()
        output = self.gen_video if segment_idx == 0 else self.gen_video[:, :, 1:]
        self.all_out_frames.append(output.cpu())
        del self.gen_video

        self._release_segment_conditioning()

    def process_images_after_vae_decoder(self):
        video = torch.cat(self.all_out_frames, dim=2)[:, :, : self.real_frame_len]
        crop_type = self.output_crop["padding_type"]
        padding = self.output_crop["padding"]
        side_long = self.output_crop["side_long"]
        if crop_type == "width":
            video = video[..., padding : padding + side_long]
        else:
            video = video[..., padding : padding + side_long, :]
        self.gen_video_final = video
        return super().process_images_after_vae_decoder()

    def end_run(self):
        for name in (
            "generation_clip",
            "generation_reference_latents",
            "reference_pixels",
            "previous_frame",
            "reference_kv_cache_manager",
            "all_out_frames",
            "driving_frames",
            "reference_image",
            "segment_plan",
            "output_crop",
        ):
            if hasattr(self, name):
                delattr(self, name)
        super().end_run()
