import json
import os
import tempfile
import time
from contextlib import closing
from fractions import Fraction
from itertools import islice, repeat
from pathlib import Path
from typing import ClassVar

import av
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from loguru import logger

from lightx2v.common.kvcache.manager import KVCacheManager
from lightx2v.models.networks.relayvsr.attention import LOCAL_ATTENTION_BACKENDS
from lightx2v.models.networks.relayvsr.model import RelayVSRModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.models.schedulers.relayvsr.scheduler import RelayVSRScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import mux_audio_from_video
from lightx2v_platform.base.global_var import AI_DEVICE

from .video_io import AsyncVideoWriter, DeviceTimings, prefetch_frames


def validate_config(config):
    backend = config.get("attention_backend", "triton")
    if backend not in LOCAL_ATTENTION_BACKENDS:
        raise ValueError(f"RelayVSR attention_backend must be one of {LOCAL_ATTENTION_BACKENDS}, got {backend!r}")
    if config.get("cpu_offload") and config.get("offload_granularity", "block") != "block":
        raise ValueError("RelayVSR supports block weight offload only")
    parallel = config.get("parallel") or {}
    if parallel.get("seq_p_size", 1) not in (1, 2):
        raise ValueError("RelayVSR sequence parallelism supports 1 or 2 GPUs")
    for key, default, minimum in (("clip_len", 16, 2), ("queue_size", 16, 1), ("encoder_threads", 16, 1)):
        if config.get(key, default) < minimum:
            raise ValueError(f"RelayVSR {key} must be >= {minimum}")


@RUNNER_REGISTER("relayvsr")
class RelayVSRRunner(DefaultRunner):
    """Official big RelayVSR: sparse Wan references + streaming Flash model decode."""

    supported_request_fields_by_task: ClassVar[dict] = {"sr": COMMON_REQUEST_FIELDS | {"video_path", "image_path", "sr_ratio", "size"}}

    WARMUP_RESOLUTIONS = ((768, 1024), (768, 1344))

    def __init__(self, config):
        for key, value in (("infer_steps", 1), ("guidance_scale", 1.0), ("enable_cfg", False), ("fps", None)):
            config.setdefault(key, value)
        validate_config(config)
        super().__init__(config)
        if AI_DEVICE != "cuda" or not torch.get_device_module(torch.device(AI_DEVICE)).is_available() or GET_DTYPE() not in (torch.bfloat16, torch.float16):
            raise ValueError("RelayVSR local attention requires a CUDA-compatible accelerator with BF16 or FP16")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.main_rank = not dist.is_initialized() or dist.get_rank() == 0
        self.seq_p_group = config["device_mesh"].get_group(mesh_dim="seq_p") if config.get("seq_parallel") else None
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size != (config.get("parallel") or {}).get("seq_p_size", 1):
            raise ValueError("RelayVSR world size must equal parallel.seq_p_size")

    def init_scheduler(self):
        self.scheduler = RelayVSRScheduler(self.config)

    @torch.no_grad()
    def init_modules(self):
        started = time.perf_counter()
        self.model = RelayVSRModel(self.config, self.seq_p_group)
        self.model.set_scheduler(self.scheduler)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.config.lock()
        logger.info("RelayVSR native 4x model loaded in {:.2f}s", self.load_seconds)

    @torch.no_grad()
    def run_warmup(self):
        if type(self) is not RelayVSRRunner or self.config["task"] != "sr":
            raise NotImplementedError("RelayVSR warmup supports the native SR runner only")
        if self.config.get("lazy_load", False):
            raise NotImplementedError("RelayVSR warmup does not support lazy loading")
        started = time.perf_counter()
        # Fill the causal Wan cache and overwrite every slot once. The Flash model's
        # condition, first-frame and history branches follow the normal stream.
        references = 2 * self.model.wan_config["temporal_context_frames"]
        frames = (references - 1) * (self.config.get("clip_len", 16) - 1) + 1
        for height, width in self.WARMUP_RESOLUTIONS:
            frame = prediction = pixels = None
            logger.info("RelayVSR warmup: {}x{}, {} reference frames", height, width, references)
            try:
                frame = torch.zeros((3, height, width), dtype=torch.uint8, pin_memory=True)
                with closing(self.restore_frames(repeat(frame, frames), 42)) as predictions:
                    for prediction in predictions:
                        for scale in (2, 4):
                            pixels = self.postprocess(prediction, (height * scale, width * scale))
                torch.cuda.synchronize()
            finally:
                frame = prediction = pixels = None
                self.timings = None
                self.input_info = None
        self.warmup_seconds = time.perf_counter() - started
        logger.info("RelayVSR warmup completed in {:.2f}s", self.warmup_seconds)
        self._maybe_freeze_gc()

    @staticmethod
    def output_size(input_info, height, width):
        if input_info.size:
            size = input_info.size
            if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in size):
                raise ValueError("RelayVSR size must be [height, width] with integer dimensions")
            target = tuple(size)
        else:
            if input_info.sr_ratio not in (2, 4):
                raise ValueError("RelayVSR sr_ratio must be 2 or 4; 2x resizes the native 4x result")
            target = (int(height * input_info.sr_ratio), int(width * input_info.sr_ratio))
        if min(target) <= 0 or target[0] > height * 4 or target[1] > width * 4:
            raise ValueError("RelayVSR output dimensions must be positive and no larger than the native 4x result")
        return target

    def preprocess(self, frame):
        height, width = frame.shape[-2:]
        x = frame[None].to(self.device, non_blocking=frame.is_pinned()).to(GET_DTYPE())
        x = x.div(127.5).sub(1.0)
        return F.pad(x, (0, -width % 8, 0, -height % 8), mode="replicate")

    @staticmethod
    def postprocess(prediction, target):
        if prediction.shape[-2:] != target:
            prediction = F.interpolate(prediction.float().clamp(-1, 1), size=target, mode="bicubic", align_corners=False, antialias=True)
        return prediction[0].add(1).mul_(127.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()

    def init_run(self, height, width, seed):
        self.scheduler.prepare(seed, self.device)
        wan = self.model.wan_config
        self.wan_cache_manager = KVCacheManager(
            {
                "num_layers": wan["num_layers"],
                "num_heads": wan["num_attention_heads"],
                "dim": wan["num_attention_heads"] * wan["attention_head_dim"],
            },
            device=self.device,
        )
        self.wan_cache = self.wan_cache_manager.create_self_attn_kv_cache(
            "wan", height // 8 * (width // 8) * wan["temporal_context_frames"], kv_cache_scheme="frame_ring", num_heads=wan["num_attention_heads"] // self.model.parallel.size
        )
        self.flash_model_cache_manager = KVCacheManager({"num_layers": 10, "num_heads": 2, "dim": 128}, device=self.device)
        flash_model_heads = 2 if self.model.flash_model_transformer.query_parallel else 2 // self.model.parallel.size
        self.flash_model_history = self.flash_model_cache_manager.create_self_attn_kv_cache("flash_model_video", height // 4 * (width // 4), kv_cache_scheme="static", num_heads=flash_model_heads)
        self.reference_count = self.flash_model_decode_count = 0

    def end_run(self):
        self.scheduler.clear()
        self.model.clear()
        self.wan_cache = self.flash_model_history = None
        self.wan_cache_manager = self.flash_model_cache_manager = None

    def reference(self, frame):
        lr = self.timings.call("preprocess", self.preprocess, frame)
        self.scheduler.prepare_latents(*lr.shape[-2:], self.device)
        self.scheduler.step_pre(0)
        self.timings.call("wan", self.model.infer, self.scheduler, lr, self.wan_cache, self.reference_count)
        self.scheduler.step_post()
        self.reference_count += 1
        latent = self.scheduler.latents[:, :, 0].to(GET_DTYPE())
        return self.timings.call("flash_model_condition", self.model.flash_model_prepare_condition, latent)

    def restore_frames(self, source, seed):
        """Yield cropped native 4x predictions; overlapping endpoint is emitted once."""
        iterator = iter(source)
        first = next(iterator, None)
        if first is None:
            raise ValueError("RelayVSR cannot restore an empty stream")
        shape = first.shape
        height, width = shape[-2:]
        self.timings = DeviceTimings()
        try:
            self.init_run(height + (-height) % 8, width + (-width) % 8, seed)
            start_condition = self.reference(first)
            first_window = True
            start = first
            while True:
                window = [start, *islice(iterator, self.config.get("clip_len", 16) - 1)]
                if any(frame.shape != shape for frame in window):
                    raise ValueError("All RelayVSR input frames must have the same dimensions")
                if len(window) == 1 and not first_window:
                    return
                self.check_stop()
                end_condition = self.reference(window[-1]) if len(window) > 1 else start_condition
                conditions = [(torch.cat((a[0], b[0]), 0), torch.cat((a[1], b[1]), 0)) for a, b in zip(start_condition, end_condition)]
                self.flash_model_history.reset()
                for offset, frame in enumerate(window):
                    self.check_stop()
                    lr = self.timings.call("preprocess", self.preprocess, frame)
                    output = self.timings.call("flash_model_decode", self.model.flash_model_decode, lr, conditions, self.flash_model_history, offset)
                    self.flash_model_decode_count += 1
                    if first_window or offset > 0:
                        yield output[..., : height * 4, : width * 4]
                first_window = False
                start, start_condition = window[-1], end_condition
                if len(window) < self.config.get("clip_len", 16):
                    return
        finally:
            self.end_run()

    @torch.no_grad()
    def run_pipeline(self, input_info):
        if bool(input_info.image_path) == bool(input_info.video_path):
            raise ValueError("RelayVSR requires exactly one image_path or video_path")
        if input_info.return_result_tensor and input_info.video_path:
            raise ValueError("RelayVSR streams videos to save_result_path; returning a whole video tensor is unsupported")
        if not input_info.save_result_path and not input_info.return_result_tensor:
            raise ValueError("RelayVSR requires save_result_path")
        self.input_info = input_info
        if input_info.video_path:
            return self.run_video(input_info)
        return self.run_image(input_info)

    def run_image(self, input_info):
        started = time.perf_counter()
        with Image.open(input_info.image_path) as image:
            frame = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).contiguous()
        target = self.output_size(input_info, *frame.shape[-2:])
        with closing(self.restore_frames([frame], input_info.seed)) as predictions:
            pixels = self.postprocess(next(predictions), target).cpu()
        if self.main_rank and input_info.save_result_path:
            path = Path(input_info.save_result_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels.numpy()).save(path)
        stats = self.timings.collect()
        stats["total_seconds"] = time.perf_counter() - started
        if self.progress_callback:
            self.progress_callback(100, 100)
        return {"images": pixels[None].float().div_(255) if input_info.return_result_tensor else None, "stats": stats}

    def run_video(self, input_info):
        started = time.perf_counter()
        source, destination = Path(input_info.video_path), Path(input_info.save_result_path)
        if source.resolve() == destination.resolve():
            raise ValueError("RelayVSR output must differ from its source video")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        writer = None
        output_frames = 0
        io_timing = {"decode_seconds": 0.0}
        torch.cuda.reset_peak_memory_stats()
        try:
            if self.main_rank:
                fd, temporary = tempfile.mkstemp(prefix=".relayvsr-", suffix=destination.suffix or ".mp4", dir=destination.parent)
                os.close(fd)
            with av.open(str(source)) as container:
                if not container.streams.video:
                    raise ValueError("RelayVSR input contains no video stream")
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                fps = Fraction(str(self.config["fps"])) if self.config.get("fps") is not None else stream.average_rate
                if fps is None or fps <= 0:
                    raise ValueError("Cannot determine source FPS; set startup fps explicitly")
                target = self.output_size(input_info, stream.height, stream.width)
                if target[0] % 2 or target[1] % 2:
                    raise ValueError("YUV420 video output requires even height and width")
                input_info.output_fps = float(fps)
                if self.main_rank:
                    writer = AsyncVideoWriter(temporary, fps, self.config)
                with closing(prefetch_frames(container, self.config.get("clip_len", 16), io_timing)) as frames, closing(self.restore_frames(frames, input_info.seed)) as predictions:
                    for prediction in predictions:
                        if self.main_rank:
                            pixels = self.timings.call("postprocess", self.postprocess, prediction, target)
                            writer.submit(pixels)
                        output_frames += 1
                        if self.main_rank and output_frames % 30 == 0:
                            logger.info("RelayVSR: {} frames written to encoder queue", output_frames)
            if writer is not None:
                writer.close()
            stages = self.timings.collect()
            mux_seconds = 0.0
            if self.main_rank:
                mux_start = time.perf_counter()
                if mux_audio_from_video(str(source), temporary, prefer_copy=True, trim_to_shortest=False) is None:
                    raise RuntimeError("RelayVSR could not mux the source audio")
                mux_seconds = time.perf_counter() - mux_start
                os.replace(temporary, destination)
                temporary = None
            stats = {
                "frames": output_frames,
                "fps": str(fps),
                "height": target[0],
                "width": target[1],
                "native_scale": 4,
                "wan_references": self.reference_count,
                "flash_model_decode_calls": self.flash_model_decode_count,
                "load_seconds": self.load_seconds,
                "cuda_stage_seconds": stages,
                "decode_seconds": io_timing["decode_seconds"],
                "output_copy_seconds": writer.copy_seconds if writer else 0,
                "encoder_queue_wait_seconds": writer.queue_wait_seconds if writer else 0,
                "encode_seconds": writer.encode_seconds if writer else 0,
                "encoder_flush_seconds": writer.flush_seconds if writer else 0,
                "audio_mux_seconds": mux_seconds,
                "total_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
            if self.main_rank:
                logger.info("RelayVSR video saved to: {}", destination.resolve())
                logger.info("RelayVSR completed: {}", json.dumps(stats, ensure_ascii=False))
                if self.progress_callback:
                    self.progress_callback(100, 100)
            return {"video": None, "stats": stats}
        finally:
            if writer is not None and temporary is not None:
                try:
                    writer.close()
                except (RuntimeError, ValueError, OSError):
                    logger.exception("RelayVSR encoder cleanup failed")
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
