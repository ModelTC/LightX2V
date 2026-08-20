import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import imageio
import torch
import torch.nn.functional as F
from decord import VideoReader
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.swiftvr import (
    RestorationAutoencoder,
    SwiftVRModel,
    SwiftVRRestorer,
    build_video_chunks,
    normalize_swiftvr_config,
    padded_frame_count,
)
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_DTYPE, GET_RECORDER_MODE
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import mux_audio_from_video


def mark_stage(device: torch.device):
    if device.type == "cuda":
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    return time.perf_counter()


def measure_stage_durations(marks, device: torch.device):
    if device.type == "cuda":
        marks[-1].synchronize()
        return [start.elapsed_time(end) / 1000 for start, end in zip(marks, marks[1:])]
    return [end - start for start, end in zip(marks, marks[1:])]


@dataclass
class PendingVideoWrite:
    chunk_index: int
    frame_count: int
    read_seconds: float
    reader_wait_seconds: float
    future: Future


@RUNNER_REGISTER("swiftvr")
class SwiftVRRunner(DefaultRunner):
    """Native LightX2V runner for SwiftVR video restoration."""

    # Two spatial shapes trigger dynamic compilation before serving requests.
    WARMUP_RESOLUTIONS = ((720, 1280), (2048, 1536))

    def __init__(self, config):
        if config["task"] != "sr":
            raise ValueError("SwiftVR only supports the `sr` task.")
        if config.get("parallel"):
            raise ValueError("SwiftVR currently supports single-GPU inference only.")
        if config.get("cpu_offload"):
            raise NotImplementedError("SwiftVR does not support CPU offload yet.")
        normalize_swiftvr_config(config)
        super().__init__(config)
        self.copy_stream = torch.cuda.Stream(device=self.init_device) if self.init_device.type == "cuda" else None

    def init_modules(self):
        logger.info(f"Loading native SwiftVR weights from {self.config['model_path']}")
        self.model = SwiftVRModel(self.config["model_path"], self.config, self.init_device)
        autoencoder = RestorationAutoencoder.from_pretrained(
            self.config["model_path"],
            self.init_device,
            GET_DTYPE(),
        )
        with safe_open(
            os.path.join(self.config["model_path"], "prompt_embedding.safetensors"),
            framework="pt",
            device="cpu",
        ) as weights:
            prompt_embedding = weights.get_tensor("prompt_emb").to(self.init_device, GET_DTYPE())

        self.restorer = SwiftVRRestorer(
            autoencoder,
            self.model,
            prompt_embedding,
            overlap=self.config.get("dit_overlap", 0),
            reae_frame_batch_size=self.config.get("reae_frame_batch_size", 1),
        )
        self.config.lock()

    @ProfilingContext4DebugL1("Warmup")
    @torch.inference_mode()
    def run_warmup(self):
        clip_length = self.config.get("clip_len", 24)
        clip_latents = clip_length // 4
        # One first, middle, and one-frame last chunk cover the 7- and 6-latent DiT paths.
        chunks = build_video_chunks(2 * clip_length + 5, clip_length)

        for height, width in self.WARMUP_RESOLUTIONS:
            padded_height = height + (-height) % 32
            padded_width = width + (-width) % 32
            logger.info(f"Warmup: {height}x{width}")
            try:
                for chunk in chunks:
                    video = torch.zeros(
                        1,
                        chunk.frame_count,
                        3,
                        padded_height,
                        padded_width,
                        dtype=GET_DTYPE(),
                        device=self.init_device,
                    )
                    restored = self.restorer.restore_chunk(video, chunk, clip_latents)
                    del video, restored
            finally:
                self.restorer.reset()

        logger.info("[Warmup] Warmup completed")
        self._maybe_freeze_gc()

    @staticmethod
    def resolve_output_size(input_info, source_height: int, source_width: int) -> tuple[int, int]:
        if input_info.target_shape:
            if len(input_info.target_shape) != 2:
                raise ValueError(f"SwiftVR target_shape must be [height, width], got {input_info.target_shape}")
            height, width = input_info.target_shape
        else:
            ratio = input_info.sr_ratio
            height, width = round(source_height * ratio), round(source_width * ratio)
        if height <= 0 or width <= 0:
            raise ValueError(f"SwiftVR output size must be positive, got {height}x{width}")
        return height, width

    @staticmethod
    def read_video_frames(reader, chunk, raw_frame_count: int, source_height: int, source_width: int, pin_memory: bool):
        started_at = time.perf_counter()
        indices = [min(index, raw_frame_count - 1) for index in range(chunk.start, chunk.start + chunk.frame_count)]
        frames = reader.get_batch(indices)
        if not torch.is_tensor(frames):
            frames = torch.from_numpy(frames.asnumpy())
        frames = frames[:, :source_height, :source_width].permute(0, 3, 1, 2).contiguous()
        if pin_memory:
            frames = frames.pin_memory()
        return frames, time.perf_counter() - started_at

    @staticmethod
    def preprocess_frames(frames, height: int, width: int, pad_height: int, pad_width: int, dtype: torch.dtype, device: torch.device, mode: str):
        frames = frames.to(device=device, dtype=dtype, non_blocking=frames.is_pinned())
        if frames.shape[-2:] != (height, width):
            interpolate_args = {"align_corners": False} if mode in {"linear", "bilinear", "bicubic", "trilinear"} else {}
            frames = F.interpolate(frames, size=(height, width), mode=mode, **interpolate_args)
        frames.div_(255)
        if pad_height or pad_width:
            frames = F.pad(frames, (0, pad_width, 0, pad_height))
        return frames.unsqueeze(0)

    @staticmethod
    def copy_frames_to_cpu(frames, copy_stream):
        if copy_stream is None:
            return frames.cpu(), time.perf_counter()

        cpu_frames = torch.empty(frames.shape, dtype=frames.dtype, device="cpu", pin_memory=True)
        copy_stream.wait_stream(torch.cuda.current_stream(frames.device))
        with torch.cuda.stream(copy_stream):
            cpu_frames.copy_(frames, non_blocking=True)
            frames.record_stream(copy_stream)
            copy_complete = torch.cuda.Event(enable_timing=True)
            copy_complete.record(copy_stream)
        return cpu_frames, copy_complete

    def open_video_writer(self, output_path: str, fps: float):
        quality = self.config.get("quality", 60)
        codec = self.config.get("video_codec", "libx265")
        ffmpeg_params = ["-crf", str(round((100 - quality) * 51 / 100))]
        if codec == "libx265":
            ffmpeg_params.extend(["-x265-params", "log-level=warning"])
        # x265 presets from fastest to slowest:
        # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo.
        preset = self.config.get("ffmpeg_preset", "")
        if preset:
            ffmpeg_params.extend(["-preset", preset])
        pixel_format = "yuv444p" if self.config.get("save_format") == "yuv444p" else "yuv420p"
        return imageio.get_writer(
            output_path,
            fps=fps,
            codec=codec,
            pixelformat=pixel_format,
            macro_block_size=None,
            ffmpeg_params=ffmpeg_params,
        )

    @staticmethod
    def write_video_frames(writer, frames, stage_marks, device: torch.device):
        stage_durations = measure_stage_durations(stage_marks, device)
        for frame in frames.numpy():
            writer.append_data(frame)
        return stage_durations

    @staticmethod
    def finish_video_write(pending_write: PendingVideoWrite, stage_seconds, chunk_count: int):
        wait_started_at = time.perf_counter()
        preprocess_seconds, restore_seconds, postprocess_seconds = pending_write.future.result()
        writer_wait_seconds = time.perf_counter() - wait_started_at

        stage_seconds["read"] += pending_write.read_seconds
        stage_seconds["reader_wait"] += pending_write.reader_wait_seconds
        stage_seconds["preprocess"] += preprocess_seconds
        stage_seconds["restore"] += restore_seconds
        stage_seconds["postprocess_d2h"] += postprocess_seconds
        stage_seconds["writer_wait"] += writer_wait_seconds
        logger.info(
            f"SwiftVR chunk {pending_write.chunk_index + 1}/{chunk_count} restored {pending_write.frame_count} frames "
            f"(read={pending_write.read_seconds:.3f}s, reader_wait={pending_write.reader_wait_seconds:.3f}s, "
            f"preprocess={preprocess_seconds:.3f}s, restore={restore_seconds:.3f}s, "
            f"postprocess_d2h={postprocess_seconds:.3f}s, writer_wait={writer_wait_seconds:.3f}s)"
        )

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["SwiftVRRunner"],
        profile_memory=True,
    )
    @torch.inference_mode()
    def run_pipeline(self, input_info):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info
        if not input_info.video_path:
            raise ValueError("SwiftVR requires `video_path`.")
        if not input_info.save_result_path:
            raise ValueError("SwiftVR requires `save_result_path`.")
        if input_info.return_result_tensor:
            raise ValueError("SwiftVR streams its result to a video file; `return_result_tensor` is not supported.")

        reader = VideoReader(input_info.video_path)
        raw_frame_count = len(reader)
        first_frame = reader[0]
        source_height = first_frame.shape[0] // 8 * 8
        source_width = first_frame.shape[1] // 8 * 8
        output_height, output_width = self.resolve_output_size(input_info, source_height, source_width)
        pad_height = (-output_height) % 32
        pad_width = (-output_width) % 32
        fps = self.config.get("fps") or reader.get_avg_fps() or 30

        clip_length = self.config.get("clip_len", 24)
        process_frame_count = padded_frame_count(raw_frame_count)
        chunks = build_video_chunks(process_frame_count, clip_length)
        clip_latents = clip_length // 4

        output_path = os.path.abspath(input_info.save_result_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.restorer.reset()
        writer = self.open_video_writer(output_path, fps)
        reader_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swiftvr-reader")
        writer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swiftvr-writer")
        max_pending = self.config.get("queue_size", 3)
        pending_reads = deque()
        pending_writes = deque()
        written = 0
        stage_seconds = {
            "read": 0.0,
            "reader_wait": 0.0,
            "preprocess": 0.0,
            "restore": 0.0,
            "postprocess_d2h": 0.0,
            "writer_wait": 0.0,
        }
        started_at = time.perf_counter()
        for chunk in chunks[:max_pending]:
            pending_reads.append(
                reader_executor.submit(
                    self.read_video_frames,
                    reader,
                    chunk,
                    raw_frame_count,
                    source_height,
                    source_width,
                    self.copy_stream is not None,
                )
            )
        try:
            for chunk_index, chunk in enumerate(chunks):
                self.check_stop()
                reader_wait_started_at = time.perf_counter()
                frames, read_seconds = pending_reads.popleft().result()
                reader_wait_seconds = time.perf_counter() - reader_wait_started_at
                next_read_index = chunk_index + max_pending
                if next_read_index < len(chunks):
                    pending_reads.append(
                        reader_executor.submit(
                            self.read_video_frames,
                            reader,
                            chunks[next_read_index],
                            raw_frame_count,
                            source_height,
                            source_width,
                            self.copy_stream is not None,
                        )
                    )

                stage_marks = [mark_stage(self.init_device)]
                video = self.preprocess_frames(
                    frames,
                    output_height,
                    output_width,
                    pad_height,
                    pad_width,
                    GET_DTYPE(),
                    self.init_device,
                    self.config.get("upscale_mode", "bilinear"),
                )
                stage_marks.append(mark_stage(self.init_device))
                restored = self.restorer.restore_chunk(video, chunk, clip_latents)
                stage_marks.append(mark_stage(self.init_device))
                restored = restored[..., :output_height, :output_width]
                restored = restored[:, : raw_frame_count - written]
                output_frames = (restored[0].permute(0, 2, 3, 1) * 255).clamp_(0, 255).to(torch.uint8)
                cpu_frames, copy_complete = self.copy_frames_to_cpu(output_frames, self.copy_stream)
                stage_marks.append(copy_complete)

                if len(pending_writes) >= max_pending:
                    self.finish_video_write(pending_writes.popleft(), stage_seconds, len(chunks))
                pending_writes.append(
                    PendingVideoWrite(
                        chunk_index=chunk_index,
                        frame_count=len(cpu_frames),
                        read_seconds=read_seconds,
                        reader_wait_seconds=reader_wait_seconds,
                        future=writer_executor.submit(
                            self.write_video_frames,
                            writer,
                            cpu_frames,
                            stage_marks,
                            self.init_device,
                        ),
                    )
                )
                written += len(cpu_frames)

                if self.progress_callback:
                    self.progress_callback((chunk.index + 1) / len(chunks) * 100, 100)
            while pending_writes:
                self.finish_video_write(pending_writes.popleft(), stage_seconds, len(chunks))
        finally:
            reader_executor.shutdown(wait=True, cancel_futures=True)
            writer_executor.shutdown(wait=True)
            writer.close()

        elapsed = time.perf_counter() - started_at
        mux_audio_from_video(
            input_info.video_path,
            output_path,
            prefer_copy=self.config.get("audio_mux_prefer_copy", True),
            trim_to_shortest=False,
        )
        stats = {
            "frames": written,
            "seconds": elapsed,
            "fps": written / elapsed if elapsed else 0.0,
            "output": output_path,
            "stage_seconds": stage_seconds,
        }
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        logger.info(f"SwiftVR restored {written} frames to {output_path} at {stats['fps']:.2f} fps")
        logger.info("SwiftVR stage totals: " + ", ".join(f"{name}={seconds:.3f}s" for name, seconds in stage_seconds.items()))
        return {"video": None, "stats": stats}
