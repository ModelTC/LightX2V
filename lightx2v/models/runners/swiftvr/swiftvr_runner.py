import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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


@RUNNER_REGISTER("swiftvr")
class SwiftVRRunner(DefaultRunner):
    """Native LightX2V runner for SwiftVR video restoration."""

    def __init__(self, config):
        if config["task"] != "sr":
            raise ValueError("SwiftVR only supports the `sr` task.")
        if config.get("parallel"):
            raise ValueError("SwiftVR currently supports single-GPU inference only.")
        normalize_swiftvr_config(config)
        super().__init__(config)

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
            overlap=int(self.config.get("dit_overlap", 0)),
            reae_frame_batch_size=int(self.config.get("reae_frame_batch_size", 1)),
        )
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = bool(self.config.get("allow_tf32", True))
            torch.backends.cudnn.allow_tf32 = bool(self.config.get("allow_tf32", True))
            torch.set_float32_matmul_precision("high")
        self.config.lock()

    @staticmethod
    def resolve_output_size(input_info, source_height: int, source_width: int) -> tuple[int, int]:
        if input_info.target_shape:
            if len(input_info.target_shape) != 2:
                raise ValueError(f"SwiftVR target_shape must be [height, width], got {input_info.target_shape}")
            height, width = map(int, input_info.target_shape)
        else:
            ratio = float(input_info.sr_ratio)
            height, width = round(source_height * ratio), round(source_width * ratio)
        if height <= 0 or width <= 0:
            raise ValueError(f"SwiftVR output size must be positive, got {height}x{width}")
        return height, width

    @staticmethod
    def preprocess_frames(frames, height: int, width: int, pad_height: int, pad_width: int, dtype: torch.dtype, device: torch.device, mode: str):
        frames = frames.permute(0, 3, 1, 2).contiguous().to(device=device, dtype=dtype)
        if frames.shape[-2:] != (height, width):
            interpolate_args = {"align_corners": False} if mode in {"linear", "bilinear", "bicubic", "trilinear"} else {}
            frames = F.interpolate(frames, size=(height, width), mode=mode, **interpolate_args)
        frames.div_(255)
        if pad_height or pad_width:
            frames = F.pad(frames, (0, pad_width, 0, pad_height))
        return frames.unsqueeze(0)

    def open_video_writer(self, output_path: str, fps: float):
        quality = max(0, min(100, int(self.config.get("quality", 60))))
        codec = self.config.get("video_codec", "libx265")
        ffmpeg_params = ["-crf", str(round((100 - quality) * 51 / 100))]
        if codec == "libx265":
            ffmpeg_params.extend(["-x265-params", "log-level=warning"])
        preset = self.config.get("ffmpeg_preset", "")
        if preset:
            ffmpeg_params.extend(["-preset", str(preset)])
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
    def write_video_frames(writer, frames):
        for frame in frames:
            writer.append_data(frame)

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
        fps = float(self.config.get("fps") or reader.get_avg_fps() or 30)

        clip_length = int(self.config.get("clip_len", 24))
        process_frame_count = padded_frame_count(raw_frame_count)
        chunks = build_video_chunks(process_frame_count, clip_length)
        clip_latents = clip_length // 4

        output_path = os.path.abspath(input_info.save_result_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.restorer.reset()
        writer = self.open_video_writer(output_path, fps)
        writer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swiftvr-writer")
        pending_writes = deque()
        max_pending_writes = max(1, int(self.config.get("queue_size", 3)))
        written = 0
        stage_seconds = {
            "read": 0.0,
            "preprocess": 0.0,
            "restore": 0.0,
            "postprocess_d2h": 0.0,
            "writer_wait": 0.0,
        }
        started_at = time.perf_counter()
        try:
            for chunk in chunks:
                self.check_stop()
                read_started_at = time.perf_counter()
                indices = [min(index, raw_frame_count - 1) for index in range(chunk.start, chunk.start + chunk.frame_count)]
                frames = reader.get_batch(indices)
                frames = torch.from_numpy(frames.asnumpy()) if hasattr(frames, "asnumpy") else torch.as_tensor(frames)
                frames = frames[:, :source_height, :source_width]
                read_seconds = time.perf_counter() - read_started_at

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
                output_frames = (restored[0].permute(0, 2, 3, 1) * 255).clamp_(0, 255).to(torch.uint8).cpu().numpy()
                stage_marks.append(mark_stage(self.init_device))
                preprocess_seconds, restore_seconds, postprocess_seconds = measure_stage_durations(stage_marks, self.init_device)

                writer_wait_seconds = 0.0
                if len(pending_writes) >= max_pending_writes:
                    writer_wait_started_at = time.perf_counter()
                    pending_writes.popleft().result()
                    writer_wait_seconds = time.perf_counter() - writer_wait_started_at
                pending_writes.append(writer_executor.submit(self.write_video_frames, writer, output_frames))
                written += len(output_frames)
                stage_seconds["read"] += read_seconds
                stage_seconds["preprocess"] += preprocess_seconds
                stage_seconds["restore"] += restore_seconds
                stage_seconds["postprocess_d2h"] += postprocess_seconds
                stage_seconds["writer_wait"] += writer_wait_seconds

                if self.progress_callback:
                    self.progress_callback((chunk.index + 1) / len(chunks) * 100, 100)
                logger.info(
                    f"SwiftVR chunk {chunk.index + 1}/{len(chunks)} restored {len(output_frames)} frames "
                    f"(read={read_seconds:.3f}s, preprocess={preprocess_seconds:.3f}s, restore={restore_seconds:.3f}s, "
                    f"postprocess_d2h={postprocess_seconds:.3f}s, writer_wait={writer_wait_seconds:.3f}s)"
                )
            while pending_writes:
                writer_wait_started_at = time.perf_counter()
                pending_writes.popleft().result()
                stage_seconds["writer_wait"] += time.perf_counter() - writer_wait_started_at
        finally:
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
