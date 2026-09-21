import os
import shutil
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from fractions import Fraction

import av
import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
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
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_DTYPE, GET_RECORDER_MODE
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import is_main_process, mux_audio_from_video, save_to_image


@RUNNER_REGISTER("swiftvr")
class SwiftVRRunner(DefaultRunner):
    """Native LightX2V runner for SwiftVR image and video restoration."""

    supported_request_fields_by_task = {
        "sr": COMMON_REQUEST_FIELDS | {"image_path", "sr_ratio", "size", "video_path"},
    }

    # Two spatial shapes trigger dynamic compilation before serving requests.
    WARMUP_RESOLUTIONS = ((720, 1280), (2048, 1536))

    def __init__(self, config):
        parallel = config.get("parallel") or {}
        self.chunk_p_size = parallel.get("chunk_p_size", 1)
        if any(parallel.get(name, 1) != 1 for name in ("seq_p_size", "cfg_p_size", "tensor_p_size")):
            raise ValueError("SwiftVR supports chunk parallelism only.")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.chunk_p_size != world_size:
            raise ValueError(f"SwiftVR requires parallel.chunk_p_size ({self.chunk_p_size}) to match world_size ({world_size}).")
        if config.get("cpu_offload"):
            raise NotImplementedError("SwiftVR does not support CPU offload yet.")
        normalize_swiftvr_config(config)
        super().__init__(config)
        self.chunk_p_group = dist.group.WORLD if self.chunk_p_size > 1 else None
        self.is_main_process = is_main_process()
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
            overlap = self.restorer.transformer.overlap
            try:
                self.restorer.transformer.overlap = 0
                self.restorer.autoencoder.chunk_p_group = self.chunk_p_group
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
                self.restorer.transformer.overlap = overlap
                self.restorer.autoencoder.chunk_p_group = None
                self.restorer.reset()

        logger.info("[Warmup] Warmup completed")
        self._maybe_freeze_gc()

    @staticmethod
    def resolve_output_size(
        input_info,
        source_height: int,
        source_width: int,
        *,
        require_even: bool = False,
    ) -> tuple[int, int]:
        if input_info.size:
            if len(input_info.size) != 2:
                raise ValueError(f"SwiftVR size must be [height, width], got {input_info.size}")
            height, width = input_info.size
        else:
            ratio = input_info.sr_ratio
            height, width = round(source_height * ratio), round(source_width * ratio)
        if height <= 0 or width <= 0:
            raise ValueError(f"SwiftVR output size must be positive, got {height}x{width}")
        if require_even:
            height = max(2, round(height / 2) * 2)
            width = max(2, round(width / 2) * 2)
        return height, width

    @staticmethod
    def resolve_input_kind(input_info) -> str:
        has_image = bool(input_info.image_path)
        has_video = bool(input_info.video_path)
        if has_image == has_video:
            raise ValueError("SwiftVR requires exactly one of `image_path` or `video_path`.")
        return "image" if has_image else "video"

    @staticmethod
    def read_image_frame(image_path: str):
        with Image.open(image_path) as image:
            frame = np.array(image.convert("RGB"), dtype=np.uint8)
        source_height = frame.shape[0] // 8 * 8
        source_width = frame.shape[1] // 8 * 8
        if source_height <= 0 or source_width <= 0:
            raise ValueError(f"SwiftVR image is too small after 8-pixel alignment: {frame.shape[:2]}.")
        frames = torch.from_numpy(frame[:source_height, :source_width]).permute(2, 0, 1).contiguous().unsqueeze(0)
        return frames, source_height, source_width

    @staticmethod
    def read_video_frames(reader, chunk, raw_frame_count: int, source_height: int, source_width: int, pin_memory: bool):
        indices = [min(index, raw_frame_count - 1) for index in range(chunk.start, chunk.start + chunk.frame_count)]
        frames = reader.get_batch(indices)
        if not torch.is_tensor(frames):
            frames = torch.from_dlpack(frames.to_dlpack())
        frames = frames[:, :source_height, :source_width].permute(0, 3, 1, 2)
        if pin_memory:
            frames = frames.pin_memory()
        return frames

    def preprocess_frames(self, frames, height: int, width: int):
        # Transfer only byte pixels; convert dtype and layout on the target device.
        frames = frames.to(device=self.init_device, non_blocking=frames.is_pinned())
        frames = frames.to(dtype=GET_DTYPE(), memory_format=torch.contiguous_format)
        if frames.shape[-2:] != (height, width):
            mode = self.config.get("upscale_mode", "bilinear")
            interpolate_args = {"align_corners": False} if mode in {"linear", "bilinear", "bicubic", "trilinear"} else {}
            frames = F.interpolate(frames, size=(height, width), mode=mode, **interpolate_args)
        frames.div_(255)
        pad_height, pad_width = (-height) % 32, (-width) % 32
        if pad_height or pad_width:
            frames = F.pad(frames, (0, pad_width, 0, pad_height))
        return frames.unsqueeze(0)

    @staticmethod
    def copy_frames_to_cpu(frames, copy_stream):
        if copy_stream is None:
            return frames.cpu(), None

        cpu_frames = torch.empty(frames.shape, dtype=frames.dtype, device="cpu", pin_memory=True)
        copy_stream.wait_stream(torch.cuda.current_stream(frames.device))
        with torch.cuda.stream(copy_stream):
            cpu_frames.copy_(frames, non_blocking=True)
            frames.record_stream(copy_stream)
            copy_complete = torch.cuda.Event()
            copy_complete.record(copy_stream)
        return cpu_frames, copy_complete

    def open_video_writer(self, output_path: str, fps: float, height: int, width: int, frame_counts=None):
        """Write one video, or a numbered segment for each count in frame_counts."""
        quality = self.config.get("quality", 60)
        codec = self.config.get("video_codec", "libx265")
        threads = min(16, max(1, len(os.sched_getaffinity(0)) // self.chunk_p_size)) if self.chunk_p_size > 1 else None
        codec_options = {"crf": str(round((100 - quality) * 51 / 100))}
        if codec == "libx265":
            codec_options["x265-params"] = "log-level=warning"
            if threads is not None:
                codec_options["x265-params"] += f":pools={threads}"
        elif threads is not None:
            codec_options["threads"] = str(threads)

        preset = self.config.get("ffmpeg_preset")
        if preset:
            codec_options["preset"] = preset

        pixel_format = "yuv444p" if self.config.get("save_format") == "yuv444p" else "yuv420p"

        def write_video(path):
            if self.chunk_p_size == 1:
                ffmpeg_params = []
                for name, value in codec_options.items():
                    ffmpeg_params.extend([f"-{name}", value])
                if codec == "libx265":
                    ffmpeg_params.extend(["-tag:v", "hvc1"])
                with closing(
                    imageio.get_writer(
                        path,
                        fps=fps,
                        codec=codec,
                        pixelformat=pixel_format,
                        macro_block_size=None,
                        ffmpeg_params=ffmpeg_params,
                    )
                ) as writer:
                    while True:
                        writer.append_data((yield))
            else:
                with av.open(path, "w") as container:
                    stream = container.add_stream(
                        codec,
                        rate=Fraction(f"{fps:.02f}"),
                        options=codec_options,
                        width=width,
                        height=height,
                        pix_fmt=pixel_format,
                    )
                    if codec == "libx265":
                        stream.codec_context.codec_tag = "hvc1"
                    reformatter = av.video.reformatter.VideoReformatter()
                    frames_written = 0
                    try:
                        while True:
                            array = yield
                            # Share the pinned RGB buffer before color conversion.
                            frame = av.VideoFrame.from_numpy_buffer(array, format="rgb24")
                            frame = reformatter.reformat(frame, format=pixel_format, interpolation="BICUBIC", threads=min(4, threads))
                            container.mux(stream.encode(frame))
                            frames_written += 1
                    except GeneratorExit:
                        if frames_written:
                            packets = stream.encode()
                            if codec == "libx265" and frames_written < 3:
                                # Short x265 segments still need the decode delay declared in their headers.
                                decoder = av.CodecContext.create("hevc", "r")
                                decoder.thread_count = 1
                                decoder.extradata = stream.codec_context.extradata
                                decoder.open()
                                for packet in packets:
                                    packet.dts = packet.pts - decoder.reorder_depth
                            container.mux(packets)

        def write_segments():
            for segment, frame_count in enumerate(frame_counts):
                # Each generator owns and releases one segment's encoder and packets.
                with closing(write_video(output_path % segment)) as writer:
                    next(writer)
                    for _ in range(frame_count):
                        writer.send((yield))
            yield

        writer = write_video(output_path) if frame_counts is None else write_segments()
        next(writer)
        return writer

    @staticmethod
    def write_video_frames(writer, frames, copy_complete):
        if copy_complete is not None:
            copy_complete.synchronize()
        for frame in frames.numpy():
            writer.send(frame)

    def join_video_chunks(self, work_dir, chunk_count, output_path, source_path):
        manifest = os.path.join(work_dir, "chunks.txt")
        with open(manifest, "w") as file:
            file.write("ffconcat version 1.0\n")
            for index in range(chunk_count):
                segment, rank = divmod(index, self.chunk_p_size)
                file.write(f"file '{rank:03d}-{segment:08d}.mp4'\n")

        result = mux_audio_from_video(
            source_path,
            manifest,
            output_path=output_path,
            prefer_copy=self.config.get("audio_mux_prefer_copy", True),
            trim_to_shortest=False,
        )
        if result is None:
            raise RuntimeError("SwiftVR failed to assemble the output video.")

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["SwiftVRRunner"],
        profile_memory=True,
    )
    @torch.inference_mode()
    def run_pipeline(self, input_info):
        self.input_info = input_info
        try:
            if self.is_main_process and GET_RECORDER_MODE():
                monitor_cli.lightx2v_worker_request_count.inc()
            if self.resolve_input_kind(input_info) == "image":
                self.check_stop()
                return self.run_image_pipeline(input_info) if self.is_main_process else {}
            self.restorer.autoencoder.chunk_p_group = self.chunk_p_group
            return self.run_video_pipeline(input_info)
        finally:
            self.restorer.autoencoder.chunk_p_group = None
            self.restorer.reset()

    def run_image_pipeline(self, input_info):
        if not input_info.return_result_tensor and not input_info.save_result_path:
            raise ValueError("SwiftVR image restoration requires `save_result_path` unless the image is returned in memory.")

        frames, source_height, source_width = self.read_image_frame(input_info.image_path)
        output_height, output_width = self.resolve_output_size(input_info, source_height, source_width)
        clip_length = self.config.get("clip_len", 24)
        chunk = build_video_chunks(1, clip_length)[0]
        clip_latents = clip_length // 4

        output_path = None if input_info.return_result_tensor else os.path.abspath(input_info.save_result_path)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        started_at = time.perf_counter()
        video = self.preprocess_frames(frames, output_height, output_width)
        restored = self.restorer.restore_chunk(video, chunk, clip_latents)[..., :output_height, :output_width]
        del video
        images = restored[0].permute(0, 2, 3, 1).contiguous()
        if input_info.return_result_tensor:
            images = images.to(device="cpu", dtype=torch.float32)
        else:
            save_to_image(images, output_path)
            images = None

        elapsed = time.perf_counter() - started_at
        stats = {
            "frames": 1,
            "seconds": elapsed,
            "fps": 1 / elapsed if elapsed else 0.0,
            "output": output_path,
        }
        if self.progress_callback:
            self.progress_callback(100, 100)
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        logger.info(f"SwiftVR restored image to {output_path or 'memory'} in {elapsed:.3f}s")
        return {"images": images, "stats": stats}

    def run_video_pipeline(self, input_info):
        reader_executor = writer_executor = writer = None
        work_dir = None
        pending_reads = deque()
        pending_writes = deque()
        started_at = time.perf_counter()
        if not input_info.save_result_path:
            raise ValueError("SwiftVR video restoration requires `save_result_path`.")
        if input_info.return_result_tensor:
            raise ValueError("SwiftVR video restoration does not support `return_result_tensor`.")

        reader = VideoReader(input_info.video_path)
        raw_frame_count = len(reader)
        first_frame = reader[0]
        source_height = first_frame.shape[0] // 8 * 8
        source_width = first_frame.shape[1] // 8 * 8
        output_height, output_width = self.resolve_output_size(input_info, source_height, source_width, require_even=True)
        fps = self.config.get("fps") or reader.get_avg_fps() or 30
        clip_length = self.config.get("clip_len", 24)
        chunks = build_video_chunks(padded_frame_count(raw_frame_count), clip_length)
        clip_latents = clip_length // 4
        frames_to_trim = self.restorer.autoencoder.autoencoder.frames_to_trim
        rank = dist.get_rank(self.chunk_p_group) if self.chunk_p_group is not None else 0
        # Idle ranks in the final batch still participate in ReAE boundary exchanges.
        local_chunks = [chunks[min(start + rank, len(chunks) - 1)] for start in range(0, len(chunks), self.chunk_p_size)]
        output_path = os.path.abspath(input_info.save_result_path)

        try:
            try:
                if self.is_main_process:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    if self.chunk_p_group is not None:
                        work_dir = tempfile.mkdtemp(prefix="swiftvr-", dir=os.path.dirname(output_path))
                    else:
                        writer = self.open_video_writer(output_path, fps, output_height, output_width)
                reader_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swiftvr-reader")
                writer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swiftvr-writer")
                max_pending = self.config.get("queue_size", 3)
                output_batch_size = 4 if self.chunk_p_group is not None else clip_length + 4
                # Keep the write queue's frame budget proportional to the original chunk budget.
                max_pending_writes = max_pending * ((clip_length + output_batch_size - 1) // output_batch_size)

                def submit_read(chunk):
                    return reader_executor.submit(self.read_video_frames, reader, chunk, raw_frame_count, source_height, source_width, self.copy_stream is not None)

                for chunk in local_chunks[:max_pending]:
                    pending_reads.append(submit_read(chunk))
                if self.chunk_p_group is not None:
                    directories = [work_dir]
                    dist.broadcast_object_list(directories, src=0, group=self.chunk_p_group)
                    work_dir = directories[0]
                    if rank < len(chunks):
                        frame_counts = [len(local_chunk.output_range(raw_frame_count, frames_to_trim)) for local_chunk in chunks[rank :: self.chunk_p_size]]
                        writer = self.open_video_writer(os.path.join(work_dir, f"{rank:03d}-%08d.mp4"), fps, output_height, output_width, frame_counts)

                for local_index, chunk in enumerate(local_chunks):
                    self.check_stop()
                    active = local_index * self.chunk_p_size + rank < len(chunks)
                    frames = pending_reads.popleft().result()
                    next_read = local_index + max_pending
                    if next_read < len(local_chunks):
                        pending_reads.append(submit_read(local_chunks[next_read]))
                    video = self.preprocess_frames(frames, output_height, output_width)
                    del frames

                    model_input, offset = self.restorer.encode_chunk(video, chunk, clip_latents)
                    del video
                    latents = model_input - self.model.predict(model_input, self.restorer.transformer.condition, offset) if active else model_input
                    restored_frames = self.restorer.decode_chunk(latents, chunk, output_batch_size)
                    del model_input, latents
                    if not active:
                        del restored_frames
                        continue

                    remaining_frames = len(chunk.output_range(raw_frame_count, frames_to_trim))
                    for restored in restored_frames:
                        if len(pending_writes) >= max_pending_writes:
                            pending_writes.popleft().result()
                        restored = restored[:, :remaining_frames, :, :output_height, :output_width]
                        # Consume the decoded pixel buffer; ReAE already clamps it to [0, 1].
                        output_frames = restored[0].permute(0, 2, 3, 1).mul_(255).to(dtype=torch.uint8, memory_format=torch.contiguous_format)
                        cpu_frames, copy_complete = self.copy_frames_to_cpu(output_frames, self.copy_stream)
                        del output_frames, restored
                        pending_writes.append(writer_executor.submit(self.write_video_frames, writer, cpu_frames, copy_complete))
                        remaining_frames -= len(cpu_frames)
                        if remaining_frames == 0:
                            break
                    if self.is_main_process and self.progress_callback:
                        self.progress_callback(min(chunk.index + self.chunk_p_size, len(chunks)) / len(chunks) * 100, 100)
                    del restored_frames
                while pending_writes:
                    pending_writes.popleft().result()
            finally:
                if reader_executor is not None:
                    reader_executor.shutdown(wait=True, cancel_futures=True)
                if writer_executor is not None:
                    writer_executor.shutdown(wait=True)
                if writer is not None:
                    writer.close()

            if self.chunk_p_group is not None:
                dist.barrier(group=self.chunk_p_group)
            if not self.is_main_process:
                return {}
            if work_dir is not None:
                self.join_video_chunks(work_dir, len(chunks), output_path, input_info.video_path)
            else:
                mux_audio_from_video(input_info.video_path, output_path, prefer_copy=self.config.get("audio_mux_prefer_copy", True), trim_to_shortest=False)
            elapsed = time.perf_counter() - started_at
            stats = {
                "frames": raw_frame_count,
                "seconds": elapsed,
                "fps": raw_frame_count / elapsed if elapsed else 0.0,
                "output": output_path,
            }
            if GET_RECORDER_MODE():
                monitor_cli.lightx2v_worker_request_success.inc()
            logger.info(f"SwiftVR restored {raw_frame_count} frames to {output_path} at {stats['fps']:.2f} fps")
            return {"video": None, "stats": stats}
        finally:
            if self.is_main_process and work_dir is not None:
                shutil.rmtree(work_dir)
