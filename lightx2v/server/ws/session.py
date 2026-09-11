import asyncio
import os
import queue
import tempfile
import threading
import traceback
from typing import Optional

import torch
from loguru import logger

from lightx2v.server.ws.protocol import (
    AudioChunk,
    ClientMessage,
    Start,
    event_message,
    video_message,
    video_start_message,
)
from lightx2v.utils.input_info import update_input_info_from_dict
from lightx2v.utils.va_reader_ws import WsAudioSource, parse_pcm_audio_format

ASPECT_RATIOS = {
    "1:1": [480, 480],
    "16:9": [360, 640],
    "9:16": [640, 360],
}
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_PROMPT = "The video features a person is saying something."
DEFAULT_NEGATIVE_PROMPT = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def resolve_aspect_ratio(value: str) -> tuple[str, list[int]]:
    ar = (value or DEFAULT_ASPECT_RATIO).strip()
    if ar not in ASPECT_RATIOS:
        raise ValueError(f"invalid aspect_ratio {ar!r}, valid: {list(ASPECT_RATIOS)}")
    return ar, list(ASPECT_RATIOS[ar])


class RunnerThread(threading.Thread):
    def __init__(self, loop, future, run_func, rank, *args, **kwargs):
        super().__init__(daemon=True)
        self.loop = loop
        self.future = future
        self.run_func = run_func
        self.args = args
        self.kwargs = kwargs
        self.rank = rank

    def run(self):
        status = False
        err = ""
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.rank if torch.cuda.device_count() > 1 else torch.cuda.current_device())
            with torch.inference_mode():
                self.run_func(*self.args, **self.kwargs)
            status = True
        except Exception:
            logger.error(f"Rank {self.rank} pipeline failed: {traceback.format_exc()}")
        finally:
            async def set_future_result():
                if not self.future.done():
                    self.future.set_result((status, err))
            asyncio.run_coroutine_threadsafe(set_future_result(), self.loop)


def launch_pipeline(runner, input_info, rank: int, loop: asyncio.AbstractEventLoop):
    if not hasattr(runner, "stop_signal"):
        runner.stop_signal = False
    if not hasattr(runner, "can_pause"):
        runner.can_pause = False
    if not hasattr(runner, "pause_signal"):
        runner.pause_signal = False
    runner.stop_signal = False
    future = loop.create_future()
    device_rank = rank if torch.cuda.is_available() else 0
    thread = RunnerThread(loop, future, runner.run_pipeline, device_rank, input_info)
    thread.start()
    return thread, future


async def join_pipeline(thread, future, rank: int, timeout: float = 5):
    if future is not None and not future.done():
        try:
            await future
        except Exception:
            logger.warning(f"Rank {rank} wait pipeline future failed: {traceback.format_exc()}")
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    if thread is not None and thread.is_alive():
        logger.error(f"Rank {rank} pipeline thread still running after join")
        return False
    return True


def _image_suffix(image_fromat: str) -> str:
    fmt = (image_fromat or "jpg").lower().lstrip(".")
    if fmt in ("jpg", "jpeg"):
        return ".jpg"
    if fmt == "png":
        return ".png"
    if fmt in ("webp", "bmp"):
        return f".{fmt}"
    return ".jpg"


class LiveSession:
    def __init__(self, runner, input_info, rank: int, loop: asyncio.AbstractEventLoop):
        self.runner = runner
        self.input_info = input_info
        self.rank = rank
        self.loop = loop
        self.audio_source: Optional[WsAudioSource] = None
        self.video_queue: queue.Queue = queue.Queue()
        self.thread: Optional[RunnerThread] = None
        self.future: Optional[asyncio.Future] = None
        self.tmp_dir = None
        self.request_id = ""
        self.busy = False

    def _on_output(self, item):
        self.video_queue.put(item)

    def apply_start(self, start: Start) -> dict:
        parse_pcm_audio_format(start.audio_format or "pcm_s16le/16000/1")
        images = start.image_data or []
        if not images or not images[0]:
            raise ValueError("Start.image_data is required")

        self.request_id = start.request_id or "ws-session"
        self.tmp_dir = tempfile.TemporaryDirectory(prefix=f"lightx2v_ws_{self.request_id}_")
        image_path = os.path.join(self.tmp_dir.name, f"ref{_image_suffix(start.image_fromat)}")
        with open(image_path, "wb") as f:
            f.write(images[0])

        prompt = start.prompt or DEFAULT_PROMPT
        negative_prompt = start.negative_prompt or DEFAULT_NEGATIVE_PROMPT
        seed = int(start.seed or 42)
        _, target_shape = resolve_aspect_ratio(start.aspect_ratio)

        seg_duration = 1.0
        try:
            fps = float(self.runner.config.get("target_fps", 16))
            latent_per_chunk = int(self.runner.config.get("ar_config", {}).get("num_frame_per_chunk", 3))
            look_ahead = float(self.runner.config.get("look_ahead", 0.0))
            seg_duration = ((latent_per_chunk - 1) * 4 + 1) / fps + look_ahead
        except Exception:
            pass

        self.audio_source = WsAudioSource(
            seg_duration=seg_duration,
            sample_rate=int(self.runner.config.get("audio_sr", 16000)),
            audio_format=start.audio_format or "pcm_s16le/16000/1",
            model_runner=self.runner,
        )

        # 设置 seed 会导致片段间有雪花点，因此暂不设置 seed
        update_input_info_from_dict(
            self.input_info,
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                # "seed": seed,
                "target_shape": target_shape,
                "image_path": image_path,
                "audio_path": {"type": "ws", "source": self.audio_source},
                "save_result_path": {
                    "type": "ws",
                    "on_output": self._on_output,
                    "chunk_frames": int(start.chunk_frames or 8),
                },
            },
        )
        logger.info(f"Rank {self.rank} input_info: {self.input_info}")
        return {
            "op": "run",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            # "seed": seed,
            "image_path": image_path,
            "target_shape": target_shape,
        }

    def start_pipeline(self):
        self.busy = True
        self.thread, self.future = launch_pipeline(self.runner, self.input_info, self.rank, self.loop)

    def handle_client_body(self, msg: ClientMessage):
        which = msg.which()
        if which == "audio_start":
            self.audio_source.on_audio_start()
        elif which == "audio":
            chunk: AudioChunk = msg.audio
            self.audio_source.on_audio_chunk(chunk.data, chunk.sequence)
        elif which == "audio_end":
            self.audio_source.on_audio_end()
        elif which == "interrupt":
            reason = msg.interrupt.reason if msg.interrupt else ""
            self.audio_source.on_clear_audio(reason)
        elif which == "set_prompt":
            logger.info("SetPrompt ignored")
        else:
            raise ValueError(f"unexpected client message {which}")

    def request_stop(self):
        self.runner.stop_signal = True

    async def wait_started(self, timeout: float = 120.0):
        if self.audio_source is None:
            raise RuntimeError("session not started")
        ok = await asyncio.to_thread(self.audio_source.started_event.wait, timeout)
        if not ok:
            raise TimeoutError("reader did not start in time")

    async def pump_video(self, websocket):
        while True:
            item = await asyncio.to_thread(self.video_queue.get)
            kind = item[0]
            if kind == "video_start":
                _, width, height = item
                await websocket.send_bytes(video_start_message(width, height))
            elif kind == "video":
                _, seq, audio_data, video_data = item
                await websocket.send_bytes(video_message(seq, audio_data, video_data))
            elif kind == "event":
                _, event_type, message = item
                await websocket.send_bytes(event_message(event_type, message))
            elif kind == "done":
                break

    def is_busy(self) -> bool:
        return self.busy or (self.thread is not None and self.thread.is_alive())

    def mark_pipeline_done(self, status: bool, err: str):
        self.video_queue.put(("done", status, err))

    async def stop_and_join(self):
        self.request_stop()
        if not await join_pipeline(self.thread, self.future, self.rank):
            return False
        self.busy = False
        logger.info("pipeline stopped, session no longer busy")
        return True

    def close(self):
        if self.tmp_dir is not None:
            try:
                self.tmp_dir.cleanup()
            except Exception:
                pass
            self.tmp_dir = None
        self.audio_source = None
