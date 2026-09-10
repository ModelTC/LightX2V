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

    def apply_start(self, start: Start) -> str:
        parse_pcm_audio_format(start.audio_format or "pcm_s16le/16000/1")
        images = start.image_data or []
        if not images or not images[0]:
            raise ValueError("Start.image_data is required")

        self.request_id = start.request_id or "ws-session"
        self.tmp_dir = tempfile.TemporaryDirectory(prefix=f"lightx2v_ws_{self.request_id}_")
        image_path = os.path.join(self.tmp_dir.name, f"ref{_image_suffix(start.image_fromat)}")
        with open(image_path, "wb") as f:
            f.write(images[0])

        prompt = start.prompt or getattr(self.input_info, "prompt", "") or ""
        negative_prompt = start.negative_prompt or getattr(self.input_info, "negative_prompt", "") or ""
        seed = int(start.seed or getattr(self.input_info, "seed", 42) or 42)
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

        update_input_info_from_dict(
            self.input_info,
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
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
        return image_path

    def start_pipeline(self):
        if not hasattr(self.runner, "stop_signal"):
            self.runner.stop_signal = False
        if not hasattr(self.runner, "can_pause"):
            self.runner.can_pause = False
        if not hasattr(self.runner, "pause_signal"):
            self.runner.pause_signal = False
        self.runner.stop_signal = False
        self.busy = True
        self.future = self.loop.create_future()
        device_rank = self.rank if torch.cuda.is_available() else 0
        self.thread = RunnerThread(self.loop, self.future, self.runner.run_pipeline, device_rank, self.input_info)
        self.thread.start()

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
        if self.future is not None and not self.future.done():
            try:
                await self.future
            except Exception:
                logger.warning(f"wait pipeline future failed: {traceback.format_exc()}")
        if self.thread is not None and self.thread.is_alive():
            await asyncio.to_thread(self.thread.join)
        if self.thread is not None and self.thread.is_alive():
            logger.error("pipeline thread still running after Interrupt")
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
