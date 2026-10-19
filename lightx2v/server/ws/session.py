import asyncio
import json
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
from lightx2v.utils.va_reader_ws import WsAudioSource, parse_pcm_audio_format

ASPECT_RATIOS = {
    "480p": {
        "1:1": [480, 480],
        "16:9": [352, 640],
        "9:16": [640, 352],
    }
}
DEFAULT_RESOLUTION = "480p"
DEFAULT_ASPECT_RATIO = "16:9"


def _parse_ratio_shapes(resolution: str, ratios: dict) -> dict:
    if not isinstance(ratios, dict) or not ratios:
        raise ValueError(f"resolution {resolution!r} must map to a non-empty aspect_ratio object")
    parsed = {}
    for ratio, shape in ratios.items():
        if not isinstance(ratio, str):
            raise ValueError(f"aspect ratio key must be str, got {type(ratio).__name__}: {ratio}")
        if not (isinstance(shape, list) and len(shape) == 2):
            raise ValueError(f"{resolution}/{ratio} must map to [height, width], got {shape}")
        height, width = shape
        if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
            raise ValueError(f"{resolution}/{ratio} must map to positive integer [height, width], got {shape}")
        parsed[ratio] = [height, width]
    return parsed


def parse_aspect_ratios(value: str) -> dict:
    data = json.loads(value)
    if not isinstance(data, dict) or not data:
        raise ValueError("aspect_ratios must be a non-empty JSON object")
    parsed = {}
    for resolution, ratios in data.items():
        if not isinstance(resolution, str) or not resolution.strip():
            raise ValueError(f"resolution key must be a non-empty str, got {resolution!r}")
        parsed[resolution.strip()] = _parse_ratio_shapes(resolution.strip(), ratios)
    return parsed


DEFAULT_PROMPT = "The video features a person is saying something."
DEFAULT_NEGATIVE_PROMPT = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
OUTPUT_URL_PREFIXES = ("http://", "https://", "rtmp://")


def resolve_aspect_ratio(value: str, resolution: Optional[str] = None) -> tuple[str, list[int]]:
    res = (resolution or DEFAULT_RESOLUTION).strip()
    if res not in ASPECT_RATIOS:
        raise ValueError(f"invalid resolution {res!r}, valid: {list(ASPECT_RATIOS)}")
    ratios = ASPECT_RATIOS[res]
    ar = (value or DEFAULT_ASPECT_RATIO).strip()
    if ar not in ratios:
        raise ValueError(f"invalid aspect_ratio {ar!r} for resolution {res!r}, valid: {list(ratios)}")
    return ar, list(ratios[ar])


def parse_stream_config(config_json: str) -> dict:
    try:
        config = json.loads(config_json) if config_json.strip() else {}
    except json.JSONDecodeError as error:
        raise ValueError(f"Start.config_json is invalid JSON: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("Start.config_json must be a JSON object")
    output_url = config.get("output_url")
    if output_url is not None and (not isinstance(output_url, str) or not output_url.startswith(OUTPUT_URL_PREFIXES)):
        raise ValueError("config_json.output_url must start with http://, https://, or rtmp://")
    if "resolution" in config:
        resolution = config["resolution"]
        if not isinstance(resolution, str) or not resolution.strip():
            raise ValueError("config_json.resolution must be a non-empty string")
        resolution = resolution.strip()
        if resolution not in ASPECT_RATIOS:
            raise ValueError(f"invalid resolution {resolution!r}, valid: {list(ASPECT_RATIOS)}")
        config["resolution"] = resolution
    return config


def apply_run_input(input_info, *, prompt, negative_prompt, image_path, size, audio_source=None, on_output=None, chunk_frames=8, stream_config=None):
    stream_config = dict(stream_config or {})
    output_url = stream_config.get("output_url")
    audio_path = {"type": "ws", "source": audio_source} if audio_source else {"type": "ws"}
    if output_url:
        save_result_path = {"type": "stream", "data": output_url}
    elif on_output is not None:
        save_result_path = {
            "type": "ws",
            "on_output": on_output,
            "chunk_frames": int(chunk_frames or 8),
        }
    else:
        save_result_path = {"type": "ws"}
    # 设置 seed 会导致片段间有雪花点，因此暂不设置 seed
    input_info.update(
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "size": size,
            "image_path": image_path,
            "audio_path": audio_path,
            "save_result_path": save_result_path,
            "action_prompts": {"0": [""]},
            "stream_config": stream_config,
        }
    )


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
        self.stream_config = {}
        self.busy = False

    def _on_output(self, item):
        self.video_queue.put(item)

    def apply_start(self, start: Start) -> dict:
        parse_pcm_audio_format(start.audio_format or "pcm_s16le/16000/1")
        images = start.image_data or []
        if not images or not images[0]:
            raise ValueError("Start.image_data is required")

        self.stream_config = parse_stream_config(start.config_json)
        logger.info(f"Rank {self.rank} stream_config: {self.stream_config}")
        self.request_id = start.request_id or "ws-session"
        self.tmp_dir = tempfile.TemporaryDirectory(prefix=f"lightx2v_ws_{self.request_id}_")
        image_path = os.path.join(self.tmp_dir.name, f"ref{_image_suffix(start.image_fromat)}")
        with open(image_path, "wb") as f:
            f.write(images[0])

        prompt = start.prompt or DEFAULT_PROMPT
        negative_prompt = start.negative_prompt or DEFAULT_NEGATIVE_PROMPT
        seed = int(start.seed or 42)
        _, size = resolve_aspect_ratio(start.aspect_ratio, self.stream_config.get("resolution"))

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

        apply_run_input(
            self.input_info,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=image_path,
            size=size,
            audio_source=self.audio_source,
            on_output=self._on_output,
            chunk_frames=start.chunk_frames,
            stream_config=self.stream_config,
        )
        logger.info(f"Rank {self.rank} input_info: {self.input_info}")
        return {
            "op": "run",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            # "seed": seed,
            "image_path": image_path,
            "size": size,
            "stream_config": self.stream_config,
        }

    @property
    def sends_ws_output(self):
        return not self.stream_config.get("output_url")

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
            prompt = msg.set_prompt.prompt if msg.set_prompt else ""
            self.audio_source.on_action_switch(prompt)
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
