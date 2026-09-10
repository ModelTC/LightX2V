import math
import threading
import time
import traceback
from typing import Callable, Optional

import torch
from loguru import logger

START_SPEAKING = 0
STOP_SPEAKING = 1


def encode_video_frames(images: torch.Tensor) -> bytes:
    out = bytearray()
    for i in range(images.shape[0]):
        frame = (images[i] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        out.extend(frame.tobytes())
    return bytes(out)


def encode_audio_pcm(audios: torch.Tensor) -> bytes:
    pcm = torch.clamp(torch.round(audios * 32767), -32768, 32767).to(torch.int16)
    return pcm[None].cpu().numpy().tobytes()


class WsVideoChunkRecorder:
    def __init__(
        self,
        fps: float = 16.0,
        sample_rate: int = 16000,
        slice_frame: int = 1,
        prev_frame: int = 1,
        on_output: Optional[Callable] = None,
        stream_config: dict = None,
        chunk_frames: int = 0,
        **_unused,
    ):
        self.fps = fps
        self.sample_rate = sample_rate
        self.slice_frame = slice_frame
        self.prev_frame = prev_frame
        self.on_output = on_output
        self.stream_config = stream_config or {}
        self.realtime = True
        self.width = None
        self.height = None
        self.stream_buffer = []
        self.send_buffer = []
        self.stream_buffer_lock = threading.Lock()
        self.stop_schedule = False
        self.schedule_thread = None
        self.audio_samples_per_frame = round(self.sample_rate / self.fps)
        self.sequence = 0
        self.stoppable_t = None
        self.speaking = False
        self.video_start_sent = False
        frames = int(chunk_frames or self.stream_config.get("chunk_frames") or 8)
        self.send_slices = max(1, math.ceil(frames / max(self.slice_frame, 1)))

    def start(self, width: int, height: int):
        self.set_video_size(width, height)

    def set_video_size(self, width: int, height: int):
        if self.width is not None and self.height is not None:
            assert self.width == width and self.height == height, "Video size already set"
            return
        self.width = width
        self.height = height
        self._emit(("video_start", width, height))
        self.video_start_sent = True
        self.schedule_thread = threading.Thread(target=self.schedule_stream_buffer, daemon=True)
        self.schedule_thread.start()

    def _emit(self, item):
        if self.on_output is None:
            return
        try:
            self.on_output(item)
        except Exception:
            logger.error(f"on_output failed: {traceback.format_exc()}")

    def buffer_stream(self, images: torch.Tensor, audios: torch.Tensor, gen_video: torch.Tensor, valid_duration=1e9):
        N, height, width, C = images.shape
        M = audios.reshape(-1).shape[0]
        assert N % self.slice_frame == 0, "Video frames must be divisible by slice_frame"
        assert C == 3, "Input must be [N, H, W, C] with C=3"
        audio_frames = round(M * self.fps / self.sample_rate)
        if audio_frames != N:
            logger.warning(f"Video and audio frames mismatch, {N} vs {audio_frames}")
        self.set_video_size(width, height)
        valid_frames = math.ceil(valid_duration * self.fps)
        rets = []
        for i in range(0, N, self.slice_frame):
            end_frame = i + self.slice_frame
            can_truncate = valid_frames < end_frame
            img = images[i:end_frame]
            aud = audios[i * self.audio_samples_per_frame : end_frame * self.audio_samples_per_frame]
            gen = gen_video[:, :, (end_frame - self.prev_frame) : end_frame]
            rets.append([img, aud, gen, can_truncate])
        with self.stream_buffer_lock:
            origin_size = len(self.stream_buffer)
            self.stream_buffer.extend(rets)
            logger.info(f"Buffered {origin_size} + {len(rets)} = {len(self.stream_buffer)} stream segments, valid_frames: {valid_frames}")

    def get_buffer_stream_size(self):
        return len(self.stream_buffer)

    def truncate_stream_buffer(self, size: int, check_can_truncate: bool = True):
        with self.stream_buffer_lock:
            idx = len(self.stream_buffer) - 1
            while check_can_truncate and idx >= size and idx >= 0:
                if not self.stream_buffer[idx][3]:
                    logger.warning(f"can not truncate frame: {idx}, truncate size: {size} -> {idx + 1}")
                    size = idx + 1
                    break
                idx -= 1
            self.stream_buffer = self.stream_buffer[:size]
            logger.info(f"Truncated stream buffer to {len(self.stream_buffer)} segments")
            if len(self.stream_buffer) > 0:
                self.stream_buffer[-1][3] = False
                return self.stream_buffer[-1][2]
            return None

    def _send_ready(self):
        if len(self.send_buffer) < self.send_slices:
            return
        batch = self.send_buffer
        self.send_buffer = []
        speaking = any(not item[3] for item in batch)
        if speaking and not self.speaking:
            self._emit(("event", START_SPEAKING, ""))
            self.speaking = True
            logger.info(f"START_SPEAKING seq={self.sequence}")
        elif not speaking and self.speaking:
            self._emit(("event", STOP_SPEAKING, ""))
            self.speaking = False
            logger.info(f"STOP_SPEAKING seq={self.sequence}")
        imgs = torch.cat([item[0] for item in batch], dim=0)
        auds = torch.cat([item[1].reshape(-1) for item in batch], dim=0)
        self._emit(("video", self.sequence, encode_audio_pcm(auds), encode_video_frames(imgs)))
        self.sequence += 1
        self.stoppable_t = time.time() + auds.numel() / self.sample_rate + 3

    def schedule_stream_buffer(self):
        interval = self.slice_frame / self.fps
        logger.info(f"Schedule stream buffer interval={interval:.3f}s send_slices={self.send_slices}")
        t = None
        while not self.stop_schedule:
            try:
                item = None
                with self.stream_buffer_lock:
                    if self.stream_buffer:
                        item = self.stream_buffer.pop(0)
                if t is not None:
                    time.sleep(max(0.0, interval - (time.time() - t)))
                t = time.time()
                if item is None:
                    continue
                self.send_buffer.append(item)
                self._send_ready()
            except Exception:
                logger.error(f"Schedule stream buffer error: {traceback.format_exc()}")
                break
        if self.speaking:
            self._emit(("event", STOP_SPEAKING, ""))
            self.speaking = False
        logger.info("WsVideoChunkRecorder schedule thread stopped")

    def pub_livestream(self, images: torch.Tensor, audios: torch.Tensor):
        self.buffer_stream(images, audios, images.permute(3, 0, 1, 2).unsqueeze(0)[:, :1], valid_duration=1e9)

    def stop(self, wait=True):
        if wait and self.stoppable_t:
            t = self.stoppable_t - time.time()
            if t > 0:
                time.sleep(min(t, 5))
        self.stop_schedule = True
        if self.schedule_thread is not None:
            self.schedule_thread.join(timeout=5)
        logger.info("WsVideoChunkRecorder stopped")
