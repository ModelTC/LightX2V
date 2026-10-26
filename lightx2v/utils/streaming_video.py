"""Bounded, incremental local video output for autoregressive requests."""

import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch


class StreamingVideoWriter:
    """Encode CPU RGB chunks while inference produces the next chunk."""

    def __init__(self, path, fps, *, queue_size=2):
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("Autoregressive streaming video output requires an .mp4 path")
        if fps <= 0 or queue_size < 1:
            raise ValueError("Streaming output requires positive FPS and queue size")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.path.with_name(f".{self.path.stem}.{uuid4().hex}.streaming.mp4")
        self.fps = fps
        self.frames = 0
        self._shape = None
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = None
        self._process = None
        self._error = None
        self._closed = False
        self._audio = None

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError("Streaming video encoding failed") from self._error

    def _put(self, item):
        while True:
            self._raise_error()
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def submit(self, frames):
        """Accept immutable uint8 [frames,height,width,RGB] CPU chunks."""
        if self._closed:
            raise RuntimeError("Streaming video writer is closed")
        if frames.device.type != "cpu" or frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("Streaming video frames must be CPU uint8 [F,H,W,3]")
        if not frames.shape[0]:
            return
        shape = tuple(frames.shape[1:])
        if self._shape is None:
            if shape[0] % 2 or shape[1] % 2:
                raise ValueError("H.264 yuv420p output requires even height and width")
            self._shape = shape
            self._thread = threading.Thread(target=self._encode, name="ar-video-writer", daemon=True)
            self._thread.start()
        elif self._shape != shape:
            raise ValueError("Video geometry changed during streaming output")
        self._put(frames.contiguous().clone())
        self.frames += frames.shape[0]

    def _encode(self):
        height, width, _ = self._shape
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            str(max(1, round(self.fps))),
            "-keyint_min",
            str(max(1, round(self.fps))),
            "-sc_threshold",
            "0",
            "-crf",
            "18",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration",
            "1000000",
            "-flush_packets",
            "1",
            str(self.temporary_path),
        ]
        process = None
        try:
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
                self._process = process
                try:
                    while True:
                        frames = self._queue.get()
                        if frames is None:
                            break
                        process.stdin.write(frames.numpy().tobytes())
                finally:
                    process.stdin.close()
                code = process.wait()
                if code:
                    errors.seek(0)
                    raise RuntimeError(f"FFmpeg exited with {code}: {errors.read().decode(errors='replace')[-4000:]}")
        except BaseException as error:
            self._error = error
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def set_audio(self, waveform, sample_rate):
        if sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive")
        waveform = waveform.detach().float().cpu()
        if waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform[0]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
            raise ValueError("Streaming audio must have shape [channels,samples], with one or two channels")
        self._audio = waveform.contiguous(), sample_rate

    def _mux_audio(self):
        waveform, sample_rate = self._audio
        # Trim audio only. Global -shortest/-t can discard reordered video
        # packets and lose the final frames of an otherwise complete stream.
        duration = self.frames / self.fps
        muxed_path = self.temporary_path.with_suffix(".mux.mp4")
        with tempfile.TemporaryDirectory(prefix="lightx2v-audio-", dir=self.path.parent) as temporary:
            audio_path = Path(temporary) / "audio.f32"
            waveform.T.contiguous().numpy().astype(np.float32, copy=False).tofile(audio_path)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(self.temporary_path),
                        "-f",
                        "f32le",
                        "-ar",
                        str(sample_rate),
                        "-ac",
                        str(waveform.shape[0]),
                        "-i",
                        str(audio_path),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-af",
                        f"apad,atrim=duration={duration},asetpts=PTS-STARTPTS",
                        "-movflags",
                        "+faststart",
                        str(muxed_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                muxed_path.replace(self.path)
            finally:
                muxed_path.unlink(missing_ok=True)
        self.temporary_path.unlink()

    def close(self):
        if self._closed:
            self._raise_error()
            return
        self._closed = True
        if self._thread is None:
            return
        self._put(None)
        self._thread.join()
        self._raise_error()
        if self._audio is None:
            self.temporary_path.replace(self.path)
        else:
            self._mux_audio()

    def abort(self):
        self._closed = True
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        # Discard queued output on cancellation rather than encoding the rest
        # of a failed request. Wake a worker that is waiting for another chunk.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put_nowait(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
            if self._thread.is_alive() and self._process is not None:
                self._process.kill()
                self._thread.join(timeout=5)
        self.temporary_path.unlink(missing_ok=True)


def video_chunk_to_uint8(pixels, *, layout="BCTHW", value_range=(-1, 1)):
    """Normalize one decoded chunk on CPU without expanding a full GPU clip."""
    pixels = pixels.detach().cpu()
    if layout == "BCTHW":
        if pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
            raise ValueError("BCTHW video output requires shape [1,3,F,H,W]")
        pixels = pixels[0].permute(1, 2, 3, 0)
    elif layout == "FCHW":
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("FCHW video output requires shape [F,3,H,W]")
        pixels = pixels.permute(0, 2, 3, 1)
    elif layout != "FHWC":
        raise ValueError(f"Unsupported decoded video layout: {layout}")
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError("Video output requires RGB channels")
    if pixels.dtype != torch.uint8:
        low, high = value_range
        if high <= low:
            raise ValueError("Video value range must be increasing")
        pixels = ((pixels.float() - low) * (255 / (high - low))).clamp_(0, 255).round_().to(torch.uint8)
    return pixels.contiguous()
