"""Bounded, incremental local video output for autoregressive requests."""

import math
import queue
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

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
        self._frame_size = None
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = None
        self._process = None
        self._error = None
        self._closed = False
        self._audio = None
        self._expected_frames = None

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

    def start(self, height, width):
        """Start the encoder for frames of the given height and width."""
        if self._closed:
            raise RuntimeError("Streaming video writer is closed")
        frame_size = (height, width)
        if self._frame_size == frame_size:
            return
        if self._frame_size is not None:
            raise ValueError("Video geometry changed during streaming output")
        if height % 2 or width % 2:
            raise ValueError("H.264 yuv420p output requires even height and width")
        self._frame_size = frame_size
        self._thread = threading.Thread(target=self._encode, name="ar-video-writer", daemon=True)
        self._thread.start()

    def submit(self, frames):
        """Queue an owned copy of a uint8 [frames,height,width,RGB] CPU chunk."""
        if self._closed:
            raise RuntimeError("Streaming video writer is closed")
        if frames.device.type != "cpu" or frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("Streaming video frames must be CPU uint8 [F,H,W,3]")
        if not frames.shape[0]:
            return
        self.start(*frames.shape[1:3])
        self._put(frames.clone(memory_format=torch.contiguous_format))
        self.frames += frames.shape[0]

    def _encode(self):
        height, width = self._frame_size
        video_input = [
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
        ]
        video_codec = [
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
        ]
        process = None
        try:
            with self._mux_options(self._expected_frames) as (audio_input, output), tempfile.TemporaryFile() as errors:
                command = ["ffmpeg", "-nostdin", "-v", "error", "-y", *video_input, *audio_input, *video_codec, *output, str(self.temporary_path)]
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

    def set_audio(self, waveform, sample_rate, *, num_frames=None):
        """Supply the final frame count before starting the encoder to mux audio in one pass."""
        if self._thread is not None and (self._expected_frames is not None or num_frames is not None):
            raise RuntimeError("Single-pass audio must be set before the encoder starts")
        if sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive")
        waveform = waveform.detach()
        if waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform[0]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
            raise ValueError("Streaming audio must have shape [channels,samples], with one or two channels")
        if num_frames is not None:
            waveform = waveform[:, : math.ceil(num_frames * sample_rate / self.fps)]
        # Copy a trimmed soundtrack so the writer can release the full recording.
        samples = waveform.T.to(device="cpu", dtype=torch.float32, memory_format=torch.contiguous_format, copy=num_frames is not None)
        self._audio = samples, sample_rate
        self._expected_frames = num_frames

    @contextmanager
    def _mux_options(self, num_frames):
        if num_frames is None:
            yield [], ["-an", "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-frag_duration", "1000000", "-flush_packets", "1"]
            return

        samples, sample_rate = self._audio
        with tempfile.TemporaryDirectory(prefix="lightx2v-audio-", dir=self.path.parent) as temporary:
            audio_path = Path(temporary) / "audio.f32"
            samples.numpy().tofile(audio_path)
            audio_input = ["-f", "f32le", "-ar", str(sample_rate), "-ac", str(samples.shape[1]), "-i", str(audio_path)]
            # Trim only audio so the final video packets are always retained.
            audio_filter = f"apad,atrim=duration={num_frames / self.fps},asetpts=PTS-STARTPTS"
            yield audio_input, ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-af", audio_filter, "-movflags", "+faststart"]

    def close(self):
        if self._closed:
            self._raise_error()
            return
        self._closed = True
        if self._thread is not None:
            self._put(None)
            self._thread.join()
            self._raise_error()
        if self._expected_frames is not None and self.frames != self._expected_frames:
            self.temporary_path.unlink(missing_ok=True)
            self._error = ValueError(f"Expected {self._expected_frames} video frames, received {self.frames}")
            raise self._error
        if self._thread is None:
            return
        if self._audio is not None and self._expected_frames is None:
            # Some runners produce the soundtrack only after video encoding.
            muxed_path = self.temporary_path.with_suffix(".mux.mp4")
            try:
                with self._mux_options(self.frames) as (audio_input, output):
                    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(self.temporary_path), *audio_input, "-c:v", "copy", *output, str(muxed_path)]
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                muxed_path.replace(self.temporary_path)
            finally:
                muxed_path.unlink(missing_ok=True)
        self.temporary_path.replace(self.path)

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


def video_chunk_to_uint8(pixels, *, layout="BCTHW", value_range=(-1, 1), device="cpu"):
    """Normalize a decoded chunk on the given device and return contiguous CPU RGB frames."""
    pixels = pixels.detach().to(device)
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
    return pixels.contiguous().cpu()
