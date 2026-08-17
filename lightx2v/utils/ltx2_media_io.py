import logging
import math
import os
import queue
import tempfile
import threading
from collections.abc import Generator, Iterator, Mapping
from fractions import Fraction
from io import BytesIO

import av
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torch._prims_common import DeviceLikeType
from tqdm import tqdm

from lightx2v.models.video_encoders.hf.ltx2.audio_vae.ops import Audio

DEFAULT_IMAGE_CRF = 33

logger = logging.getLogger(__name__)


def resize_aspect_ratio_preserving(image: torch.Tensor, long_side: int) -> torch.Tensor:
    """
    Resize image preserving aspect ratio (filling target long side).
    Preserves the input dimensions order.
    Args:
        image: Input image tensor with shape (F (optional), H, W, C)
        long_side: Target long side size.
    Returns:
        Tensor with shape (F (optional), H, W, C) F = 1 if input is 3D, otherwise input shape[0]
    """
    height, width = image.shape[-3:2]
    max_side = max(height, width)
    scale = long_side / float(max_side)
    target_height = int(height * scale)
    target_width = int(width * scale)
    resized = resize_and_center_crop(image, target_height, target_width)
    # rearrange and remove batch dimension
    result = rearrange(resized, "b c f h w -> b f h w c")[0]
    # preserve input dimensions
    return result[0] if result.shape[0] == 1 else result


def resize_and_center_crop(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Resize tensor preserving aspect ratio (filling target), then center crop to exact dimensions.
    Args:
        latent: Input tensor with shape (H, W, C) or (F, H, W, C)
        height: Target height
        width: Target width
    Returns:
        Tensor with shape (1, C, 1, height, width) for 3D input or (1, C, F, height, width) for 4D input
    """
    if tensor.ndim == 3:
        tensor = rearrange(tensor, "h w c -> 1 c h w")
    elif tensor.ndim == 4:
        tensor = rearrange(tensor, "f h w c -> f c h w")
    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")

    _, _, src_h, src_w = tensor.shape

    scale = max(height / src_h, width / src_w)
    # Use ceil to avoid floating-point rounding causing new_h/new_w to be
    # slightly smaller than target, which would result in negative crop offsets.
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)

    tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]

    tensor = rearrange(tensor, "f c h w -> 1 c f h w")
    return tensor


def normalize_latent(latent: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (latent / 127.5 - 1.0).to(device=device, dtype=dtype)


def load_image_conditioning(
    image_path: str,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    crf: int = DEFAULT_IMAGE_CRF,
) -> torch.Tensor:
    """
    Loads an image from a path and preprocesses it for conditioning.
    Note: The image is resized to the nearest multiple of 2 for compatibility with video codecs.
    """
    image = decode_image(image_path=image_path)
    image = preprocess(image=image, crf=crf)
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image = resize_and_center_crop(image, height, width)
    image = normalize_latent(image, device, dtype)
    return image


def load_video_conditioning(video_path: str, height: int, width: int, frame_cap: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Loads a video from a path and preprocesses it for conditioning.
    Note: The video is resized to the nearest multiple of 2 for compatibility with video codecs.
    """
    frames = decode_video_from_file(path=video_path, frame_cap=frame_cap, device=device)
    result = None
    for f in frames:
        frame = resize_and_center_crop(f.to(torch.float32), height, width)
        frame = normalize_latent(frame, device, dtype)
        result = frame if result is None else torch.cat([result, frame], dim=2)
    return result


def decode_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def _write_audio(container: av.container.Container, audio_stream: av.audio.AudioStream, audio: Audio) -> None:
    samples = audio.waveform
    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio.sampling_rate

    _resample_audio(container, audio_stream, frame_in)


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame) -> None:
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def encode_video(
    video: torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    audio: Audio | None,
    output_path: str,
    video_chunks_number: int,
    video_codec_options: Mapping[str, str] | None = None,
) -> None:
    if isinstance(video, torch.Tensor):
        video = iter([video])

    first_chunk = next(video)

    _, height, width, _ = first_chunk.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    if video_codec_options:
        stream.options = dict(video_codec_options)

    if audio is not None:
        audio_stream = _prepare_audio_stream(container, audio.sampling_rate)

    def all_tiles(first_chunk: torch.Tensor, tiles_generator: Generator[tuple[torch.Tensor, int], None, None]) -> Generator[tuple[torch.Tensor, int], None, None]:
        yield first_chunk
        yield from tiles_generator

    for video_chunk in tqdm(all_tiles(first_chunk, video), total=video_chunks_number):
        video_chunk_cpu = video_chunk.to("cpu").numpy()
        for frame_array in video_chunk_cpu:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio)

    container.close()
    logger.info(f"Video saved to {output_path}")


class AsyncAudioVideoWriter:
    """Encode CPU RGB chunks on a background thread and atomically publish an MP4.

    The worker thread exclusively owns the PyAV container.  Producers submit
    bounded video chunks while the accelerator decodes the following chunk,
    then submit the final audio waveform.  A bounded queue provides
    backpressure without retaining the complete decoded video in host memory.
    """

    _FINISH = object()

    def __init__(
        self,
        *,
        output_path: str,
        fps: int,
        audio_sample_rate: int | None,
        queue_size: int = 2,
        video_codec_options: Mapping[str, str] | None = None,
    ) -> None:
        if queue_size <= 0:
            raise ValueError(f"queue_size must be positive, got {queue_size}")
        self.output_path = os.fspath(output_path)
        self.fps = int(fps)
        self.audio_sample_rate = int(audio_sample_rate) if audio_sample_rate is not None else None
        self.video_codec_options = dict(video_codec_options) if video_codec_options else None
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._closed = False

        output_dir = os.path.dirname(os.path.abspath(self.output_path))
        os.makedirs(output_dir, exist_ok=True)
        fd, self._temporary_path = tempfile.mkstemp(prefix=f".{os.path.basename(self.output_path)}.", suffix=".tmp.mp4", dir=output_dir)
        os.close(fd)

        self._thread = threading.Thread(target=self._worker, name="lightx2v-av-writer")
        self._thread.start()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Asynchronous audio-video encoding failed for {self.output_path}") from self._error

    def _put(self, item: object) -> None:
        if self._closed:
            raise RuntimeError("Cannot submit data to a closed audio-video writer")
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if self._done.is_set():
                    self._raise_if_failed()
                    raise RuntimeError("Audio-video writer stopped before accepting all data")

    def submit_video(self, frames: torch.Tensor) -> None:
        """Submit one CPU ``[F,H,W,3]`` RGB uint8 chunk."""

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"video chunk must be [F,H,W,3], got {tuple(frames.shape)}")
        if frames.shape[0] <= 0:
            return
        if frames.dtype != torch.uint8:
            raise TypeError(f"video chunk must use torch.uint8, got {frames.dtype}")
        if frames.device.type != "cpu":
            raise ValueError(f"video chunk must be on CPU, got {frames.device}")
        self._put(("video", frames.contiguous()))

    def submit_audio(self, audio: Audio) -> None:
        """Submit the final audio waveform after the video producer is done."""

        if self.audio_sample_rate is None:
            raise RuntimeError("This writer was created without an audio stream")
        if int(audio.sampling_rate) != self.audio_sample_rate:
            raise ValueError(f"audio sampling rate mismatch: writer={self.audio_sample_rate}, payload={audio.sampling_rate}")
        self._put(("audio", audio))

    def finish(self) -> None:
        """Drain queued chunks, finalize the MP4, and atomically publish it."""

        if self._closed:
            self._raise_if_failed()
            return
        try:
            self._put(self._FINISH)
        except BaseException:
            self.abort()
            raise
        self._thread.join()
        self._closed = True
        self._raise_if_failed()

    def abort(self) -> None:
        """Stop after the current encode operation and remove the partial file."""

        if self._closed:
            return
        self._cancelled.set()
        while not self._done.is_set():
            try:
                self._queue.put(self._FINISH, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        self._closed = True

    def _worker(self) -> None:
        container = None
        video_stream = None
        audio_stream = None
        audio_payload = None
        success = False
        try:
            while True:
                item = self._queue.get()
                if item is self._FINISH:
                    break
                if self._cancelled.is_set():
                    continue

                kind, payload = item
                if kind == "video":
                    frames = payload
                    if container is None:
                        _, height, width, _ = frames.shape
                        container = av.open(self._temporary_path, mode="w")
                        video_stream = container.add_stream("libx264", rate=self.fps)
                        video_stream.width = width
                        video_stream.height = height
                        video_stream.pix_fmt = "yuv420p"
                        if self.video_codec_options:
                            video_stream.options = dict(self.video_codec_options)
                        if self.audio_sample_rate is not None:
                            audio_stream = _prepare_audio_stream(container, self.audio_sample_rate)
                    elif frames.shape[1] != video_stream.height or frames.shape[2] != video_stream.width:
                        raise ValueError(
                            f"video chunk size changed from {video_stream.height}x{video_stream.width} to {frames.shape[1]}x{frames.shape[2]}"
                        )

                    for frame_array in frames.numpy():
                        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                        for packet in video_stream.encode(frame):
                            container.mux(packet)
                elif kind == "audio":
                    if audio_payload is not None:
                        raise RuntimeError("Audio was submitted more than once")
                    audio_payload = payload
                else:
                    raise RuntimeError(f"Unknown audio-video writer message: {kind!r}")

            if self._cancelled.is_set():
                return
            if container is None or video_stream is None:
                raise RuntimeError("No video frames were submitted")
            for packet in video_stream.encode():
                container.mux(packet)
            if self.audio_sample_rate is not None:
                if audio_payload is None or audio_stream is None:
                    raise RuntimeError("No audio waveform was submitted")
                _write_audio(container, audio_stream, audio_payload)
            container.close()
            container = None
            os.replace(self._temporary_path, self.output_path)
            success = True
            logger.info(f"Video saved to {self.output_path}")
        except BaseException as exc:
            self._error = exc
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            if not success and os.path.exists(self._temporary_path):
                try:
                    os.remove(self._temporary_path)
                except OSError:
                    pass
            self._done.set()


_INT_FORMAT_MAX: dict[str, float] = {
    "u8": 128.0,
    "u8p": 128.0,
    "s16": 32768.0,
    "s16p": 32768.0,
    "s32": 2147483648.0,
    "s32p": 2147483648.0,
}


def _audio_frame_to_float(frame: av.AudioFrame) -> np.ndarray:
    """Convert an audio frame to a float32 ndarray with values in [-1, 1] and shape (channels, samples)."""
    fmt = frame.format.name
    arr = frame.to_ndarray().astype(np.float32)
    if fmt in _INT_FORMAT_MAX:
        arr = arr / _INT_FORMAT_MAX[fmt]
    if not frame.format.is_planar:
        # Interleaved formats have shape (1, samples * channels) — reshape to (channels, samples).
        channels = len(frame.layout.channels)
        arr = arr.reshape(-1, channels).T
    return arr


def get_videostream_metadata(path: str) -> tuple[float, int, int, int]:
    """Read video stream metadata: (fps, num_frames, width, height).
    If frame count is missing in the container, decodes the stream to count frames.
    """
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        fps = float(video_stream.average_rate)
        num_frames = video_stream.frames or 0
        if num_frames == 0:
            num_frames = sum(1 for _ in container.decode(video_stream))
        width = video_stream.codec_context.width
        height = video_stream.codec_context.height
        return fps, num_frames, width, height
    finally:
        container.close()


def decode_audio_from_file(path: str, device: torch.device, start_time: float = 0.0, max_duration: float | None = None) -> Audio | None:
    """Decodes audio from a file, optionally seeking to a start time and limiting duration.
    Args:
        path: Path to the audio/video file containing an audio stream.
        device: Device to place the resulting tensor on.
        start_time: Start time in seconds to begin reading audio from.
        max_duration: Maximum audio duration in seconds. If None, reads to end of stream.
    Returns:
        An Audio object with waveform of shape (1, channels, samples), or None if no audio stream.
    """
    container = av.open(path)
    try:
        audio_stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration:
        container.close()
        return None

    sample_rate = audio_stream.rate
    start_pts = int(start_time / audio_stream.time_base)
    end_time = start_time + max_duration if max_duration else audio_stream.duration * audio_stream.time_base
    container.seek(start_pts, stream=audio_stream)

    samples = []
    first_frame_time = None
    for frame in container.decode(audio=0):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts * audio_stream.time_base)
        frame_end = frame_time + frame.samples / frame.sample_rate
        if frame_end < start_time:
            continue
        if frame_time > end_time:
            break
        if first_frame_time is None:
            first_frame_time = frame_time
        samples.append(_audio_frame_to_float(frame))

    container.close()

    if not samples:
        return None

    audio = np.concatenate(samples, axis=-1)

    # Trim samples that fall outside the requested [start_time, start_time + max_duration] window.
    # Audio codecs decode in fixed-size frames whose boundaries may not align with the requested
    # time range, so the first frame can start before start_time and the last frame can end after
    # start_time + max_duration.
    skip_samples = round((start_time - first_frame_time) * sample_rate)
    if skip_samples > 0:
        audio = audio[..., skip_samples:]

    if max_duration is not None:
        max_samples = round(max_duration * sample_rate)
        audio = audio[..., :max_samples]

    waveform = torch.from_numpy(audio).to(device).unsqueeze(0)

    return Audio(waveform=waveform, sampling_rate=sample_rate)


def decode_video_from_file(path: str, frame_cap: int, device: DeviceLikeType) -> Generator[torch.Tensor]:
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        for frame in container.decode(video_stream):
            tensor = torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.uint8, device=device).unsqueeze(0)
            yield tensor
            frame_cap = frame_cap - 1
            if frame_cap == 0:
                break
    finally:
        container.close()


def encode_single_frame(output_file: str, image_array: np.ndarray, crf: float) -> None:
    if image_array.ndim == 2:
        image_array = np.stack([image_array, image_array, image_array], axis=-1)
    elif image_array.ndim != 3 or image_array.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image; got shape {image_array.shape}.")

    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
        # Round to nearest multiple of 2 for compatibility with video codecs
        height = image_array.shape[0] // 2 * 2
        width = image_array.shape[1] // 2 * 2
        image_array = image_array[:height, :width]
        stream.height = height
        stream.width = width
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file: str) -> np.array:
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def preprocess(image: np.array, crf: float = DEFAULT_IMAGE_CRF) -> np.array:
    if crf == 0:
        return image

    with BytesIO() as output_file:
        encode_single_frame(output_file, image, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    return image_array
