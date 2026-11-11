import queue
import threading
from concurrent.futures import CancelledError
from pathlib import Path
from uuid import uuid4

import av
import torch

from lightx2v.models.networks.minimax_h3.packing import FRAMES_PER_CHUNK
from lightx2v.utils.ltx2_media_io import _prepare_audio_stream, _write_audio


class MiniMaxH3StreamingVideoWriter:
    """Encode CPU RGB as it arrives, reusing three chunk-sized host buffers."""

    def __init__(self, output_path, num_frames, fps, *, audio_sampling_rate, pinned_memory=False, video_codec_options=None):
        self.output_path = output_path
        path = Path(output_path)
        self.temporary_path = path.with_name(f".{path.stem}.{uuid4().hex}.streaming.mp4")
        self.num_frames = num_frames
        self.fps = fps
        self.audio = None
        self.audio_sampling_rate = audio_sampling_rate
        self.pinned_memory = pinned_memory
        self.video_codec_options = video_codec_options
        self.buffers = []
        self.available = queue.SimpleQueue()
        # The three available slots bound RGB in flight, including the encoder's
        # current chunk. A control message can always wake the consumer.
        self.pending = queue.SimpleQueue()
        self.cancelled = threading.Event()
        self.thread = None
        self.error = None
        self.on_rgb_ready = None

    def write(self, rgb):
        # Report encoder errors at finish(), once all ranks leave DiT collectives.
        if self.error is not None:
            return
        if self.thread is None:
            b, c, _, h, w = rgb.shape
            self.buffers = [torch.empty((b, c, FRAMES_PER_CHUNK, h, w), dtype=torch.uint8, memory_format=torch.channels_last_3d, pin_memory=self.pinned_memory) for _ in range(3)]
            for index in range(len(self.buffers)):
                self.available.put(index)
            self.thread = threading.Thread(target=self.consume, name="h3-video-encoder")
            self.thread.start()
        while True:
            try:
                index = self.available.get(timeout=0.1)
                break
            except queue.Empty:
                if self.error is not None:
                    return
        frames = rgb.shape[2]
        output = self.buffers[index][:, :, :frames]
        output.copy_(rgb, non_blocking=self.pinned_memory)
        if rgb.is_cuda:
            torch.cuda.current_stream(rgb.device).synchronize()
        if self.on_rgb_ready is not None:
            # The callback borrows this slot synchronously. A consumer retaining
            # RGB after returning must copy it before the encoder reuses the slot.
            try:
                self.on_rgb_ready(output)
            except Exception as error:
                self.error = error
                self.available.put(index)
                return
        self.pending.put((index, frames))

    def consume(self):
        def chunks():
            while True:
                item = self.pending.get()
                if self.cancelled.is_set():
                    raise CancelledError
                if item is None:
                    return
                index, frames = item
                try:
                    yield self.buffers[index][0, :, :frames].permute(1, 2, 3, 0)
                finally:
                    # PyAV has copied these frames into the codec before
                    # advancing the iterator, so the producer may reuse this slot.
                    self.available.put(index)

        try:
            self.temporary_path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(self.temporary_path), mode="w") as container:
                stream = container.add_stream("libx264", rate=int(self.fps))
                stream.height, stream.width = self.buffers[0].shape[-2:]
                stream.pix_fmt = "yuv420p"
                if self.video_codec_options:
                    stream.options = dict(self.video_codec_options)
                # MP4 streams must exist before the first video packet is muxed.
                # Both tasks supply their waveform in finish(), before EOS.
                audio_stream = _prepare_audio_stream(container, self.audio_sampling_rate)
                encoded_frames = 0
                for chunk in chunks():
                    for frame_array in chunk.numpy():
                        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                        encoded_frames += 1
                if encoded_frames != self.num_frames:
                    raise RuntimeError(f"Expected {self.num_frames} video frames, received {encoded_frames}")
                for packet in stream.encode():
                    container.mux(packet)
                if self.audio.sampling_rate != self.audio_sampling_rate:
                    raise ValueError("Streaming AV output requires audio at the declared sampling rate")
                _write_audio(container, audio_stream, self.audio)
        except CancelledError:
            pass
        except Exception as error:
            if self.error is None:
                self.error = error

    def finish(self, audio):
        self.audio = audio
        self.pending.put(None)
        self.thread.join()
        if self.error is not None:
            raise self.error
        # Publish only a complete file; failed requests leave existing output intact.
        self.temporary_path.replace(self.output_path)

    def close(self):
        """Wake and join the consumer even when inference fails or is cancelled."""
        self.cancelled.set()
        if self.thread is not None:
            self.pending.put(None)
            self.thread.join()
        self.buffers.clear()
        self.audio = None
        self.on_rgb_ready = None
        self.error = None
        self.temporary_path.unlink(missing_ok=True)
