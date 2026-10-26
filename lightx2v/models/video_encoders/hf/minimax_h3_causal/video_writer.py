import math
import queue
import threading
from concurrent.futures import CancelledError
from pathlib import Path
from uuid import uuid4

import torch

from lightx2v.models.networks.minimax_h3.packing import FRAMES_PER_CHUNK
from lightx2v.utils.ltx2_media_io import encode_video


class MiniMaxH3StreamingVideoWriter:
    """Encode CPU RGB as it arrives, reusing three chunk-sized host buffers."""

    def __init__(self, output_path, num_frames, fps, audio, *, pinned_memory=False, video_codec_options=None):
        self.output_path = output_path
        path = Path(output_path)
        self.temporary_path = path.with_name(f".{path.stem}.{uuid4().hex}.streaming.mp4")
        self.num_chunks = math.ceil(num_frames / FRAMES_PER_CHUNK)
        self.fps = fps
        self.audio = audio
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
                    # encode_video has copied these frames into the codec before
                    # advancing the iterator, so the producer may reuse this slot.
                    self.available.put(index)

        try:
            encode_video(
                chunks(),
                fps=self.fps,
                audio=self.audio,
                output_path=str(self.temporary_path),
                video_chunks_number=self.num_chunks,
                video_codec_options=self.video_codec_options,
                show_progress=False,
            )
        except CancelledError:
            pass
        except Exception as error:
            self.error = error

    def finish(self):
        if self.thread is None:
            return
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
        self.error = None
        self.temporary_path.unlink(missing_ok=True)
