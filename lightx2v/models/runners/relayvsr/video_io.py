import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import av
import torch


def prefetch_frames(container, batch_size, timing):
    def frames():
        for frame in container.decode(video=0):
            yield torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).contiguous().pin_memory()

    iterator = iter(frames())

    def read_batch():
        started = time.perf_counter()
        batch = list(islice(iterator, batch_size))
        timing["decode_seconds"] += time.perf_counter() - started
        return batch

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="relayvsr-decode") as executor:
        pending = executor.submit(read_batch)
        while True:
            batch = pending.result()
            if not batch:
                return
            pending = executor.submit(read_batch)
            yield from batch


class AsyncVideoWriter:
    def __init__(self, path, fps, config):
        self.path, self.fps, self.config = path, fps, config
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="relayvsr-encode")
        self.pending = deque()
        self.copy_stream = torch.cuda.Stream()
        self.container = self.stream = None
        self.frames = 0
        self.encode_seconds = self.flush_seconds = 0.0
        self.copy_seconds = self.queue_wait_seconds = 0.0
        self.queue_size = config.get("queue_size", 16)
        self.closed = False

    def _write(self, pixels, copy_start, event):
        event.synchronize()
        self.copy_seconds += copy_start.elapsed_time(event) / 1000
        start = time.perf_counter()
        if self.container is None:
            self.container = av.open(str(self.path), mode="w")
            codec = self.config.get("video_codec", "libx265")
            self.stream = self.container.add_stream(codec, rate=self.fps)
            self.stream.width, self.stream.height = pixels.shape[1], pixels.shape[0]
            self.stream.pix_fmt = "yuv420p"
            threads = self.config.get("encoder_threads", 16)
            options = {"crf": str(self.config.get("video_crf", 18)), "preset": self.config.get("ffmpeg_preset", "ultrafast")}
            if codec == "libx265":
                options["x265-params"] = f"log-level=warning:pools={threads}"
                if self.stream.width < 64:
                    # x265's SIMD path can corrupt memory on very narrow frames.
                    options["x265-params"] += ":asm=0"
                self.stream.codec_context.codec_tag = "hvc1"
            else:
                self.stream.codec_context.thread_count = threads
            self.stream.options = options
        # Share the pinned RGB buffer with PyAV.
        frame = av.VideoFrame.from_numpy_buffer(pixels.numpy(), format="rgb24")
        frame.pts = self.frames
        frame.time_base = 1 / self.fps
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1
        self.encode_seconds += time.perf_counter() - start

    def submit(self, pixels):
        if len(self.pending) >= self.queue_size:
            started = time.perf_counter()
            self.pending.popleft().result()
            self.queue_wait_seconds += time.perf_counter() - started
        host = torch.empty(pixels.shape, device="cpu", dtype=torch.uint8, pin_memory=True)
        self.copy_stream.wait_stream(torch.cuda.current_stream(pixels.device))
        with torch.cuda.stream(self.copy_stream):
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_start.record()
            host.copy_(pixels, non_blocking=True)
            pixels.record_stream(self.copy_stream)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        self.pending.append(self.executor.submit(self._write, host, copy_start, event))

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            for future in self.pending:
                future.result()
            start = time.perf_counter()
            if self.stream is not None:
                packets = self.stream.encode()
                if self.stream.codec_context.name == "libx265" and self.frames < 3:
                    # Match SwiftVR's DTS correction for very short HEVC videos.
                    decoder = av.CodecContext.create("hevc", "r")
                    decoder.thread_count = 1
                    decoder.extradata = self.stream.codec_context.extradata
                    decoder.open()
                    for packet in packets:
                        packet.dts = packet.pts - decoder.reorder_depth
                for packet in packets:
                    self.container.mux(packet)
            self.flush_seconds = time.perf_counter() - start
        finally:
            self.executor.shutdown(wait=True)
            if self.container is not None:
                self.container.close()


class DeviceTimings:
    """CUDA elapsed spans overlap CPU I/O; they must not be summed into wall time."""

    def __init__(self):
        self.events = []

    def call(self, name, function, *args):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args)
        end.record()
        self.events.append((name, start, end))
        return result

    def collect(self):
        torch.cuda.synchronize()
        totals = {}
        for name, start, end in self.events:
            totals[name] = totals.get(name, 0.0) + start.elapsed_time(end) / 1000
        self.events.clear()
        return totals
