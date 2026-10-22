"""Collect or encode the RGB frames published by a StreamRender session."""

from pathlib import Path

import av
import torch


class StreamRenderVideoOutput:
    def __init__(self, path, return_tensor, size):
        self.path = Path(path) if path else None
        self.return_tensor = return_tensor
        self.size = size
        self.frames = 0
        self.tensors = []
        self.writer = None
        self.stream = None

    def append(self, video):
        if video is None:
            return
        self.frames += video.shape[2]
        if self.path is None and not self.return_tensor:
            return
        frames = video.permute(0, 2, 1, 3, 4).cpu()
        if self.return_tensor:
            self.tensors.append(frames)
        if self.path is not None:
            if self.writer is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                writer = av.open(str(self.path), mode="w")
                try:
                    stream = writer.add_stream("libx264", rate=24)
                    stream.width, stream.height = self.size[1], self.size[0]
                    stream.pix_fmt = "yuv420p"
                    stream.options = {"preset": "ultrafast", "crf": "18"}
                except BaseException:
                    writer.close()
                    raise
                self.writer = writer
                self.stream = stream
            pixels = frames[0].permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().numpy()
            for frame in pixels:
                for packet in self.stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    self.writer.mux(packet)

    def close(self):
        if self.writer is not None:
            writer, stream = self.writer, self.stream
            self.writer = None
            self.stream = None
            try:
                for packet in stream.encode():
                    writer.mux(packet)
            finally:
                writer.close()

    def tensor(self):
        return torch.cat(self.tensors, dim=1) if self.tensors else None
