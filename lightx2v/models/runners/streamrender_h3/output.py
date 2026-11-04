"""Count published frames and optionally retain StreamRender's tensor result."""

import torch


class StreamRenderVideoOutput:
    def __init__(self, return_tensor):
        self.return_tensor = return_tensor
        self.frames = 0
        self.tensors = []

    def append(self, video):
        if video is None:
            return
        self.frames += video.shape[2]
        if self.return_tensor:
            self.tensors.append(video.permute(0, 2, 1, 3, 4).cpu())

    def tensor(self):
        return torch.cat(self.tensors, dim=1) if self.tensors else None
