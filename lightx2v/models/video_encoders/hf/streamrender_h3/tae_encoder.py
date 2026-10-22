# SPDX-License-Identifier: MIT
# Adapted from the released StreamRender-H3 TAE-H3 encoder (Ollin Boer Bohan).
"""Native causal TAE-H3 encoder for StreamRender's RGB reference stream."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _conv(inputs: int, outputs: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(inputs, outputs, 3, padding=1, **kwargs)


class _MemoryBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(channels * 2, channels),
            nn.ReLU(inplace=True),
            _conv(channels, channels),
            nn.ReLU(inplace=True),
            _conv(channels, channels),
        )
        self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat((current, previous), dim=1)) + self.skip(current))


class _TemporalPool(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels * stride, channels, 1, bias=False)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = frames.shape
        return self.conv(frames.reshape(-1, self.stride * channels, height, width))


@dataclass
class _WorkItem:
    tensor: torch.Tensor
    block_index: int


def _run_encoder(model: nn.Sequential, memory: list, frames: torch.Tensor) -> torch.Tensor | None:
    queue = deque(_WorkItem(frame, 0) for frame in frames.unbind(1))
    outputs = []
    while queue:
        item = queue.popleft()
        if item.block_index == len(model):
            outputs.append(item.tensor.unsqueeze(1))
            continue
        block = model[item.block_index]
        if isinstance(block, _MemoryBlock):
            previous = item.tensor * 0 if memory[item.block_index] is None else memory[item.block_index]
            result = block(item.tensor, previous)
            memory[item.block_index] = item.tensor
            queue.appendleft(_WorkItem(result, item.block_index + 1))
        elif isinstance(block, _TemporalPool):
            if memory[item.block_index] is None:
                memory[item.block_index] = []
            pending = memory[item.block_index]
            pending.append(item.tensor)
            if len(pending) == block.stride:
                batch, channels, height, width = item.tensor.shape
                result = block(torch.cat(pending, dim=1).view(batch * block.stride, channels, height, width))
                memory[item.block_index] = []
                queue.appendleft(_WorkItem(result, item.block_index + 1))
        else:
            queue.appendleft(_WorkItem(block(item.tensor), item.block_index + 1))
    return torch.cat(outputs, dim=1) if outputs else None


class StreamRenderH3TAEEncodeStream:
    """One request's 17-RGB-frame/5-latent causal encoder state."""

    def __init__(self, codec: "StreamRenderH3TAEEncoder"):
        self.codec = codec
        self.memory = [None] * len(codec.encoder)
        self.frame_in_block = 0

    @torch.no_grad()
    def encode(self, frame: torch.Tensor) -> torch.Tensor | None:
        """Consume one NTCHW RGB frame and return its published latent, if any."""
        if frame.ndim != 5 or frame.shape[1:3] != (1, 3):
            raise ValueError(f"TAE encoder expects one NTCHW RGB frame, got {tuple(frame.shape)}")
        if self.frame_in_block == 0:
            self.memory = [None] * len(self.codec.encoder)
            prefix = frame.new_zeros((frame.shape[0], 3, 3, frame.shape[3], frame.shape[4]))
            _run_encoder(self.codec.encoder, self.memory, F.pixel_unshuffle(prefix, 2))
        latent = _run_encoder(self.codec.encoder, self.memory, F.pixel_unshuffle(frame, 2))
        self.frame_in_block = (self.frame_in_block + 1) % 17
        return latent


class StreamRenderH3TAEEncoder(nn.Module):
    """Released TAE encoder weights, executed entirely by LightX2V modules."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cuda"):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv(12, 64),
            nn.ReLU(inplace=True),
            _TemporalPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            _MemoryBlock(64),
            _MemoryBlock(64),
            _MemoryBlock(64),
            _TemporalPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            _MemoryBlock(64),
            _MemoryBlock(64),
            _MemoryBlock(64),
            _TemporalPool(64, 1),
            _conv(64, 64, stride=2, bias=False),
            _MemoryBlock(64),
            _MemoryBlock(64),
            _MemoryBlock(64),
            _conv(64, 24),
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict({key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")}, strict=True)
        self.to(device).eval()

    def create_stream(self) -> StreamRenderH3TAEEncodeStream:
        return StreamRenderH3TAEEncodeStream(self)

    @torch.no_grad()
    def encode_picture(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an NCHW [0,1] picture as a single CTHW latent anchor."""
        device_type = next(self.encoder.parameters()).device.type
        with torch.autocast(device_type, dtype=torch.float16, enabled=device_type == "cuda"):
            latent = self.create_stream().encode(image.unsqueeze(1))
        if latent is None or latent.shape[1] != 1:
            raise RuntimeError("TAE picture encoding did not publish one latent")
        return latent[0].permute(1, 0, 2, 3).contiguous()


__all__ = ["StreamRenderH3TAEEncoder", "StreamRenderH3TAEEncodeStream"]
