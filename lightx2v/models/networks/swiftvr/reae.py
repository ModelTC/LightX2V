"""SwiftVR restoration-aware autoencoder and its causal state."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


def convolution(input_channels: int, output_channels: int, **kwargs):
    return nn.Conv2d(input_channels, output_channels, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(hidden_states / 3) * 3


class MemoryBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            convolution(input_channels * 2, output_channels),
            nn.ReLU(inplace=True),
            convolution(output_channels, output_channels),
            nn.ReLU(inplace=True),
            convolution(output_channels, output_channels),
        )
        self.skip = nn.Conv2d(input_channels, output_channels, 1, bias=False) if input_channels != output_channels else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, hidden_states: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(torch.cat([hidden_states, previous], dim=1)) + self.skip(hidden_states))


def run_frame_batches(function, hidden_states: torch.Tensor, frame_batch_size: int | None) -> torch.Tensor:
    """Run a frame-independent operation in small batches with one output buffer."""

    if not frame_batch_size or hidden_states.shape[0] <= frame_batch_size:
        return function(hidden_states)

    output = None
    output_frames_per_input = 0
    for start in range(0, hidden_states.shape[0], frame_batch_size):
        end = min(start + frame_batch_size, hidden_states.shape[0])
        batch = function(hidden_states[start:end])
        if output is None:
            output_frames_per_input = batch.shape[0] // (end - start)
            output = batch.new_empty((hidden_states.shape[0] * output_frames_per_input, *batch.shape[1:]))
        output[start * output_frames_per_input : end * output_frames_per_input].copy_(batch)
    return output


def run_frame_layers(layers, hidden_states: torch.Tensor, frame_batch_size: int) -> torch.Tensor:
    """Run consecutive frame-independent layers without full-size intermediate buffers."""

    def apply_layers(frames):
        for layer in layers:
            frames = layer(frames)
        return frames

    return run_frame_batches(apply_layers, hidden_states, frame_batch_size)


class TemporalPool(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels * stride, channels, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, frame_batch_size: int | None = None) -> torch.Tensor:
        frame_groups, channels, height, width = hidden_states.shape

        def pool_frames(frames):
            return self.conv(frames.reshape(-1, self.stride * channels, height, width))

        output_frame_count = frame_groups // self.stride
        if not frame_batch_size or output_frame_count <= frame_batch_size:
            return pool_frames(hidden_states)

        output = None
        for start in range(0, output_frame_count, frame_batch_size):
            end = min(start + frame_batch_size, output_frame_count)
            batch = pool_frames(hidden_states[start * self.stride : end * self.stride])
            if output is None:
                output = batch.new_empty((output_frame_count, *batch.shape[1:]))
            output[start:end].copy_(batch)
        return output


class TemporalGrow(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        if stride == 1:
            self.proj = nn.Conv2d(channels, channels, 1, bias=False)
            self.conv3d = None
        else:
            self.conv3d = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False)
            self.proj = None

    def forward(self, hidden_states: torch.Tensor, frame_batch_size: int | None = None) -> torch.Tensor:
        if self.stride == 1:
            return run_frame_batches(self.proj, hidden_states, frame_batch_size)

        def grow_frames(frames):
            frame_groups, channels, height, width = frames.shape
            frames = F.interpolate(frames.unsqueeze(2), size=(self.stride, height, width), mode="nearest")
            frames = self.conv3d(frames)
            return frames.permute(0, 2, 1, 3, 4).reshape(frame_groups * self.stride, channels, height, width)

        return run_frame_batches(grow_frames, hidden_states, frame_batch_size)


class RestorationAutoencoder(nn.Module):
    patch_size = 2
    frames_to_trim = 3

    def __init__(self):
        super().__init__()
        encoder_channels = 64
        self.encoder = nn.Sequential(
            convolution(12, encoder_channels),
            nn.ReLU(inplace=True),
            TemporalPool(encoder_channels, 2),
            convolution(encoder_channels, encoder_channels, stride=2, bias=False),
            MemoryBlock(encoder_channels, encoder_channels),
            MemoryBlock(encoder_channels, encoder_channels),
            MemoryBlock(encoder_channels, encoder_channels),
            TemporalPool(encoder_channels, 2),
            convolution(encoder_channels, encoder_channels, stride=2, bias=False),
            MemoryBlock(encoder_channels, encoder_channels),
            MemoryBlock(encoder_channels, encoder_channels),
            MemoryBlock(encoder_channels, encoder_channels),
            TemporalPool(encoder_channels, 1),
            convolution(encoder_channels, encoder_channels, stride=2, bias=False),
            MemoryBlock(encoder_channels, encoder_channels),
            MemoryBlock(encoder_channels, encoder_channels),
            MemoryBlock(encoder_channels, encoder_channels),
            convolution(encoder_channels, 48),
        )

        widths = (512, 256, 128, 64)
        self.decoder = nn.Sequential(
            Clamp(),
            convolution(48, widths[0]),
            nn.ReLU(inplace=True),
            MemoryBlock(widths[0], widths[0]),
            MemoryBlock(widths[0], widths[0]),
            MemoryBlock(widths[0], widths[0]),
            nn.Upsample(scale_factor=2),
            TemporalGrow(widths[0], 1),
            convolution(widths[0], widths[1], bias=False),
            MemoryBlock(widths[1], widths[1]),
            MemoryBlock(widths[1], widths[1]),
            MemoryBlock(widths[1], widths[1]),
            nn.Upsample(scale_factor=2),
            TemporalGrow(widths[1], 2),
            convolution(widths[1], widths[2], bias=False),
            MemoryBlock(widths[2], widths[2]),
            MemoryBlock(widths[2], widths[2]),
            MemoryBlock(widths[2], widths[2]),
            nn.Upsample(scale_factor=2),
            TemporalGrow(widths[2], 2),
            convolution(widths[2], widths[3], bias=False),
            nn.ReLU(inplace=True),
            convolution(widths[3], 12),
        )

    @classmethod
    def from_pretrained(cls, model_path: str | Path, device: torch.device, dtype: torch.dtype):
        with torch.device("meta"):
            model = cls()
        checkpoint = Path(model_path) / "reae.safetensors"
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            state_dict = {name: weights.get_tensor(name).to(device=device, dtype=dtype) for name in weights.keys()}
        model.load_state_dict(state_dict, strict=True, assign=True)
        return model.requires_grad_(False).eval()


def run_causal_layers(
    layers: nn.Sequential,
    video: torch.Tensor,
    state: dict | None,
    frame_batch_size: int | None = None,
):
    state = state or {}
    next_state = {}
    batch, frames, channels, height, width = video.shape
    hidden_states = video.reshape(batch * frames, channels, height, width)

    index = 0
    while index < len(layers):
        layer = layers[index]
        if frame_batch_size and not isinstance(layer, (MemoryBlock, TemporalPool)):
            frame_layers = []
            while index < len(layers) and not isinstance(layers[index], (MemoryBlock, TemporalPool)):
                frame_layers.append(layers[index])
                index += 1
            hidden_states = run_frame_layers(frame_layers, hidden_states, frame_batch_size)
            continue
        if isinstance(layer, TemporalPool):
            hidden_states = layer(hidden_states, frame_batch_size)
            index += 1
            continue
        if not isinstance(layer, MemoryBlock):
            hidden_states = layer(hidden_states)
            index += 1
            continue

        _, channels, height, width = hidden_states.shape
        layer_frames = hidden_states.shape[0] // batch
        sequence = hidden_states.reshape(batch, layer_frames, channels, height, width)
        state_key = f"memory_{index}"
        if state_key in state:
            previous = torch.cat([state[state_key], sequence[:, :-1]], dim=1)
        else:
            previous = F.pad(sequence, (0, 0, 0, 0, 0, 0, 1, 0))[:, :layer_frames]
        next_state[state_key] = sequence[:, -1:].detach().clone()
        hidden_states = layer(hidden_states, previous.reshape_as(hidden_states))
        index += 1

    _, channels, height, width = hidden_states.shape
    return hidden_states.view(batch, hidden_states.shape[0] // batch, channels, height, width), next_state


class StreamingAutoencoder:
    def __init__(self, autoencoder: RestorationAutoencoder, frame_batch_size: int = 1):
        self.autoencoder = autoencoder
        self.frame_batch_size = frame_batch_size
        self.reset()

    def reset(self):
        self.encoder_state = None
        self.decoder_state = None

    @torch.inference_mode()
    def encode(self, video: torch.Tensor, is_last: bool) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        video = F.pixel_unshuffle(video.reshape(batch * frames, channels, height, width), self.autoencoder.patch_size)
        video = video.reshape(batch, frames, *video.shape[1:])
        if is_last:
            video = torch.cat([video, video[:, -1:].expand(-1, 3, -1, -1, -1)], dim=1)
        latents, self.encoder_state = run_causal_layers(
            self.autoencoder.encoder,
            video,
            self.encoder_state,
            self.frame_batch_size,
        )
        return latents

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor, is_first: bool) -> torch.Tensor:
        video, self.decoder_state = run_causal_layers(
            self.autoencoder.decoder,
            latents,
            self.decoder_state,
            self.frame_batch_size,
        )
        video = video.clamp_(0, 1)
        batch, frames, channels, height, width = video.shape
        video = F.pixel_shuffle(video.reshape(batch * frames, channels, height, width), self.autoencoder.patch_size)
        video = video.reshape(batch, frames, *video.shape[1:])
        return video[:, self.autoencoder.frames_to_trim :] if is_first else video
