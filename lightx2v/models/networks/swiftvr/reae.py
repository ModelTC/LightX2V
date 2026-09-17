"""SwiftVR restoration-aware autoencoder and its causal state."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .parallel import exchange_chunk_boundary


def convolution(input_channels: int, output_channels: int, **kwargs):
    return nn.Conv2d(input_channels, output_channels, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return (hidden_states / 3).tanh_() * 3


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

    def forward(self, sequence: torch.Tensor, previous_frame: torch.Tensor | None) -> torch.Tensor:
        if previous_frame is None:
            previous_frame = torch.zeros_like(sequence[:, :1])
        previous = torch.cat([previous_frame, sequence[:, :-1]], dim=1)
        inputs = torch.cat([sequence, previous], dim=2).flatten(0, 1)
        del previous
        return self.activation(self.conv(inputs).add_(self.skip(sequence.flatten(0, 1))))


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
        _, channels, height, width = hidden_states.shape
        frame_groups = hidden_states.reshape(-1, self.stride * channels, height, width)
        return run_frame_batches(self.conv, frame_groups, frame_batch_size)


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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return self.proj(hidden_states)

        frames, channels, height, width = hidden_states.shape
        hidden_states = F.interpolate(hidden_states.unsqueeze(2), size=(self.stride, height, width), mode="nearest")
        hidden_states = self.conv3d(hidden_states)
        return hidden_states.permute(0, 2, 1, 3, 4).reshape(frames * self.stride, channels, height, width)


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
        # TemporalGrow is spatially pointwise, so run it before nearest upsampling.
        # List layers in execution order; numeric names preserve checkpoint keys.
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    ("0", Clamp()),
                    ("1", convolution(48, widths[0])),
                    ("2", nn.ReLU(inplace=True)),
                    ("3", MemoryBlock(widths[0], widths[0])),
                    ("4", MemoryBlock(widths[0], widths[0])),
                    ("5", MemoryBlock(widths[0], widths[0])),
                    ("7", TemporalGrow(widths[0], 1)),
                    ("6", nn.Upsample(scale_factor=2)),
                    ("8", convolution(widths[0], widths[1], bias=False)),
                    ("9", MemoryBlock(widths[1], widths[1])),
                    ("10", MemoryBlock(widths[1], widths[1])),
                    ("11", MemoryBlock(widths[1], widths[1])),
                    ("13", TemporalGrow(widths[1], 2)),
                    ("12", nn.Upsample(scale_factor=2)),
                    ("14", convolution(widths[1], widths[2], bias=False)),
                    ("15", MemoryBlock(widths[2], widths[2])),
                    ("16", MemoryBlock(widths[2], widths[2])),
                    ("17", MemoryBlock(widths[2], widths[2])),
                    ("19", TemporalGrow(widths[2], 2)),
                    ("18", nn.Upsample(scale_factor=2)),
                    ("20", convolution(widths[2], widths[3], bias=False)),
                    ("21", nn.ReLU(inplace=True)),
                    ("22", convolution(widths[3], 12)),
                ]
            )
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
    chunk_p_group=None,
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
        previous_frame = state.get(state_key)
        if chunk_p_group is not None:
            boundary = exchange_chunk_boundary(sequence[:, -1:].contiguous(), chunk_p_group)
            # Rank 0 carries the last rank's boundary into the next batch.
            if dist.get_rank(chunk_p_group) != 0:
                previous_frame = boundary
            else:
                next_state[state_key] = boundary
        else:
            next_state[state_key] = sequence[:, -1:].detach().clone()
        hidden_states = layer(sequence, previous_frame)
        index += 1

    _, channels, height, width = hidden_states.shape
    return hidden_states.view(batch, hidden_states.shape[0] // batch, channels, height, width), next_state


class StreamingAutoencoder:
    def __init__(self, autoencoder: RestorationAutoencoder, frame_batch_size: int = 1):
        self.autoencoder = autoencoder
        self.frame_batch_size = frame_batch_size
        self.chunk_p_group = None
        # The final temporal expansion precedes the frame-independent output layers.
        output_start = max(index for index, layer in enumerate(autoencoder.decoder) if isinstance(layer, TemporalGrow)) + 1
        self.decoder_layers = autoencoder.decoder[:output_start]
        self.output_layers = autoencoder.decoder[output_start:]
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
            self.chunk_p_group,
        )
        return latents

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor, is_first: bool, output_batch_size: int = 0) -> Iterator[torch.Tensor]:
        video, self.decoder_state = run_causal_layers(
            self.decoder_layers,
            latents,
            self.decoder_state,
            self.frame_batch_size,
            self.chunk_p_group,
        )
        # Causal state is complete; omit discarded frames from the high-resolution layers.
        if is_first:
            video = video[:, self.autoencoder.frames_to_trim :]

        @torch.inference_mode()
        def decode_frames():
            for frame_batch in video.split(output_batch_size or video.shape[1], dim=1):
                batch, frames, channels, height, width = frame_batch.shape
                pixels = run_frame_layers(self.output_layers, frame_batch.reshape(batch * frames, channels, height, width), self.frame_batch_size)
                pixels = F.pixel_shuffle(pixels.clamp_(0, 1), self.autoencoder.patch_size)
                yield pixels.reshape(batch, frames, *pixels.shape[1:])

        # Issue all causal exchanges even if the caller does not consume every pixel batch.
        return decode_frames()
