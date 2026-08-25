# Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
# Copyright 2026 The LightX2V Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native MiniMax-H3 waveform VAE encoder/decoder.

The released audio VAE is mono.  H3 represents stereo as two batch items:
``[2, 32, frames] -> [2, 1, samples] -> [1, 2, samples]``.  The public
:meth:`decode` API keeps that contract, including per-channel latent
denormalization and FP32 DAC/BigVGAN execution.
"""

from __future__ import annotations

import gc
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from lightx2v.models.video_encoders.hf.minimax_h3.weights import (
    SafetensorsSubsetReport,
    load_safetensors_subset,
)
from lightx2v_platform.base.global_var import AI_DEVICE


def _empty_device_cache(device: torch.device) -> None:
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "empty_cache"):
        backend.empty_cache()


def _component_dir(model_path: str | Path, component: str) -> Path:
    model_path = Path(model_path)
    nested = model_path / component
    if nested.is_dir():
        return nested
    if model_path.name == component and model_path.is_dir():
        return model_path
    raise FileNotFoundError(f"Cannot find MiniMax-H3 {component!r} below {model_path}")


def _wn_conv1d(*args, **kwargs) -> nn.Module:
    # The original checkpoint uses the legacy weight_g/weight_v spelling.
    return weight_norm(nn.Conv1d(*args, **kwargs))


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Kaiser-windowed sinc filter, arithmetically identical to the release."""

    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class MiniMaxH3AudioSnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)


class MiniMaxH3AudioSnake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states):
        return hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)


class MiniMaxH3AudioResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxH3AudioSnake1d(dim),
            _wn_conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=3 * dilation),
            MiniMaxH3AudioSnake1d(dim),
            _wn_conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, hidden_states):
        residual = self.block(hidden_states)
        pad = (hidden_states.shape[-1] - residual.shape[-1]) // 2
        if pad > 0:
            hidden_states = hidden_states[..., pad:-pad]
        return hidden_states + residual


class MiniMaxH3AudioEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxH3AudioResidualUnit(dim // 2, 1),
            MiniMaxH3AudioResidualUnit(dim // 2, 3),
            MiniMaxH3AudioResidualUnit(dim // 2, 9),
            MiniMaxH3AudioSnake1d(dim // 2),
            _wn_conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, hidden_states):
        return self.block(hidden_states)


class MiniMaxH3AudioEncoder(nn.Module):
    def __init__(self, d_model: int, strides: tuple[int, ...], d_latent: int) -> None:
        super().__init__()
        blocks = [_wn_conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            blocks.append(MiniMaxH3AudioEncoderBlock(d_model, stride))
        blocks += [MiniMaxH3AudioSnake1d(d_model), _wn_conv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*blocks)

    def forward(self, hidden_states):
        return self.block(hidden_states)


class MiniMaxH3AudioGeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        return self.w2(self.act(self.w0(hidden_states)) * self.w1(hidden_states))


class MiniMaxH3AudioCausalAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states):
        batch, length, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self.qkv.weight, torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        query, key, value = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4).unbind(0)
        output = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0, is_causal=True).transpose(1, 2)
        output = output.mean(dim=2)
        output = F.adaptive_avg_pool1d(output, self.out_dim)
        return self.proj(output)


class MiniMaxH3AudioAttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MiniMaxH3AudioCausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = MiniMaxH3AudioGeGluMlp(out_dim, out_dim * mlp_ratio)

    def forward(self, hidden_states):
        hidden_states = self.proj(self.norm3(hidden_states)) + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3AudioLowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int) -> None:
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            hidden_states,
            self.filter.expand(num_channels, -1, -1),
            stride=self.stride,
            groups=num_channels,
        )


class MiniMaxH3AudioUpSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states,
            self.filter.expand(num_channels, -1, -1),
            stride=self.stride,
            groups=num_channels,
        )
        return hidden_states[..., self.pad_left : -self.pad_right]


class MiniMaxH3AudioDownSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int) -> None:
        super().__init__()
        self.lowpass = MiniMaxH3AudioLowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lowpass(hidden_states)


class MiniMaxH3AudioActivation1d(nn.Module):
    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.act = activation
        self.upsample = MiniMaxH3AudioUpSample1d(ratio, kernel_size)
        self.downsample = MiniMaxH3AudioDownSample1d(ratio, kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.downsample(hidden_states)


class MiniMaxH3AudioAMPBlock(nn.Module):
    """BigVGAN anti-aliased multi-periodicity residual block."""

    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                _wn_conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=value,
                    padding=(kernel_size * value - value) // 2,
                )
                for value in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                _wn_conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                )
                for _ in dilation
            ]
        )
        self.activations = nn.ModuleList([MiniMaxH3AudioActivation1d(MiniMaxH3AudioSnakeBeta(channels)) for _ in range(2 * len(dilation))])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = conv1(act1(hidden_states))
            residual = conv2(act2(residual))
            hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxH3AudioBigVGANDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        if self.num_upsamples == 0 or self.num_kernels == 0:
            raise ValueError("MiniMax-H3 audio decoder requires at least one upsampler and one residual kernel")

        self.conv_pre = _wn_conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

        # Preserve the released ``ups.<i>.0`` nesting for direct key parity.
        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            nn.ConvTranspose1d(
                                upsample_initial_channel // (2**index),
                                upsample_initial_channel // (2 ** (index + 1)),
                                kernel,
                                rate,
                                padding=(kernel - rate) // 2,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        channels = upsample_initial_channel
        for index in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(MiniMaxH3AudioAMPBlock(channels, kernel, tuple(dilation)))

        self.activation_post = MiniMaxH3AudioActivation1d(MiniMaxH3AudioSnakeBeta(channels))
        self.conv_post = _wn_conv1d(channels, 1, 7, 1, padding=3, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_pre(hidden_states)
        for upsample_index in range(self.num_upsamples):
            hidden_states = self.ups[upsample_index][0](hidden_states)
            residual = None
            for kernel_index in range(self.num_kernels):
                block = self.resblocks[upsample_index * self.num_kernels + kernel_index](hidden_states)
                residual = block if residual is None else residual + block
            hidden_states = residual / self.num_kernels

        hidden_states = self.activation_post(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        return torch.clamp(hidden_states, min=-1.0, max=1.0)


class MiniMaxH3AudioVAE(nn.Module):
    """H3 audio VAE that loads encoder and decoder from the original file."""

    def __init__(
        self,
        config: dict,
        *,
        device: str | torch.device | None = None,
        cpu_offload: bool = False,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.execution_device = torch.device(device or AI_DEVICE)
        self.cpu_offload = cpu_offload

        encoder_rates = tuple(int(value) for value in config.get("encoder_rates", (2, 4, 4, 5, 5)))
        decoder_rates = tuple(int(value) for value in config.get("decoder_rates", (5, 5, 2, 2, 2, 2, 2)))
        self.hop_length = math.prod(encoder_rates)
        if math.prod(decoder_rates) != self.hop_length:
            raise ValueError(f"decoder_rates must upsample by the encoder hop length {self.hop_length}, got {math.prod(decoder_rates)}")

        latent_dim = int(config.get("latent_dim", 2048))
        latent_channels = int(config.get("latent_channels", 32))
        self.sampling_rate = int(config.get("sampling_rate", 32000))
        if latent_dim % latent_channels:
            raise ValueError(f"latent_dim ({latent_dim}) must be a multiple of latent_channels ({latent_channels})")
        self.encoder = MiniMaxH3AudioEncoder(int(config.get("encoder_dim", 64)), encoder_rates, latent_dim)
        self.pre_block = MiniMaxH3AudioAttnProjection(latent_dim, latent_channels, int(config.get("num_attention_heads", 8)))
        self.mean_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        self.logs_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        self.dec_in_proj = nn.Conv1d(latent_channels, latent_dim, 1)
        self.decoder = MiniMaxH3AudioBigVGANDecoder(
            in_channels=latent_dim,
            upsample_initial_channel=int(config.get("decoder_dim", 1024)),
            upsample_rates=decoder_rates,
            upsample_kernel_sizes=tuple(int(value) for value in config.get("decoder_kernel_sizes", (9, 9, 4, 4, 4, 4, 4))),
            resblock_kernel_sizes=tuple(int(value) for value in config.get("resblock_kernel_sizes", (3, 7, 11))),
            resblock_dilation_sizes=tuple(tuple(int(value) for value in dilation) for dilation in config.get("resblock_dilation_sizes", ((1, 3, 5), (1, 3, 5), (1, 3, 5)))),
        )

        self.register_buffer("latents_mean", torch.empty(latent_channels), persistent=False)
        self.register_buffer("latents_std", torch.empty(latent_channels), persistent=False)
        self._reset_runtime_buffers()
        self.load_report: SafetensorsSubsetReport | None = None

    def _reset_runtime_buffers(self) -> None:
        latent_channels = self.dec_in_proj.in_channels
        mean = self.config.get("latents_mean", [0.0] * latent_channels)
        std = self.config.get("latents_std", [1.0] * latent_channels)
        if len(mean) != latent_channels or len(std) != latent_channels:
            raise ValueError(f"Audio latent statistics must contain {latent_channels} values, got mean={len(mean)}, std={len(std)}")
        self._buffers["latents_mean"] = torch.tensor(mean, dtype=torch.float32)
        self._buffers["latents_std"] = torch.tensor(std, dtype=torch.float32)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device | None = None,
        cpu_offload: bool = False,
    ) -> "MiniMaxH3AudioVAE":
        vae_dir = _component_dir(model_path, "audio_vae")
        with (vae_dir / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        with torch.device("meta"):
            model = cls(config, device=device, cpu_offload=cpu_offload)
        model._reset_runtime_buffers()
        model.load_report = load_safetensors_subset(model, vae_dir)
        model.eval().requires_grad_(False)
        if not cpu_offload:
            model.to(model.execution_device)
        return model

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(device=latents.device).view(1, -1, 1)
        std = self.latents_std.to(device=latents.device).view(1, -1, 1)
        return latents.float() * std + mean

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(device=latents.device).view(1, -1, 1)
        std = self.latents_std.to(device=latents.device).view(1, -1, 1)
        return (latents.float() - mean) / std

    def encode(self, waveform: torch.Tensor, *, return_cpu: bool = True) -> torch.Tensor:
        """Encode stereo as two mono batch items and return normalized posterior means."""
        try:
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(1)
            if waveform.ndim != 3 or waveform.shape[1] != 1:
                raise ValueError(f"audio waveform must be [batch,1,samples] or [batch,samples], got {tuple(waveform.shape)}")
            device = self._activate()
            waveform = waveform.to(device=device, dtype=torch.float32)
            right_pad = math.ceil(waveform.shape[-1] / self.hop_length) * self.hop_length - waveform.shape[-1]
            if right_pad:
                waveform = F.pad(waveform, (0, right_pad))
            with torch.no_grad():
                hidden_states = self.encoder(waveform)
                hidden_states = self.pre_block(hidden_states.transpose(1, 2)).transpose(1, 2)
                mean = self.mean_proj(hidden_states).float()
                # The released Ref2AV path consumes posterior.mode(); logs_proj
                # is intentionally not evaluated there.
                mean = self.normalize_latents(mean)
            return mean.cpu() if return_cpu else mean
        finally:
            if self.cpu_offload:
                self.offload()

    def _activate(self) -> torch.device:
        if self.cpu_offload:
            self.to(self.execution_device)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if dtype != torch.float32:
            raise RuntimeError(f"MiniMax-H3 audio VAE weights must remain float32 for parity; found {dtype}. Move the module by device only, without a dtype cast.")
        return device

    def offload(self) -> None:
        self.to("cpu")
        _empty_device_cache(self.execution_device)
        gc.collect()

    def _prepare_stereo_latents(
        self,
        latents: torch.Tensor,
        stereo_batch: bool,
    ) -> tuple[torch.Tensor, int | None]:
        if latents.ndim == 4:
            if not stereo_batch or latents.shape[1] != 2:
                raise ValueError(f"rank-4 audio latents must use stereo_batch=True and have shape [batch, 2, channels, frames], got {tuple(latents.shape)}")
            batch_size, _, channels, frames = latents.shape
            return latents.reshape(batch_size * 2, channels, frames), batch_size
        if latents.ndim != 3:
            raise ValueError(f"audio latents must have shape [2 * batch, channels, frames] or [batch, 2, channels, frames], got {tuple(latents.shape)}")
        if stereo_batch:
            if latents.shape[0] % 2:
                raise ValueError(f"stereo audio uses two mono batch items; leading dimension must be even, got {latents.shape[0]}")
            return latents, latents.shape[0] // 2
        return latents, None

    def _run_decode(
        self,
        latents: torch.Tensor,
        *,
        denormalize: bool,
        stereo_batch: bool,
        return_cpu: bool | None,
    ) -> torch.Tensor:
        try:
            latents, stereo_groups = self._prepare_stereo_latents(latents, stereo_batch)
            if latents.shape[1] != self.dec_in_proj.in_channels:
                raise ValueError(f"audio latents must have {self.dec_in_proj.in_channels} channels, got {latents.shape[1]}")
            device = self._activate()
            latents = latents.to(device=device, dtype=torch.float32)
            if denormalize:
                latents = self.denormalize_latents(latents)
            # Disable an ambient CUDA autocast: the released DAC/BigVGAN weights
            # and arithmetic stay FP32 (BF16 decodes are roughly 20 dB quieter).
            autocast_context = torch.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()
            with torch.no_grad(), autocast_context:
                decoded = self.decoder(self.dec_in_proj(latents)).float()

            if stereo_groups is not None:
                decoded = decoded.squeeze(1).reshape(stereo_groups, 2, decoded.shape[-1])

            if return_cpu is None:
                return_cpu = self.cpu_offload
            if return_cpu:
                decoded = decoded.cpu()
            return decoded
        finally:
            if self.cpu_offload:
                self.offload()

    def decode_raw(
        self,
        denormalized_latents: torch.Tensor,
        *,
        stereo_batch: bool = False,
        return_cpu: bool | None = None,
    ) -> torch.Tensor:
        """Decode VAE-space latents using the low-level mono-as-batch contract.

        By default ``[2, 32, T]`` returns ``[2, 1, T * 800]``, exactly like
        the released autoencoder. Set ``stereo_batch=True`` to fold each pair
        into pipeline-facing ``[B, 2, samples]`` output.
        """

        return self._run_decode(
            denormalized_latents,
            denormalize=False,
            stereo_batch=stereo_batch,
            return_cpu=return_cpu,
        )

    def decode(
        self,
        latents: torch.Tensor,
        *,
        stereo_batch: bool = True,
        return_cpu: bool | None = None,
    ) -> torch.Tensor:
        """Decode normalized H3 audio latents to a waveform in ``[-1, 1]``.

        ``[2, 32, T]`` follows the official mono-as-batch convention and
        returns ``[1, 2, T * 800]``.  Batched stereo input may alternatively be
        supplied as ``[B, 2, 32, T]``.
        """

        return self._run_decode(
            latents,
            denormalize=True,
            stereo_batch=stereo_batch,
            return_cpu=return_cpu,
        )


AutoencoderKLMiniMaxH3AudioNative = MiniMaxH3AudioVAE
