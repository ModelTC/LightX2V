from __future__ import annotations

import functools
import math
from contextlib import nullcontext

import torch
import torch.nn as nn

from lightx2v.models.networks.minimax_h3.infer.sgl import rope as _registered_rope  # noqa: F401
from lightx2v.models.networks.minimax_h3.infer.sglang_fused import (
    apply_vae_silu_mul_sglang,
    scaled_residual_add_vae_sglang,
)
from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MiniMaxH3VideoAttention,
    MiniMaxH3VideoRotaryPosEmbed,
    MiniMaxH3VideoTransformerBlock,
    MiniMaxH3VideoVAE,
    MiniMaxH3VideoViTDecoder3d,
    _FeedForward,
    _SwiGLU,
)
from lightx2v.utils.registry_factory import ROPE_REGISTER

_SGL_ROPE_TYPE = "h3ref_sgl_rope"


def _cuda_autocast_disabled(tensor: torch.Tensor):
    return torch.autocast("cuda", enabled=False) if tensor.is_cuda else nullcontext()


def _linear_with_module_dtype(
    linear: nn.Linear,
    tensor: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return linear(tensor.to(linear.weight.dtype)).to(out_dtype)


def _apply_qk_norm(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    if (
        isinstance(module, (nn.LayerNorm, nn.RMSNorm))
        and module.weight is None
        and (not isinstance(module, nn.LayerNorm) or module.bias is None)
        and hidden_states.is_cuda
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    ):
        with torch.autocast("cuda", enabled=False):
            return module(hidden_states)
    return module(hidden_states.float()).to(hidden_states.dtype)


@functools.lru_cache(maxsize=1)
def _is_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def _linear_without_fused_bias(linear: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    if linear.bias is None or not hidden_states.is_cuda or hidden_states.dtype != linear.weight.dtype or not _is_sm120():
        return linear(hidden_states)
    output = torch.matmul(hidden_states, linear.weight.t())
    output += linear.bias
    return output


class _SGLSwiGLU(_SwiGLU):
    def _pack_after_load(self) -> None:
        if getattr(self, "_sgl_layout_packed", False):
            raise RuntimeError("MiniMax-H3 SGL VAE SwiGLU weights were already packed")
        value_weight, gate_weight = self.proj.weight.chunk(2, dim=0)
        self.proj.weight = nn.Parameter(
            torch.cat((gate_weight, value_weight), dim=0).contiguous(),
            requires_grad=self.proj.weight.requires_grad,
        )
        if self.proj.bias is not None:
            value_bias, gate_bias = self.proj.bias.chunk(2, dim=0)
            self.proj.bias = nn.Parameter(
                torch.cat((gate_bias, value_bias), dim=0).contiguous(),
                requires_grad=self.proj.bias.requires_grad,
            )
        self._sgl_layout_packed = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return apply_vae_silu_mul_sglang(self.proj(hidden_states))


class _SGLFeedForward(_FeedForward):
    swiglu_cls = _SGLSwiGLU

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net[0](hidden_states)
        return _linear_without_fused_bias(self.net[2], hidden_states)


class MiniMaxH3SGLVideoRotaryPosEmbed(MiniMaxH3VideoRotaryPosEmbed):
    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__(dim=dim, theta=theta, num_axes=num_axes)
        inv_freq = 1 / self.theta ** torch.arange(
            0,
            1,
            2 * self.num_axes / self.dim,
            dtype=torch.float32,
            device="cpu",
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.shape[-1] != self.num_axes:
            raise ValueError(f"Expected {self.num_axes} dimensions, got {position_ids.shape[-1]}")
        with _cuda_autocast_disabled(position_ids):
            angles = 2.0 * math.pi * position_ids[:, :, :, None]
            angles = angles * self.inv_freq.to(position_ids.device)[None, None, None, :]
            angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
            cos = torch.cos(angles)
            sin = torch.sin(angles)
        return cos.to(dtype=position_ids.dtype), sin.to(dtype=position_ids.dtype)

    @staticmethod
    def prepare(
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope = ROPE_REGISTER[_SGL_ROPE_TYPE](compute_dtype=dtype)
        return rope.prepare_freqs(rotary_emb, rotary_dim=rotary_emb[0].shape[-1])


class MiniMaxH3SGLVideoAttention(MiniMaxH3VideoAttention):
    rope = ROPE_REGISTER[_SGL_ROPE_TYPE]()

    def _pack_after_load(self) -> None:
        if getattr(self, "_sgl_layout_packed", False):
            raise RuntimeError("MiniMax-H3 SGL VAE QKV weights were already packed")
        linears = (self.to_q, self.to_k, self.to_v)
        in_features = linears[0].in_features
        with torch.device("meta"):
            self.to_qkv = nn.Linear(
                in_features,
                self.inner_dim * 3,
                bias=linears[0].bias is not None,
                dtype=linears[0].weight.dtype,
            )
        packed_weight = torch.stack(
            tuple(linear.weight.reshape(self.heads, self.dim_head, in_features) for linear in linears),
            dim=1,
        ).reshape(self.inner_dim * 3, in_features)
        self.to_qkv.weight = nn.Parameter(
            packed_weight.contiguous(),
            requires_grad=linears[0].weight.requires_grad,
        )
        if linears[0].bias is not None:
            packed_bias = torch.stack(
                tuple(linear.bias.reshape(self.heads, self.dim_head) for linear in linears),
                dim=1,
            ).reshape(self.inner_dim * 3)
            self.to_qkv.bias = nn.Parameter(
                packed_bias.contiguous(),
                requires_grad=linears[0].bias.requires_grad,
            )
        self.to_q = None
        self.to_k = None
        self.to_v = None
        self._sgl_layout_packed = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if self.to_qkv is None:
            raise RuntimeError("MiniMax-H3 SGL VAE QKV weights must be packed after checkpoint loading")
        qkv = self.to_qkv(hidden_states)
        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        query = _apply_qk_norm(self.norm_q, query)
        key = _apply_qk_norm(self.norm_k, key)
        if rotary_emb is not None:
            query, key = self.rope.apply(query, key, rotary_emb, materialize=True)

        hidden_states = self.calculate.apply(
            query,
            key,
            value,
            max_seqlen_q=query.shape[1],
            max_seqlen_kv=key.shape[1],
            softmax_scale=self.dim_head**-0.5,
        ).view(batch_size, seq_len, self.inner_dim)
        return self.to_out[0](hidden_states)


class MiniMaxH3SGLVideoTransformerBlock(MiniMaxH3VideoTransformerBlock):
    attention_cls = MiniMaxH3SGLVideoAttention
    feed_forward_cls = _SGLFeedForward

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        attention_output = self.attn(norm_hidden_states, rotary_emb)
        hidden_states = scaled_residual_add_vae_sglang(hidden_states, attention_output, self.scale1)

        norm_hidden_states = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        feed_forward_output = self.ff(norm_hidden_states)
        return scaled_residual_add_vae_sglang(hidden_states, feed_forward_output, self.scale2)


class MiniMaxH3SGLVideoViTDecoder3d(MiniMaxH3VideoViTDecoder3d):
    rope_cls = MiniMaxH3SGLVideoRotaryPosEmbed
    block_cls = MiniMaxH3SGLVideoTransformerBlock

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(
            batch_size,
            num_channels,
            num_frames,
            1,
            height,
            1,
            width,
            1,
        )
        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7)
        hidden_states = hidden_states.reshape(batch_size, num_frames * height * width, num_channels)

        with _cuda_autocast_disabled(hidden_states):
            hidden_states = _linear_with_module_dtype(self.proj_in, hidden_states, input_dtype)
        num_patches = hidden_states.shape[1]
        hidden_states = torch.cat(
            (
                hidden_states,
                self.register_tokens.expand(batch_size, -1, -1),
                torch.zeros_like(hidden_states[:, 0:1, :]),
            ),
            dim=1,
        )

        coords = []
        for size in (num_frames, height, width):
            axis = torch.arange(0.5, size, dtype=input_dtype, device=hidden_states.device)
            axis = axis / size
            axis = 2.0 * axis - 1.0
            coords.append(axis)
        position_ids = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
        position_ids = position_ids.flatten(0, 2).unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = torch.zeros(
            (batch_size, self.num_register_tokens + 1, 3),
            device=hidden_states.device,
            dtype=position_ids.dtype,
        )
        position_ids = torch.cat((position_ids, suffix_ids), dim=1)
        rotary_dtype = torch.get_autocast_dtype("cuda") if hidden_states.is_cuda and torch.is_autocast_enabled("cuda") else hidden_states.dtype
        rotary_emb = self.rope.prepare(self.rope(position_ids), dtype=rotary_dtype)

        for block_index, block in enumerate(self.transformer_blocks):
            hidden_states = self._run_block(block_index, block, hidden_states, rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        with _cuda_autocast_disabled(hidden_states):
            output = _linear_with_module_dtype(self.proj_out, hidden_states, hidden_states.dtype)
        output = output[:, :num_patches, :]

        video_frames = num_frames * self.patch_size_t
        video_height = height * self.patch_size
        video_width = width * self.patch_size
        output = output.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            self.patch_size_t,
            self.patch_size,
            self.patch_size,
        )
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return output.reshape(batch_size, self.out_channels, video_frames, video_height, video_width)


class MiniMaxH3SGLVideoVAE(MiniMaxH3VideoVAE):
    decoder_cls = MiniMaxH3SGLVideoViTDecoder3d
    encoder_infer_dtype = torch.float32

    def _validate_execution_profile(
        self,
        *,
        quant_scheme: str | None,
        attn_type: str,
        use_compile: bool,
        sensitive_layer_dtype: torch.dtype,
    ) -> None:
        if quant_scheme is not None:
            raise ValueError("MiniMax-H3 SGL video VAE requires the unquantized checkpoint")
        if attn_type != "torch_sdpa":
            raise ValueError("MiniMax-H3 SGL video VAE requires vae_attn_type='torch_sdpa'")
        if use_compile:
            raise ValueError("MiniMax-H3 SGL video VAE requires vae_use_compile=false")
        if sensitive_layer_dtype != torch.float32:
            raise ValueError("MiniMax-H3 SGL video VAE requires vae_sensitive_layer_dtype='fp32'")

    def _post_load(self) -> None:
        for block in self.decoder.transformer_blocks:
            block.attn._pack_after_load()
            block.ff.net[0]._pack_after_load()

    def _prepare_inference_dtypes(self) -> None:
        for block in self.decoder.transformer_blocks:
            for linear in (
                block.attn.to_qkv,
                block.attn.to_out[0],
                block.ff.net[0].proj,
                block.ff.net[2],
            ):
                linear.to(dtype=self.infer_dtype)

    def _cast_decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def _decode_context(self, latents: torch.Tensor):
        return torch.autocast("cuda", dtype=self.infer_dtype) if latents.is_cuda else nullcontext()

    def _return_cpu_by_default(self) -> bool:
        return False

    @staticmethod
    def prepare_reference_pixels(pixels: torch.Tensor) -> torch.Tensor:
        return pixels

    @staticmethod
    def _sample_posterior(moments: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        parameters = moments.to(dtype=torch.float32)
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = logvar.mul(0.5).exp_()
        noise = torch.randn(mean.shape, generator=generator)
        noise = noise.to(device=parameters.device)
        return noise.mul_(std).add_(mean)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        result_device = latents.device
        latents_cpu = latents.detach().to(device="cpu", dtype=torch.float32)
        mean = self.latents_mean.detach().to(device="cpu", dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = self.latents_std.detach().to(device="cpu", dtype=torch.float32).view(1, -1, 1, 1, 1)
        return latents_cpu.sub_(mean).div_(std).to(result_device)

    def preprocess(self, pixels: torch.Tensor, *, video: bool = False) -> torch.Tensor:
        if pixels.dtype == torch.uint8:
            if video:
                frames = pixels[0].transpose(0, 1).to(torch.float32).div_(255.0)
                mean = self.pixel_mean.to(frames.device).view(1, -1, 1, 1)
                std = self.pixel_std.to(frames.device).view(1, -1, 1, 1)
                frames.sub_(mean).div_(std)
                return frames.contiguous().transpose(0, 1).unsqueeze(0)
            images = pixels.squeeze(2).to(torch.float32).div_(255.0)
            mean = self.pixel_mean.to(images.device).view(1, -1, 1, 1)
            std = self.pixel_std.to(images.device).view(1, -1, 1, 1)
            images.sub_(mean).div_(std)
            return images.contiguous().unsqueeze(2)

        mean = self.pixel_mean.to(pixels.device).view(1, -1, 1, 1, 1)
        std = self.pixel_std.to(pixels.device).view(1, -1, 1, 1, 1)
        return pixels.to(self.sensitive_layer_dtype).sub_(mean).div_(std)

    def postprocess(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frames, height, width = video.shape
        inverse_mean = video.new_tensor(tuple(-mean / std for mean, std in zip(MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD)))
        inverse_std = video.new_tensor(tuple(1.0 / std for std in MINIMAX_H3_PIXEL_STD))
        video = video.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        video = video.clone().sub_(inverse_mean[:, None, None]).div_(inverse_std[:, None, None])
        video.clamp_(0, 1)
        return video.reshape(batch_size, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def to_uint8_frames(video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
            raise ValueError(f"decoded H3 video must be [1,3,F,H,W], got {tuple(video.shape)}")
        pixels = video[0].permute(1, 2, 3, 0).float() * 255.0
        return pixels.clamp_(0, 255).to(torch.uint8).contiguous().cpu()

    @staticmethod
    def _blend_values(
        a: torch.Tensor,
        b: torch.Tensor,
        weight_a: torch.Tensor,
        weight_b: torch.Tensor,
    ) -> torch.Tensor:
        blended = a * weight_a
        blended.add_(b * weight_b)
        return blended


__all__ = [
    "MiniMaxH3SGLVideoAttention",
    "MiniMaxH3SGLVideoRotaryPosEmbed",
    "MiniMaxH3SGLVideoTransformerBlock",
    "MiniMaxH3SGLVideoVAE",
    "MiniMaxH3SGLVideoViTDecoder3d",
]
