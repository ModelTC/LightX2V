"""Native SwiftVR DiT built from LightX2V's Wan inference components."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.models.networks.wan.infer.utils import sinusoidal_embedding_1d
from lightx2v.models.networks.wan.model import WanModel

from .attention import SwiftVRShiftedWindowAttention

INFERENCE_TIMESTEP = 1000.0


def normalize_swiftvr_config(config):
    """Translate the official transformer config into LightX2V's Wan fields."""

    config.update(
        {
            "dim": config["num_attention_heads"] * config["attention_head_dim"],
            "num_heads": config["num_attention_heads"],
            "in_dim": config["in_channels"],
            "out_dim": config["out_channels"],
            "self_attn_1_type": "swiftvr_mfswa",
            "cross_attn_1_type": config.get("cross_attention_backend", "torch_sdpa"),
            "cross_attn_2_type": config.get("cross_attention_backend", "torch_sdpa"),
            "rope_type": config.get("rope_type", "torch_real_rope"),
            "rms_norm_type": "torch",
            "layer_norm_type": "torch",
            "modulate_type": "torch",
            "use_image_encoder": False,
            "enable_cfg": False,
            "mxfp8_fuse_enable": False,
            "lazy_load": False,
        }
    )
    if not config.get("dit_original_ckpt"):
        config["dit_original_ckpt"] = os.path.join(config["model_path"], "transformer")


@dataclass(frozen=True)
class SwiftVRCondition:
    timestep_embedding: torch.Tensor
    block_modulation: torch.Tensor
    text_context: torch.Tensor


class SwiftVRPreInfer(WanPreInfer):
    """Wan pre-inference with cached conditioning and temporal RoPE offsets."""

    @torch.no_grad()
    def prepare_condition(self, weights, prompt_embedding: torch.Tensor) -> SwiftVRCondition:
        timestep = torch.full((1,), INFERENCE_TIMESTEP, device=prompt_embedding.device, dtype=torch.float32)
        embedding = sinusoidal_embedding_1d(self.freq_dim, timestep)
        embedding = weights.time_embedding_0.apply(embedding)
        embedding = torch.nn.functional.silu(embedding)
        embedding = weights.time_embedding_2.apply(embedding)
        block_modulation = weights.time_projection_1.apply(torch.nn.functional.silu(embedding)).unflatten(1, (6, self.dim))

        text_context = prompt_embedding.squeeze(0)
        if self.sensitive_layer_dtype != self.infer_dtype:
            text_context = text_context.to(self.sensitive_layer_dtype)
        text_context = weights.text_embedding_0.apply(text_context)
        text_context = torch.nn.functional.gelu(text_context, approximate="tanh")
        text_context = weights.text_embedding_2.apply(text_context)
        return SwiftVRCondition(embedding, block_modulation.squeeze(0), text_context)

    def get_frequency_table(self, required_length: int) -> torch.Tensor:
        if required_length <= self.freqs.shape[0]:
            return self.freqs
        return torch.cat(
            [
                self.rope_params(required_length, self.head_size - 4 * (self.head_size // 6)),
                self.rope_params(required_length, 2 * (self.head_size // 6)),
                self.rope_params(required_length, 2 * (self.head_size // 6)),
            ],
            dim=1,
        ).to(self.freqs.device)

    def prepare_rope(self, grid_sizes: tuple[int, int, int], temporal_offset: int):
        frames, height, width = grid_sizes
        frequencies = self.get_frequency_table(max(temporal_offset + frames, height, width))
        half_head = self.head_size // 2
        temporal, vertical, horizontal = frequencies.split(
            [half_head - 2 * (half_head // 3), half_head // 3, half_head // 3],
            dim=1,
        )
        frequencies = torch.cat(
            [
                temporal[temporal_offset : temporal_offset + frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                vertical[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                horizontal[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(frames * height * width, -1)
        return self.prepare_rope_cache((frequencies.real, frequencies.imag))

    @torch.no_grad()
    def infer(self, weights, latents: torch.Tensor, condition: SwiftVRCondition, temporal_offset: int):
        hidden_states = weights.patch_embedding.apply(latents)
        grid_tuple = tuple(hidden_states.shape[2:])
        hidden_states = hidden_states.flatten(2).transpose(1, 2).squeeze(0).contiguous()
        grid_sizes = GridOutput(
            tensor=torch.tensor([grid_tuple], dtype=torch.int32, device=hidden_states.device),
            tuple=grid_tuple,
        )
        cos_sin = self.prepare_rope(grid_tuple, temporal_offset)
        return WanPreInferModuleOutput(
            embed=condition.timestep_embedding,
            grid_sizes=grid_sizes,
            x=hidden_states,
            embed0=condition.block_modulation,
            context=condition.text_context,
            cos_sin=cos_sin,
            rope_positions=self.rope_positions,
        )


class SwiftVRPostInfer(WanPostInfer):
    @torch.no_grad()
    def infer(self, x, pre_infer_out):
        return self.unpatchify(x, pre_infer_out.grid_sizes.tuple)


class SwiftVRModel(WanModel):
    """One-step SwiftVR transformer using native LightX2V Wan weights/inference."""

    def __init__(self, model_path, config, device):
        SwiftVRShiftedWindowAttention.configure(
            window_size=config.get("self_attn_window_size", (16, 16)),
            backend=config.get("attention_backend", "torch_sdpa"),
        )
        super().__init__(model_path, config, device, model_type="swiftvr")
        self.set_scheduler(SimpleNamespace())

    def _init_infer_class(self):
        self.pre_infer_class = SwiftVRPreInfer
        self.transformer_infer_class = WanTransformerInfer
        self.post_infer_class = SwiftVRPostInfer

    @torch.no_grad()
    def prepare_condition(self, prompt_embedding: torch.Tensor) -> SwiftVRCondition:
        return self.pre_infer.prepare_condition(self.pre_weight, prompt_embedding)

    @torch.no_grad()
    def predict(self, latents: torch.Tensor, condition: SwiftVRCondition, temporal_offset: int) -> torch.Tensor:
        pre_infer_output = self.pre_infer.infer(self.pre_weight, latents, condition, temporal_offset)
        prediction = self.transformer_infer.infer(self.transformer_weights, pre_infer_output)
        return self.post_infer.infer(prediction, pre_infer_output)[0].unsqueeze(0)
