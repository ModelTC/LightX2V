"""Residual-branch pruning of the released H3 causal video encoder."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoEncoder3d
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..pruning_search import build_residual_search_masks
from .vae_protocol import MiniMaxH3VideoVAE


class SearchLoRAConv3d(nn.Conv3d):
    """Temporary low-rank update with the teacher's causal/spatial padding."""

    def __init__(self, convolution, rank: int, alpha: float):
        nn.Module.__init__(self)
        self.in_channels = convolution.in_channels
        self.out_channels = convolution.out_channels
        self.kernel_size = convolution.kernel_size
        self.stride = convolution.stride
        self.padding = convolution.padding
        self.padding_mode = convolution.padding_mode
        self.dilation = convolution.dilation
        self.groups = convolution.groups
        self.transposed = convolution.transposed
        self.output_padding = convolution.output_padding
        self._reversed_padding_repeated_twice = convolution._reversed_padding_repeated_twice
        self.spatial_padding = convolution.spatial_padding
        self.temporal_padding = convolution.temporal_padding
        self.spatial_padding_mode = convolution.spatial_padding_mode
        self.weight = convolution.weight
        self.bias = convolution.bias
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(convolution.weight.new_empty(rank, convolution.in_channels, *self.kernel_size))
        self.lora_B = nn.Parameter(convolution.weight.new_zeros(convolution.out_channels, rank, 1, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, hidden_states):
        if self.spatial_padding:
            padding = self.spatial_padding
            hidden_states = F.pad(hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode)
        if self.temporal_padding:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0))
        output = F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, dilation=self.dilation)
        update = F.conv3d(hidden_states, self.lora_A, stride=self.stride, dilation=self.dilation)
        return output + F.conv3d(update, self.lora_B) * self.scale


class ResidualShortcut(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv_shortcut = block.conv_shortcut

    def forward(self, hidden_states):
        return hidden_states if self.conv_shortcut is None else self.conv_shortcut(hidden_states)


class GatedResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.norm1 = block.norm1
        self.conv1 = block.conv1
        self.norm2 = block.norm2
        self.conv2 = block.conv2
        self.conv_shortcut = block.conv_shortcut

    def forward(self, hidden_states, gate):
        residual = hidden_states if self.conv_shortcut is None else self.conv_shortcut(hidden_states)
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(F.silu(self.norm2(hidden_states)))
        return residual + gate.to(hidden_states.dtype).view(-1, 1, 1, 1, 1) * hidden_states


def encoder_config(teacher_config: Mapping) -> dict:
    keys = (
        "in_channels", "block_out_channels", "layers_per_block", "spatial_downsample_factors",
        "temporal_downsample_factors", "norm_num_groups", "norm_eps", "spatial_padding_mode",
    )
    return {**{name: teacher_config[name] for name in keys}, "out_channels": 2 * teacher_config["latent_channels"]}


def _run(module, hidden_states, *, gradient_checkpointing=False):
    if torch.is_grad_enabled() and gradient_checkpointing:
        return checkpoint(module, hidden_states, use_reentrant=False)
    return module(hidden_states)


def teacher_encoder_forward(teacher_vae, pixel_clip, *, return_features=False):
    hidden_states = teacher_vae.encoder.conv_in(pixel_clip)
    features = []
    for block in teacher_vae.encoder.down_blocks:
        hidden_states = block(hidden_states)
        if return_features:
            features.append(hidden_states)
    hidden_states = teacher_vae.encoder.conv_out(F.silu(teacher_vae.encoder.norm_out(hidden_states)))
    moments = teacher_vae.quant_conv(hidden_states)
    return (moments, tuple(features)) if return_features else moments


def teacher_encoder_suffix(teacher_vae, stage_feature, stage_index: int, *, gradient_checkpointing=True):
    """A frozen suffix transmits gradients to its post-downsample stage input."""
    if not 0 <= stage_index < len(teacher_vae.encoder.down_blocks):
        raise ValueError("Teacher encoder suffix anchor is outside the encoder stages.")
    hidden_states = stage_feature
    for block in teacher_vae.encoder.down_blocks[stage_index + 1 :]:
        hidden_states = _run(block, hidden_states, gradient_checkpointing=gradient_checkpointing)
    hidden_states = F.silu(teacher_vae.encoder.norm_out(hidden_states))
    hidden_states = _run(teacher_vae.encoder.conv_out, hidden_states, gradient_checkpointing=gradient_checkpointing)
    return _run(teacher_vae.quant_conv, hidden_states, gradient_checkpointing=gradient_checkpointing)


class PrunableVideoEncoder(MiniMaxH3VideoEncoder3d):
    def __init__(self, config, *, kept_residual_indices=None, search=None):
        super().__init__(**config)
        self.search_config = search
        self.gradient_checkpointing = False
        self._search_noise = None
        self.depth = sum(len(stage.resnets) for stage in self.down_blocks)
        self.residual_groups = []
        offset = 0
        for stage in self.down_blocks:
            self.residual_groups.append(tuple(range(offset, offset + len(stage.resnets))))
            offset += len(stage.resnets)
        if search is not None:
            self.search_grouping = search.get("grouping", "global")
            masks = build_residual_search_masks(self.depth, search, self.residual_groups, default_keep=3)
            self.keep_residuals = int(masks[0].sum())
            self.temperature = float(search.get("temperature", 4.0))
            self.gate_scale = float(search.get("gate_scale", 100.0))
            if self.temperature <= 0 or self.gate_scale <= 0:
                raise ValueError("Search temperature and gate_scale must be positive.")
            self.gate_logits = nn.Parameter(torch.zeros(self.depth))
            self.register_buffer("candidate_masks", masks, persistent=False)
        kept = set(range(self.depth) if kept_residual_indices is None else kept_residual_indices)
        index = 0
        for stage in self.down_blocks:
            for local_index, block in enumerate(stage.resnets):
                if search is not None:
                    stage.resnets[local_index] = GatedResidualBlock(block)
                elif index not in kept:
                    stage.resnets[local_index] = ResidualShortcut(block)
                index += 1

    def configure_search_trainable(self):
        self.requires_grad_(False)
        rank = int(self.search_config.get("lora_rank", 16))
        if rank < 1:
            raise ValueError("Search lora_rank must be positive.")
        alpha = float(self.search_config.get("lora_alpha", 2 * rank))
        for stage in self.down_blocks:
            for block in stage.resnets:
                for name in ("conv1", "conv2"):
                    convolution = getattr(block, name)
                    if not isinstance(convolution, SearchLoRAConv3d):
                        setattr(block, name, SearchLoRAConv3d(convolution, rank, alpha))
                    getattr(block, name).lora_A.requires_grad_(True)
                    getattr(block, name).lora_B.requires_grad_(True)
        self.gate_logits.requires_grad_(True)

    def prepare_search_step(self, batch_size: int):
        if self.search_config is not None:
            self._search_noise = -torch.empty(
                batch_size, len(self.candidate_masks), device=self.gate_logits.device,
            ).exponential_().log()

    def clear_search_step(self):
        self._search_noise = None

    def selected_layers(self, logits=None):
        values = self.gate_logits if logits is None else logits
        masks = self.candidate_masks.to(device=values.device, dtype=torch.float32)
        scores = (values.detach().float().unsqueeze(0) * masks).sum(-1)
        return masks[scores.argmax()].nonzero().flatten().cpu().tolist()

    def _search_mask(self, batch_size: int):
        candidates = self.candidate_masks.float()
        scores = (self.gate_logits.float().unsqueeze(0) * candidates).sum(-1) * self.gate_scale
        scores = scores.unsqueeze(0).expand(batch_size, -1)
        if not self.training:
            return self.candidate_masks[scores.argmax(-1)]
        noise = self._search_noise
        if noise is None:
            noise = -torch.empty_like(scores).exponential_().log()
        elif noise.shape[0] != batch_size:
            raise ValueError("prepare_search_step batch size differs from the encoder clip batch.")
        soft = ((scores + noise) / self.temperature).softmax(-1)
        hard = F.one_hot(soft.argmax(-1), len(self.candidate_masks)).to(soft.dtype)
        weights = hard - soft.detach() + soft
        return (weights.unsqueeze(-1) * candidates.unsqueeze(0)).sum(-2)

    def forward(self, pixel_clip, *, return_features=False, auxiliary_index=None):
        if auxiliary_index is not None and not 0 <= auxiliary_index < len(self.down_blocks):
            raise ValueError("Auxiliary feature index is outside the encoder stages.")
        hidden_states = _run(self.conv_in, pixel_clip, gradient_checkpointing=self.gradient_checkpointing)
        mask = self._search_mask(pixel_clip.shape[0]) if self.search_config is not None else None
        features, auxiliary_feature = [], None
        residual_index = 0
        for stage_index, stage in enumerate(self.down_blocks):
            for block in stage.resnets:
                if mask is None:
                    hidden_states = _run(block, hidden_states, gradient_checkpointing=self.gradient_checkpointing)
                elif torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = checkpoint(block, hidden_states, mask[:, residual_index], use_reentrant=False)
                else:
                    hidden_states = block(hidden_states, mask[:, residual_index])
                residual_index += 1
            if stage.downsamplers is not None:
                for downsampler in stage.downsamplers:
                    hidden_states = _run(downsampler, hidden_states, gradient_checkpointing=self.gradient_checkpointing)
            if return_features:
                features.append(hidden_states)
            if stage_index == auxiliary_index:
                auxiliary_feature = hidden_states
        hidden_states = F.silu(self.norm_out(hidden_states))
        hidden_states = _run(self.conv_out, hidden_states, gradient_checkpointing=self.gradient_checkpointing)
        return hidden_states, tuple(features), auxiliary_feature


class MiniMaxH3PrunedVideoEncoder(nn.Module):
    config_name = "pruned_encoder_config.json"
    weights_name = "minimax_h3_pruned_encoder.safetensors"
    blend = staticmethod(MiniMaxH3VideoVAE.blend)
    stitch_tiles = MiniMaxH3VideoVAE.stitch_tiles

    def __init__(
        self, teacher_config: dict, *, kept_residual_indices=None, search=None, use_tiling=True,
        tile_sample_min_height=256, tile_sample_min_width=256,
        tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64,
    ):
        super().__init__()
        depth = len(teacher_config["block_out_channels"]) * teacher_config["layers_per_block"]
        if search is not None and kept_residual_indices is not None:
            raise ValueError("Encoder search starts from all teacher residual branches.")
        kept = list(range(depth)) if kept_residual_indices is None else list(kept_residual_indices)
        if kept != sorted(set(kept)) or any(index < 0 or index >= depth for index in kept):
            raise ValueError("kept_residual_indices must be unique, in range, and in teacher order.")
        self.kept_residual_indices = kept
        self.search_config = search
        self.spatial_compression_ratio = math.prod(teacher_config["spatial_downsample_factors"])
        self.temporal_compression_ratio = math.prod(teacher_config["temporal_downsample_factors"])
        self.clip_length = teacher_config["clip_length"]
        self.token_drop = teacher_config["token_drop"]
        self.encoder = PrunableVideoEncoder(encoder_config(teacher_config), kept_residual_indices=kept, search=search)
        self.latent_channels = teacher_config["latent_channels"]
        self.quant_conv = nn.Conv3d(2 * self.latent_channels, 2 * self.latent_channels, 1)
        self.register_buffer("latents_mean", torch.tensor(teacher_config["latents_mean"], dtype=torch.float32))
        self.register_buffer("latents_std", torch.tensor(teacher_config["latents_std"], dtype=torch.float32))
        self.use_tiling = use_tiling
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_sample_min_overlap_height = tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = tile_sample_min_overlap_width
        self.architecture_config = {
            "teacher_config": dict(teacher_config),
            "kept_residual_indices": kept if search is None else None,
            "search": dict(search) if search is not None else None,
            "use_tiling": use_tiling,
            "tile_sample_min_height": tile_sample_min_height,
            "tile_sample_min_width": tile_sample_min_width,
            "tile_sample_min_overlap_height": tile_sample_min_overlap_height,
            "tile_sample_min_overlap_width": tile_sample_min_overlap_width,
        }
        if search is not None:
            self.configure_search_trainable()

    @property
    def pruning_module(self):
        return self.encoder

    def forward(self, pixel_clip, *, return_features=False, auxiliary_index=None):
        hidden_states, features, auxiliary = self.encoder(
            pixel_clip, return_features=return_features, auxiliary_index=auxiliary_index,
        )
        moments = _run(self.quant_conv, hidden_states, gradient_checkpointing=self.encoder.gradient_checkpointing)
        if auxiliary_index is not None:
            return moments, features, auxiliary
        return (moments, features) if return_features else moments

    def normalize_latents(self, latents):
        mean = self.latents_mean.float().view(1, -1, 1, 1, 1)
        std = self.latents_std.float().view(1, -1, 1, 1, 1)
        return (latents.float() - mean) / std

    def denormalize_latents(self, latents):
        return latents.float() * self.latents_std.float().view(1, -1, 1, 1, 1) + self.latents_mean.float().view(1, -1, 1, 1, 1)

    def initialize_from_teacher_state(self, teacher_state: Mapping[str, torch.Tensor]):
        target = self.state_dict()
        copied = {
            name: teacher_state[name]
            for name in target
            if name not in ("latents_mean", "latents_std") and not name.endswith(("lora_A", "lora_B", "gate_logits"))
        }
        self.load_state_dict({**target, **copied}, strict=True)

    def initialize_from_teacher(self, teacher_vae):
        self.initialize_from_teacher_state(teacher_vae.state_dict())

    def configure_search_trainable(self):
        self.requires_grad_(False)
        self.encoder.configure_search_trainable()

    def enable_gradient_checkpointing(self):
        self.encoder.gradient_checkpointing = True

    def prepare_search_step(self, batch_size):
        self.encoder.prepare_search_step(batch_size)

    def clear_search_step(self):
        self.encoder.clear_search_step()

    def selected_layers(self, logits=None):
        return self.encoder.selected_layers(logits) if self.search_config is not None else list(self.kept_residual_indices)

    def fsdp_modules(self):
        return tuple(
            resnet for stage in self.encoder.down_blocks for resnet in stage.resnets
            if next(resnet.parameters(), None) is not None
        )

    def save_config(self, save_directory):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        with (save_directory / self.config_name).open("w", encoding="utf-8") as handle:
            json.dump(self.architecture_config, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def save_pretrained(self, save_directory, *, state_dict=None, safe_serialization=True):
        if not safe_serialization:
            raise ValueError("MiniMax-H3 encoder checkpoints use safetensors.")
        self.save_config(save_directory)
        weights = self.state_dict() if state_dict is None else state_dict
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in weights.items()},
            str(Path(save_directory) / self.weights_name), metadata={"format": "pt"},
        )

    @classmethod
    def from_pretrained(cls, model_path, *, device=None, dtype=None):
        model_path = Path(model_path)
        with (model_path / cls.config_name).open("r", encoding="utf-8") as handle:
            model = cls(**json.load(handle))
        model.load_state_dict(load_file(str(model_path / cls.weights_name), device="cpu"), strict=True)
        return model.to(device=device, dtype=dtype)
