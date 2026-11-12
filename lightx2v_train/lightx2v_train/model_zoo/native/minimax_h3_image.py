# The adapted transformer forward is Copyright 2026 The MiniMax and
# HuggingFace Teams, licensed under Apache-2.0:
# https://www.apache.org/licenses/LICENSE-2.0
"""Trainable H3-Image loading and the released T2I sequence/sampling geometry.

The released model has six distinct output heads, not one continuous-flow
head. Off-grid score queries use the head at the start of the enclosing
released interval. This is an explicit distillation adaptation; it does not
turn the six-step checkpoint into an independently validated score teacher.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from safetensors import safe_open

from .minimax_h3.packing import patchify_video_latents

DIFFUSERS_REVISION = "875373ac7f5856071e345a3e394c74efccc6b9e6"
HEAD_POLICY = "released_interval_start"


def transformer_class():
    try:
        from diffusers import MiniMaxH3ImageTransformer2DModel
    except ImportError as error:
        raise ImportError(f"H3-Image requires Diffusers with MiniMaxH3ImageTransformer2DModel; install git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_REVISION}.") from error
    return MiniMaxH3ImageTransformer2DModel


def resolve_transformer_dir(model_path):
    path = Path(model_path).expanduser().resolve()
    if (path / "transformer_diffusers" / "config.json").is_file():
        path = path / "transformer_diffusers"
    with (path / "config.json").open() as handle:
        config = json.load(handle)
    if config.get("_class_name") != "MiniMaxH3ImageTransformer2DModel":
        raise ValueError(f"{path} is not the H3-Image Diffusers transformer; use transformer_diffusers, not the AV H3 or native transformer directory.")
    return path


def original_image_head_state(model_path):
    """Read only the retained native output head, not another 33B transformer.

    The public release guarantees its six-step fused-head sampler. The retained
    single head enables an explicit experimental continuous-teacher path; it
    is not a separately released undistilled/SFT checkpoint.
    """
    root = Path(model_path).expanduser().resolve()
    if root.name == "transformer_diffusers":
        root = root.parent
    native = root / "transformer"
    index = json.loads((native / "model.safetensors.index.json").read_text())["weight_map"]
    values = {}
    for name in ("weight", "bias"):
        key = f"final_layer.video_out.{name}"
        with safe_open(str(native / index[key]), framework="pt", device="cpu") as handle:
            values[f"proj_out.{name}"] = handle.get_tensor(key)
    return values


def load_transformer(model_path, *, torch_dtype, meta=False, attention_backend=None, image_head_mode="released"):
    if image_head_mode not in {"released", "base"}:
        raise ValueError("image_head_mode must be 'released' or the experimental retained-single-head 'base'.")
    path = resolve_transformer_dir(model_path)
    cls = transformer_class()
    if meta:
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch_dtype)
            with torch.device("meta"):
                transformer = cls.from_config(cls.load_config(str(path), local_files_only=True))
        finally:
            torch.set_default_dtype(original_dtype)
        # Match from_pretrained's mixed-precision checkpoint loading without
        # materializing a full 33B model on any rank before FSDP sharding.
        for name, module in transformer.named_modules():
            if name and any(pattern in name for pattern in transformer._keep_in_fp32_modules):
                module.to(dtype=torch.float32)
        transformer.register_to_config(_name_or_path=str(path))
    else:
        transformer = cls.from_pretrained(str(path), torch_dtype=torch_dtype, local_files_only=True, low_cpu_mem_usage=True)
    if attention_backend not in (None, "torch_sdpa", "sdpa", "native"):
        transformer.set_attention_backend(attention_backend)
    transformer.pdd_proj_out = PDDHeadBank(transformer.pdd_proj_out)
    transformer._image_head_mode = image_head_mode
    if image_head_mode == "base":
        reference = transformer.pdd_proj_out[0]
        transformer.proj_out = torch.nn.Linear(reference.in_features, reference.out_features, device=reference.weight.device, dtype=torch.float32)
        if not meta:
            transformer.load_state_dict(original_image_head_state(model_path), strict=False)
    transformer.forward = MethodType(_image_forward, transformer)
    return transformer.eval()


class PDDHeadBank(torch.nn.Module):
    """One FSDP unit for rank-local head selection, retaining checkpoint keys."""

    def __init__(self, heads):
        super().__init__()
        for index, head in enumerate(heads):
            self.add_module(str(index), head)

    def __getitem__(self, index):
        return self._modules[str(index)]

    def forward(self, hidden_states, index):
        prediction = self[index](hidden_states)
        if torch.is_grad_enabled():
            # PyTorch 2.8 FSDP only reduces parameters with non-None gradients.
            # Different ranks query different PDD intervals. Give each unused
            # head a zero gradient while the whole bank is still unsharded,
            # so every rank reduces the same ordered list of parameters.
            zero = prediction.new_zeros(())
            for name, head in self._modules.items():
                if int(name) != index:
                    zero = zero + head.weight.flatten()[0] * 0 + head.bias[0] * 0
            prediction = prediction + zero
        return prediction


def _image_forward(
    self, hidden_states, encoder_hidden_states, timestep, timestep_indices, token_tags, position_ids, image_indices, text_indices, pdd_step_index, return_dict=True, return_features_block=None
):
    """Official H3-Image forward with one FSDP-safe call to the PDD head bank.

    Adapted from diffusers@875373ac7f5856071e345a3e394c74efccc6b9e6,
    transformer_minimax_h3_image.py (MiniMax/HuggingFace, Apache-2.0).
    Only the final head invocation changes; the T2I numerical path is intact.
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.modeling_utils import get_parameter_dtype

    if return_features_block is not None and (
        isinstance(return_features_block, bool) or int(return_features_block) != return_features_block or not 0 <= return_features_block < len(self.transformer_blocks)
    ):
        raise ValueError(f"Invalid H3-Image critic feature block {return_features_block!r}.")
    rotary_emb = self.rope(position_ids)
    image_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
    text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
    text_embeds = self.token_refiner(text_embeds)
    hidden_states = text_embeds.new_zeros((text_embeds.shape[0], position_ids.shape[0], text_embeds.shape[-1]))
    hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
    hidden_states = hidden_states.index_copy(1, image_indices, image_embeds.to(text_embeds.dtype))
    temb = self.time_embedder(self.time_proj(timestep).to(get_parameter_dtype(self.time_embedder)))
    adaln_indices = timestep_indices * 3 + token_tags
    for index, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(block, hidden_states, temb, adaln_indices, rotary_emb)
        else:
            hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)
        if index == return_features_block:
            # The critic consumes only generated image tokens. Return through
            # the normal module call so FSDP hooks and checkpoint recomputation
            # remain active, including input gradients with a frozen fake model.
            return (hidden_states.index_select(1, image_indices),)
    projection = self.proj_out if self._image_head_mode == "base" else self.pdd_proj_out
    hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(projection))
    output = (projection(hidden_states) if self._image_head_mode == "base" else projection(hidden_states, pdd_step_index)).index_select(1, image_indices)
    return Transformer2DModelOutput(sample=output) if return_dict else (output,)


@dataclass(frozen=True)
class PDDHeadSchedule:
    sigmas: tuple[float, ...]
    policy: str = HEAD_POLICY

    @classmethod
    def from_model(cls, model_path):
        root = Path(model_path).expanduser().resolve()
        if not (root / "pdd_config.json").is_file():
            root = root.parent
        with (root / "pdd_config.json").open() as handle:
            config = json.load(handle)
        sigmas = tuple(float(value) for value in config["sigmas"])
        if int(config["num_inference_steps"]) != len(sigmas) - 1:
            raise ValueError("H3-Image pdd_config step count does not match its sigma grid.")
        return cls(sigmas)

    def __post_init__(self):
        if self.policy != HEAD_POLICY or len(self.sigmas) < 2 or self.sigmas[0] != 1 or self.sigmas[-1] != 0 or any(not left > right for left, right in zip(self.sigmas, self.sigmas[1:])):
            raise ValueError("H3-Image requires a strictly descending released sigma grid from 1 to 0 and released_interval_start head selection.")

    def physical_sigmas(self, *, device=None):
        # The official sampler rounds through float32 (1 - sigma) and back.
        timesteps = torch.tensor([1.0 - value for value in self.sigmas[:-1]], device=device, dtype=torch.float32)
        return torch.cat((1 - timesteps, timesteps.new_zeros(1)))

    def head_index(self, sigma):
        value = torch.as_tensor(sigma, dtype=torch.float32).reshape(-1)
        if value.numel() != 1 or not bool(torch.isfinite(value).all()) or not 0 <= float(value[0]) <= 1:
            raise ValueError("H3-Image requires one finite physical noise sigma in [0, 1].")
        boundaries = self.physical_sigmas(device=value.device)[1:-1]
        return int(torch.bucketize(-value[0], -boundaries, right=True).item())

    def metadata(self):
        return {"released_sigmas": list(self.sigmas), "head_policy": self.policy, "score_off_grid_is_experimental": True}


def packed_t2i_layout(text_length, latent_height, latent_width, *, patch_size=(1, 2, 2), device=None):
    """Official [text | generated image] layout; no image-edit reference rows."""
    pt, ph, pw = patch_size
    if text_length < 1 or pt != 1 or latent_height % ph or latent_width % pw:
        raise ValueError("Invalid H3-Image text length, latent spatial size or patch size.")
    area = np.sqrt(latent_height * latent_width)

    def axis(size, patch):
        ratio = size / area
        left = (1 - ratio) / 2
        return torch.from_numpy(np.linspace(left, left + ratio, size // patch, endpoint=False) * 32)

    hh, ww = torch.meshgrid(axis(latent_height, ph), axis(latent_width, pw), indexing="ij")
    image_rows = hh.numel()
    positions = torch.zeros((text_length + image_rows, 3), dtype=torch.float64)
    positions[:text_length, 0] = torch.arange(text_length)
    positions[text_length:, 0] = text_length
    positions[text_length:, 1] = hh.flatten()
    positions[text_length:, 2] = ww.flatten()
    tags = torch.zeros(text_length + image_rows, dtype=torch.long)
    tags[:text_length] = 1
    return {
        "position_ids": positions.to(device),
        "token_tags": tags.to(device),
        "text_indices": torch.arange(text_length, device=device),
        "image_indices": torch.arange(text_length, text_length + image_rows, device=device),
        "timestep_indices": torch.zeros(text_length + image_rows, dtype=torch.long, device=device),
    }


def unpatchify_video_latents(rows, shape, patch_size=(1, 2, 2)):
    batch, channels, frames, height, width = shape
    pt, ph, pw = patch_size
    values = rows.reshape(batch, frames // pt, height // ph, width // pw, channels, pt, ph, pw)
    return values.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(shape)


__all__ = ["PDDHeadSchedule", "load_transformer", "packed_t2i_layout", "patchify_video_latents", "resolve_transformer_dir", "unpatchify_video_latents"]
