"""Optional MetaX BF16 training fusions for LongCat-Image."""

from types import MethodType

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.models.transformers.transformer_longcat_image import LongCatImageAttnProcessor
from loguru import logger

from lightx2v_train.model_zoo.qwen_image.modulated_layer_norm import _ModulatedLayerNorm
from lightx2v_train.model_zoo.qwen_image.shared_qkv_input import project_image_qkv
from lightx2v_train.runtime.backend import get_backend

from .rotary import LongCatRotary, supports_rotary


def _bf16_autocast(tensor):
    return tensor.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16


def _has_hooks(module):
    return bool(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks)


def _modulate(norm, states, shift, scale):
    if (
        type(norm) is torch.nn.LayerNorm
        and norm.normalized_shape == (states.size(-1),)
        and norm.weight is None
        and norm.bias is None
        and not _has_hooks(norm)
        and not (shift.requires_grad or scale.requires_grad)
        and states.ndim == 3
        and states.size(0) == 1
        and shift.shape == scale.shape == (1, states.size(-1))
        and states.is_contiguous()
        and shift.is_contiguous()
        and scale.is_contiguous()
        and states.dtype == scale.dtype == shift.dtype == torch.bfloat16
    ):
        return _ModulatedLayerNorm.apply(states, scale, shift, norm.eps, torch.float32)
    return norm(states) * (1 + scale[:, None]) + shift[:, None]


def _ada_layer_norm(self, x, timestep=None, class_labels=None, hidden_dtype=None, emb=None):
    if not self.training or not _bf16_autocast(x) or self.emb is not None:
        return AdaLayerNormZero.forward(self, x, timestep, class_labels, hidden_dtype, emb)
    params = self.linear(self.silu(emb))
    shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=1)
    return _modulate(self.norm, x, shift, scale), gate, shift_mlp, scale_mlp, gate_mlp


def _ada_layer_norm_single(self, x, emb=None):
    if not self.training or not _bf16_autocast(x):
        return AdaLayerNormZeroSingle.forward(self, x, emb)
    shift, scale, gate = self.linear(self.silu(emb)).chunk(3, dim=1)
    return _modulate(self.norm, x, shift, scale), gate


class LongCatTrainingAttnProcessor(LongCatImageAttnProcessor):
    def __init__(self, original, fusions):
        super().__init__()
        self._attention_backend = original._attention_backend
        self._parallel_config = original._parallel_config
        self.fusions = fusions

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        if not attn.training or not _bf16_autocast(hidden_states) or attn.fused_projections or self._parallel_config is not None or (encoder_hidden_states is None) != (attn.added_kv_proj_dim is None):
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        def heads(x):
            return x.unflatten(-1, (attn.heads, -1))

        projections = (attn.to_q, attn.to_k, attn.to_v)
        query, key, value = map(heads, project_image_qkv(hidden_states, projections, "shared_qkv_input" in self.fusions))
        query, key = attn.norm_q(query), attn.norm_k(key)
        text_q = text_k = None
        if encoder_hidden_states is not None:
            text_q = attn.norm_added_q(heads(attn.add_q_proj(encoder_hidden_states)))
            text_k = attn.norm_added_k(heads(attn.add_k_proj(encoder_hidden_states)))
            value = torch.cat((heads(attn.add_v_proj(encoder_hidden_states)), value), dim=1)

        if "joint_rope" in self.fusions and supports_rotary(query, key, text_q, text_k, image_rotary_emb):
            query, key = LongCatRotary.apply(query, key, text_q, text_k, *image_rotary_emb)
        else:
            if text_q is not None:
                query, key = torch.cat((text_q, query), dim=1), torch.cat((text_k, key), dim=1)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        result = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, backend=self._attention_backend, parallel_config=self._parallel_config)
        result = result.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is None:
            return result
        text_length = encoder_hidden_states.size(1)
        text, image = result.split((text_length, result.size(1) - text_length), dim=1)
        return attn.to_out[1](attn.to_out[0](image)), attn.to_add_out(text)


def configure_training_fusions(transformer, fusions):
    supported = {"joint_rope", "modulated_layer_norm", "shared_qkv_input"}
    if not isinstance(fusions, list) or any(not isinstance(name, str) or name not in supported for name in fusions):
        raise ValueError(f"longcat_image_fusions must be a list of names from {sorted(supported)}.")
    if get_backend().name != "metax_cuda":
        raise ValueError("longcat_image_fusions require the metax_cuda backend.")
    features = frozenset(fusions)
    for block in (*transformer.transformer_blocks, *transformer.single_transformer_blocks):
        if features - {"modulated_layer_norm"} and type(block.attn.processor) is LongCatImageAttnProcessor:
            block.attn.set_processor(LongCatTrainingAttnProcessor(block.attn.processor, features))
        if "modulated_layer_norm" in features:
            for norm in (getattr(block, name, None) for name in ("norm1", "norm1_context", "norm")):
                if type(norm) is AdaLayerNormZero:
                    norm.forward = MethodType(_ada_layer_norm, norm)
                elif type(norm) is AdaLayerNormZeroSingle:
                    norm.forward = MethodType(_ada_layer_norm_single, norm)
    logger.info("[train] longcat_image_fusions={}", sorted(features))
