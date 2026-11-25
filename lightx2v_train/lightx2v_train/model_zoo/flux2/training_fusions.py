"""Optional BF16 MetaX training operators for the Flux2-dev denoiser."""

from functools import partial

import torch
from diffusers.models.transformers import transformer_flux2 as flux_layers
from loguru import logger

from ..qwen_image.modulated_layer_norm import _ModulatedLayerNorm
from ..qwen_image.shared_qkv_input import project_image_qkv
from .training_ops import JointRoPE, can_rotate


FUSIONS = {"joint_rope", "modulated_layer_norm", "shared_qkv_input"}


def _bf16_training(module, states):
    return (
        module.training
        and states.is_cuda
        and states.size(0) == 1
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )


def _rotate(q, k, text_q, text_k, frequencies, fused):
    if fused and can_rotate(q, k, text_q, text_k, frequencies):
        return JointRoPE.apply(q, k, text_q, text_k, *frequencies)
    if text_q is not None:
        q, k = torch.cat((text_q, q), dim=1), torch.cat((text_k, k), dim=1)
    if frequencies is not None:
        q = flux_layers.apply_rotary_emb(q, frequencies, sequence_dim=1)
        k = flux_layers.apply_rotary_emb(k, frequencies, sequence_dim=1)
    return q, k


class _TrainingAttention:
    def __init__(self, original, fusions, deterministic):
        super().__init__()
        self._attention_backend = original._attention_backend
        self._parallel_config = original._parallel_config
        self.fusions = fusions
        self.deterministic = deterministic

    def supported(self, attn, hidden_states, attention_mask):
        return (
            _bf16_training(attn, hidden_states)
            and attention_mask is None
            and self._parallel_config is None
            and (not self.deterministic or self._attention_backend in (None, "native"))
        )

    def attend(self, q, k, v):
        if self.deterministic and q.dtype == k.dtype == v.dtype == torch.bfloat16:
            from flash_attn import flash_attn_func

            result = flash_attn_func(q, k, v, deterministic=True)
        else:
            result = flux_layers.dispatch_attention_fn(q, k, v, backend=self._attention_backend)
        return result.flatten(2, 3).to(q.dtype)


class DoubleStreamTrainingAttention(_TrainingAttention, flux_layers.Flux2AttnProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        if (
            not self.supported(attn, hidden_states, attention_mask)
            or attn.fused_projections
            or attn.added_kv_proj_dim is None
            or encoder_hidden_states is None
        ):
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        projections = (attn.to_q, attn.to_k, attn.to_v)
        q, k, v = project_image_qkv(hidden_states, projections, "shared_qkv_input" in self.fusions)
        text_q = attn.add_q_proj(encoder_hidden_states)
        text_k = attn.add_k_proj(encoder_hidden_states)
        text_v = attn.add_v_proj(encoder_hidden_states)

        q, k, v, text_q, text_k, text_v = (
            tensor.unflatten(-1, (-1, attn.head_dim))
            for tensor in (q, k, v, text_q, text_k, text_v)
        )
        q, k = attn.norm_q(q), attn.norm_k(k)
        text_q, text_k = attn.norm_added_q(text_q), attn.norm_added_k(text_k)
        q, k = _rotate(q, k, text_q, text_k, image_rotary_emb, "joint_rope" in self.fusions)
        result = self.attend(q, k, torch.cat((text_v, v), dim=1))
        text_tokens = encoder_hidden_states.size(1)
        text, image = result.split((text_tokens, result.size(1) - text_tokens), dim=1)
        return attn.to_out[1](attn.to_out[0](image)), attn.to_add_out(text)


class SingleStreamTrainingAttention(_TrainingAttention, flux_layers.Flux2ParallelSelfAttnProcessor):
    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        if not self.supported(attn, hidden_states, attention_mask):
            return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)

        projected = attn.to_qkv_mlp_proj(hidden_states)
        qkv_dim = 3 * attn.inner_dim
        mlp_dim = attn.mlp_hidden_dim * attn.mlp_mult_factor
        local_qkv = projected.size(-1) * qkv_dim // (qkv_dim + mlp_dim)
        qkv, mlp = projected.split((local_qkv, projected.size(-1) - local_qkv), dim=-1)
        q, k, v = (tensor.unflatten(-1, (-1, attn.head_dim)) for tensor in qkv.chunk(3, dim=-1))
        q, k = attn.norm_q(q), attn.norm_k(k)
        q, k = _rotate(q, k, None, None, image_rotary_emb, "joint_rope" in self.fusions)
        return attn.to_out(torch.cat((self.attend(q, k, v), attn.mlp_act_fn(mlp)), dim=-1))


def _modulate(norm, states, shift, scale):
    if (
        type(norm) is torch.nn.LayerNorm
        and norm.weight is None
        and norm.bias is None
        and norm.normalized_shape == (states.size(-1),)
        and not (norm._forward_hooks or norm._forward_pre_hooks or norm._backward_hooks or norm._backward_pre_hooks)
        and not (shift.requires_grad or scale.requires_grad)
        and states.ndim == 3
        and shift.shape == scale.shape == (1, 1, states.size(-1))
        and all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in (states, shift, scale))
    ):
        return _ModulatedLayerNorm.apply(states, scale, shift, norm.eps, torch.float32)
    return norm(states) * (1 + scale) + shift


def _double_modulated(original, block, hidden_states, encoder_hidden_states, temb_mod_img, temb_mod_txt,
                      image_rotary_emb=None, joint_attention_kwargs=None):
    if not _bf16_training(block, hidden_states):
        return original(hidden_states, encoder_hidden_states, temb_mod_img, temb_mod_txt, image_rotary_emb, joint_attention_kwargs)

    (shift_attn, scale_attn, gate_attn), (shift_ff, scale_ff, gate_ff) = flux_layers.Flux2Modulation.split(temb_mod_img, 2)
    (text_shift_attn, text_scale_attn, text_gate_attn), (text_shift_ff, text_scale_ff, text_gate_ff) = flux_layers.Flux2Modulation.split(temb_mod_txt, 2)
    image, text = block.attn(
        hidden_states=_modulate(block.norm1, hidden_states, shift_attn, scale_attn),
        encoder_hidden_states=_modulate(block.norm1_context, encoder_hidden_states, text_shift_attn, text_scale_attn),
        image_rotary_emb=image_rotary_emb,
        **(joint_attention_kwargs or {}),
    )
    hidden_states = hidden_states + gate_attn * image
    hidden_states = hidden_states + gate_ff * block.ff(_modulate(block.norm2, hidden_states, shift_ff, scale_ff))
    encoder_hidden_states = encoder_hidden_states + text_gate_attn * text
    encoder_hidden_states = encoder_hidden_states + text_gate_ff * block.ff_context(
        _modulate(block.norm2_context, encoder_hidden_states, text_shift_ff, text_scale_ff)
    )
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def _single_modulated(original, block, hidden_states, encoder_hidden_states, temb_mod, image_rotary_emb=None,
                      joint_attention_kwargs=None, split_hidden_states=False, text_seq_len=None):
    if not _bf16_training(block, hidden_states) or encoder_hidden_states is not None or split_hidden_states:
        return original(hidden_states, encoder_hidden_states, temb_mod, image_rotary_emb, joint_attention_kwargs, split_hidden_states, text_seq_len)

    shift, scale, gate = flux_layers.Flux2Modulation.split(temb_mod, 1)[0]
    result = block.attn(
        hidden_states=_modulate(block.norm, hidden_states, shift, scale),
        image_rotary_emb=image_rotary_emb,
        **(joint_attention_kwargs or {}),
    )
    result = hidden_states + gate * result
    return result.clip(-65504, 65504) if result.dtype == torch.float16 else result


def configure_training_fusions(transformer, fusions, deterministic):
    if not isinstance(fusions, list) or any(not isinstance(name, str) or name not in FUSIONS for name in fusions):
        raise ValueError(f"flux2_fusions must be a list of names from {sorted(FUSIONS)}.")
    features = frozenset(fusions)
    shared_features = features - {"shared_qkv_input", "modulated_layer_norm"}
    for block in transformer.transformer_blocks:
        if (shared_features or "shared_qkv_input" in features or deterministic) and type(block.attn.processor) is flux_layers.Flux2AttnProcessor:
            block.attn.set_processor(DoubleStreamTrainingAttention(block.attn.processor, features, deterministic))
        if "modulated_layer_norm" in features and type(block) is flux_layers.Flux2TransformerBlock and "forward" not in vars(block):
            block.forward = partial(_double_modulated, block.forward, block)
    for block in transformer.single_transformer_blocks:
        if (shared_features or deterministic) and type(block.attn.processor) is flux_layers.Flux2ParallelSelfAttnProcessor:
            block.attn.set_processor(SingleStreamTrainingAttention(block.attn.processor, features, deterministic))
        if "modulated_layer_norm" in features and type(block) is flux_layers.Flux2SingleTransformerBlock and "forward" not in vars(block):
            block.forward = partial(_single_modulated, block.forward, block)
    logger.info("[train] Flux2 fusions={} deterministic_fa2={}", sorted(features), deterministic)
