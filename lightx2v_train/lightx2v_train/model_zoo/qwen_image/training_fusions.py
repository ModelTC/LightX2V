"""Optional MetaX training operators for the Qwen Image double-stream blocks."""

from types import MethodType

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0, QwenImageTransformerBlock, apply_rotary_emb_qwen

from .joint_rope import JointRoPE, supports_joint_rope
from .modulated_layer_norm import forward_with_modulated_norm
from .qk_rms_norm import normalize_image_qk
from .shared_qkv_input import project_image_qkv


def _rotate_and_join(image_q, image_k, text_q, text_k, frequencies, fused):
    if frequencies is not None:
        image_freqs, text_freqs = frequencies
        if fused and supports_joint_rope(image_q, image_k, text_q, text_k, image_freqs, text_freqs):
            return JointRoPE.apply(text_q, text_k, image_q, image_k, text_freqs, image_freqs)
        image_q = apply_rotary_emb_qwen(image_q, image_freqs, use_real=False)
        image_k = apply_rotary_emb_qwen(image_k, image_freqs, use_real=False)
        text_q = apply_rotary_emb_qwen(text_q, text_freqs, use_real=False)
        text_k = apply_rotary_emb_qwen(text_k, text_freqs, use_real=False)
    return torch.cat((text_q, image_q), dim=1), torch.cat((text_k, image_k), dim=1)


class QwenTrainingAttnProcessor(QwenDoubleStreamAttnProcessor2_0):
    def __init__(self, original, fusions, deterministic):
        super().__init__()
        self._attention_backend = original._attention_backend
        self._parallel_config = original._parallel_config
        self.fusions = fusions
        self.deterministic = deterministic

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None, attention_mask=None, image_rotary_emb=None):
        if (
            not attn.training
            or not hidden_states.is_cuda
            or not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.bfloat16
            or attn.heads != 24
            or attn.inner_dim != 3072
            or hidden_states.shape[0] != 1
            or encoder_hidden_states is None
            or encoder_hidden_states_mask is not None
            or attention_mask is not None
            or self._parallel_config is not None
            or self._attention_backend not in (None, "native")
        ):
            return super().__call__(attn, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, attention_mask, image_rotary_emb)

        head_dim = attn.inner_dim // attn.heads
        projections = (attn.to_q, attn.to_k, attn.to_v)
        image_q, image_k, image_v = (t.unflatten(-1, (-1, head_dim)) for t in project_image_qkv(hidden_states, projections, "shared_qkv_input" in self.fusions))
        text_q = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (-1, head_dim))
        text_k = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (-1, head_dim))
        text_v = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (-1, head_dim))

        image_q, image_k = normalize_image_qk(image_q, image_k, attn.norm_q, attn.norm_k, fused="qk_rms_norm" in self.fusions)
        text_q = attn.norm_added_q(text_q) if attn.norm_added_q is not None else text_q
        text_k = attn.norm_added_k(text_k) if attn.norm_added_k is not None else text_k

        query, key = _rotate_and_join(image_q, image_k, text_q, text_k, image_rotary_emb, "joint_rope" in self.fusions)
        value = torch.cat((text_v, image_v), dim=1)
        if self.deterministic:
            from flash_attn import flash_attn_func

            output = flash_attn_func(query, key, value, deterministic=True)
        else:
            output = dispatch_attention_fn(query, key, value, dropout_p=0.0, is_causal=False, backend=self._attention_backend)
        output = output.flatten(2, 3).to(query.dtype)
        text, image = output.split((text_q.shape[1], image_q.shape[1]), dim=1)
        image = attn.to_out[0](image.contiguous())
        if len(attn.to_out) > 1:
            image = attn.to_out[1](image)
        return image, attn.to_add_out(text.contiguous())


def configure_training_fusions(transformer, fusions, deterministic):
    supported = {"qk_rms_norm", "joint_rope", "modulated_layer_norm", "shared_qkv_input"}
    if not isinstance(fusions, list) or any(not isinstance(name, str) or name not in supported for name in fusions):
        raise ValueError(f"qwen_image_fusions must be a list of names from {sorted(supported)}.")
    features = frozenset(fusions)
    for block in transformer.transformer_blocks:
        if type(block) is not QwenImageTransformerBlock:
            continue
        if "modulated_layer_norm" in features and "forward" not in vars(block):
            block.forward = MethodType(forward_with_modulated_norm, block)
        if (deterministic or features - {"modulated_layer_norm"}) and type(block.attn.processor) is QwenDoubleStreamAttnProcessor2_0:
            processor = QwenTrainingAttnProcessor(block.attn.processor, features, deterministic)
            block.attn.set_processor(processor)
