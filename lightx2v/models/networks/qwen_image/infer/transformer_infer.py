from typing import Tuple, Union

import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class QwenImageTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.infer_conditional = True
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.infer_func = self.infer_calculating

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _modulate(self, x, mod_params):
        """Apply modulation to input tensor"""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def attn(self, block_weight, hidden_states, encoder_hidden_states, image_rotary_emb):
        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = block_weight.attn.to_q.apply(hidden_states[0])
        img_key = block_weight.attn.to_k.apply(hidden_states[0])
        img_value = block_weight.attn.to_v.apply(hidden_states[0])

        # Compute QKV for text stream (context projections)
        txt_query = block_weight.attn.add_q_proj.apply(encoder_hidden_states[0])
        txt_key = block_weight.attn.add_k_proj.apply(encoder_hidden_states[0])
        txt_value = block_weight.attn.add_v_proj.apply(encoder_hidden_states[0])

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (block_weight.attn.heads, -1))
        img_key = img_key.unflatten(-1, (block_weight.attn.heads, -1))
        img_value = img_value.unflatten(-1, (block_weight.attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (block_weight.attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (block_weight.attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (block_weight.attn.heads, -1))

        # Apply QK normalization
        if block_weight.attn.norm_q is not None:
            img_query = block_weight.attn.norm_q.apply(img_query)
        if block_weight.attn.norm_k is not None:
            img_key = block_weight.attn.norm_k.apply(img_key)
        if block_weight.attn.norm_added_q is not None:
            txt_query = block_weight.attn.norm_added_q.apply(txt_query)
        if block_weight.attn.norm_added_k is not None:
            txt_key = block_weight.attn.norm_added_k.apply(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs1 = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query.unsqueeze(0), img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key.unsqueeze(0), img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query.unsqueeze(0), txt_freqs1, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key.unsqueeze(0), txt_freqs1, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value.unsqueeze(0), img_value.unsqueeze(0)], dim=1)

        # Compute joint attention
        joint_hidden_states = block_weight.attn.torch_sdpa.apply(q=joint_query, k=joint_key, v=joint_value)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = block_weight.attn.to_out.apply(img_attn_output)
        txt_attn_output = block_weight.attn.to_add_out.apply(txt_attn_output)

        return img_attn_output, txt_attn_output

    def infer_block(self, block_weight, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb, joint_attention_kwargs):
        # Get modulation parameters for both streams
        img_mod_params = block_weight.img_mod.apply(F.silu(temb))
        txt_mod_params = block_weight.txt_mod.apply(F.silu(temb))

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        # Process image stream - norm1 + modulation
        img_normed = block_weight.img_norm1.apply(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = block_weight.txt_norm1.apply(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            block_weight=block_weight,
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            image_rotary_emb=image_rotary_emb,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = block_weight.img_norm2.apply(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = F.silu(block_weight.img_mlp.mlp_0.apply(img_modulated2.squeeze(0)))
        img_mlp_output = block_weight.img_mlp.mlp_2.apply(img_mlp_output)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = block_weight.txt_norm2.apply(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = F.silu(block_weight.txt_mlp.mlp_0.apply(txt_modulated2.squeeze(0)))
        txt_mlp_output = block_weight.txt_mlp.mlp_2.apply(txt_mlp_output)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def infer_calculating(self, block_weights, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs):
        for idx in range(len(block_weights.blocks)):
            block_weight = block_weights.blocks[idx]
            encoder_hidden_states, hidden_states = self.infer_block(
                block_weight=block_weight,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )
        return encoder_hidden_states, hidden_states

    def infer(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, pre_infer_out, attention_kwargs, block_weights):
        temb, image_rotary_emb = pre_infer_out
        encoder_hidden_states, hidden_states = self.infer_func(block_weights, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs)
        return encoder_hidden_states, hidden_states
