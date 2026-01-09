import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer


def apply_rotary_emb(x, freqs_cis, use_real=True, sequence_dim=1):
    """Apply rotary position embedding to query/key tensors.

    Follows the diffusers implementation for LongCat/Flux.

    Args:
        x: Input tensor [B, H, S, D] where H=heads, S=seq_len, D=head_dim
        freqs_cis: Tuple of (cos, sin) each [S, D]
        use_real: Whether freqs_cis is real (cos, sin) format
        sequence_dim: Which dimension contains sequence (1 or 2)

    Returns:
        Tensor with rotary embedding applied [B, H, S, D]
    """
    cos, sin = freqs_cis  # [S, D]
    if sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:
        raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

    cos, sin = cos.to(x.device), sin.to(x.device)

    # Split into real and imaginary parts (interleaved format)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class LongCatImageTransformerInfer(BaseTransformerInfer):
    """Transformer inference for LongCat Image model.

    Handles both double-stream blocks (10 layers) and single-stream blocks (20 layers).
    """

    def __init__(self, config):
        self.config = config
        self.infer_conditional = True
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.attn_type = config.get("attn_type", "flash_attn3")

        # Sequence parallel settings
        if self.config.get("seq_parallel", False):
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
            self.seq_p_fp8_comm = self.config["parallel"].get("seq_p_fp8_comm", False)
            self.enable_head_parallel = self.config["parallel"].get("seq_p_head_parallel", False)
        else:
            self.seq_p_group = None
            self.seq_p_fp8_comm = False
            self.enable_head_parallel = False

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _ada_layer_norm_zero(self, hidden_states, temb, linear_weight, linear_bias):
        """AdaLayerNormZero: compute shift, scale, gate from temb.

        For double-stream blocks: returns (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        # Linear projection of silu(temb)
        emb = F.linear(F.silu(temb), linear_weight, linear_bias)
        # Split into 6 components: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)

        # Apply layer norm and modulation
        norm_hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        return norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _ada_layer_norm_zero_single(self, hidden_states, temb, linear_weight, linear_bias):
        """AdaLayerNormZeroSingle: for single-stream blocks.

        Returns (norm_hidden_states, gate)
        """
        # Linear projection of silu(temb)
        emb = F.linear(F.silu(temb), linear_weight, linear_bias)
        # Split into 3 components: shift, scale, gate
        shift, scale, gate = emb.chunk(3, dim=-1)

        # Apply layer norm and modulation
        norm_hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
        norm_hidden_states = norm_hidden_states * (1 + scale) + shift

        return norm_hidden_states, gate

    def infer_double_stream_block(
        self,
        block_weights,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
    ):
        """Inference for a single double-stream transformer block.

        Args:
            block_weights: Weights for this block
            hidden_states: Image stream [L_img, D]
            encoder_hidden_states: Text stream [L_txt, D]
            temb: Timestep embedding [B, D]
            image_rotary_emb: (freqs_cos, freqs_sin) tuple

        Returns:
            Updated (encoder_hidden_states, hidden_states)
        """
        heads = self.config["num_attention_heads"]
        head_dim = self.config["attention_head_dim"]

        # ===== Image stream: norm1 =====
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._ada_layer_norm_zero(
            hidden_states, temb,
            block_weights.norm1_linear_weight,
            block_weights.norm1_linear_bias
        )

        # ===== Text stream: norm1_context =====
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self._ada_layer_norm_zero(
            encoder_hidden_states, temb,
            block_weights.norm1_context_linear_weight,
            block_weights.norm1_context_linear_bias
        )

        # ===== Attention projections =====
        # Image stream QKV
        img_query = F.linear(norm_hidden_states, block_weights.attn_to_q_weight, block_weights.attn_to_q_bias)
        img_key = F.linear(norm_hidden_states, block_weights.attn_to_k_weight, block_weights.attn_to_k_bias)
        img_value = F.linear(norm_hidden_states, block_weights.attn_to_v_weight, block_weights.attn_to_v_bias)

        # Text stream QKV (added projections)
        txt_query = F.linear(norm_encoder_hidden_states, block_weights.attn_add_q_proj_weight, block_weights.attn_add_q_proj_bias)
        txt_key = F.linear(norm_encoder_hidden_states, block_weights.attn_add_k_proj_weight, block_weights.attn_add_k_proj_bias)
        txt_value = F.linear(norm_encoder_hidden_states, block_weights.attn_add_v_proj_weight, block_weights.attn_add_v_proj_bias)

        # Reshape for multi-head attention: [L, D] -> [L, heads, head_dim]
        img_query = img_query.unflatten(-1, (heads, head_dim))
        img_key = img_key.unflatten(-1, (heads, head_dim))
        img_value = img_value.unflatten(-1, (heads, head_dim))
        txt_query = txt_query.unflatten(-1, (heads, head_dim))
        txt_key = txt_key.unflatten(-1, (heads, head_dim))
        txt_value = txt_value.unflatten(-1, (heads, head_dim))

        # RMSNorm on Q/K
        img_query = F.rms_norm(img_query, (head_dim,), block_weights.attn_norm_q_weight)
        img_key = F.rms_norm(img_key, (head_dim,), block_weights.attn_norm_k_weight)
        txt_query = F.rms_norm(txt_query, (head_dim,), block_weights.attn_norm_added_q_weight)
        txt_key = F.rms_norm(txt_key, (head_dim,), block_weights.attn_norm_added_k_weight)

        # Concatenate [text, image] for joint attention: [L_txt + L_img, heads, head_dim]
        query = torch.cat([txt_query, img_query], dim=0)
        key = torch.cat([txt_key, img_key], dim=0)
        value = torch.cat([txt_value, img_value], dim=0)

        # Reshape for rotary embedding: [L, heads, head_dim] -> [1, L, heads, head_dim]
        # Match diffusers layout [B, S, H, D]
        query = query.unsqueeze(0)  # [1, L, H, D]
        key = key.unsqueeze(0)      # [1, L, H, D]
        value = value.unsqueeze(0)  # [1, L, H, D]

        # Apply rotary embedding (expects [B, S, H, D] with sequence_dim=1)
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Transpose for SDPA: [B, S, H, D] -> [B, H, S, D]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(query, key, value)

        # Reshape back: [1, H, L, D] -> [1, L, H, D] -> [L, H*D]
        attn_output = attn_output.transpose(1, 2).squeeze(0).flatten(-2)

        # Split back to text and image
        txt_len = encoder_hidden_states.shape[0]
        txt_attn_output = attn_output[:txt_len]
        img_attn_output = attn_output[txt_len:]

        # Output projections
        img_attn_output = F.linear(img_attn_output, block_weights.attn_to_out_0_weight, block_weights.attn_to_out_0_bias)
        txt_attn_output = F.linear(txt_attn_output, block_weights.attn_to_add_out_weight, block_weights.attn_to_add_out_bias)

        # Apply gates and residual
        hidden_states = hidden_states + gate_msa * img_attn_output
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * txt_attn_output

        # ===== FFN for image stream =====
        # Use norm2 with learned parameters
        norm_hidden_states2 = F.layer_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            weight=block_weights.norm2_weight,
            bias=block_weights.norm2_bias
        )
        norm_hidden_states2 = norm_hidden_states2 * (1 + scale_mlp) + shift_mlp
        ff_output = F.linear(norm_hidden_states2, block_weights.ff_net_0_proj_weight, block_weights.ff_net_0_proj_bias)
        ff_output = F.gelu(ff_output, approximate="tanh")
        ff_output = F.linear(ff_output, block_weights.ff_net_2_weight, block_weights.ff_net_2_bias)
        hidden_states = hidden_states + gate_mlp * ff_output

        # ===== FFN for text stream =====
        # Use norm2_context with learned parameters
        norm_encoder_hidden_states2 = F.layer_norm(
            encoder_hidden_states,
            (encoder_hidden_states.shape[-1],),
            weight=block_weights.norm2_context_weight,
            bias=block_weights.norm2_context_bias
        )
        norm_encoder_hidden_states2 = norm_encoder_hidden_states2 * (1 + c_scale_mlp) + c_shift_mlp
        context_ff_output = F.linear(norm_encoder_hidden_states2, block_weights.ff_context_net_0_proj_weight, block_weights.ff_context_net_0_proj_bias)
        context_ff_output = F.gelu(context_ff_output, approximate="tanh")
        context_ff_output = F.linear(context_ff_output, block_weights.ff_context_net_2_weight, block_weights.ff_context_net_2_bias)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output

        # Clip for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def infer_single_stream_block(
        self,
        block_weights,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
    ):
        """Inference for a single single-stream transformer block.

        Args:
            block_weights: Weights for this block
            hidden_states: Image stream [L_img, D]
            encoder_hidden_states: Text stream [L_txt, D]
            temb: Timestep embedding [B, D]
            image_rotary_emb: (freqs_cos, freqs_sin) tuple

        Returns:
            Updated (encoder_hidden_states, hidden_states)
        """
        heads = self.config["num_attention_heads"]
        head_dim = self.config["attention_head_dim"]

        txt_len = encoder_hidden_states.shape[0]

        # Concatenate text and image
        combined = torch.cat([encoder_hidden_states, hidden_states], dim=0)
        residual = combined

        # AdaLayerNormZeroSingle
        norm_combined, gate = self._ada_layer_norm_zero_single(
            combined, temb,
            block_weights.norm_linear_weight,
            block_weights.norm_linear_bias
        )

        # MLP branch
        mlp_hidden_states = F.linear(norm_combined, block_weights.proj_mlp_weight, block_weights.proj_mlp_bias)
        mlp_hidden_states = F.gelu(mlp_hidden_states, approximate="tanh")

        # Attention projections
        query = F.linear(norm_combined, block_weights.attn_to_q_weight, block_weights.attn_to_q_bias)
        key = F.linear(norm_combined, block_weights.attn_to_k_weight, block_weights.attn_to_k_bias)
        value = F.linear(norm_combined, block_weights.attn_to_v_weight, block_weights.attn_to_v_bias)

        # Reshape for multi-head attention
        query = query.unflatten(-1, (heads, head_dim))
        key = key.unflatten(-1, (heads, head_dim))
        value = value.unflatten(-1, (heads, head_dim))

        # RMSNorm on Q/K
        query = F.rms_norm(query, (head_dim,), block_weights.attn_norm_q_weight)
        key = F.rms_norm(key, (head_dim,), block_weights.attn_norm_k_weight)

        # Reshape for rotary embedding: [L, heads, head_dim] -> [1, L, heads, head_dim]
        # Match diffusers layout [B, S, H, D]
        query = query.unsqueeze(0)  # [1, L, H, D]
        key = key.unsqueeze(0)      # [1, L, H, D]
        value = value.unsqueeze(0)  # [1, L, H, D]

        # Apply rotary embedding (expects [B, S, H, D] with sequence_dim=1)
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Transpose for SDPA: [B, S, H, D] -> [B, H, S, D]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(query, key, value)
        # Reshape back: [1, H, L, D] -> [1, L, H, D] -> [L, H*D]
        attn_output = attn_output.transpose(1, 2).squeeze(0).flatten(-2)

        # Concatenate attention output and MLP output, then project
        combined_output = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        combined_output = F.linear(combined_output, block_weights.proj_out_weight, block_weights.proj_out_bias)

        # Apply gate and residual
        combined = residual + gate * combined_output

        # Clip for fp16
        if combined.dtype == torch.float16:
            combined = combined.clip(-65504, 65504)

        # Split back
        encoder_hidden_states = combined[:txt_len]
        hidden_states = combined[txt_len:]

        return encoder_hidden_states, hidden_states

    def infer(self, block_weights, pre_infer_out):
        """Run transformer inference through all blocks.

        Args:
            block_weights: LongCatImageTransformerWeights containing all block weights
            pre_infer_out: Output from pre-inference stage

        Returns:
            Final hidden states for post-processing
        """
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        temb = pre_infer_out.temb
        image_rotary_emb = pre_infer_out.image_rotary_emb

        # Process double-stream blocks
        for block in block_weights.double_blocks:
            encoder_hidden_states, hidden_states = self.infer_double_stream_block(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
            )

        # Process single-stream blocks
        for block in block_weights.single_blocks:
            encoder_hidden_states, hidden_states = self.infer_single_stream_block(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
            )

        return hidden_states
