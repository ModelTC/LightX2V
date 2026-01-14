import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer


def apply_rotary_emb(x, freqs_cis, sequence_dim=1):
    """Apply rotary position embedding to query/key tensors.

    Follows the diffusers implementation for LongCat/Flux.

    Args:
        x: Input tensor [B, H, S, D] where H=heads, S=seq_len, D=head_dim
        freqs_cis: Tuple of (cos, sin) each [S, D]
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

    def _ada_layer_norm_zero(self, hidden_states, temb, norm_linear):
        """AdaLayerNormZero: compute shift, scale, gate from temb.

        For double-stream blocks: returns (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        # Linear projection of silu(temb)
        emb = norm_linear.apply(F.silu(temb))
        # Split into 6 components: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)

        # Apply layer norm and modulation
        norm_hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        return norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _ada_layer_norm_zero_single(self, hidden_states, temb, norm_linear):
        """AdaLayerNormZeroSingle: for single-stream blocks.

        Returns (norm_hidden_states, gate)
        """
        # Linear projection of silu(temb)
        emb = norm_linear.apply(F.silu(temb))
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

        Note:
            Current implementation only supports batch_size=1. The unsqueeze(0) operations
            add a batch dimension for attention computation.
        """
        heads = self.config["num_attention_heads"]
        head_dim = self.config["attention_head_dim"]

        # ===== Image stream: norm1 =====
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._ada_layer_norm_zero(hidden_states, temb, block_weights.norm1_linear)

        # ===== Text stream: norm1_context =====
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self._ada_layer_norm_zero(encoder_hidden_states, temb, block_weights.norm1_context_linear)

        # ===== Attention projections =====
        # Image stream QKV
        img_query = block_weights.to_q.apply(norm_hidden_states)
        img_key = block_weights.to_k.apply(norm_hidden_states)
        img_value = block_weights.to_v.apply(norm_hidden_states)

        # Text stream QKV (added projections)
        txt_query = block_weights.add_q_proj.apply(norm_encoder_hidden_states)
        txt_key = block_weights.add_k_proj.apply(norm_encoder_hidden_states)
        txt_value = block_weights.add_v_proj.apply(norm_encoder_hidden_states)

        # Reshape for multi-head attention: [L, D] -> [L, heads, head_dim]
        img_query = img_query.unflatten(-1, (heads, head_dim))
        img_key = img_key.unflatten(-1, (heads, head_dim))
        img_value = img_value.unflatten(-1, (heads, head_dim))
        txt_query = txt_query.unflatten(-1, (heads, head_dim))
        txt_key = txt_key.unflatten(-1, (heads, head_dim))
        txt_value = txt_value.unflatten(-1, (heads, head_dim))

        # RMSNorm on Q/K
        img_query = block_weights.norm_q.apply(img_query)
        img_key = block_weights.norm_k.apply(img_key)
        txt_query = block_weights.norm_added_q.apply(txt_query)
        txt_key = block_weights.norm_added_k.apply(txt_key)

        # Concatenate [text, image] for joint attention: [L_txt + L_img, heads, head_dim]
        query = torch.cat([txt_query, img_query], dim=0)
        key = torch.cat([txt_key, img_key], dim=0)
        value = torch.cat([txt_value, img_value], dim=0)

        # Apply rotary embedding: [L, H, D] -> [1, L, H, D] -> apply rope -> [L, H, D]
        query = apply_rotary_emb(query.unsqueeze(0), image_rotary_emb, sequence_dim=1).squeeze(0)
        key = apply_rotary_emb(key.unsqueeze(0), image_rotary_emb, sequence_dim=1).squeeze(0)

        # Calculate cu_seqlens for flash attention (batch_size=1)
        total_len = query.shape[0]
        cu_seqlens = torch.tensor([0, total_len], dtype=torch.int32, device=query.device)

        # Use registered attention module
        attn_output = block_weights.calculate.apply(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=total_len,
            max_seqlen_kv=total_len,
            model_cls="longcat_image",
        )

        # Split back to text and image
        txt_len = encoder_hidden_states.shape[0]
        txt_attn_output = attn_output[:txt_len]
        img_attn_output = attn_output[txt_len:]

        # Output projections
        img_attn_output = block_weights.to_out.apply(img_attn_output)
        txt_attn_output = block_weights.to_add_out.apply(txt_attn_output)

        # Apply gates and residual
        hidden_states = hidden_states + gate_msa * img_attn_output
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * txt_attn_output

        # ===== FFN for image stream =====
        # Layer norm without learnable parameters (LongCat/Flux architecture)
        norm_hidden_states2 = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
        norm_hidden_states2 = norm_hidden_states2 * (1 + scale_mlp) + shift_mlp
        ff_output = block_weights.ff_net_0_proj.apply(norm_hidden_states2)
        ff_output = F.gelu(ff_output, approximate="tanh")
        ff_output = block_weights.ff_net_2.apply(ff_output)
        hidden_states = hidden_states + gate_mlp * ff_output

        # ===== FFN for text stream =====
        # Layer norm without learnable parameters (LongCat/Flux architecture)
        norm_encoder_hidden_states2 = F.layer_norm(encoder_hidden_states, (encoder_hidden_states.shape[-1],))
        norm_encoder_hidden_states2 = norm_encoder_hidden_states2 * (1 + c_scale_mlp) + c_shift_mlp
        context_ff_output = block_weights.ff_context_net_0_proj.apply(norm_encoder_hidden_states2)
        context_ff_output = F.gelu(context_ff_output, approximate="tanh")
        context_ff_output = block_weights.ff_context_net_2.apply(context_ff_output)
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

        Note:
            Current implementation only supports batch_size=1. The unsqueeze(0) operations
            add a batch dimension for attention computation.
        """
        heads = self.config["num_attention_heads"]
        head_dim = self.config["attention_head_dim"]

        txt_len = encoder_hidden_states.shape[0]

        # Concatenate text and image
        combined = torch.cat([encoder_hidden_states, hidden_states], dim=0)
        residual = combined

        # AdaLayerNormZeroSingle
        norm_combined, gate = self._ada_layer_norm_zero_single(combined, temb, block_weights.norm_linear)

        # MLP branch
        mlp_hidden_states = block_weights.proj_mlp.apply(norm_combined)
        mlp_hidden_states = F.gelu(mlp_hidden_states, approximate="tanh")

        # Attention projections
        query = block_weights.to_q.apply(norm_combined)
        key = block_weights.to_k.apply(norm_combined)
        value = block_weights.to_v.apply(norm_combined)

        # Reshape for multi-head attention
        query = query.unflatten(-1, (heads, head_dim))
        key = key.unflatten(-1, (heads, head_dim))
        value = value.unflatten(-1, (heads, head_dim))

        # RMSNorm on Q/K
        query = block_weights.norm_q.apply(query)
        key = block_weights.norm_k.apply(key)

        # Apply rotary embedding: [L, H, D] -> [1, L, H, D] -> apply rope -> [L, H, D]
        query = apply_rotary_emb(query.unsqueeze(0), image_rotary_emb, sequence_dim=1).squeeze(0)
        key = apply_rotary_emb(key.unsqueeze(0), image_rotary_emb, sequence_dim=1).squeeze(0)

        # Calculate cu_seqlens for flash attention (batch_size=1)
        total_len = query.shape[0]
        cu_seqlens = torch.tensor([0, total_len], dtype=torch.int32, device=query.device)

        # Use registered attention module
        attn_output = block_weights.calculate.apply(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=total_len,
            max_seqlen_kv=total_len,
            model_cls="longcat_image",
        )

        # Concatenate attention output and MLP output, then project
        combined_output = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        combined_output = block_weights.proj_out.apply(combined_output)

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

        # For I2I task: concatenate output latents with input image latents
        output_seq_len = None
        if pre_infer_out.input_image_latents is not None:
            output_seq_len = pre_infer_out.output_seq_len
            hidden_states = torch.cat([hidden_states, pre_infer_out.input_image_latents], dim=0)

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

        # For I2I task: only return output image latents (not input image latents)
        if output_seq_len is not None:
            hidden_states = hidden_states[:output_seq_len]

        return hidden_states
