import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer

from .utils import apply_rope_with_flashinfer, apply_rope_with_torch


class Flux2KleinTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.infer_conditional = True
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)

        # Calculate inner dimension
        self.inner_dim = config.get("num_attention_heads", 24) * config.get("attention_head_dim", 64)

        # Sequence parallel settings
        if self.config.get("seq_parallel", False):
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
            self.seq_p_fp8_comm = self.config["parallel"].get("seq_p_fp8_comm", False)
            self.seq_p_fp4_comm = self.config["parallel"].get("seq_p_fp4_comm", False)
            self.enable_head_parallel = self.config["parallel"].get("seq_p_head_parallel", False)
        else:
            self.seq_p_group = None
            self.seq_p_fp8_comm = False
            self.seq_p_fp4_comm = False
            self.enable_head_parallel = False

        # RoPE function selection
        rope_funcs = {
            "flashinfer": apply_rope_with_flashinfer,
            "torch": apply_rope_with_torch,
        }
        rope_type = config.get("rope_type", "flashinfer")
        self.apply_rope_func = rope_funcs.get(rope_type, apply_rope_with_torch)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _split_double_modulation(self, mod):
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 6, dim=-1)
        return mod_params[0:3], mod_params[3:6]

    def _split_single_modulation(self, mod):
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3, dim=-1)
        return mod_params

    def infer_double_stream_block(
        self,
        block_weights,
        hidden_states,
        encoder_hidden_states,
        temb_mod_img,
        temb_mod_txt,
        image_rotary_emb,
    ):
        """Inference for a single double-stream transformer block.

        Args:
            block_weights: Weights for this block
            hidden_states: Image stream [L_img, D]
            encoder_hidden_states: Text stream [L_txt, D]
            temb_mod_img: Image stream modulation
            temb_mod_txt: Text stream modulation
            image_rotary_emb: (freqs_cos, freqs_sin) tuple

        Returns:
            Updated (encoder_hidden_states, hidden_states)

        Note:
            Current implementation only supports batch_size=1. The unsqueeze(0) operations
            add a batch dimension for attention computation.
        """
        heads = self.config["num_attention_heads"]
        head_dim = self.config["attention_head_dim"]

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = self._split_double_modulation(temb_mod_img)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = self._split_double_modulation(temb_mod_txt)
        # ===== Image stream: norm1 =====
        norm_hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
        norm_hidden_states = (norm_hidden_states * (1 + scale_msa) + shift_msa).squeeze(0)

        # ===== Text stream: norm1_context =====
        norm_encoder_hidden_states = F.layer_norm(encoder_hidden_states, (encoder_hidden_states.shape[-1],))
        norm_encoder_hidden_states = (norm_encoder_hidden_states * (1 + c_scale_msa) + c_shift_msa).squeeze(0)

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

        # Apply rotary embedding: [L, H, D]
        query, key = self.apply_rope_func(query, key, image_rotary_emb)

        # Calculate cu_seqlens for flash attention (batch_size=1)
        total_len = query.shape[0]
        cu_seqlens = torch.tensor([0, total_len], dtype=torch.int32)

        # Use registered attention module
        if self.seq_p_group is not None:
            txt_len = encoder_hidden_states.shape[0]
            attn_output = block_weights.calculate_parallel.apply(
                q=query,
                k=key,
                v=value,
                slice_qkv_len=txt_len,
                cu_seqlens_qkv=cu_seqlens,
                attention_module=block_weights.calculate,
                seq_p_group=self.seq_p_group,
                use_fp8_comm=self.seq_p_fp8_comm,
                use_fp4_comm=self.seq_p_fp4_comm,
                enable_head_parallel=self.enable_head_parallel,
                img_first=False,
                model_cls="flux2_klein",
            )
        else:
            attn_output = block_weights.calculate.apply(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=total_len,
                max_seqlen_kv=total_len,
                model_cls="flux2_klein",
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
        norm_hidden_states2 = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
        norm_hidden_states2 = (norm_hidden_states2 * (1 + scale_mlp) + shift_mlp).squeeze(0)
        ff_output = block_weights.ff_net_0.apply(norm_hidden_states2)
        ff_1, ff_2 = ff_output.chunk(2, dim=-1)
        ff_output = F.silu(ff_1) * ff_2
        ff_output = block_weights.ff_net_2.apply(ff_output)
        hidden_states = hidden_states + gate_mlp * ff_output

        # ===== FFN for text stream =====
        # Layer norm without learnable parameters (LongCat/Flux architecture)
        norm_encoder_hidden_states2 = F.layer_norm(encoder_hidden_states, (encoder_hidden_states.shape[-1],))
        norm_encoder_hidden_states2 = (norm_encoder_hidden_states2 * (1 + c_scale_mlp) + c_shift_mlp).squeeze(0)
        context_ff_output = block_weights.ff_context_net_0.apply(norm_encoder_hidden_states2)
        ctx_ff_1, ctx_ff_2 = context_ff_output.chunk(2, dim=-1)
        context_ff_output = F.silu(ctx_ff_1) * ctx_ff_2
        context_ff_output = block_weights.ff_context_net_2.apply(context_ff_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        # Clip for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states.squeeze(0), hidden_states.squeeze(0)

    def infer_single_stream_block(
        self,
        block_weights,
        hidden_states,
        encoder_hidden_states,
        temb_mod,
        image_rotary_emb,
        num_txt_tokens=0,
    ):
        """Inference for a single single-stream transformer block.

        Args:
            block_weights: Weights for this block
            hidden_states: Image stream [L_img, D]
            encoder_hidden_states: Text stream [L_txt, D]
            temb_mod: Modulation
            image_rotary_emb: (freqs_cos, freqs_sin) tuple

        Returns:
            Updated (encoder_hidden_states, hidden_states)

        Note:
            Current implementation only supports batch_size=1. The unsqueeze(0) operations
            add a batch dimension for attention computation.
        """
        heads = self.config["num_attention_heads"]
        head_dim = self.config["attention_head_dim"]

        if encoder_hidden_states is not None:
            raise ValueError("Encoder hidden states already cat in hidden states for single-stream blocks in Flux2-Klein, should be None here")

        residual = hidden_states

        # AdaLayerNormZeroSingle
        shift_msa, scale_msa, gate_msa = self._split_single_modulation(temb_mod)

        # Apply layer norm and modulation
        norm_combined = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
        norm_combined = (norm_combined * (1 + scale_msa) + shift_msa).squeeze(0)

        # Parallel QKV and MLP projection
        hidden_states_proj = block_weights.to_qkv_mlp_proj.apply(norm_combined)
        inner_dim = heads * head_dim
        qkv, mlp_hidden_states = torch.split(hidden_states_proj, [3 * inner_dim, hidden_states_proj.shape[-1] - 3 * inner_dim], dim=-1)
        query, key, value = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        query = query.unflatten(-1, (heads, head_dim))
        key = key.unflatten(-1, (heads, head_dim))
        value = value.unflatten(-1, (heads, head_dim))

        query = block_weights.norm_q.apply(query)
        key = block_weights.norm_k.apply(key)

        query, key = self.apply_rope_func(query, key, image_rotary_emb)

        total_len = query.shape[0]
        cu_seqlens = torch.tensor([0, total_len], dtype=torch.int32)

        if self.seq_p_group is not None:
            attn_output = block_weights.calculate_parallel.apply(
                q=query,
                k=key,
                v=value,
                slice_qkv_len=num_txt_tokens,
                cu_seqlens_qkv=cu_seqlens,
                attention_module=block_weights.calculate,
                seq_p_group=self.seq_p_group,
                use_fp8_comm=self.seq_p_fp8_comm,
                use_fp4_comm=self.seq_p_fp4_comm,
                enable_head_parallel=self.enable_head_parallel,
                img_first=False,
                model_cls="flux2_klein",
            )
        else:
            attn_output = block_weights.calculate.apply(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=total_len,
                max_seqlen_kv=total_len,
                model_cls="flux2_klein",
            )

        # SwiGLU MLP activation
        mlp_1, mlp_2 = mlp_hidden_states.chunk(2, dim=-1)
        mlp_hidden_states = F.silu(mlp_1) * mlp_2

        # Concatenate attention output and MLP output, then project
        combined_output = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        combined_output = block_weights.to_out.apply(combined_output)

        # Apply gate and residual
        hidden_states = residual + gate_msa * combined_output
        hidden_states = hidden_states.squeeze(0)
        # Clip for fp16
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        """Run transformer inference through all blocks.
        Maps Flux2KleinPipeline.__call__ transformer logic to inference:
        - Receives embedded latents, text, timestep, and positions from pre_infer
        - Processes through double-stream blocks (8 layers)
        - Processes through single-stream blocks (24 layers)
        - Computes modulation from timestep embedding (internal to transformer)
        Args:
            block_weights: Flux2KleinTransformerWeights containing all block weights
            pre_infer_out: Flux2KleinPreInferModuleOutput from pre-inference stage
        Returns:
            Final hidden states [L, D] for post-processing
        """
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        timestep = pre_infer_out.timestep
        image_rotary_emb = pre_infer_out.image_rotary_emb

        num_txt_tokens = encoder_hidden_states.shape[0]

        if self.seq_p_group is not None and image_rotary_emb is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            cur_rank = dist.get_rank(self.seq_p_group)

            if isinstance(image_rotary_emb, tuple):
                # For torch rope_type: (cos, sin) tuple
                freqs_cos, freqs_sin = image_rotary_emb

                # Split text and image portions
                txt_cos = freqs_cos[:num_txt_tokens]
                img_cos = freqs_cos[num_txt_tokens:]
                txt_sin = freqs_sin[:num_txt_tokens]
                img_sin = freqs_sin[num_txt_tokens:]

                # Pad and chunk only the image portion
                seqlen = img_cos.shape[0]
                padding_size = (world_size - (seqlen % world_size)) % world_size
                if padding_size > 0:
                    img_cos = F.pad(img_cos, (0, 0, 0, padding_size))
                    img_sin = F.pad(img_sin, (0, 0, 0, padding_size))
                img_cos = torch.chunk(img_cos, world_size, dim=0)[cur_rank]
                img_sin = torch.chunk(img_sin, world_size, dim=0)[cur_rank]

                # Concatenate text and image portions back
                freqs_cos = torch.cat([txt_cos, img_cos], dim=0)
                freqs_sin = torch.cat([txt_sin, img_sin], dim=0)
                image_rotary_emb = (freqs_cos, freqs_sin)
            else:
                # For flashinfer rope_type: concatenated tensor
                txt_emb = image_rotary_emb[:num_txt_tokens]
                img_emb = image_rotary_emb[num_txt_tokens:]

                # Pad and chunk only the image portion
                seqlen = img_emb.shape[0]
                padding_size = (world_size - (seqlen % world_size)) % world_size
                if padding_size > 0:
                    img_emb = F.pad(img_emb, (0, 0, 0, padding_size))
                img_emb = torch.chunk(img_emb, world_size, dim=0)[cur_rank]

                # Concatenate text and image portions back
                image_rotary_emb = torch.cat([txt_emb, img_emb], dim=0)

        # Compute modulations from timestep embedding using weight modules
        # timestep shape: [D] (already embedded by pre_infer)
        # modulation_* will be tensors of shape [3*inner_dim] or [6*inner_dim]
        timestep_act = F.silu(timestep)
        double_stream_mod_img = block_weights.double_stream_modulation_img_linear.apply(timestep_act)
        double_stream_mod_txt = block_weights.double_stream_modulation_txt_linear.apply(timestep_act)
        single_stream_mod = block_weights.single_stream_modulation_linear.apply(timestep_act)

        # Process double-stream blocks
        for block in block_weights.double_blocks:
            encoder_hidden_states, hidden_states = self.infer_double_stream_block(
                block,
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                image_rotary_emb,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        # Process single-stream blocks
        for block in block_weights.single_blocks:
            hidden_states = self.infer_single_stream_block(
                block,
                hidden_states,
                None,
                single_stream_mod,
                image_rotary_emb,
                num_txt_tokens=num_txt_tokens,
            )
        hidden_states = hidden_states[num_txt_tokens:, ...]
        return hidden_states
