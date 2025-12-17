import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer

from .triton_ops import fuse_scale_shift_kernel
from .utils import apply_rotary_emb_qwen, apply_wan_rope_with_flashinfer


def calculate_q_k_len(q, k_lens):
    q_lens = torch.tensor([q.size(0)], dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
    return cu_seqlens_q, cu_seqlens_k


class QwenImageTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.infer_conditional = True
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.infer_func = self.infer_calculating
        self.attn_type = config.get("attn_type", "flash_attn3")
        self.zero_cond_t = config.get("zero_cond_t", False)
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None
        self.seq_p_fp8_comm = False
        if self.config.get("modulate_type", "triton") == "triton":
            self.modulate_func = fuse_scale_shift_kernel
        else:
            self.modulate_func = lambda x, scale, shift: x * (1 + scale) + shift
        if self.config.get("rope_type", "flashinfer") == "flashinfer":
            self.apply_rope_func = apply_wan_rope_with_flashinfer
        else:
            self.apply_rope_func = apply_rotary_emb_qwen

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _modulate(self, x, mod_params, index=None):
        """Apply modulation to input tensor"""
        # x: b l d, shift: b d, scale: b d, gate: b d
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return self.modulate_func(x, scale_result, shift_result), gate_result

    def apply_attn(self, block_weight, hidden_states, encoder_hidden_states, image_rotary_emb):
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
        img_value = img_value.unflatten(-1, (block_weight.attn.heads, -1)).unsqueeze(0)

        txt_query = txt_query.unflatten(-1, (block_weight.attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (block_weight.attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (block_weight.attn.heads, -1)).unsqueeze(0)

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
        img_freqs, txt_freqs = image_rotary_emb
        img_query, img_key = self.apply_rope_func(img_query, img_key, img_freqs)
        txt_query, txt_key = self.apply_rope_func(txt_query, txt_key, txt_freqs)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=0).unsqueeze(0)
        joint_key = torch.cat([txt_key, img_key], dim=0).unsqueeze(0)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_query = joint_query.squeeze(0)
        joint_key = joint_key.squeeze(0)
        joint_value = joint_value.squeeze(0)

        img_qkv_len = joint_query.shape[0]
        cu_seqlens_qkv = torch.tensor([0, img_qkv_len], dtype=torch.int32, device="cpu").to(joint_query.device, non_blocking=True)

        if self.config["seq_parallel"]:
            txt_qkv_len = txt_query.shape(1)
            joint_hidden_states = block_weight.attn.calculate_parallel.apply(
                q=joint_query,
                k=joint_key,
                v=joint_value,
                slice_qkv_len=txt_qkv_len,
                cu_seqlens_qkv=cu_seqlens_qkv,
                attention_module=block_weight.attn.calculate,
                seq_p_group=self.seq_p_group,
                use_fp8_comm=self.seq_p_fp8_comm,
                model_cls=self.config["model_cls"],
                img_first=False,
            )
        else:
            joint_hidden_states = block_weight.attn.calculate.apply(
                q=joint_query,
                k=joint_key,
                v=joint_value,
                cu_seqlens_q=cu_seqlens_qkv,
                cu_seqlens_kv=cu_seqlens_qkv,
                max_seqlen_q=img_qkv_len,
                max_seqlen_kv=img_qkv_len,
                model_cls="qwen_image",
            )

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = block_weight.attn.to_out.apply(img_attn_output)
        txt_attn_output = block_weight.attn.to_add_out.apply(txt_attn_output)

        return img_attn_output, txt_attn_output

    def infer_block(self, block_weight, hidden_states, encoder_hidden_states, temb, image_rotary_emb, modulate_index=None):
        # Get modulation parameters for both streams
        img_mod_params = block_weight.img_mod.apply(F.silu(temb))
        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = block_weight.txt_mod.apply(F.silu(temb))

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        # Process image stream - norm1 + modulation
        img_normed = block_weight.img_norm1.apply(hidden_states.squeeze(0))
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)

        # Process text stream - norm1 + modulation
        txt_normed = block_weight.txt_norm1.apply(encoder_hidden_states.view(-1, encoder_hidden_states.shape[-1]))
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        attn_output = self.apply_attn(
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
        img_normed2 = block_weight.img_norm2.apply(hidden_states.squeeze(0))
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
        img_mlp_output = F.gelu(block_weight.img_mlp.mlp_0.apply(img_modulated2.squeeze(0)), approximate="tanh")
        img_mlp_output = block_weight.img_mlp.mlp_2.apply(img_mlp_output)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = block_weight.txt_norm2.apply(encoder_hidden_states.squeeze(0))
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = F.gelu(block_weight.txt_mlp.mlp_0.apply(txt_modulated2.squeeze(0)), approximate="tanh")
        txt_mlp_output = block_weight.txt_mlp.mlp_2.apply(txt_mlp_output)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def infer_calculating(self, block_weights, hidden_states, encoder_hidden_states, temb, image_rotary_emb, modulate_index):
        for idx in range(len(block_weights.blocks)):
            block_weight = block_weights.blocks[idx]
            encoder_hidden_states, hidden_states = self.infer_block(
                block_weight=block_weight, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, image_rotary_emb=image_rotary_emb, modulate_index=modulate_index
            )
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        embed0 = pre_infer_out.embed0
        image_rotary_emb = pre_infer_out.image_rotary_emb
        hidden_states = self.infer_func(block_weights, hidden_states, encoder_hidden_states, embed0, image_rotary_emb, self.scheduler.modulate_index)
        return hidden_states
