import torch
import torch.distributed as dist

from lightx2v.models.networks.hidream_o1_image.infer.module_io import HidreamTransformerInferOutput
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE


class HidreamO1ImageTransformerInfer:
    def __init__(self, config):
        self.config = config
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
            self.seq_p_fp8_comm = self.config["parallel"].get("seq_p_fp8_comm", False)
            self.seq_p_fp4_comm = self.config["parallel"].get("seq_p_fp4_comm", False)
            self.enable_head_parallel = self.config["parallel"].get("seq_p_head_parallel", False)
        else:
            self.seq_p_group = None
            self.seq_p_fp8_comm = False
            self.seq_p_fp4_comm = False
            self.enable_head_parallel = False

    def _apply_rope(self, rope, q, k, rope_cos_sin):
        positions = rope_cos_sin[2] if len(rope_cos_sin) > 2 else None
        return rope.apply(q, k, tuple(rope_cos_sin[:2]), positions=positions, unsqueeze_dim=-2)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, block_weights, pre_infer_out):
        if self.config["seq_parallel"]:
            return self._infer_seq_parallel(block_weights, pre_infer_out)

        hidden_states = pre_infer_out.inputs_embeds
        for layer_idx, decoder_block in enumerate(block_weights.blocks):
            hidden_states = self._infer_decoder_block(decoder_block, hidden_states, pre_infer_out.rope_cos_sin, pre_infer_out.idx_ar)
            if pre_infer_out.deepstack_visual_embeds is not None and pre_infer_out.visual_pos_masks is not None and layer_idx < len(pre_infer_out.deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    pre_infer_out.visual_pos_masks,
                    pre_infer_out.deepstack_visual_embeds[layer_idx],
                )
        hidden_states = block_weights.norm.apply(hidden_states)
        x_vis = self._infer_final_linear(block_weights, hidden_states, pre_infer_out.vinput_mask)
        return HidreamTransformerInferOutput(
            hidden_states=x_vis,
            vinput_mask=pre_infer_out.vinput_mask,
            tgt_image_len=pre_infer_out.tgt_image_len,
        )

    def _infer_seq_parallel(self, block_weights, pre_infer_out):
        hidden_ar = pre_infer_out.inputs_embeds_ar
        hidden_gen = pre_infer_out.inputs_embeds_gen
        for layer_idx, decoder_block in enumerate(block_weights.blocks):
            hidden_ar, hidden_gen = self._infer_decoder_block_seq_parallel(
                decoder_block,
                hidden_ar,
                hidden_gen,
                pre_infer_out.rope_cos_sin_ar,
                pre_infer_out.rope_cos_sin_gen,
            )
            if pre_infer_out.deepstack_visual_embeds is not None and pre_infer_out.visual_pos_masks is not None and layer_idx < len(pre_infer_out.deepstack_visual_embeds):
                hidden_ar = self._deepstack_process(
                    hidden_ar,
                    pre_infer_out.visual_pos_masks,
                    pre_infer_out.deepstack_visual_embeds[layer_idx],
                )
        hidden_gen = block_weights.norm.apply(hidden_gen)
        x_vis = self._infer_final_linear(block_weights, hidden_gen, pre_infer_out.vinput_mask_gen)
        return HidreamTransformerInferOutput(
            hidden_states=x_vis,
            vinput_mask=pre_infer_out.vinput_mask_gen,
            tgt_image_len=pre_infer_out.tgt_image_len,
            seq_p_padding_size=pre_infer_out.seq_p_padding_size,
        )

    def _infer_final_linear(self, weights, hidden_states, vinput_mask):
        hidden_vis = hidden_states[0, vinput_mask[0].to(hidden_states.device)]
        return self._apply_linear(weights.final_linear, hidden_vis).unsqueeze(0)

    def _infer_decoder_block(self, weights, hidden_states, rope_cos_sin, idx_ar):
        residual = hidden_states
        normed = weights.input_layernorm.apply(hidden_states)
        attn_output = self._infer_self_attn(weights, normed, rope_cos_sin, idx_ar)
        hidden_states = residual + attn_output

        residual = hidden_states
        normed = weights.post_attention_layernorm.apply(hidden_states)
        mlp_output = self._infer_mlp(weights, normed)
        return residual + mlp_output

    def _infer_decoder_block_seq_parallel(self, weights, hidden_ar, hidden_gen, rope_ar, rope_gen):
        residual_ar, residual_gen = hidden_ar, hidden_gen
        normed_ar = weights.input_layernorm.apply(hidden_ar)
        normed_gen = weights.input_layernorm.apply(hidden_gen)
        attn_ar, attn_gen = self._infer_self_attn_seq_parallel(weights, normed_ar, normed_gen, rope_ar, rope_gen)
        hidden_ar = residual_ar + attn_ar
        hidden_gen = residual_gen + attn_gen

        residual_ar, residual_gen = hidden_ar, hidden_gen
        hidden_ar = residual_ar + self._infer_mlp(weights, weights.post_attention_layernorm.apply(hidden_ar))
        hidden_gen = residual_gen + self._infer_mlp(weights, weights.post_attention_layernorm.apply(hidden_gen))
        return hidden_ar, hidden_gen

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        return hidden_states

    def _infer_self_attn(self, weights, hidden_states, rope_cos_sin, idx_ar):
        batch, seq_len, _ = hidden_states.shape
        q, k, v = self._project_qkv(weights, hidden_states, rope_cos_sin)
        attn_output = self._two_pass_attn(weights, q, k, v, idx_ar)
        attn_output = attn_output.reshape(-1, attn_output.shape[-1])
        attn_output = self._apply_linear(weights.o_proj, attn_output)
        return attn_output.reshape(batch, seq_len, -1)

    def _infer_self_attn_seq_parallel(self, weights, hidden_ar, hidden_gen, rope_ar, rope_gen):
        world_size = dist.get_world_size(self.seq_p_group)
        if weights.heads % world_size != 0 or weights.kv_heads % world_size != 0:
            raise ValueError(f"HiDream Ulysses requires heads and kv_heads divisible by seq_p_size={world_size}.")

        q_ar, k_ar, v_ar = self._project_qkv(weights, hidden_ar, rope_ar)
        q_gen, k_gen, v_gen = self._project_qkv(weights, hidden_gen, rope_gen)
        softmax_scale = weights.head_dim**-0.5

        out_ar = weights.attn.apply(
            q_ar[0].to(self.infer_dtype),
            k_ar[0].to(self.infer_dtype),
            v_ar[0].to(self.infer_dtype),
            causal=True,
            softmax_scale=softmax_scale,
            max_seqlen_q=q_ar.shape[1],
            max_seqlen_kv=k_ar.shape[1],
            model_cls="hidream_o1_image",
        )

        q_gen = q_gen.to(self.infer_dtype)
        k = torch.cat([k_gen[0], k_ar[0]], dim=0).to(self.infer_dtype)
        v = torch.cat([v_gen[0], v_ar[0]], dim=0).to(self.infer_dtype)
        cu_seqlens_qkv = torch.tensor([0, q_gen.shape[1] + k_ar.shape[1]], dtype=torch.int32, device="cpu")
        out_gen = weights.attn_parallel.apply(
            q=q_gen[0],
            k=k,
            v=v,
            slice_qkv_len=q_gen.shape[1],
            cu_seqlens_qkv=cu_seqlens_qkv,
            attention_module=weights.attn,
            seq_p_group=self.seq_p_group,
            use_fp8_comm=self.seq_p_fp8_comm,
            use_fp4_comm=self.seq_p_fp4_comm,
            enable_head_parallel=self.enable_head_parallel,
            img_first=True,
            q_only_img=True,
            causal=False,
            softmax_scale=softmax_scale,
            model_cls="hidream_o1_image",
        )

        out_ar = self._apply_linear(weights.o_proj, out_ar).reshape(1, q_ar.shape[1], -1)
        out_gen = self._apply_linear(weights.o_proj, out_gen).reshape(1, q_gen.shape[1], -1)
        return out_ar, out_gen

    def _project_qkv(self, weights, hidden_states, rope_cos_sin):
        batch, seq_len, _ = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        q = self._apply_linear(weights.q_proj, flat_hidden).reshape(batch, seq_len, weights.heads, weights.head_dim)
        k = self._apply_linear(weights.k_proj, flat_hidden).reshape(batch, seq_len, weights.kv_heads, weights.head_dim)
        v = self._apply_linear(weights.v_proj, flat_hidden).reshape(batch, seq_len, weights.kv_heads, weights.head_dim)
        q = weights.q_norm.apply(q)
        k = weights.k_norm.apply(k)
        q, k = self._apply_rope(weights.rope, q, k, rope_cos_sin)
        return q, k, v

    def _two_pass_attn(self, weights, q, k, v, idx_ar):
        if q.shape[0] != 1:
            raise NotImplementedError("HiDream common-op attention currently expects batch=1 CFG forwards")
        softmax_scale = weights.head_dim**-0.5
        q_ar = q[0, idx_ar].contiguous()
        k_ar = k[0, idx_ar].contiguous()
        v_ar = v[0, idx_ar].contiguous()
        out_ar = weights.attn.apply(
            q_ar.to(self.infer_dtype),
            k_ar.to(self.infer_dtype),
            v_ar.to(self.infer_dtype),
            causal=True,
            softmax_scale=softmax_scale,
            max_seqlen_q=q_ar.shape[0],
            max_seqlen_kv=k_ar.shape[0],
            model_cls="hidream_o1_image",
        )
        out_full = weights.attn.apply(
            q[0].to(self.infer_dtype),
            k[0].to(self.infer_dtype),
            v[0].to(self.infer_dtype),
            causal=False,
            softmax_scale=softmax_scale,
            max_seqlen_q=q.shape[1],
            max_seqlen_kv=k.shape[1],
            model_cls="hidream_o1_image",
        )
        out_full = out_full.clone()
        out_full[idx_ar] = out_ar
        return out_full.unsqueeze(0)

    def _infer_mlp(self, weights, hidden_states):
        shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, shape[-1])
        gate = self._apply_linear(weights.gate_proj, flat_hidden)
        up = self._apply_linear(weights.up_proj, flat_hidden)
        hidden = weights.act_fn(gate) * up
        hidden = self._apply_linear(weights.down_proj, hidden)
        return hidden.reshape(*shape)

    def _apply_linear(self, weights, hidden_states):
        if self.infer_dtype != self.sensitive_layer_dtype:
            hidden_states = hidden_states.to(self.infer_dtype)
        return weights.apply(hidden_states)
