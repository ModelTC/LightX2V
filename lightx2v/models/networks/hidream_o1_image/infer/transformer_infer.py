import torch

from lightx2v.models.networks.hidream_o1_image.infer.module_io import HidreamTransformerInferOutput
from lightx2v.models.networks.hidream_o1_image.qwen3_vl import apply_rotary_pos_emb


class HidreamO1ImageTransformerInfer:
    def __init__(self, config):
        self.config = config

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, block_weights, pre_infer_out, dtype):
        device = block_weights.device
        device_type = torch.device(device).type
        position_ids = pre_infer_out.position_ids.to(device)
        token_types = pre_infer_out.token_types.to(device)

        with torch.autocast(device_type, dtype=dtype, cache_enabled=False):
            position_embeddings = self._position_embeddings(block_weights, pre_infer_out.inputs_embeds, position_ids)
            hidden_states = pre_infer_out.inputs_embeds
            for layer_idx, decoder_block in enumerate(block_weights.blocks):
                hidden_states = self._infer_decoder_block(decoder_block, hidden_states, position_embeddings, token_types)
                if pre_infer_out.deepstack_visual_embeds is not None and pre_infer_out.visual_pos_masks is not None and layer_idx < len(pre_infer_out.deepstack_visual_embeds):
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        pre_infer_out.visual_pos_masks,
                        pre_infer_out.deepstack_visual_embeds[layer_idx],
                    )
            hidden_states = block_weights.norm.apply(hidden_states)
        return HidreamTransformerInferOutput(
            hidden_states=hidden_states,
            vinput_mask=pre_infer_out.vinput_mask,
            tgt_image_len=pre_infer_out.tgt_image_len,
        )

    def _position_embeddings(self, block_weights, inputs_embeds, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        return block_weights.rotary_emb(inputs_embeds, position_ids)

    def _infer_decoder_block(self, weights, hidden_states, position_embeddings, token_types):
        residual = hidden_states
        normed = weights.input_layernorm.apply(hidden_states)
        attn_output = self._infer_self_attn(weights, normed, position_embeddings, token_types)
        hidden_states = residual + attn_output

        residual = hidden_states
        normed = weights.post_attention_layernorm.apply(hidden_states)
        mlp_output = self._infer_mlp(weights, normed)
        return residual + mlp_output

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        return hidden_states

    def _infer_self_attn(self, weights, hidden_states, position_embeddings, token_types):
        batch, seq_len, _ = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        q = weights.q_proj.apply(flat_hidden).reshape(batch, seq_len, weights.heads, weights.head_dim)
        k = weights.k_proj.apply(flat_hidden).reshape(batch, seq_len, weights.kv_heads, weights.head_dim)
        v = weights.v_proj.apply(flat_hidden).reshape(batch, seq_len, weights.kv_heads, weights.head_dim)
        q = weights.q_norm.apply(q)
        k = weights.k_norm.apply(k)

        cos, sin = position_embeddings
        q_rope = q.transpose(1, 2)
        k_rope = k.transpose(1, 2)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        q = q_rope.transpose(1, 2).contiguous()
        k = k_rope.transpose(1, 2).contiguous()

        attn_output = self._two_pass_attn(weights, q, k, v, token_types)
        attn_output = attn_output.reshape(-1, attn_output.shape[-1])
        attn_output = weights.o_proj.apply(attn_output)
        return attn_output.reshape(batch, seq_len, -1)

    def _two_pass_attn(self, weights, q, k, v, token_types):
        if q.shape[0] != 1:
            raise NotImplementedError("HiDream common-op attention currently expects batch=1 CFG forwards")
        k = self._repeat_kv(k, weights.heads)
        v = self._repeat_kv(v, weights.heads)
        idx_ar = torch.nonzero(~token_types[0].bool(), as_tuple=False).squeeze(-1)
        q_ar = q[0, idx_ar].contiguous()
        k_ar = k[0, idx_ar].contiguous()
        v_ar = v[0, idx_ar].contiguous()
        out_ar = weights.attn.apply(q_ar, k_ar, v_ar, causal=True, max_seqlen_q=q_ar.shape[0], max_seqlen_kv=k_ar.shape[0], model_cls="hidream_o1_image")
        out_full = weights.attn.apply(q[0], k[0], v[0], causal=False, max_seqlen_q=q.shape[1], max_seqlen_kv=k.shape[1], model_cls="hidream_o1_image")
        out_full = out_full.clone()
        out_full[idx_ar] = out_ar
        return out_full.unsqueeze(0)

    def _repeat_kv(self, states, target_heads):
        if states.shape[2] == target_heads:
            return states
        repeat = target_heads // states.shape[2]
        return states.repeat_interleave(repeat, dim=2)

    def _infer_mlp(self, weights, hidden_states):
        shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, shape[-1])
        gate = weights.gate_proj.apply(flat_hidden)
        up = weights.up_proj.apply(flat_hidden)
        hidden = weights.act_fn(gate) * up
        hidden = weights.down_proj.apply(hidden)
        return hidden.reshape(*shape)
