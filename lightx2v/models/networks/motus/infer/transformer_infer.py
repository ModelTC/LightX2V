import torch

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.wan.infer.triton_ops import fuse_scale_shift_kernel
from lightx2v.common.ops.norm.triton_ops import apply_rotary_embedding

from ..ops import RegistryAttention


class MotusTransformerInfer(BaseTransformerInfer):
    def __init__(self, adapter, config):
        self.adapter = adapter
        self.config = config
        self.self_attn_1_type = config.get("self_attn_1_type", config.get("attention_type", "flash_attn2"))
        self.self_attn_2_type = config.get("self_attn_2_type", config.get("attention_type", "flash_attn2"))
        self.cross_attn_1_type = config.get("cross_attn_1_type", config.get("attention_type", "flash_attn2"))
        self.modulate_type = config.get("modulate_type", "triton")
        self.rope_type = config.get("rope_type", "triton")
        self.self_attn = RegistryAttention(self.self_attn_1_type)
        self.joint_self_attn = RegistryAttention(self.self_attn_2_type)
        self.cross_attn = RegistryAttention(self.cross_attn_1_type)

    def _get_wan_layer(self, layer_idx):
        return self.adapter.model.video_module.video_model.wan_model.blocks[layer_idx]

    def _get_action_block(self, layer_idx):
        return self.adapter.model.action_expert.blocks[layer_idx]

    def _get_und_block(self, layer_idx):
        return self.adapter.model.und_expert.blocks[layer_idx]

    def _modulate(self, x, modulation):
        scale = modulation[1].squeeze(2)
        shift = modulation[0].squeeze(2)
        x = x.float().contiguous()
        if self.modulate_type == "triton" and x.is_cuda:
            return fuse_scale_shift_kernel(x, scale.contiguous(), shift.contiguous())
        return x * (1 + scale) + shift

    def _video_self_attention_qkv(self, video_tokens, pre_infer_out, layer_idx):
        model = self.adapter.model
        wan_layer = self._get_wan_layer(layer_idx)
        batch, seq_len, dim = video_tokens.shape
        num_heads = model.video_model.wan_model.num_heads
        head_dim = dim // num_heads

        video_q = wan_layer.self_attn.norm_q(wan_layer.self_attn.q(video_tokens)).view(batch, seq_len, num_heads, head_dim)
        video_k = wan_layer.self_attn.norm_k(wan_layer.self_attn.k(video_tokens)).view(batch, seq_len, num_heads, head_dim)
        video_v = wan_layer.self_attn.v(video_tokens).view(batch, seq_len, num_heads, head_dim)
        video_q, video_k = self._apply_video_rope(video_q, video_k, pre_infer_out)
        return video_q, video_k, video_v

    def _apply_video_rope(self, video_q, video_k, pre_infer_out):
        grid_sizes = pre_infer_out.grid_sizes.tensor if hasattr(pre_infer_out.grid_sizes, "tensor") else pre_infer_out.grid_sizes
        if self.rope_type == "triton" and video_q.is_cuda and video_q.dtype == torch.bfloat16 and video_k.dtype == torch.bfloat16:
            q_out = video_q.clone()
            k_out = video_k.clone()
            for batch_idx, grid_size in enumerate(grid_sizes.tolist()):
                grid_size_tuple = tuple(int(v) for v in grid_size)
                seq_len = grid_size_tuple[0] * grid_size_tuple[1] * grid_size_tuple[2]
                cos, sin = self.adapter.get_wan_rotary_cos_sin(grid_size_tuple)
                q_out[batch_idx, :seq_len] = apply_rotary_embedding(q_out[batch_idx, :seq_len].contiguous(), cos, sin)
                k_out[batch_idx, :seq_len] = apply_rotary_embedding(k_out[batch_idx, :seq_len].contiguous(), cos, sin)
            return q_out, k_out

        freqs = self.adapter.get_wan_freqs()
        return self.adapter.rope_apply(video_q, grid_sizes, freqs), self.adapter.rope_apply(video_k, grid_sizes, freqs)

    def _joint_attention(self, pre_infer_out, video_tokens, action_tokens, und_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx):
        model = self.adapter.model
        wan_layer = self._get_wan_layer(layer_idx)
        action_block = self._get_action_block(layer_idx)
        und_block = self._get_und_block(layer_idx)

        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation
        norm_video = self._modulate(wan_layer.norm1(video_tokens), v_mod)
        norm_action = self._modulate(action_block.norm1(action_tokens), a_mod)
        norm_und = und_block.norm1(und_tokens)

        batch, video_len, video_dim = norm_video.shape
        action_len = norm_action.shape[1]
        und_len = norm_und.shape[1]
        num_heads = model.video_model.wan_model.num_heads
        head_dim = video_dim // num_heads

        video_q, video_k, video_v = self._video_self_attention_qkv(norm_video, pre_infer_out, layer_idx)

        action_q, action_k, action_v = action_block.wan_action_qkv_mm(norm_action)
        action_q = action_block.wan_action_norm_q(action_q.flatten(-2)).view(batch, action_len, num_heads, head_dim)
        action_k = action_block.wan_action_norm_k(action_k.flatten(-2)).view(batch, action_len, num_heads, head_dim)

        und_q, und_k, und_v = und_block.wan_und_qkv_mm(norm_und)
        und_q = und_block.wan_und_norm_q(und_q.flatten(-2)).view(batch, und_len, num_heads, head_dim)
        und_k = und_block.wan_und_norm_k(und_k.flatten(-2)).view(batch, und_len, num_heads, head_dim)

        q_all = torch.cat([video_q, action_q, und_q], dim=1)
        k_all = torch.cat([video_k, action_k, und_k], dim=1)
        v_all = torch.cat([video_v, action_v, und_v], dim=1)
        attn_out = self.joint_self_attn(q_all, k_all, v_all)

        video_out = wan_layer.self_attn.o(attn_out[:, :video_len, :])
        action_out = action_block.wan_action_o(attn_out[:, video_len : video_len + action_len, :])
        und_out = und_block.wan_und_o(attn_out[:, video_len + action_len :, :])

        video_tokens = video_tokens + video_out * v_mod[2].squeeze(2)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)
        und_tokens = und_tokens + und_out
        return video_tokens, action_tokens, und_tokens

    def _cross_attention(self, video_tokens, processed_t5_context, layer_idx):
        wan_layer = self._get_wan_layer(layer_idx)
        batch, q_len, dim = video_tokens.shape
        ctx_len = processed_t5_context.shape[1]
        num_heads = wan_layer.cross_attn.num_heads
        head_dim = dim // num_heads

        norm_video = wan_layer.norm3(video_tokens)
        q = wan_layer.cross_attn.norm_q(wan_layer.cross_attn.q(norm_video)).view(batch, q_len, num_heads, head_dim)
        k = wan_layer.cross_attn.norm_k(wan_layer.cross_attn.k(processed_t5_context)).view(batch, ctx_len, num_heads, head_dim)
        v = wan_layer.cross_attn.v(processed_t5_context).view(batch, ctx_len, num_heads, head_dim)
        return video_tokens + wan_layer.cross_attn.o(self.cross_attn(q, k, v))

    def _prepare_action_tokens(self, pre_infer_out, action_latents):
        model = self.adapter.model
        state_tokens = pre_infer_out.state.unsqueeze(1).to(model.dtype)
        registers = model.action_expert.registers
        if registers is not None:
            registers = registers.expand(state_tokens.shape[0], -1, -1)
        action_tokens = model.action_expert.input_encoder(state_tokens, action_latents, registers)
        return state_tokens, action_tokens

    def _prepare_step_embeddings(self, batch_size, video_tokens, action_tokens, timestep):
        model = self.adapter.model
        timestep_scaled = (timestep * 1000).expand(batch_size).to(model.dtype)
        video_head_time_emb, video_adaln_params = model.video_module.get_time_embedding(timestep_scaled, video_tokens.shape[1])
        action_head_time_emb, action_adaln_params = model.action_module.get_time_embedding(timestep_scaled, action_tokens.shape[1])
        return video_head_time_emb, video_adaln_params, action_head_time_emb, action_adaln_params

    def _run_layer(self, pre_infer_out, video_tokens, action_tokens, und_tokens, processed_t5_context, video_adaln_params, action_adaln_params, layer_idx):
        model = self.adapter.model
        video_adaln_modulation = model.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
        action_adaln_modulation = model.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
        video_tokens, action_tokens, und_tokens = self._joint_attention(
            pre_infer_out,
            video_tokens,
            action_tokens,
            und_tokens,
            video_adaln_modulation,
            action_adaln_modulation,
            layer_idx,
        )
        video_tokens = self._cross_attention(video_tokens, processed_t5_context, layer_idx)
        video_tokens = model.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
        action_tokens = model.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
        und_tokens = model.und_module.process_ffn(und_tokens, layer_idx)
        return video_tokens, action_tokens, und_tokens

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        model = self.adapter.model
        scheduler = self.scheduler
        processed_t5_context = pre_infer_out.processed_t5_context
        image_context = pre_infer_out.image_context
        und_tokens_base = pre_infer_out.und_tokens

        for step_index, t, t_next, dt in scheduler.iter_steps():
            scheduler.step_pre(step_index)
            video_tokens = model.video_module.prepare_input(scheduler.video_latents.to(model.dtype))
            state_tokens, action_tokens = self._prepare_action_tokens(pre_infer_out, scheduler.action_latents)
            und_tokens = und_tokens_base.clone()

            with torch.autocast(device_type="cuda", dtype=model.video_model.precision):
                video_head_time_emb, video_adaln_params, action_head_time_emb, action_adaln_params = self._prepare_step_embeddings(
                    state_tokens.shape[0],
                    video_tokens,
                    action_tokens,
                    t,
                )

                for layer_idx in range(model.config.num_layers):
                    video_tokens, action_tokens, und_tokens = self._run_layer(
                        pre_infer_out,
                        video_tokens,
                        action_tokens,
                        und_tokens,
                        processed_t5_context,
                        video_adaln_params,
                        action_adaln_params,
                        layer_idx,
                    )

                video_velocity = model.video_module.apply_output_head(video_tokens, video_head_time_emb)
                action_pred_full = model.action_expert.decoder(action_tokens, action_head_time_emb)
                # in case for the registers is set to 0
                num_regs = model.action_expert.config.num_registers
                action_velocity = action_pred_full[:, 1:-num_regs, :] if num_regs > 0 else action_pred_full[:, 1:, :]

            scheduler.step(video_velocity=video_velocity, action_velocity=action_velocity, dt=dt, condition_frame_latent=pre_infer_out.condition_frame_latent)

        return scheduler.video_latents, scheduler.action_latents
