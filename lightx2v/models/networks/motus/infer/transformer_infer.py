import torch

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer

from ..ops import RegistryAttention


class MotusTransformerInfer(BaseTransformerInfer):
    def __init__(self, adapter, config):
        self.adapter = adapter
        self.config = config
        self.self_attn_1_type = config.get("self_attn_1_type", config.get("attention_type", "flash_attn2"))
        self.self_attn_2_type = config.get("self_attn_2_type", config.get("attention_type", "flash_attn2"))
        self.cross_attn_1_type = config.get("cross_attn_1_type", config.get("attention_type", "flash_attn2"))
        self.self_attn = RegistryAttention(self.self_attn_1_type)
        self.joint_self_attn = RegistryAttention(self.self_attn_2_type)
        self.cross_attn = RegistryAttention(self.cross_attn_1_type)

    def _joint_attention(self, pre_infer_out, video_tokens, action_tokens, und_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx):
        model = self.adapter.model
        wan_layer = model.video_module.video_model.wan_model.blocks[layer_idx]
        action_block = model.action_expert.blocks[layer_idx]
        und_block = model.und_expert.blocks[layer_idx]

        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation
        norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens).float() * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)
        norm_und = und_block.norm1(und_tokens)

        batch, video_len, video_dim = norm_video.shape
        action_len = norm_action.shape[1]
        und_len = norm_und.shape[1]
        num_heads = model.video_model.wan_model.num_heads
        head_dim = video_dim // num_heads

        video_q = wan_layer.self_attn.norm_q(wan_layer.self_attn.q(norm_video)).view(batch, video_len, num_heads, head_dim)
        video_k = wan_layer.self_attn.norm_k(wan_layer.self_attn.k(norm_video)).view(batch, video_len, num_heads, head_dim)
        video_v = wan_layer.self_attn.v(norm_video).view(batch, video_len, num_heads, head_dim)
        freqs = self.adapter.get_wan_freqs()
        video_q = self.adapter.rope_apply(video_q, pre_infer_out.grid_sizes, freqs)
        video_k = self.adapter.rope_apply(video_k, pre_infer_out.grid_sizes, freqs)

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
        wan_layer = self.adapter.model.video_module.video_model.wan_model.blocks[layer_idx]
        batch, q_len, dim = video_tokens.shape
        ctx_len = processed_t5_context.shape[1]
        num_heads = wan_layer.cross_attn.num_heads
        head_dim = dim // num_heads

        norm_video = wan_layer.norm3(video_tokens)
        q = wan_layer.cross_attn.norm_q(wan_layer.cross_attn.q(norm_video)).view(batch, q_len, num_heads, head_dim)
        k = wan_layer.cross_attn.norm_k(wan_layer.cross_attn.k(processed_t5_context)).view(batch, ctx_len, num_heads, head_dim)
        v = wan_layer.cross_attn.v(processed_t5_context).view(batch, ctx_len, num_heads, head_dim)
        return video_tokens + wan_layer.cross_attn.o(self.cross_attn(q, k, v))

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
            state_tokens = pre_infer_out.state.unsqueeze(1).to(model.dtype)
            # in case for the registers is set to 0
            registers = model.action_expert.registers
            if registers is not None:
                registers = registers.expand(state_tokens.shape[0], -1, -1)
            action_tokens = model.action_expert.input_encoder(state_tokens, scheduler.action_latents, registers)
            und_tokens = und_tokens_base.clone()

            video_t_scaled = (t * 1000).expand(state_tokens.shape[0]).to(model.dtype)
            action_t_scaled = (t * 1000).expand(state_tokens.shape[0]).to(model.dtype)

            with torch.autocast(device_type="cuda", dtype=model.video_model.precision):
                video_head_time_emb, video_adaln_params = model.video_module.get_time_embedding(video_t_scaled, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = model.action_module.get_time_embedding(action_t_scaled, action_tokens.shape[1])

                for layer_idx in range(model.config.num_layers):
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

                video_velocity = model.video_module.apply_output_head(video_tokens, video_head_time_emb)
                action_pred_full = model.action_expert.decoder(action_tokens, action_head_time_emb)
                # in case for the registers is set to 0
                num_regs = model.action_expert.config.num_registers
                action_velocity = action_pred_full[:, 1:-num_regs, :] if num_regs > 0 else action_pred_full[:, 1:, :]

            scheduler.step(video_velocity=video_velocity, action_velocity=action_velocity, dt=dt, condition_frame_latent=pre_infer_out.condition_frame_latent)

        return scheduler.video_latents, scheduler.action_latents
