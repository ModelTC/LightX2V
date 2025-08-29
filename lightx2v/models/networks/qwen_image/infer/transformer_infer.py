import torch

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer


class QwenImageTransformerInfer(BaseTransformerInfer):
    def __init__(self, config, blocks):
        self.config = config
        self.blocks = blocks
        self.infer_conditional = True
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.infer_func = self.infer_calculating

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer_block(self, block, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb, joint_attention_kwargs):
        # Get modulation parameters for both streams
        img_mod_params = block.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = block.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = block.img_norm1(hidden_states)
        img_modulated, img_gate1 = block._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = block.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = block._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = block.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = block.img_norm2(hidden_states)
        img_modulated2, img_gate2 = block._modulate(img_normed2, img_mod2)
        img_mlp_output = block.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = block.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = block._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = block.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def infer_calculating(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs):
        for index_block, block in enumerate(self.blocks):
            encoder_hidden_states, hidden_states = self.infer_block(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )
        return encoder_hidden_states, hidden_states

    def infer(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, pre_infer_out, attention_kwargs):
        _, temb, image_rotary_emb = pre_infer_out
        encoder_hidden_states, hidden_states = self.infer_func(hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs)
        return encoder_hidden_states, hidden_states
