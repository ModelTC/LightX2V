import torch
from diffusers.models.embeddings import get_3d_sincos_pos_embed, get_timestep_embedding


class CogvideoxPreInfer:
    def __init__(self, config):
        self.config = config
        self.use_positional_embeddings = not self.config.use_rotary_positional_embeddings
        self.inner_dim = self.config.transformer_num_attention_heads * self.config.transformer_attention_head_dim
        self.freq_shift = 0
        self.flip_sin_to_cos = True
        self.scale = 1
        self.act = "silu"

    def _get_positional_embeddings(self, sample_height, sample_width, sample_frames, device):
        post_patch_height = sample_height // self.config.patch_size
        post_patch_width = sample_width // self.config.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.config.transformer_temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.inner_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.config.transformer_spatial_interpolation_scale,
            self.config.transformer_temporal_interpolation_scale,
            device=device,
            output_type="pt",
        )
        pos_embedding = pos_embedding.flatten(0, 1)
        joint_pos_embedding = pos_embedding.new_zeros(1, self.config.text_len + num_patches, self.inner_dim, requires_grad=False)
        joint_pos_embedding.data[:, self.config.text_len :].copy_(pos_embedding)

        return joint_pos_embedding

    def infer(self, weights, hidden_states, timestep, encoder_hidden_states):
        t_emb = get_timestep_embedding(
            timestep,
            self.inner_dim,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.freq_shift,
            scale=self.scale,
        )
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        sample = weights.time_embedding_linear_1.apply(t_emb)
        sample = torch.nn.functional.silu(sample)
        emb = weights.time_embedding_linear_2.apply(sample)

        text_embeds = weights.patch_embed_text_proj.apply(encoder_hidden_states)
        num_frames, channels, height, width = hidden_states.shape
        infer_shapes = (num_frames, channels, height, width)

        p = self.config.patch_size
        p_t = self.config.patch_size_t

        image_embeds = hidden_states.permute(0, 2, 3, 1)
        image_embeds = image_embeds.reshape(num_frames // p_t, p_t, height // p, p, width // p, p, channels)
        image_embeds = image_embeds.permute(0, 2, 4, 6, 1, 3, 5).flatten(3, 6).flatten(0, 2)
        image_embeds = weights.patch_embed_proj.apply(image_embeds)

        embeds = torch.cat([text_embeds, image_embeds], dim=0).contiguous()

        if self.use_positional_embeddings or self.config.transformer_use_learned_positional_embeddings:
            if self.config.transformer_use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.config.transformer_temporal_compression_ratio + 1

            if self.config.transformer_sample_height != height or self.config.transformer_sample_width != width or self.config.transformer_sample_frames != pre_time_compression_frames:
                pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames, device=embeds.device)[0]
            else:
                pos_embedding = self.pos_embedding[0]
            pos_embedding = pos_embedding.to(dtype=embeds.dtype)
            embeds = embeds + pos_embedding

        hidden_states = embeds
        text_seq_length = encoder_hidden_states.shape[0]
        encoder_hidden_states = hidden_states[:text_seq_length, :]
        hidden_states = hidden_states[text_seq_length:, :]

        return hidden_states, encoder_hidden_states, emb, infer_shapes
