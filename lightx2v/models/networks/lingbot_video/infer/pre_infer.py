import math

import torch
import torch.nn.functional as F

from lightx2v.models.networks.lingbot_video.infer.module_io import LingBotVideoPreInferOutput
from lightx2v.models.networks.lingbot_video.infer.utils import (
    LingBotVideoRotaryEmbedding,
    get_timestep_embedding,
    make_joint_position_ids,
)
from lightx2v.utils.envs import GET_DTYPE


class LingBotVideoPreInfer:
    def __init__(self, config):
        self.config = config
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        self.freq_dim = int(config.get("freq_dim", 256))
        self.rope = LingBotVideoRotaryEmbedding(
            tuple(config.get("axes_dims", (32, 48, 48))),
            tuple(config.get("axes_lens", (4096, 512, 512))),
            float(config.get("rope_theta", 256.0)),
        )

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _patchify(self, hidden_states):
        if hidden_states.ndim != 5 or hidden_states.shape[0] != 1:
            raise ValueError(f"LingBot-Video expects latent shape [1,C,T,H,W], got {tuple(hidden_states.shape)}")
        batch, channels, frames, height, width = hidden_states.shape
        p_f, p_h, p_w = self.patch_size
        if frames % p_f or height % p_h or width % p_w:
            raise ValueError(f"Latent shape {(frames, height, width)} is not divisible by patch size {self.patch_size}")

        grid_t, grid_h, grid_w = frames // p_f, height // p_h, width // p_w
        patch_tokens = hidden_states.reshape(batch, channels, grid_t, p_f, grid_h, p_h, grid_w, p_w)
        patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
            grid_t * grid_h * grid_w,
            math.prod(self.patch_size) * channels,
        )
        return patch_tokens, grid_t, grid_h, grid_w

    def infer(self, weights, hidden_states, encoder_hidden_states):
        patch_tokens, grid_t, grid_h, grid_w = self._patchify(hidden_states)
        bulk_dtype = GET_DTYPE()
        patch_tokens = patch_tokens.to(dtype=bulk_dtype)
        video_tokens = weights.patch_embedder.apply(patch_tokens)

        text_tokens = encoder_hidden_states.squeeze(0).to(dtype=bulk_dtype)
        text_tokens = weights.text_norm.apply(text_tokens)
        text_tokens = weights.text_linear_2.apply(F.silu(weights.text_linear_1.apply(text_tokens)))

        hidden_states = torch.cat([video_tokens, text_tokens], dim=0)
        text_len = text_tokens.shape[0]
        n_video = video_tokens.shape[0]
        rotary_ids = make_joint_position_ids(text_len, grid_t, grid_h, grid_w, hidden_states.device)
        rotary_emb = self.rope(rotary_ids)

        timestep_proj = get_timestep_embedding(
            self.scheduler.current_timestep,
            embedding_dim=self.freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1,
        )
        t_emb = weights.time_linear_2.apply(F.silu(weights.time_linear_1.apply(timestep_proj.float())))
        temb_input = t_emb.expand(hidden_states.shape[0], -1).contiguous()
        temb6 = weights.time_modulation.apply(F.silu(temb_input.float()))

        return LingBotVideoPreInferOutput(
            hidden_states=hidden_states,
            rotary_emb=rotary_emb,
            temb6=temb6,
            temb_input=temb_input,
            n_video=n_video,
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            latent_shape=tuple(hidden_states.shape),
        )
