import torch
import torch.nn.functional as F

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

from .module_io import QwenPreInferModuleOutput


class QwenImagePreInfer:
    def __init__(self, config):
        self.config = config
        self.attention_kwargs = {}
        self.cpu_offload = config.get("cpu_offload", False)
        self.zero_cond_t = config.get("zero_cond_t", False)
        self.use_additional_t_cond = config.get("use_additional_t_cond", False)
        self.img_rope = None
        self.txt_rope = None
        self.clear_rope_cache()

    def clear_rope_cache(self):
        self._rope_cache = {True: None, False: None}

    def set_rope(self, img_rope, txt_rope):
        self.img_rope = img_rope
        self.txt_rope = txt_rope
        self.clear_rope_cache()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.clear_rope_cache()

    def prepare_rope_cache(self, image_rotary_emb):
        if self.img_rope is None or self.txt_rope is None:
            raise RuntimeError("QwenImagePreInfer RoPE is not initialized.")

        img_freqs, txt_freqs = image_rotary_emb
        img_freqs = self.img_rope.prepare_freqs(img_freqs)
        txt_freqs = self.txt_rope.prepare_freqs(txt_freqs)
        positions = (
            self.img_rope.prepare_positions(img_freqs),
            self.txt_rope.prepare_positions(txt_freqs),
        )
        return (img_freqs, txt_freqs), positions

    def get_rope_cache(self, image_rotary_emb):
        branch = bool(self.scheduler.infer_condition)
        img_source, txt_source = image_rotary_emb
        cached = self._rope_cache[branch]
        if cached is not None:
            cached_sources, cached_value = cached
            if cached_sources[0] is img_source and cached_sources[1] is txt_source:
                return cached_value

        prepared = self.prepare_rope_cache(image_rotary_emb)
        self._rope_cache[branch] = ((img_source, txt_source), prepared)
        return prepared

    def infer(self, weights, hidden_states, encoder_hidden_states):
        hidden_states = weights.img_in.apply(hidden_states.squeeze(0))

        encoder_hidden_states = weights.txt_norm.apply(encoder_hidden_states.squeeze(0))
        encoder_hidden_states = weights.txt_in.apply(encoder_hidden_states)

        embed0 = weights.time_text_embed_timestep_embedder_linear_1.apply(self.scheduler.timesteps_proj)
        embed0 = torch.nn.functional.silu(embed0)
        embed0 = weights.time_text_embed_timestep_embedder_linear_2.apply(embed0)

        if self.use_additional_t_cond:
            is_rgb = torch.tensor([0] * 1).to(device=AI_DEVICE, dtype=torch.long)
            addition_t_emb = weights.time_text_embed_addition_t_embedding.apply(is_rgb)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            embed0 = embed0 + addition_t_emb

        if self.scheduler.infer_condition:
            image_rotary_emb = self.scheduler.image_rotary_emb
        else:
            image_rotary_emb = self.scheduler.negative_image_rotary_emb
        image_rotary_emb, image_rotary_positions = self.get_rope_cache(image_rotary_emb)

        temb_img_silu = F.silu(embed0)
        if self.zero_cond_t:
            temb_txt_silu = F.silu(torch.chunk(embed0, 2, dim=0)[0])
        else:
            temb_txt_silu = temb_img_silu
        return QwenPreInferModuleOutput(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_img_silu=temb_img_silu,
            temb_txt_silu=temb_txt_silu,
            image_rotary_emb=image_rotary_emb,
            image_rotary_positions=image_rotary_positions,
        )
