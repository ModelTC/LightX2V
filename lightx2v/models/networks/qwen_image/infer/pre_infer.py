import torch

from lightx2v.utils.envs import *

from .module_io import QwenPreInferModuleOutput


class QwenImagePreInfer:
    def __init__(self, config):
        self.config = config
        self.attention_kwargs = {}
        self.cpu_offload = config.get("cpu_offload", False)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, encoder_hidden_states):
        hidden_states = weights.img_in.apply(hidden_states.squeeze(0))

        encoder_hidden_states = weights.txt_norm.apply(encoder_hidden_states.squeeze(0))
        encoder_hidden_states = weights.txt_in.apply(encoder_hidden_states)

        embed0 = weights.time_text_embed_timestep_embedder_linear_1.apply(self.scheduler.timesteps_proj)
        embed0 = torch.nn.functional.silu(embed0)
        embed0 = weights.time_text_embed_timestep_embedder_linear_2.apply(embed0)

        if self.scheduler.infer_condition:
            image_rotary_emb = self.scheduler.image_rotary_emb
        else:
            image_rotary_emb = self.scheduler.negative_image_rotary_emb

        return QwenPreInferModuleOutput(hidden_states=hidden_states.unsqueeze(0), encoder_hidden_states=encoder_hidden_states.unsqueeze(0), embed0=embed0, image_rotary_emb=image_rotary_emb)
