import torch
import torch.nn.functional as F

from lightx2v.models.networks.hunyuan3d.infer.module_io import Hunyuan3DPreInferOutput
from lightx2v.models.networks.hunyuan3d.infer.utils import apply_timesteps_embedding


class Hunyuan3DPreInfer:
    """Pre-processing for Hunyuan3D shape DiT: timestep + latent embed + token concat."""

    def __init__(self, config):
        self.config = config
        hidden_size = config["hidden_size"]
        self.scheduler = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, cond, timestep, guidance_cond=None):
        t_freq = apply_timesteps_embedding(timestep, self.config["hidden_size"])
        weight_dtype = weights.t_embedder_mlp_0.weight.dtype
        t_freq = t_freq.to(dtype=weight_dtype)
        if guidance_cond is not None and weights.t_embedder_cond_proj is not None:
            t_freq = t_freq + weights.t_embedder_cond_proj.apply(guidance_cond)

        t_emb = weights.t_embedder_mlp_0.apply(t_freq)
        t_emb = F.gelu(t_emb)
        c = weights.t_embedder_mlp_2.apply(t_emb).unsqueeze(dim=1)

        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        x = weights.x_embedder.apply(x)
        x = x.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)

        if self.config.get("with_decoupled_ca", False):
            raise NotImplementedError("with_decoupled_ca is not implemented in Hunyuan3DPreInfer")

        hidden_states = torch.cat([c, x], dim=1)
        return Hunyuan3DPreInferOutput(hidden_states=hidden_states, cond=cond, c=c)
