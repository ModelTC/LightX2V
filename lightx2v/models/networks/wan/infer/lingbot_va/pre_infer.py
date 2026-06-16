import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.utils import sinusoidal_embedding_1d
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

from .module_io import LingbotVAPreInferOutput


class LingbotVAPreInfer(WanPreInfer):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        self.theta = 10000.0
        self._rope_cache = {}

    def _patchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = latents.shape
        p_t, p_h, p_w = self.patch_size
        if f % p_t != 0 or h % p_h != 0 or w % p_w != 0:
            raise ValueError(f"latent shape {tuple(latents.shape)} is not divisible by patch_size={self.patch_size}")
        latents = latents.reshape(b, c, f // p_t, p_t, h // p_h, p_h, w // p_w, p_w)
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return latents.reshape(b, -1, c * p_t * p_h * p_w).squeeze(0)

    @staticmethod
    def _flatten_actions(actions: torch.Tensor) -> torch.Tensor:
        return actions.permute(0, 2, 3, 4, 1).contiguous().reshape(actions.shape[0], -1, actions.shape[1]).squeeze(0)

    def _rope_base(self, dim: int, device: torch.device) -> torch.Tensor:
        key = (dim, str(device))
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        base = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float64)[: (dim // 2)] / dim))
        self._rope_cache[key] = base
        return base

    def _rotary_emb(self, grid_id: torch.Tensor) -> torch.Tensor:
        head_dim = self.config["dim"] // self.config["num_heads"]
        f_dim = head_dim - 2 * (head_dim // 3)
        h_dim = head_dim // 3
        w_dim = head_dim // 3
        grid_id = grid_id.to(torch.device(AI_DEVICE)).squeeze(0)
        f_freqs = grid_id[0].unsqueeze(-1) * self._rope_base(f_dim, grid_id.device)
        h_freqs = grid_id[1].unsqueeze(-1) * self._rope_base(h_dim, grid_id.device)
        w_freqs = grid_id[2].unsqueeze(-1) * self._rope_base(w_dim, grid_id.device)
        freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
        return torch.polar(torch.ones_like(freqs), freqs)[:, None]

    def _time_embed(self, weights, timesteps: torch.Tensor, noisy_latents: torch.Tensor, action_mode: bool, dtype: torch.dtype):
        timesteps = timesteps.to(noisy_latents.device).squeeze(0)
        patch_scale_h, patch_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        repeat = (noisy_latents.shape[-2] // patch_scale_h) * (noisy_latents.shape[-1] // patch_scale_w)
        latent_time_steps = torch.repeat_interleave(timesteps, repeat, dim=0)

        embed = sinusoidal_embedding_1d(self.freq_dim, latent_time_steps)
        if action_mode:
            embed = weights.action_time_embedding_0.apply(embed.to(GET_SENSITIVE_DTYPE() if GET_SENSITIVE_DTYPE() != GET_DTYPE() else embed.dtype))
            embed = F.silu(embed)
            temb = weights.action_time_embedding_2.apply(embed)
            timestep_proj = weights.action_time_projection.apply(F.silu(temb))
        else:
            embed = weights.time_embedding_0.apply(embed.to(GET_SENSITIVE_DTYPE() if GET_SENSITIVE_DTYPE() != GET_DTYPE() else embed.dtype))
            embed = F.silu(embed)
            temb = weights.time_embedding_2.apply(embed)
            timestep_proj = weights.time_projection_1.apply(F.silu(temb))

        temb = temb.to(dtype=dtype)
        timestep_proj = timestep_proj.reshape(latent_time_steps.shape[0], 6, -1).to(dtype=dtype)
        return temb, timestep_proj

    @torch.no_grad()
    def infer(self, weights, inputs, action_mode=False):
        noisy_latents = inputs["noisy_latents"].to(torch.device(AI_DEVICE)).to(GET_DTYPE())
        if action_mode:
            x = self._flatten_actions(noisy_latents)
            x = weights.action_embedder.apply(x)
            grid_shape = tuple(noisy_latents.shape[-3:])
        else:
            x = self._patchify_latents(noisy_latents)
            x = weights.patch_embedding_mlp.apply(x)
            grid_shape = (
                noisy_latents.shape[-3] // self.patch_size[0],
                noisy_latents.shape[-2] // self.patch_size[1],
                noisy_latents.shape[-1] // self.patch_size[2],
            )

        text_emb = inputs["text_emb"].to(x.device).to(GET_DTYPE()).squeeze(0)
        context = weights.text_embedding_0.apply(text_emb.to(GET_SENSITIVE_DTYPE() if GET_SENSITIVE_DTYPE() != GET_DTYPE() else text_emb.dtype))
        context = F.gelu(context, approximate="tanh")
        context = weights.text_embedding_2.apply(context)

        temb, timestep_proj = self._time_embed(
            weights,
            inputs["timesteps"],
            noisy_latents,
            action_mode=action_mode,
            dtype=x.dtype,
        )
        rotary_emb = self._rotary_emb(inputs["grid_id"])

        return LingbotVAPreInferOutput(
            x=x,
            context=context,
            temb=temb,
            timestep_proj=timestep_proj,
            rotary_emb=rotary_emb,
            grid_shape=grid_shape,
            action_mode=action_mode,
        )
