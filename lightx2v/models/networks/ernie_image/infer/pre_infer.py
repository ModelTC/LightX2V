import math

import torch
import torch.nn.functional as F

from .module_io import ErnieImagePreInferOutput


def get_ernie_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def _rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    return torch.einsum("...n,d->...nd", pos, omega).float()


class ErnieImagePreInfer:
    def __init__(self, config):
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.patch_size = config.get("patch_size", 1)
        self.head_dim = config["hidden_size"] // config["num_attention_heads"]
        self.rope_theta = config.get("rope_theta", 256)
        self.rope_axes_dim = config.get("rope_axes_dim", (32, 48, 48))
        self.rotary_dim = sum(self.rope_axes_dim)
        self.rope = None
        self.clear_rope_cache()

    def clear_rope_cache(self):
        self._rope_cache = {}

    def set_rope(self, rope):
        self.rope = rope
        self.clear_rope_cache()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.clear_rope_cache()

    def prepare_rope_cache(self, image_hw, text_len, device):
        if self.rope is None:
            raise RuntimeError("ErnieImagePreInfer RoPE is not initialized.")

        rotary_pos_emb = self._pos_embed(self._build_position_ids(image_hw, text_len, device))
        raw_freqs = (rotary_pos_emb.cos().squeeze(1), rotary_pos_emb.sin().squeeze(1))
        rotary_freqs = self.rope.prepare_freqs(raw_freqs, rotary_dim=self.rotary_dim)
        rotary_positions = self.rope.prepare_positions(rotary_freqs)
        return rotary_freqs, rotary_positions

    def get_rope_cache(self, image_hw, text_len, device):
        cache_key = (*image_hw, text_len, str(device))
        cached = self._rope_cache.get(cache_key)
        if cached is None:
            cached = self.prepare_rope_cache(image_hw, text_len, device)
            self._rope_cache[cache_key] = cached
        return cached

    def _pos_embed(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([_rope(ids[..., i], self.rope_axes_dim[i], self.rope_theta) for i in range(3)], dim=-1)
        emb = emb.unsqueeze(1)
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)

    def _build_position_ids(self, image_hw, text_len, device):
        height, width = image_hw
        grid_yx = torch.stack(
            torch.meshgrid(
                torch.arange(height, device=device, dtype=torch.float32),
                torch.arange(width, device=device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        image_ids = torch.cat(
            [
                torch.full((height * width, 1), float(text_len), device=device),
                grid_yx,
            ],
            dim=-1,
        )
        text_ids = torch.cat(
            [
                torch.arange(text_len, device=device, dtype=torch.float32).view(text_len, 1),
                torch.zeros((text_len, 2), device=device),
            ],
            dim=-1,
        )
        return torch.cat([image_ids, text_ids], dim=0)

    def infer(self, weights, hidden_states, encoder_hidden_states):
        device = hidden_states.device
        dtype = hidden_states.dtype

        image_tokens = weights.x_embedder.apply(hidden_states)
        _, dim, height, width = image_tokens.shape
        image_tokens = image_tokens.reshape(1, dim, height * width).transpose(1, 2).squeeze(0).contiguous()

        if encoder_hidden_states.dim() == 3:
            encoder_hidden_states = encoder_hidden_states.squeeze(0)
        encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=dtype)
        if getattr(weights, "text_proj", None) is not None:
            encoder_hidden_states = weights.text_proj.apply(encoder_hidden_states)

        hidden_states = torch.cat([image_tokens, encoder_hidden_states], dim=0)
        text_len = encoder_hidden_states.shape[0]
        rotary_freqs, rotary_positions = self.get_rope_cache((height, width), text_len, device)

        timestep = self.scheduler.timestep.reshape(1).to(device=device)
        time_emb = get_ernie_timestep_embedding(
            timestep,
            self.hidden_size,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
            scale=1,
        ).to(dtype=dtype)
        conditioning = weights.time_embedding_linear_1.apply(time_emb)
        conditioning = F.silu(conditioning)
        conditioning = weights.time_embedding_linear_2.apply(conditioning)

        adaln = weights.adaln_modulation.apply(F.silu(conditioning)).chunk(6, dim=-1)
        temb = tuple(item.squeeze(0).contiguous() for item in adaln)

        return ErnieImagePreInferOutput(
            hidden_states=hidden_states,
            image_tokens_len=height * width,
            image_hw=(height, width),
            rotary_freqs=rotary_freqs,
            rotary_positions=rotary_positions,
            temb=temb,
            conditioning=conditioning,
        )
