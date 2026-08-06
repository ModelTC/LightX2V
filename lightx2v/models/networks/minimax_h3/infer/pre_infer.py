import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3PreInferOutput
from lightx2v.utils.envs import GET_DTYPE


def timestep_embedding(timesteps: torch.Tensor, embedding_dim: int = 256) -> torch.Tensor:
    """Diffusers Timesteps(..., flip_sin_to_cos=True, shift=0), reproduced locally."""
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be one-dimensional, got {tuple(timesteps.shape)}")
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim
    phases = timesteps[:, None].float() * torch.exp(exponent)[None]
    embedding = torch.cat((torch.cos(phases), torch.sin(phases)), dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class MiniMaxH3PreInfer:
    def __init__(self, config):
        self.config = config
        global_num_heads = int(config.get("num_attention_heads", 56))
        if config.get("tensor_parallel", False):
            tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
            tp_size = dist.get_world_size(tp_group)
        else:
            tp_size = 1
        self.num_heads = global_num_heads // tp_size
        self.head_dim = int(config.get("attention_head_dim", 128))
        self.hidden_size = int(config.get("hidden_size", 5376))
        self.rope_freq_dim = int(config.get("rope_freq_dim", 16))
        self.rope_theta = float(config.get("rope_theta", 10000.0))
        self.freq_dim = int(config.get("freq_dim", 256))

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _attention(self, weights, hidden_states):
        q = weights.to_q.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        k = weights.to_k.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        v = weights.to_v.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        q = weights.norm_q.apply(q)
        k = weights.norm_k.apply(k)
        seq_len = q.shape[0]
        cu_seqlens = torch.tensor((0, seq_len), dtype=torch.int32, device=q.device)
        out = weights.calculate.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
            causal=False,
        )
        return weights.to_out.apply(out.to(GET_DTYPE()))

    @staticmethod
    def _ff(weights, hidden_states):
        value, gate = weights.in_proj.apply(hidden_states).chunk(2, dim=-1)
        return weights.out_proj.apply(value * F.silu(gate))

    def _refine_text(self, weights, text_embeds):
        for block in weights.refiner_blocks:
            text_embeds = text_embeds + self._attention(block.attn, block.norm1.apply(text_embeds))
            text_embeds = text_embeds + self._ff(block.ff, block.norm2.apply(text_embeds))
        return weights.refiner_final_norm.apply(text_embeds)

    def _rotary_embedding(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = position_ids.to(torch.float32)
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    2 * self.rope_freq_dim,
                    2,
                    dtype=torch.float32,
                    device=position_ids.device,
                )
                / (2 * self.rope_freq_dim)
            )
        )
        freqs = position_ids.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
        freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()

    def infer(self, weights, prompt_embeds):
        layout = self.scheduler.layout
        bulk_dtype = GET_DTYPE()

        video_embeds = weights.proj_in.apply(self.scheduler.video_latents.float()).to(bulk_dtype)
        audio_embeds = weights.audio_proj_in.apply(self.scheduler.audio_latents.float()).to(bulk_dtype)
        text_embeds = weights.context_embedder.apply(prompt_embeds.to(bulk_dtype))
        text_embeds = self._refine_text(weights, text_embeds)

        hidden_states = text_embeds.new_zeros((layout.sequence_length, self.hidden_size))
        hidden_states.index_copy_(0, layout.text_indices, text_embeds)
        hidden_states.index_copy_(0, layout.audio_indices, audio_embeds)
        hidden_states.index_copy_(0, layout.video_indices, video_embeds)

        temb = timestep_embedding(self.scheduler.unique_timesteps, self.freq_dim)
        temb = weights.time_linear_2.apply(F.silu(weights.time_linear_1.apply(temb.float())))
        timestep_indices = self.scheduler.timestep_indices
        adaln_indices = timestep_indices * 3 + layout.token_tags.clamp(min=0)

        return MiniMaxH3PreInferOutput(
            hidden_states=hidden_states,
            temb=temb,
            timestep_indices=timestep_indices,
            adaln_indices=adaln_indices,
            rotary_emb=self._rotary_embedding(layout.position_ids),
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
        )
