from types import SimpleNamespace

import torch
import torch.nn.functional as F


def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    freqs = torch.pow(
        10000,
        -torch.arange(half, dtype=torch.float64, device=position.device).div(half),
    )
    sinusoid = torch.outer(position.to(torch.float64), freqs)
    emb = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return emb.to(position.dtype)


def precompute_freqs_cis(dim, end=1024, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).double()[: dim // 2] / dim))
    grid = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(grid), grid)


def precompute_freqs_cis_3d(dim, end=1024, theta=10000.0):
    f_dim = dim - 2 * (dim // 3)
    hw_dim = dim // 3
    return (
        precompute_freqs_cis(f_dim, end, theta),
        precompute_freqs_cis(hw_dim, end, theta),
        precompute_freqs_cis(hw_dim, end, theta),
    )


class FastWAMPreInfer:
    def __init__(self, config):
        self.config = config
        self.freq_dim = int(config.get("freq_dim", 256))
        self.video_hidden_dim = int(config["dim"])
        self.action_hidden_dim = int(config.get("action_dim_hidden", 1024))
        self.num_heads = int(config["num_heads"])
        self.head_dim = self.video_hidden_dim // self.num_heads
        self.patch_size = tuple(config.get("patch_size", [1, 2, 2]))
        self.video_freqs = precompute_freqs_cis_3d(self.head_dim)
        self.action_freqs = precompute_freqs_cis(self.head_dim, end=int(config.get("max_action_tokens", 1024)))

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @staticmethod
    def _mlp2(x, fc0, fc2, activation="gelu"):
        x = fc0.apply(x)
        if activation == "silu":
            x = F.silu(x)
        else:
            x = F.gelu(x, approximate="tanh")
        return fc2.apply(x)

    def _time_embedding(self, weight, timestep, hidden_dim):
        emb = sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1))
        t = self._mlp2(emb, weight.time_embedding_0, weight.time_embedding_2, activation="silu")
        t_mod = weight.time_projection_1.apply(F.silu(t)).reshape(t.shape[0], 6, hidden_dim)
        return t, t_mod

    def _text_embedding(self, weight, context):
        return self._mlp2(context, weight.text_embedding_0, weight.text_embedding_2)

    def infer_video(self, pre_weight, first_frame_latents, context, context_mask):
        if first_frame_latents.ndim != 5 or first_frame_latents.shape[0] != 1:
            raise ValueError(f"FastWAM video latents must be [1,C,F,H,W], got {tuple(first_frame_latents.shape)}")

        video = pre_weight.video
        x = video.patch_embedding.apply(first_frame_latents)
        _, channels, frames, height, width = x.shape
        if channels != self.video_hidden_dim:
            raise ValueError(f"Unexpected video hidden dim {channels}, expected {self.video_hidden_dim}")

        tokens_per_frame = height * width
        token_timesteps = torch.zeros(
            (frames, tokens_per_frame),
            dtype=first_frame_latents.dtype,
            device=first_frame_latents.device,
        )
        t, t_mod = self._time_embedding(video, token_timesteps.reshape(-1), self.video_hidden_dim)
        tokens = x.permute(0, 2, 3, 4, 1).reshape(-1, channels).contiguous()
        context = self._text_embedding(video, context)

        freqs = torch.cat(
            [
                self.video_freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                self.video_freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                self.video_freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(frames * height * width, 1, -1)
        freqs = freqs.to(tokens.device)

        if context_mask is None:
            context_mask = torch.ones((tokens.shape[0], context.shape[0]), dtype=torch.bool, device=tokens.device)
        elif context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0).expand(tokens.shape[0], -1)

        return SimpleNamespace(
            tokens=tokens,
            freqs=freqs,
            t=t,
            t_mod=t_mod,
            context=context,
            context_mask=context_mask,
            tokens_per_frame=tokens_per_frame,
            grid_size=(frames, height, width),
        )

    def infer_action(self, pre_weight, action_latents, timestep, context, context_mask):
        if action_latents.ndim != 3 or action_latents.shape[0] != 1:
            raise ValueError(f"FastWAM action latents must be [1,T,D], got {tuple(action_latents.shape)}")

        action = pre_weight.action
        tokens = action.action_encoder.apply(action_latents[0])
        t, t_mod = self._time_embedding(action, timestep.reshape(-1), self.action_hidden_dim)
        context = self._text_embedding(action, context)
        seq_len = tokens.shape[0]
        freqs = self.action_freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        if context_mask is None:
            context_mask = torch.ones((seq_len, context.shape[0]), dtype=torch.bool, device=tokens.device)
        elif context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0).expand(seq_len, -1)

        return SimpleNamespace(
            tokens=tokens,
            freqs=freqs,
            t=t,
            t_mod=t_mod[0],
            context=context,
            context_mask=context_mask,
            seq_len=seq_len,
        )
