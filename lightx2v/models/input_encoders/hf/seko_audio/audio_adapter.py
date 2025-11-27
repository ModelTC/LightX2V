try:
    import flash_attn
except ModuleNotFoundError:
    flash_attn = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from einops import rearrange


def linear_interpolation(features, output_len: int):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=output_len, align_corners=False, mode="linear")
    return output_features.transpose(1, 2)


@torch.compiler.disable
def get_max_int(q_lens, k_lens):
    max_seqlen_q = int(q_lens.max().item())
    max_seqlen_k = int(k_lens.max().item())
    return max_seqlen_q, max_seqlen_k


def get_qk_lens_audio_range(
    n_tokens_per_rank: torch.Tensor,
    n_query_tokens: torch.Tensor,
    n_tokens_per_frame: torch.Tensor,
    sp_rank: torch.Tensor,
    num_tokens_x4,
):
    device = n_tokens_per_rank.device
    dtype = torch.int32

    if n_query_tokens == 0:
        q_lens = torch.ones(1, dtype=dtype, device=device)
        t0 = torch.tensor(0, device=device)
        t1 = torch.tensor(1, device=device)
        k_lens = torch.full((t1 - t0,), num_tokens_x4, dtype=dtype, device=device)
        max_seqlen_q, max_seqlen_k = get_max_int(q_lens, k_lens)
        return q_lens, k_lens, max_seqlen_q, max_seqlen_k, t0, t1

    idx0 = n_tokens_per_rank * sp_rank

    first_length = n_tokens_per_frame - idx0 % n_tokens_per_frame
    first_length = torch.minimum(first_length, n_query_tokens)

    n_frames = torch.div(n_query_tokens - first_length, n_tokens_per_frame, rounding_mode="floor")

    last_length = n_query_tokens - n_frames * n_tokens_per_frame - first_length

    first_tensor = first_length.unsqueeze(0)  # [1]
    frame_tensor = n_tokens_per_frame.repeat(n_frames)  # [n_frames]
    last_tensor = last_length.unsqueeze(0)  # [1]

    q_lens_all = torch.cat([first_tensor, frame_tensor, last_tensor])
    q_lens = q_lens_all[q_lens_all > 0].to(dtype)

    t0 = idx0 // n_tokens_per_frame
    t1 = t0 + q_lens.numel()

    k_lens = torch.full((t1 - t0,), num_tokens_x4, dtype=dtype, device=device)

    assert q_lens.shape == k_lens.shape
    max_seqlen_q, max_seqlen_k = get_max_int(q_lens, k_lens)

    return q_lens, k_lens, max_seqlen_q, max_seqlen_k, t0, t1


def calculate_n_query_tokens(hidden_states, person_mask_latens, sp_rank, sp_size, n_tokens_per_rank, n_tokens):
    tail_length = n_tokens_per_rank * sp_size - n_tokens
    n_unused_ranks = tail_length // n_tokens_per_rank

    if sp_rank > sp_size - n_unused_ranks - 1:
        n_query_tokens = 0
    elif sp_rank == sp_size - n_unused_ranks - 1:
        val = n_tokens_per_rank - (tail_length % n_tokens_per_rank)
        n_query_tokens = val
    else:
        n_query_tokens = n_tokens_per_rank

    if n_query_tokens > 0:
        hidden_states_aligned = hidden_states[:n_query_tokens]
        hidden_states_tail = hidden_states[n_query_tokens:]
        if person_mask_latens is not None:
            person_mask_aligned = person_mask_latens[:, :n_query_tokens]
        else:
            person_mask_aligned = None
    else:
        # for ranks that should be excluded from cross-attn, fake cross-attn will be applied so that FSDP works.
        hidden_states_aligned = hidden_states[:1]
        hidden_states_tail = hidden_states[1:]
        if person_mask_latens is not None:
            person_mask_aligned = person_mask_latens[:, :1]
        else:
            person_mask_aligned = None

    return n_query_tokens, hidden_states_aligned, hidden_states_tail, person_mask_aligned


'''
class PerceiverAttentionCA(nn.Module):
    def __init__(self, dim_head=128, heads=16, kv_dim=2048, adaLN: bool = False, quantized=False, quant_scheme=None):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        kv_dim = inner_dim if kv_dim is None else kv_dim
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.norm_q = nn.LayerNorm(inner_dim, elementwise_affine=not adaLN)

        if quantized:
            if quant_scheme == "fp8":
                self.to_q = SglQuantLinearFp8(inner_dim, inner_dim)
                self.to_kv = nn.Linear(kv_dim, inner_dim * 2)
                self.to_out = SglQuantLinearFp8(inner_dim, inner_dim)
            else:
                raise ValueError(f"Unsupported quant_scheme: {quant_scheme}")
        else:
            self.to_q = nn.Linear(inner_dim, inner_dim)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2)
            self.to_out = nn.Linear(inner_dim, inner_dim)
        if adaLN:
            self.shift_scale_gate = nn.Parameter(torch.randn(1, 3, inner_dim) / inner_dim**0.5)
        else:
            shift_scale_gate = torch.zeros((1, 3, inner_dim))
            shift_scale_gate[:, 2] = 1
            self.register_buffer("shift_scale_gate", shift_scale_gate, persistent=False)

    def forward(self, x, latents, t_emb, q_lens, k_lens, max_seqlen_q, max_seqlen_k):
        """x shape (batchsize, latent_frame, audio_tokens_per_latent,
        model_dim) latents (batchsize, length, model_dim)"""
        batchsize = len(x)
        x = self.norm_kv(x)
        shift, scale, gate = (t_emb + self.shift_scale_gate).chunk(3, dim=1)
        norm_q = self.norm_q(latents)
        if scale.shape[0] != norm_q.shape[0]:
            scale = scale.transpose(0, 1)  # (1, 5070, 3072)
            shift = shift.transpose(0, 1)
            gate = gate.transpose(0, 1)
        latents = norm_q * (1 + scale) + shift
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q = rearrange(q, "B L (H C) -> (B L) H C", H=self.heads)
        k = rearrange(k, "B T L (H C) -> (B T L) H C", H=self.heads)
        v = rearrange(v, "B T L (H C) -> (B T L) H C", H=self.heads)

        out = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
        )
        out = rearrange(out, "(B L) H C -> B L (H C)", B=batchsize)
        return self.to_out(out) * gate
'''


class AudioProjection(nn.Module):
    def __init__(
        self,
        audio_feature_dim: int = 768,
        n_neighbors: tuple = (2, 2),
        num_tokens: int = 32,
        mlp_dims: tuple = (1024, 1024, 32 * 768),
        transformer_layers: int = 4,
    ):
        super().__init__()
        mlp = []
        self.left, self.right = n_neighbors
        self.audio_frames = sum(n_neighbors) + 1
        in_dim = audio_feature_dim * self.audio_frames
        for i, out_dim in enumerate(mlp_dims):
            mlp.append(nn.Linear(in_dim, out_dim))
            if i != len(mlp_dims) - 1:
                mlp.append(nn.ReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp)
        self.norm = nn.LayerNorm(mlp_dims[-1] // num_tokens)
        self.num_tokens = num_tokens
        if transformer_layers > 0:
            decoder_layer = nn.TransformerDecoderLayer(d_model=audio_feature_dim, nhead=audio_feature_dim // 64, dim_feedforward=4 * audio_feature_dim, dropout=0.0, batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=transformer_layers,
            )
        else:
            self.transformer_decoder = None

    def forward(self, audio_feature, latent_frame):
        video_frame = (latent_frame - 1) * 4 + 1
        audio_feature_ori = audio_feature
        audio_feature = linear_interpolation(audio_feature_ori, video_frame)
        if self.transformer_decoder is not None:
            audio_feature = self.transformer_decoder(audio_feature, audio_feature_ori)
        audio_feature = F.pad(audio_feature, pad=(0, 0, self.left, self.right), mode="replicate")
        audio_feature = audio_feature.unfold(dimension=1, size=self.audio_frames, step=1)
        audio_feature = rearrange(audio_feature, "B T C W -> B T (W C)")
        audio_feature = self.mlp(audio_feature)  # (B, video_frame, C)
        audio_feature = rearrange(audio_feature, "B T (N C) -> B T N C", N=self.num_tokens)  # (B, video_frame, num_tokens, C)
        return self.norm(audio_feature)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, time_freq_dim, time_proj_dim):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)

    def forward(self, timestep: torch.Tensor):
        # Project timestep
        if timestep.dim() == 2:
            timestep = self.timesteps_proj(timestep.squeeze(0)).unsqueeze(0)
        else:
            timestep = self.timesteps_proj(timestep)

        # Match dtype with time_embedder (except int8)
        target_dtype = next(self.time_embedder.parameters()).dtype
        if timestep.dtype != target_dtype and target_dtype != torch.int8:
            timestep = timestep.to(target_dtype)

        # Time embedding projection
        temb = self.time_embedder(timestep)
        timestep_proj = self.time_proj(self.act_fn(temb))

        return timestep_proj.squeeze(0) if timestep_proj.dim() == 3 else timestep_proj


class AudioAdapter(nn.Module):
    def __init__(
        self,
        attention_head_dim=64,
        num_attention_heads=40,
        base_num_layers=30,
        interval=1,
        audio_feature_dim: int = 768,
        num_tokens: int = 32,
        mlp_dims: tuple = (1024, 1024, 32 * 768),
        time_freq_dim: int = 256,
        projection_transformer_layers: int = 4,
        quantized: bool = False,
        quant_scheme: str = None,
        cpu_offload: bool = False,
        run_device=torch.device("cuda"),
    ):
        super().__init__()
        self.cpu_offload = cpu_offload
        self.audio_proj = AudioProjection(
            audio_feature_dim=audio_feature_dim,
            n_neighbors=(2, 2),
            num_tokens=num_tokens,
            mlp_dims=mlp_dims,
            transformer_layers=projection_transformer_layers,
        )
        self.run_device = run_device
        # self.num_tokens = num_tokens * 4
        self.num_tokens_x4 = num_tokens * 4
        self.audio_pe = nn.Parameter(torch.randn(self.num_tokens_x4, mlp_dims[-1] // num_tokens) * 0.02)
        # ca_num = math.ceil(base_num_layers / interval)
        self.base_num_layers = base_num_layers
        self.interval = interval
        """
        self.ca = nn.ModuleList(
            [
                PerceiverAttentionCA(
                    dim_head=attention_head_dim,
                    heads=num_attention_heads,
                    kv_dim=mlp_dims[-1] // num_tokens,
                    adaLN=time_freq_dim > 0,
                    quantized=quantized,
                    quant_scheme=quant_scheme,
                )
                for _ in range(ca_num)
            ]
        )
        """
        self.dim = attention_head_dim * num_attention_heads
        if time_freq_dim > 0:
            self.time_embedding = TimeEmbedding(self.dim, time_freq_dim, self.dim * 3)
        else:
            self.time_embedding = None

    def rearange_audio_features(self, audio_feature: torch.Tensor):
        # audio_feature (B, video_frame, num_tokens, C)
        audio_feature_0 = audio_feature[:, :1]
        audio_feature_0 = torch.repeat_interleave(audio_feature_0, repeats=4, dim=1)
        audio_feature = torch.cat([audio_feature_0, audio_feature[:, 1:]], dim=1)  # (B, 4 * latent_frame, num_tokens, C)
        audio_feature = rearrange(audio_feature, "B (T S) N C -> B T (S N) C", S=4)
        return audio_feature

    @torch.no_grad()
    def forward_audio_proj(self, audio_feat, latent_frame):
        if self.cpu_offload:
            self.audio_proj.to(self.run_device)
        x = self.audio_proj(audio_feat, latent_frame)
        x = self.rearange_audio_features(x)
        x = x + self.audio_pe.to(self.run_device)
        if self.cpu_offload:
            self.audio_proj.to("cpu")
        return x
