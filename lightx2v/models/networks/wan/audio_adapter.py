try:
    import flash_attn
except ModuleNotFoundError:
    flash_attn = None
import math
import os

import safetensors
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from einops import rearrange
from loguru import logger
from transformers import AutoModel

from lightx2v.utils.envs import *


def load_safetensors(in_path: str):
    if os.path.isdir(in_path):
        return load_safetensors_from_dir(in_path)
    elif os.path.isfile(in_path):
        return load_safetensors_from_path(in_path)
    else:
        raise ValueError(f"{in_path} does not exist")


def load_safetensors_from_path(in_path: str):
    tensors = {}
    with safetensors.safe_open(in_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_safetensors_from_dir(in_dir: str):
    tensors = {}
    safetensors = os.listdir(in_dir)
    safetensors = [f for f in safetensors if f.endswith(".safetensors")]
    for f in safetensors:
        tensors.update(load_safetensors_from_path(os.path.join(in_dir, f)))
    return tensors


def load_pt_safetensors(in_path: str):
    ext = os.path.splitext(in_path)[-1]
    if ext in (".pt", ".pth", ".tar"):
        state_dict = torch.load(in_path, map_location="cpu", weights_only=True)
    else:
        state_dict = load_safetensors(in_path)
    return state_dict


def rank0_load_state_dict_from_path(model, in_path: str, strict: bool = True):
    model = model.to("cuda")
    # 确定当前进程是否是（负责加载权重）
    is_leader = False
    if dist.is_initialized():
        current_rank = dist.get_rank()
        if current_rank == 0:
            is_leader = True
    elif not dist.is_initialized() or dist.get_rank() == 0:
        is_leader = True

    if is_leader:
        logger.info(f"Loading model state from {in_path}")
        state_dict = load_pt_safetensors(in_path)
        model.load_state_dict(state_dict, strict=strict)

    # 将模型状态从领导者同步到组内所有其他进程
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
        src_global_rank = 0
        for param in model.parameters():
            dist.broadcast(param.data, src=src_global_rank)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=src_global_rank)
    elif dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)

    return model.to(dtype=GET_DTYPE())


def linear_interpolation(features, output_len: int):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=output_len, align_corners=False, mode="linear")
    return output_features.transpose(1, 2)


def get_q_lens_audio_range(
    batchsize: int,
    n_tokens_per_rank: int,
    n_query_tokens: int,
    n_tokens_per_frame: int,
    sp_rank: int,
):
    if n_query_tokens == 0:
        q_lens = [1] * batchsize
        return q_lens, 0, 1
    idx0 = n_tokens_per_rank * sp_rank
    first_length = n_tokens_per_frame - idx0 % n_tokens_per_frame
    first_length = min(first_length, n_query_tokens)
    n_frames = (n_query_tokens - first_length) // n_tokens_per_frame
    last_length = n_query_tokens - n_frames * n_tokens_per_frame - first_length
    q_lens = []
    if first_length > 0:
        q_lens.append(first_length)
    q_lens += [n_tokens_per_frame] * n_frames
    if last_length > 0:
        q_lens.append(last_length)
    t0 = idx0 // n_tokens_per_frame
    t1 = t0 + len(q_lens)
    return q_lens * batchsize, t0, t1


class PerceiverAttentionCA(nn.Module):
    def __init__(self, dim_head=128, heads=16, kv_dim=2048, adaLN: bool = False):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        kv_dim = inner_dim if kv_dim is None else kv_dim
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.norm_q = nn.LayerNorm(inner_dim, elementwise_affine=not adaLN)

        self.to_q = nn.Linear(inner_dim, inner_dim)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2)
        self.to_out = nn.Linear(inner_dim, inner_dim)
        if adaLN:
            self.shift_scale_gate = nn.Parameter(torch.randn(1, 3, inner_dim) / inner_dim**0.5)
        else:
            shift_scale_gate = torch.zeros((1, 3, inner_dim))
            shift_scale_gate[:, 2] = 1
            self.register_buffer("shift_scale_gate", shift_scale_gate, persistent=False)

    def forward(self, x, latents, t_emb, q_lens, k_lens):
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
        q = self.to_q(latents.to(GET_DTYPE()))
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
            max_seqlen_q=q_lens.max().item(),
            max_seqlen_k=k_lens.max().item(),
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
        )
        out = rearrange(out, "(B L) H C -> B L (H C)", B=batchsize)
        return self.to_out(out) * gate


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
    ):
        super().__init__()
        self.audio_proj = AudioProjection(
            audio_feature_dim=audio_feature_dim,
            n_neighbors=(2, 2),
            num_tokens=num_tokens,
            mlp_dims=mlp_dims,
            transformer_layers=projection_transformer_layers,
        )
        # self.num_tokens = num_tokens * 4
        self.num_tokens_x4 = num_tokens * 4
        self.audio_pe = nn.Parameter(torch.randn(self.num_tokens_x4, mlp_dims[-1] // num_tokens) * 0.02)
        ca_num = math.ceil(base_num_layers / interval)
        self.base_num_layers = base_num_layers
        self.interval = interval
        self.ca = nn.ModuleList(
            [
                PerceiverAttentionCA(
                    dim_head=attention_head_dim,
                    heads=num_attention_heads,
                    kv_dim=mlp_dims[-1] // num_tokens,
                    adaLN=time_freq_dim > 0,
                )
                for _ in range(ca_num)
            ]
        )
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

    def forward(self, audio_feat: torch.Tensor, timestep: torch.Tensor, latent_frame: int, weight: float = 1.0, seq_p_group=None):
        def modify_hidden_states(hidden_states, grid_sizes, ca_block: PerceiverAttentionCA, x, t_emb, dtype, weight, seq_p_group):
            """thw specify the latent_frame, latent_height, latenf_width after
            hidden_states is patchified.

            latent_frame does not include the reference images so that the
            audios and hidden_states are strictly aligned
            """
            if len(hidden_states.shape) == 2:  # 扩展batchsize dim
                hidden_states = hidden_states.unsqueeze(0)  # bs = 1
            t, h, w = grid_sizes[0].tolist()
            n_tokens = t * h * w
            ori_dtype = hidden_states.dtype
            device = hidden_states.device
            bs, n_tokens_per_rank = hidden_states.shape[:2]

            if seq_p_group is not None:
                sp_size = dist.get_world_size(seq_p_group)
                sp_rank = dist.get_rank(seq_p_group)
            else:
                sp_size = 1
                sp_rank = 0

            tail_length = n_tokens_per_rank * sp_size - n_tokens
            n_unused_ranks = tail_length // n_tokens_per_rank
            if sp_rank > sp_size - n_unused_ranks - 1:
                n_query_tokens = 0
            elif sp_rank == sp_size - n_unused_ranks - 1:
                n_query_tokens = n_tokens_per_rank - tail_length % n_tokens_per_rank
            else:
                n_query_tokens = n_tokens_per_rank

            if n_query_tokens > 0:
                hidden_states_aligned = hidden_states[:, :n_query_tokens]
                hidden_states_tail = hidden_states[:, n_query_tokens:]
            else:
                # for ranks that should be excluded from cross-attn, fake cross-attn will be applied so that FSDP works.
                hidden_states_aligned = hidden_states[:, :1]
                hidden_states_tail = hidden_states[:, 1:]

            q_lens, t0, t1 = get_q_lens_audio_range(batchsize=bs, n_tokens_per_rank=n_tokens_per_rank, n_query_tokens=n_query_tokens, n_tokens_per_frame=h * w, sp_rank=sp_rank)
            q_lens = torch.tensor(q_lens, device=device, dtype=torch.int32)
            """
            processing audio features in sp_state can be moved outside.
            """
            x = x[:, t0:t1]
            x = x.to(dtype)
            k_lens = torch.tensor([self.num_tokens_x4] * (t1 - t0) * bs, device=device, dtype=torch.int32)
            assert q_lens.shape == k_lens.shape
            # ca_block:CrossAttention函数
            residual = ca_block(x, hidden_states_aligned, t_emb, q_lens, k_lens) * weight

            residual = residual.to(ori_dtype)  # audio做了CrossAttention之后以Residual的方式注入
            if n_query_tokens == 0:
                residual = residual * 0.0
            hidden_states = torch.cat([hidden_states_aligned + residual, hidden_states_tail], dim=1)

            if len(hidden_states.shape) == 3:  #
                hidden_states = hidden_states.squeeze(0)  # bs = 1
            return hidden_states

        x = self.audio_proj(audio_feat, latent_frame)
        x = self.rearange_audio_features(x)
        x = x + self.audio_pe
        if self.time_embedding is not None:
            t_emb = self.time_embedding(timestep).unflatten(1, (3, -1))
        else:
            t_emb = torch.zeros((len(x), 3, self.dim), device=x.device, dtype=x.dtype)
        ret_dict = {}
        for block_idx, base_idx in enumerate(range(0, self.base_num_layers, self.interval)):
            block_dict = {
                "kwargs": {
                    "ca_block": self.ca[block_idx],
                    "x": x,
                    "weight": weight,
                    "t_emb": t_emb,
                    "dtype": x.dtype,
                    "seq_p_group": seq_p_group,
                },
                "modify_func": modify_hidden_states,
            }
            ret_dict[base_idx] = block_dict
        return ret_dict

    @classmethod
    def from_transformer(
        cls,
        transformer,
        audio_feature_dim: int = 1024,
        interval: int = 1,
        time_freq_dim: int = 256,
        projection_transformer_layers: int = 4,
    ):
        num_attention_heads = transformer.config["num_heads"]
        base_num_layers = transformer.config["num_layers"]
        attention_head_dim = transformer.config["dim"] // num_attention_heads

        audio_adapter = AudioAdapter(
            attention_head_dim,
            num_attention_heads,
            base_num_layers,
            interval=interval,
            audio_feature_dim=audio_feature_dim,
            time_freq_dim=time_freq_dim,
            projection_transformer_layers=projection_transformer_layers,
            mlp_dims=(1024, 1024, 32 * audio_feature_dim),
        )
        return audio_adapter

    def get_fsdp_wrap_module_list(
        self,
    ):
        ret_list = list(self.ca)
        return ret_list

    def enable_gradient_checkpointing(
        self,
    ):
        pass


class AudioAdapterPipe:
    def __init__(
        self,
        audio_adapter: AudioAdapter,
        audio_encoder_repo: str = "microsoft/wavlm-base-plus",
        dtype=torch.float32,
        device="cuda",
        tgt_fps: int = 15,
        weight: float = 1.0,
        cpu_offload: bool = False,
        seq_p_group=None,
    ) -> None:
        self.seq_p_group = seq_p_group
        self.audio_adapter = audio_adapter
        self.dtype = dtype
        self.audio_encoder_dtype = torch.float16
        self.cpu_offload = cpu_offload
        ##音频编码器
        self.audio_encoder = AutoModel.from_pretrained(audio_encoder_repo)

        self.audio_encoder.eval()
        self.audio_encoder.to(device, self.audio_encoder_dtype)
        self.tgt_fps = tgt_fps
        self.weight = weight
        if "base" in audio_encoder_repo:
            self.audio_feature_dim = 768
        else:
            self.audio_feature_dim = 1024

    def update_model(self, audio_adapter):
        self.audio_adapter = audio_adapter

    def __call__(self, audio_input_feat, timestep, latent_shape: tuple, dropout_cond: callable = None):
        # audio_input_feat is from AudioPreprocessor
        latent_frame = latent_shape[2]
        if len(audio_input_feat.shape) == 1:  # 扩展batchsize = 1
            audio_input_feat = audio_input_feat.unsqueeze(0)
            latent_frame = latent_shape[1]

        video_frame = (latent_frame - 1) * 4 + 1
        audio_length = int(50 / self.tgt_fps * video_frame)

        with torch.no_grad():
            try:
                if self.cpu_offload:
                    self.audio_encoder = self.audio_encoder.to("cuda")
                audio_feat = self.audio_encoder(audio_input_feat.to(self.audio_encoder_dtype), return_dict=True).last_hidden_state
                if self.cpu_offload:
                    self.audio_encoder = self.audio_encoder.to("cpu")
            except Exception as err:
                audio_feat = torch.rand(1, audio_length, self.audio_feature_dim).to("cuda")
                print(err)
            audio_feat = audio_feat.to(self.dtype)
            if dropout_cond is not None:
                audio_feat = dropout_cond(audio_feat)

        return self.audio_adapter(audio_feat=audio_feat, timestep=timestep, latent_frame=latent_frame, weight=self.weight, seq_p_group=self.seq_p_group)
