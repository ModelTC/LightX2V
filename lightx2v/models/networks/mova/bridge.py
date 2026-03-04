# lightx2v/models/networks/mova/bridge.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange


# -------------------- 辅助模块：RMSNorm --------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# -------------------- 注意力模块（简化版，使用 PyTorch 原生 SDPA）--------------------
class AttentionModule(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        # q, k, v: [B, L, D]
        B, L, D = q.shape
        head_dim = D // self.num_heads
        q = q.view(B, L, self.num_heads, head_dim).transpose(1, 2)  # [B, H, L, D/H]
        k = k.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return out


# -------------------- RoPE 相关 --------------------
class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, base: float, dim: int, device=None):
        super().__init__()
        self.base = base
        self.dim = dim
        self.attention_scaling = 1.0

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(fullgraph=True)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -------------------- PerFrameAttentionPooling --------------------
class PerFrameAttentionPooling(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        self.probe = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.normal_(self.probe, std=0.02)

        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        B, L, D = x.shape
        T, H, W = grid_size
        assert D == self.dim
        assert L == T * H * W

        S = H * W
        x_bt_s_d = x.view(B, T, S, D).contiguous().view(B * T, S, D)
        probe = self.probe.expand(B * T, -1, -1)
        pooled_bt_1_d = self.attention(probe, x_bt_s_d, x_bt_s_d, need_weights=False)[0]
        pooled_bt_d = pooled_bt_1_d.squeeze(1)
        pooled = pooled_bt_d.view(B, T, D)
        pooled = self.layernorm(pooled)
        return pooled


# -------------------- CrossModalInteractionController --------------------
class CrossModalInteractionController:
    def __init__(self, visual_layers: int = 30, audio_layers: int = 30):
        self.visual_layers = visual_layers
        self.audio_layers = audio_layers
        self.min_layers = min(visual_layers, audio_layers)

    def get_interaction_layers(self, strategy: str = "shallow_focus") -> Dict[str, List[Tuple[int, int]]]:
        if strategy == "shallow_focus":
            num_interact = min(10, self.min_layers // 3)
            interact_layers = list(range(0, num_interact))
        elif strategy == "distributed":
            step = 3
            interact_layers = list(range(0, self.min_layers, step))
        elif strategy == "progressive":
            shallow = list(range(0, min(8, self.min_layers)))
            if self.min_layers > 8:
                deep = list(range(8, self.min_layers, 3))
                interact_layers = shallow + deep
            else:
                interact_layers = shallow
        elif strategy == "custom":
            interact_layers = [0, 2, 4, 6, 8, 12, 16, 20]
            interact_layers = [i for i in interact_layers if i < self.min_layers]
        elif strategy == "full":
            interact_layers = list(range(0, self.min_layers))
        else:
            raise ValueError(f"Unknown interaction strategy: {strategy}")

        mapping = {
            'v2a': [(i, i) for i in interact_layers],
            'a2v': [(i, i) for i in interact_layers]
        }
        return mapping

    def should_interact(self, layer_idx: int, direction: str, interaction_mapping: Dict) -> bool:
        if direction not in interaction_mapping:
            return False
        return any(src == layer_idx for src, _ in interaction_mapping[direction])


# -------------------- ConditionalCrossAttention --------------------
class ConditionalCrossAttention(nn.Module):
    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.q_dim = dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        # 修正：norm_k 应作用于 k 的输出，维度应为 dim
        self.norm_k = RMSNorm(dim, eps=eps)  # 原来是 kv_dim
        self.attn = AttentionModule(self.num_heads)
    def _apply_rope(self, x, cos, sin):
        # x: [B, L, D]
        B, L, D = x.shape
        head_dim = D // self.num_heads
        x = x.view(B, L, self.num_heads, head_dim)
        x, _ = apply_rotary_pos_emb(x, x, cos, sin, unsqueeze_dim=2)
        return x.view(B, L, D)
    def forward(self, x: torch.Tensor, y: torch.Tensor,
            x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # 确保输入 dtype 与线性层权重匹配
        target_dtype = self.q.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if y.dtype != target_dtype:
            y = y.to(target_dtype)

        q = self.norm_q(self.q(x))          # [B, L_q, D]
        k = self.norm_k(self.k(y))          # [B, L_k, D]
        v = self.v(y)                        # [B, L_k, D]

        # 应用 RoPE
        if x_freqs is not None:
            q = self._apply_rope(q, x_freqs[0], x_freqs[1])
        if y_freqs is not None:
            k = self._apply_rope(k, y_freqs[0], y_freqs[1])

        B, L_q, D = q.shape
        L_k = k.shape[1]
        head_dim = D // self.num_heads

        # 重塑为多头格式
        q = q.view(B, L_q, self.num_heads, head_dim).transpose(1, 2)  # [B, H, L_q, head_dim]
        k = k.view(B, L_k, self.num_heads, head_dim).transpose(1, 2)  # [B, H, L_k, head_dim]
        v = v.view(B, L_k, self.num_heads, head_dim).transpose(1, 2)

        # 使用 PyTorch 的 scaled_dot_product_attention，支持不同长度
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_q, D)

        return self.o(attn_out)

# -------------------- AdaLayerNorm (简化版，来自 diffusers) --------------------
class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim * 2)
        self.norm = nn.LayerNorm(output_dim, eps=eps)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # temb: [B, T, D] or [B, D]
        if temb.dim() == 2:
            temb = temb.unsqueeze(1)  # [B, 1, D]
        temb = self.linear(self.silu(temb))  # [B, T, 2*D]
        shift, scale = temb.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


# -------------------- ConditionalCrossAttentionBlock --------------------
class ConditionalCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6, pooled_adaln: bool = False):
        super().__init__()
        self.y_norm = nn.LayerNorm(kv_dim, eps=eps)
        self.inner = ConditionalCrossAttention(dim=dim, kv_dim=kv_dim, num_heads=num_heads, eps=eps)
        self.pooled_adaln = pooled_adaln
        if pooled_adaln:
            self.per_frame_pooling = PerFrameAttentionPooling(kv_dim, num_heads=num_heads, eps=eps)
            self.adaln = AdaLayerNorm(embedding_dim=kv_dim, output_dim=dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if self.pooled_adaln:
            assert video_grid_size is not None
            pooled_y = self.per_frame_pooling(y, video_grid_size)
            if pooled_y.shape[1] != x.shape[1]:
                pooled_y = F.interpolate(
                    pooled_y.permute(0, 2, 1),
                    size=x.shape[1],
                    mode='linear',
                    align_corners=False,
                ).permute(0, 2, 1)
            x = self.adaln(x, temb=pooled_y)

        # 确保 y 的 dtype 与 LayerNorm 参数匹配
        if y.dtype != self.y_norm.weight.dtype:
            y = y.to(self.y_norm.weight.dtype)
        y = self.y_norm(y)

        return self.inner(x=x, y=y, x_freqs=x_freqs, y_freqs=y_freqs)


# -------------------- 主桥接模块 --------------------

class MOVADualTowerBridge(nn.Module):
    def __init__(
        self,
        visual_layers: int = 30,
        audio_layers: int = 30,
        visual_hidden_dim: int = 3072,
        audio_hidden_dim: int = 1536,
        audio_fps: float = 44100.0 / 2048.0,
        head_dim: int = 128,
        interaction_strategy: str = "shallow_focus",
        apply_cross_rope: bool = False,
        apply_first_frame_bias_in_rope: bool = False,
        trainable_condition_scale: bool = False,
        pooled_adaln: bool = False,
        **kwargs
    ):
        super().__init__()
        self.visual_hidden_dim = visual_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.audio_fps = audio_fps
        self.head_dim = head_dim
        self.apply_cross_rope = apply_cross_rope
        self.apply_first_frame_bias_in_rope = apply_first_frame_bias_in_rope
        self.trainable_condition_scale = trainable_condition_scale
        self.pooled_adaln = pooled_adaln

        if self.trainable_condition_scale:
            self.condition_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        else:
            self.condition_scale = 1.0

        self.controller = CrossModalInteractionController(visual_layers, audio_layers)
        self.interaction_mapping = self.controller.get_interaction_layers(interaction_strategy)

        self.audio_to_video_conditioners = nn.ModuleDict()
        self.video_to_audio_conditioners = nn.ModuleDict()

        if self.apply_cross_rope:
            self.rotary = RotaryEmbedding(base=10000.0, dim=head_dim)

        for v_layer, _ in self.interaction_mapping['a2v']:
            self.audio_to_video_conditioners[str(v_layer)] = ConditionalCrossAttentionBlock(
                dim=visual_hidden_dim,
                kv_dim=audio_hidden_dim,
                num_heads=visual_hidden_dim // head_dim,
                pooled_adaln=False
            )

        for a_layer, _ in self.interaction_mapping['v2a']:
            self.video_to_audio_conditioners[str(a_layer)] = ConditionalCrossAttentionBlock(
                dim=audio_hidden_dim,
                kv_dim=visual_hidden_dim,
                num_heads=audio_hidden_dim // head_dim,
                pooled_adaln=self.pooled_adaln
            )

    @torch.no_grad()
    def build_aligned_freqs(
        self,
        video_fps: float,
        grid_size: Tuple[int, int, int],
        audio_steps: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        f_v, h, w = grid_size
        L_v = f_v * h * w
        L_a = int(audio_steps)

        device = device or next(self.parameters()).device
        dtype = dtype or torch.float32

        audio_pos = torch.arange(L_a, device=device, dtype=torch.float32).unsqueeze(0)

        if self.apply_first_frame_bias_in_rope:
            video_effective_fps = float(video_fps) / 4.0
            if f_v > 0:
                t_starts = torch.zeros((f_v,), device=device, dtype=torch.float32)
                if f_v > 1:
                    t_starts[1:] = (1.0 / float(video_fps)) + torch.arange(f_v - 1, device=device, dtype=torch.float32) * (1.0 / video_effective_fps)
            else:
                t_starts = torch.zeros((0,), device=device, dtype=torch.float32)
            video_pos_per_frame = t_starts * float(self.audio_fps)
        else:
            scale = float(self.audio_fps) / float(video_fps / 4.0)
            video_pos_per_frame = torch.arange(f_v, device=device, dtype=torch.float32) * scale

        video_pos = video_pos_per_frame.repeat_interleave(h * w).unsqueeze(0)

        dummy_v = torch.zeros((1, L_v, self.head_dim), device=device, dtype=dtype)
        dummy_a = torch.zeros((1, L_a, self.head_dim), device=device, dtype=dtype)

        cos_v, sin_v = self.rotary(dummy_v, position_ids=video_pos)
        cos_a, sin_a = self.rotary(dummy_a, position_ids=audio_pos)

        return (cos_v, sin_v), (cos_a, sin_a)

    def should_interact(self, layer_idx: int, direction: str) -> bool:
        return self.controller.should_interact(layer_idx, direction, self.interaction_mapping)

    def forward(
        self,
        layer_idx: int,
        visual_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        *,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        a2v_condition_scale: Optional[float] = None,
        v2a_condition_scale: Optional[float] = None,
        condition_scale: Optional[float] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = condition_scale if condition_scale is not None else self.condition_scale

        # Audio -> Visual
        if self.should_interact(layer_idx, 'a2v'):
            conditioner = self.audio_to_video_conditioners[str(layer_idx)]
            conditioned = conditioner(
                x=visual_hidden_states,
                y=audio_hidden_states,
                x_freqs=x_freqs,
                y_freqs=y_freqs,
                video_grid_size=video_grid_size,
            )
            visual_hidden_states = visual_hidden_states + conditioned * scale

        # Visual -> Audio
        if self.should_interact(layer_idx, 'v2a'):
            conditioner = self.video_to_audio_conditioners[str(layer_idx)]
            conditioned = conditioner(
                x=audio_hidden_states,
                y=visual_hidden_states,
                x_freqs=y_freqs,
                y_freqs=x_freqs,
                video_grid_size=video_grid_size,
            )
            audio_hidden_states = audio_hidden_states + conditioned * scale

        return visual_hidden_states, audio_hidden_states