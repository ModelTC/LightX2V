import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class LinearWithMM(nn.Module):
    """nn.Linear-compatible module with optional LightX2V MM backend."""

    def __init__(self, in_features, out_features, bias=True, quant_scheme="Default", quantized=False, config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.quant_scheme = quant_scheme
        self.quantized = quantized and quant_scheme != "Default"
        self.config = copy.deepcopy(config or {})
        self.mm = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, quant_scheme="Default", quantized=False, config=None):
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            quant_scheme=quant_scheme,
            quantized=quantized,
            config=config,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight.detach())
            if linear.bias is not None:
                module.bias.copy_(linear.bias.detach())
        module = module.to(device=linear.weight.device, dtype=linear.weight.dtype)
        module._build_mm()
        return module

    @classmethod
    def from_tensor(cls, weight: torch.Tensor, bias: torch.Tensor | None = None, quant_scheme="Default", quantized=False, config=None):
        out_features, in_features = weight.shape
        module = cls(
            in_features,
            out_features,
            bias=bias is not None,
            quant_scheme=quant_scheme,
            quantized=quantized,
            config=config,
        )
        with torch.no_grad():
            module.weight.copy_(weight.detach())
            if bias is not None:
                module.bias.copy_(bias.detach())
        module = module.to(device=weight.device, dtype=weight.dtype)
        module._build_mm()
        return module

    def _build_mm(self):
        scheme = self.quant_scheme if self.quantized else "Default"
        self.mm = MM_WEIGHT_REGISTER[scheme]("__motus_weight__", "__motus_bias__" if self.bias is not None else None)
        if hasattr(self.mm, "set_config"):
            cfg = copy.deepcopy(self.config)
            if self.quantized:
                cfg["dit_quantized"] = True
                cfg["dit_quant_scheme"] = self.quant_scheme
                cfg.setdefault("weight_auto_quant", True)
            self.mm.set_config(cfg)
        weight_dict = {"__motus_weight__": self.weight.detach()}
        if self.bias is not None:
            weight_dict["__motus_bias__"] = self.bias.detach()
        self.mm.load(weight_dict)

    def _mm_apply(self, x):
        if self.mm is None:
            self._build_mm()
        x2d = x.reshape(-1, x.shape[-1])
        y2d = self.mm.apply(x2d.to(self.weight.dtype))
        if y2d.dtype != x.dtype:
            y2d = y2d.to(x.dtype)
        return y2d.reshape(*x.shape[:-1], self.out_features)

    def forward(self, x):
        input_dtype = x.dtype
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        if not self.quantized:
            y = F.linear(x, self.weight, self.bias)
            return y if y.dtype == input_dtype else y.to(input_dtype)
        return self._mm_apply(x)


class TripleQKVProjector(nn.Module):
    """Three-way linear projection for q/k/v from a packed tensor."""

    def __init__(self, packed_qkv: torch.Tensor, quant_scheme="Default", quantized=False, config=None):
        super().__init__()
        assert packed_qkv.dim() == 4
        self.num_heads = packed_qkv.shape[1]
        self.in_features = packed_qkv.shape[2]
        self.head_dim = packed_qkv.shape[3]
        self.out_features = self.num_heads * self.head_dim

        q_w = packed_qkv[0].permute(0, 2, 1).reshape(self.out_features, self.in_features).contiguous()
        k_w = packed_qkv[1].permute(0, 2, 1).reshape(self.out_features, self.in_features).contiguous()
        v_w = packed_qkv[2].permute(0, 2, 1).reshape(self.out_features, self.in_features).contiguous()

        self.q = LinearWithMM.from_tensor(q_w, None, quant_scheme=quant_scheme, quantized=quantized, config=config)
        self.k = LinearWithMM.from_tensor(k_w, None, quant_scheme=quant_scheme, quantized=quantized, config=config)
        self.v = LinearWithMM.from_tensor(v_w, None, quant_scheme=quant_scheme, quantized=quantized, config=config)

    def forward(self, x):
        q = self.q(x).reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        k = self.k(x).reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        v = self.v(x).reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        return q, k, v


class RegistryAttention(nn.Module):
    """LightX2V attention-kernel wrapper with Wan-style varlen arguments."""

    def __init__(self, attn_type: str):
        super().__init__()
        self.attn_type = attn_type
        self.kernel = ATTN_WEIGHT_REGISTER[attn_type]()

    def _build_cu_seqlens(self, batch: int, seq_len: int, device: torch.device):
        return torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)

    def _normalize_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype in (torch.float16, torch.bfloat16):
            return tensor
        if tensor.device.type == "cuda":
            return tensor.to(torch.bfloat16)
        return tensor.to(torch.float32)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_kv: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_kv: int | None = None,
        **kwargs,
    ):
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError("RegistryAttention expects q/k/v with shape [B, L, H, D].")

        q = self._normalize_dtype(q)
        k = self._normalize_dtype(k)
        v = self._normalize_dtype(v)

        batch, q_len = q.shape[:2]
        kv_len = k.shape[1]
        if cu_seqlens_q is None:
            cu_seqlens_q = self._build_cu_seqlens(batch, q_len, q.device)
        if cu_seqlens_kv is None:
            cu_seqlens_kv = self._build_cu_seqlens(batch, kv_len, k.device)
        if max_seqlen_q is None:
            max_seqlen_q = q_len
        if max_seqlen_kv is None:
            max_seqlen_kv = kv_len
        out = self.kernel.apply(
            q=q,
            k=k,
            v=v,
            causal=causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            **kwargs,
        )
        if out.dim() == 2:
            out = out.view(batch, q_len, -1)
        return out
