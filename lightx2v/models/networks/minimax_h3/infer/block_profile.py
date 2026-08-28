"""MiniMax-H3 block shapes and FLOPs for targeted transformer profiling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from lightx2v.common.ops.mm.mm_weight import unwrap_tp_weight
from lightx2v.utils import op_shape_trace as ost


@dataclass(frozen=True)
class _GemmSpec:
    region: str
    tag: str
    n: int
    k: int
    lora_rank: int


class MiniMaxH3BlockProfile:
    """Bind one H3 block's weights and packed runtime token counts."""

    block_profile_report_module = "lightx2v.models.networks.minimax_h3.infer.block_profile_report"

    _GEMM_ORDER = (
        "adaln",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "ffn_in",
        "ffn_out",
    )

    def __init__(self, config: dict, *, num_heads: int | None = None, seq_p_size: int = 1):
        self.config = config
        self.num_heads = int(num_heads or config.get("num_attention_heads", 56))
        self.head_dim = int(config.get("attention_head_dim", 128))
        self.seq_p_size = int(seq_p_size)
        self._tokens = 0
        self._temb_rows = 0
        self._attention_seq_len = 0
        self._attention_heads = self.num_heads
        self._gemms: dict[str, _GemmSpec] = {}

    @staticmethod
    def _spec(region: str, tag: str, linear) -> _GemmSpec:
        linear = unwrap_tp_weight(linear)
        weight = linear._get_actual_weight()
        if weight.ndim != 2:
            raise ValueError(f"MiniMax-H3 profile expected a matrix for {tag}, got shape={tuple(weight.shape)}")

        lora_rank = 0
        if getattr(linear, "has_lora_branch", False):
            lora_rank = int(linear.lora_down.shape[0])

        # H3 linear weights expose their compute layout as [K, N].
        return _GemmSpec(region, tag, int(weight.shape[1]), int(weight.shape[0]), lora_rank)

    def bind(
        self,
        block,
        hidden_states: torch.Tensor,
        pre_infer_out,
        *,
        include_adaln: bool = True,
    ) -> None:
        self._tokens = hidden_states.numel() // hidden_states.shape[-1]
        self._temb_rows = pre_infer_out.temb.numel() // pre_infer_out.temb.shape[-1]

        sp_state = pre_infer_out.sequence_parallel_state
        if sp_state is None:
            self._attention_seq_len = self._tokens
            self._attention_heads = self.num_heads
        else:
            self._attention_seq_len = int(sp_state.aux_length + sp_state.main_shard_length * self.seq_p_size)
            self._attention_heads = self.num_heads // self.seq_p_size

        gemms = {
            "attn_q": self._spec("self_attn", "attn_q", block.attn.to_q),
            "attn_k": self._spec("self_attn", "attn_k", block.attn.to_k),
            "attn_v": self._spec("self_attn", "attn_v", block.attn.to_v),
            "attn_o": self._spec("self_attn", "attn_o", block.attn.to_out),
            "ffn_in": self._spec("dense_ffn", "ffn_in_fused", block.ff.in_proj),
            "ffn_out": self._spec("dense_ffn", "ffn_out", block.ff.out_proj),
        }
        if include_adaln:
            gemms["adaln"] = self._spec("adaln", "adaln", block.adaln)
        self._gemms = gemms

    def write_inventory(self, path: Path, block_idx: int) -> None:
        operations = []
        total_linear_flops = 0
        for tag in self._GEMM_ORDER:
            if tag not in self._gemms:
                continue
            spec = self._gemms[tag]
            m = self._temb_rows if tag == "adaln" else self._tokens
            main_flops = 2 * m * spec.n * spec.k
            lora_flops = 2 * m * spec.lora_rank * (spec.n + spec.k)
            operations.append(
                {
                    "region": spec.region,
                    "tag": spec.tag,
                    "shape_mnk": [m, spec.n, spec.k],
                    "main_flops": main_flops,
                    "lora_flops": lora_flops,
                    "total_flops": main_flops + lora_flops,
                }
            )
            total_linear_flops += main_flops + lora_flops

        attention = {
            "tag": "joint_attention",
            "shape_bhsd": [1, self._attention_heads, self._attention_seq_len, self.head_dim],
            "flops": 4 * self._attention_heads * self._attention_seq_len**2 * self.head_dim,
        }
        if self.config.get("attn_type") == "dynamic_sparse_attn":
            attention["flops_semantics"] = "dense-equivalent"

        inventory = {
            "block": block_idx,
            "local_tokens": self._tokens,
            "sequence_parallel_size": self.seq_p_size,
            "linear_operations": operations,
            "total_linear_flops": total_linear_flops,
            "attention": attention,
        }
        path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    def _emit_gemm(self, tag: str, m: int) -> None:
        if not ost.is_recording():
            return
        spec = self._gemms[tag]
        ost.log_gemm(spec.region, spec.tag, m, spec.n, spec.k)

    def adaln(self) -> None:
        self._emit_gemm("adaln", self._temb_rows)

    def self_attn(self) -> None:
        if not ost.is_recording():
            return
        for tag in ("attn_q", "attn_k", "attn_v"):
            self._emit_gemm(tag, self._tokens)
        ost.log_attn(
            "self_attn",
            "joint_attention",
            batch=1,
            num_heads=self._attention_heads,
            seq_q=self._attention_seq_len,
            seq_k=self._attention_seq_len,
            head_dim=self.head_dim,
            flops_semantics="dense-equivalent" if self.config.get("attn_type") == "dynamic_sparse_attn" else None,
        )
        self._emit_gemm("attn_o", self._tokens)

    def dense_ffn(self) -> None:
        self._emit_gemm("ffn_in", self._tokens)
        self._emit_gemm("ffn_out", self._tokens)


__all__ = ["MiniMaxH3BlockProfile"]
