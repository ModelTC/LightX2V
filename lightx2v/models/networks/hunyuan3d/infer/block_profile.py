"""Hunyuan3D logical-op shapes for targeted block profiling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lightx2v.utils import op_shape_trace as ost


def _op_shape_logging_enabled() -> bool:
    return ost.is_recording()


__all__ = [
    "Hunyuan3DBlockProfile",
]


@dataclass(frozen=True)
class _GemmSpec:
    region: str
    tag: str
    n: int
    k: int


class Hunyuan3DBlockProfile:
    """Bind static GEMM N/K and runtime M for op-shape emit hooks."""

    block_profile_report_module = "lightx2v.models.networks.hunyuan3d.infer.block_profile_report"

    def __init__(self, config: dict):
        self.num_heads = int(config["num_heads"])
        self.hidden = int(config["hidden_size"])
        self.head_dim = self.hidden // self.num_heads
        self._moe_top_k = int(config.get("moe_top_k", 2))
        self._moe_backend = str(config.get("moe_backend", "pytorch_loop"))
        self._moe_fc_schema = str(config.get("moe_fc_schema", "fc1_fc2"))
        self._cond_len = 0
        self._moe_intermediate = 0
        self._gemms: dict[str, _GemmSpec] = {}
        self._batch = 1
        self._seq_len = 0
        self._m = 0

    @staticmethod
    def _register(store: dict[str, _GemmSpec], tag: str, region: str, linear) -> None:
        w = linear._get_actual_weight()
        store[tag] = _GemmSpec(region, tag, int(w.shape[0]), int(w.shape[1]))

    def bind(self, block_weights, cond_len: int, hidden_states: torch.Tensor) -> None:
        self._batch = int(hidden_states.shape[0])
        self._seq_len = int(hidden_states.shape[1])
        self._m = self._batch * self._seq_len
        self._cond_len = int(cond_len)
        g: dict[str, _GemmSpec] = {}
        if block_weights.skip_linear is not None:
            self._register(g, "skip_linear", "skip_connection", block_weights.skip_linear)
        self._register(g, "self_q", "self_attn", block_weights.attn1.to_q)
        self._register(g, "self_k", "self_attn", block_weights.attn1.to_k)
        self._register(g, "self_v", "self_attn", block_weights.attn1.to_v)
        self._register(g, "self_o", "self_attn", block_weights.attn1.out_proj)
        self._register(g, "cross_q", "cross_attn", block_weights.attn2.to_q)
        self._register(g, "cross_k", "cross_attn", block_weights.attn2.to_k)
        self._register(g, "cross_v", "cross_attn", block_weights.attn2.to_v)
        self._register(g, "cross_o", "cross_attn", block_weights.attn2.out_proj)
        if block_weights.moe is not None:
            self._register(g, "moe_gate", "moe", block_weights.moe.gate)
            self._moe_intermediate = int(block_weights.moe.experts[0].fc1._get_actual_weight().shape[0])
            self._register(g, "moe_shared.fc1", "moe", block_weights.moe.shared_experts.fc1)
            self._register(g, "moe_shared.fc2", "moe", block_weights.moe.shared_experts.fc2)
        elif block_weights.mlp is not None:
            self._register(g, "ffn_fc1", "dense_ffn", block_weights.mlp.fc1)
            self._register(g, "ffn_fc2", "dense_ffn", block_weights.mlp.fc2)
        self._gemms = g

    def _emit_gemm(self, tag: str) -> None:
        if not _op_shape_logging_enabled():
            return
        spec = self._gemms[tag]
        ost.log_gemm(spec.region, spec.tag, self._m, spec.n, spec.k)

    def skip_connection(self) -> None:
        if not _op_shape_logging_enabled():
            return
        self._emit_gemm("skip_linear")

    def self_attn(self) -> None:
        if not _op_shape_logging_enabled():
            return
        for tag in ("self_q", "self_k", "self_v"):
            self._emit_gemm(tag)
        ost.log_attn(
            "self_attn",
            "self_sdpa",
            batch=self._batch,
            num_heads=self.num_heads,
            seq_q=self._seq_len,
            seq_k=self._seq_len,
            head_dim=self.head_dim,
        )
        self._emit_gemm("self_o")

    def cross_attn(self) -> None:
        if not _op_shape_logging_enabled():
            return
        ost.log_gemm(
            self._gemms["cross_q"].region,
            "cross_q",
            self._batch * self._seq_len,
            self._gemms["cross_q"].n,
            self._gemms["cross_q"].k,
        )
        m_kv = self._batch * self._cond_len
        ost.log_gemm(self._gemms["cross_k"].region, "cross_k", m_kv, self._gemms["cross_k"].n, self._gemms["cross_k"].k)
        ost.log_gemm(self._gemms["cross_v"].region, "cross_v", m_kv, self._gemms["cross_v"].n, self._gemms["cross_v"].k)
        ost.log_attn(
            "cross_attn",
            "cross_sdpa",
            batch=self._batch,
            num_heads=self.num_heads,
            seq_q=self._seq_len,
            seq_k=self._cond_len,
            head_dim=self.head_dim,
        )
        self._emit_gemm("cross_o")

    def dense_ffn(self) -> None:
        if not _op_shape_logging_enabled():
            return
        self._emit_gemm("ffn_fc1")
        self._emit_gemm("ffn_fc2")

    def moe(self, expert_tokens: list[int]) -> None:
        self.log_moe_gate()
        self.log_moe_routed(expert_tokens)
        self.log_moe_shared()

    def log_moe_gate(self) -> None:
        if not _op_shape_logging_enabled():
            return
        self._emit_gemm("moe_gate")

    def log_moe_routed(self, expert_tokens: list[int]) -> None:
        if not _op_shape_logging_enabled():
            return
        ost.log_moe_routed(
            "moe",
            "moe_routed",
            num_tokens=self._m,
            top_k=self._moe_top_k,
            hidden=self.hidden,
            intermediate=self._moe_intermediate,
            backend=self._moe_backend,
            fc_schema=self._moe_fc_schema,
            expert_tokens=expert_tokens,
        )

    def log_moe_shared(self) -> None:
        if not _op_shape_logging_enabled():
            return
        for tag in ("moe_shared.fc1", "moe_shared.fc2"):
            self._emit_gemm(tag)
