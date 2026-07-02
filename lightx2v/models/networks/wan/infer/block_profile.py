"""Wan / SekoTalk block profile: ``@region_profile`` regions + op-shape JSONL.

Env ``SEKO_TALK_BLOCK_PROFILE=1`` (also used by ``wyr_seko.sh``) enables
``record_function`` region labels and logical-op shape logging for Wan DiT blocks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

import torch

from lightx2v.utils import op_shape_trace as ost
from lightx2v.utils.region_profile import (
    active_profile,
    get_active_profile,
)
from lightx2v.utils.region_profile import (
    region_profile as _region_profile,
)

BLOCK_PROFILE_ENV = "SEKO_TALK_BLOCK_PROFILE"

region_profile = partial(_region_profile, annotate_env=BLOCK_PROFILE_ENV)


def _op_shape_logging_enabled() -> bool:
    return os.environ.get(BLOCK_PROFILE_ENV) == "1" and ost.is_recording()


__all__ = [
    "BLOCK_PROFILE_ENV",
    "WanBlockProfile",
    "active_profile",
    "get_active_profile",
    "region_profile",
]


@dataclass(frozen=True)
class _GemmSpec:
    region: str
    tag: str
    n: int
    k: int


class WanBlockProfile:
    """Bind Wan block phase weights + runtime token counts for op-shape hooks."""

    profile_env = BLOCK_PROFILE_ENV
    block_profile_report_module = "lightx2v.models.networks.wan.infer.block_profile_report"

    def __init__(self, config: dict):
        self.config = config
        self.num_heads = int(config["num_heads"])
        self.hidden = int(config["dim"])
        self.head_dim = self.hidden // self.num_heads
        self._m = 0
        self._context_len = 0
        self._context_img_len = 0
        self._use_cross_img = False
        self._gemms: dict[str, _GemmSpec] = {}
        self._has_audio_adapter = False
        self._audio_q_len = 0
        self._audio_kv_len = 0

    @staticmethod
    def _register(store: dict[str, _GemmSpec], tag: str, region: str, linear) -> None:
        w = linear._get_actual_weight()
        store[tag] = _GemmSpec(region, tag, int(w.shape[0]), int(w.shape[1]))

    def bind(self, block, x: torch.Tensor, pre_infer_out) -> None:
        self._m = int(x.shape[0])
        g: dict[str, _GemmSpec] = {}
        p0, p1, p2 = block.compute_phases[0], block.compute_phases[1], block.compute_phases[2]
        self._register(g, "self_q", "self_attn", p0.self_attn_q)
        self._register(g, "self_k", "self_attn", p0.self_attn_k)
        self._register(g, "self_v", "self_attn", p0.self_attn_v)
        self._register(g, "self_o", "self_attn", p0.self_attn_o)
        self._register(g, "cross_q", "cross_attn", p1.cross_attn_q)
        self._register(g, "cross_k", "cross_attn", p1.cross_attn_k)
        self._register(g, "cross_v", "cross_attn", p1.cross_attn_v)
        self._register(g, "cross_o", "cross_attn", p1.cross_attn_o)
        self._use_cross_img = self.config.get("task") in ("i2v", "flf2v", "animate", "s2v", "rs2v") and self.config.get("use_image_encoder", True) and hasattr(p1, "cross_attn_k_img")
        if self._use_cross_img:
            self._context_img_len = 257
            self._register(g, "cross_k_img", "cross_attn", p1.cross_attn_k_img)
            self._register(g, "cross_v_img", "cross_attn", p1.cross_attn_v_img)
        else:
            self._context_img_len = 0
        text_len = int(pre_infer_out.context.shape[0])
        self._context_len = text_len - self._context_img_len if self._context_img_len else text_len
        self._register(g, "ffn_0", "dense_ffn", p2.ffn_0)
        self._register(g, "ffn_2", "dense_ffn", p2.ffn_2)
        self._has_audio_adapter = len(block.compute_phases) > 3
        if self._has_audio_adapter:
            p3 = block.compute_phases[3]
            self._register(g, "audio_q", "audio_adapter", p3.to_q)
            self._register(g, "audio_kv", "audio_adapter", p3.to_kv)
            self._register(g, "audio_o", "audio_adapter", p3.to_out)
        self._gemms = g

    def _emit_gemm(self, tag: str) -> None:
        if not _op_shape_logging_enabled():
            return
        spec = self._gemms[tag]
        ost.log_gemm(spec.region, spec.tag, self._m, spec.n, spec.k)

    def self_attn(self) -> None:
        if not _op_shape_logging_enabled():
            return
        for tag in ("self_q", "self_k", "self_v"):
            self._emit_gemm(tag)
        ost.log_attn(
            "self_attn",
            "self_sdpa",
            batch=1,
            num_heads=self.num_heads,
            seq_q=self._m,
            seq_k=self._m,
            head_dim=self.head_dim,
        )
        self._emit_gemm("self_o")

    def cross_attn(self) -> None:
        if not _op_shape_logging_enabled():
            return
        self._emit_gemm("cross_q")
        ost.log_gemm(
            self._gemms["cross_k"].region,
            "cross_k",
            self._context_len,
            self._gemms["cross_k"].n,
            self._gemms["cross_k"].k,
        )
        ost.log_gemm(
            self._gemms["cross_v"].region,
            "cross_v",
            self._context_len,
            self._gemms["cross_v"].n,
            self._gemms["cross_v"].k,
        )
        if self._use_cross_img:
            ost.log_gemm(
                self._gemms["cross_k_img"].region,
                "cross_k_img",
                self._context_img_len,
                self._gemms["cross_k_img"].n,
                self._gemms["cross_k_img"].k,
            )
            ost.log_gemm(
                self._gemms["cross_v_img"].region,
                "cross_v_img",
                self._context_img_len,
                self._gemms["cross_v_img"].n,
                self._gemms["cross_v_img"].k,
            )
        ost.log_attn(
            "cross_attn",
            "cross_sdpa",
            batch=1,
            num_heads=self.num_heads,
            seq_q=self._m,
            seq_k=self._context_len,
            head_dim=self.head_dim,
        )
        if self._use_cross_img:
            ost.log_attn(
                "cross_attn",
                "cross_sdpa_img",
                batch=1,
                num_heads=self.num_heads,
                seq_q=self._m,
                seq_k=self._context_img_len,
                head_dim=self.head_dim,
            )
        self._emit_gemm("cross_o")

    def dense_ffn(self) -> None:
        if not _op_shape_logging_enabled():
            return
        self._emit_gemm("ffn_0")
        self._emit_gemm("ffn_2")

    def _emit_gemm_m(self, tag: str, m: int) -> None:
        if not _op_shape_logging_enabled():
            return
        spec = self._gemms[tag]
        ost.log_gemm(spec.region, spec.tag, m, spec.n, spec.k)

    def log_audio_adapter(self, n_q: int, n_kv: int) -> None:
        if not _op_shape_logging_enabled() or not self._has_audio_adapter:
            return
        self._audio_q_len = int(n_q)
        self._audio_kv_len = int(n_kv)
        self._emit_gemm_m("audio_q", self._audio_q_len)
        self._emit_gemm_m("audio_kv", self._audio_kv_len)
        ost.log_attn(
            "audio_adapter",
            "audio_perceiver",
            batch=1,
            num_heads=self.num_heads,
            seq_q=self._audio_q_len,
            seq_k=self._audio_kv_len,
            head_dim=self.head_dim,
        )
        self._emit_gemm_m("audio_o", self._audio_q_len)
