"""Opt-in Sage search keeps fixed defaults and caches measured layouts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lightx2v.common.kvcache.attn import sage_tuning
from lightx2v.common.kvcache.attn.sage import SageRingAttention


@pytest.mark.parametrize("candidates", [[], [0], [-1], [True], [64, 64], [64.0], "64,128"])
def test_candidate_list_is_explicit_and_unique(candidates):
    with pytest.raises(ValueError, match="sage_segment_candidates"):
        SageRingAttention(profile="minimax_h3", segment_candidates=candidates)


def test_encoded_value_candidates_keep_physical_alignment():
    candidates = (32, 64, 96, 128)
    assert sage_tuning.legal_segment_candidates(candidates, 192, 64, 64, value_only=True) == (32, 64)
    assert sage_tuning.legal_segment_candidates(candidates, 192, 64, 64) == candidates


def test_unspecified_candidates_use_fixed_policy():
    key = SimpleNamespace(shape=(1024, 2, 128), device=torch.device("cpu"))
    assert SageRingAttention(profile="minimax_h3").select_segment_tokens(key, write_tokens=128, architecture=90) == 256


def test_requested_candidates_are_measured_once_per_geometry(monkeypatch):
    key = SimpleNamespace(shape=(1024, 2, 128), device=torch.device("cuda:0"), dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    calls = []

    def measure(key, tokens, **options):
        calls.append(tokens)
        return {64: 2.0, 128: 1.0}[tokens]

    monkeypatch.setattr(sage_tuning, "_benchmark_segment", measure)
    monkeypatch.setattr(sage_tuning, "_SEGMENT_CHOICES", {})
    options = dict(baseline=256, write_tokens=128, visible_tokens=512)
    choice = sage_tuning.select_sage_segment_tokens(key, [64, 128], **options)
    assert choice.segment_tokens == 128
    assert choice.timings_ms == ((64, 2.0), (128, 1.0))
    assert sage_tuning.select_sage_segment_tokens(key, [64, 128], **options) is choice
    assert calls == [64, 128]
    sage_tuning.select_sage_segment_tokens(key, [64, 128], **(options | {"visible_tokens": 768}))
    assert calls == [64, 128, 64, 128]


def test_attention_opt_in_uses_measured_candidate(monkeypatch):
    key = SimpleNamespace(shape=(256, 2, 128), device=torch.device("cuda:0"), dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(sage_tuning, "_SEGMENT_CHOICES", {})
    calls = []

    def measure(key, tokens, **options):
        calls.append(tokens)
        return {64: 2.0, 128: 1.0}[tokens]

    monkeypatch.setattr(sage_tuning, "_benchmark_segment", measure)
    attention = SageRingAttention(profile="minimax_h3", segment_candidates=[64, 128])
    assert attention.select_segment_tokens(key, write_tokens=64, visible_tokens=128) == 128
    assert attention.select_segment_tokens(key, write_tokens=64, visible_tokens=128) == 128
    assert calls == [64, 128]


def test_uncached_search_rejects_graph_capture(monkeypatch):
    key = SimpleNamespace(shape=(1024, 2, 128), device=torch.device("cuda:0"), dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(sage_tuning, "_SEGMENT_CHOICES", {})
    with pytest.raises(RuntimeError, match="during warmup"):
        sage_tuning.select_sage_segment_tokens(key, [64, 128], baseline=256, write_tokens=128)


def test_no_legal_encoded_value_candidate_uses_policy_fallback():
    key = SimpleNamespace(shape=(192, 2, 128), device=torch.device("cuda:0"), dtype=torch.bfloat16)
    choice = sage_tuning.select_sage_segment_tokens(key, [128], baseline=64, write_tokens=64, sink_tokens=64, value_only=True)
    assert choice.segment_tokens == 64
    assert choice.timings_ms == ()
