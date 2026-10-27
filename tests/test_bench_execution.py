"""CPU runner regressions; CUDA context and timing are explicit substitutes."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools.benchmarks import operator_bench as bench
from tools.benchmarks.operator_backends import BackendDescriptor, PreparedOperation


@pytest.mark.parametrize("stage,fault", [(s, f) for s in ("preflight", "measure") for f in ("none", "nonfinite", "shape")] + [("measure", "valid")])
def test_runner_checks_structure_without_math_reference(monkeypatch, stage, fault):
    actual = torch.ones(2)
    bad = {"none": None, "nonfinite": torch.full((2,), float("nan")), "shape": torch.ones(3), "valid": actual}[fault]
    calls = []
    prepared = PreparedOperation(lambda: bad if stage == "preflight" else actual, 8, "tflops", "fixture", expected_output_shape=(2,))
    adapter = SimpleNamespace(
        descriptor=BackendDescriptor("fixture", "gemm", "test", ("fp32",), "CPU runner"), support_error=lambda _: None, precision=lambda c: c["precision"], prepare=lambda *args: prepared
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: None)

    def measure(fn, *args, **kwargs):
        calls.append("measure")
        return [1.0], bad

    monkeypatch.setattr(bench, "_measure", measure)
    case = {"case_id": "fixture", "operator_family": "gemm", "shape": {"m": 1, "n": 2, "k": 2, "bias": False}, "precision": {"input_dtype": "fp32"}}
    options = dict(measurement_seed=42, seed=42, device="cpu", warmup=0, iterations=1, prewarm_seconds=0, moe_routing="balanced", timing_mode=bench.TIMING_MODE, rate_schema=bench.RATE_SCHEMA)
    row = bench.run_case(case, adapter, "repeat000", {}, options)
    assert calls == ([] if stage == "preflight" else ["measure"])
    if fault == "valid":
        assert row["status"] == "ok" and row["correctness"]["checked"] is False
    else:
        assert row["status"] == "error" and row["error"]["stage"] == stage and "metrics" not in row
