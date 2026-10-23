"""Real MoE operands and production preparation; CPU tests do not prove CUDA execution."""

import hashlib
import json
import sys
import weakref

import pytest
import torch

from lightx2v.common.ops.moe.fused_moe import TorchExpertLoopFusedMoE
from lightx2v.utils import workload_stats
from tools.benchmarks.operator_backends import TorchExpertLoopAdapter, TorchGroupedMMAdapter
from tools.benchmarks.operator_bench import validate_suite


def capture(tmp_path, activation="swiglu", split=True, dtype=torch.bfloat16, scale_dtype=torch.float32):
    torch.manual_seed(7)
    fc1 = torch.randn(3, 4 if split or activation == "gelu" else 8, 3, dtype=dtype) * 0.1
    fc2 = torch.randn(3, 3, 4, dtype=dtype) * 0.1
    gate = torch.randn_like(fc1) if split else None
    b1 = None if split else torch.randn(3, fc1.shape[1], dtype=dtype) * 0.1
    b2 = torch.randn(3, 3, dtype=dtype) * 0.1
    op = TorchExpertLoopFusedMoE(fc1, fc2, activation, b1, b2, gate)
    x = torch.randn(4, 3, dtype=dtype)
    ids = torch.tensor([[0, 2], [0, 2], [0, 1], [2, 1]], dtype=torch.int64)
    scales = torch.tensor([[0.0, 0.3], [0.1, 0.7], [0.2, 0.5], [0.3, 0.9]], dtype=scale_dtype)
    artifacts = []

    def save(sample):
        # This test owns its file format, just as a task-local plugin would.
        path = tmp_path / "operands.pt"
        torch.save(sample["inputs"], path)
        artifact = {"manifest": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        artifacts.append(artifact)
        return artifact

    with workload_stats.WorkloadStats(tmp_path / "stats", families=["moe"], max_calls=1, on_moe_inputs=save):
        actual = op.apply(x, ids, scales)
        op.apply(x + 1, ids, scales)
    assert len(artifacts) == 1 and workload_stats._active is None
    case = {"case_id": "actual", "operator_family": "moe",
            "shape": {"tokens": 4, "hidden_size": 3, "intermediate_size": 4, "num_experts": 3,
                      "top_k": 2, "activation": activation, "expert_bias": b1 is not None or b2 is not None},
            "precision": {"input_dtype": {torch.float32: "fp32", torch.bfloat16: "bf16"}[dtype]},
            "replay": artifacts[0], "source": {"origin": "synthetic_cpu_control_flow"}}
    validate_suite({"schema_version": 1, "kind": "operator_benchmark_shape_suite_v1", "suite_id": "actual", "cases": [case]})
    return case, {"activation": x, "fc1": fc1, "fc2": fc2, "fc1_gate": gate, "fc1_bias": b1,
                  "fc2_bias": b2, "expert_ids": ids, "scales": scales}, actual


@pytest.mark.parametrize("activation,split", [("gelu", False), ("swiglu", False), ("swiglu", True)])
def test_actual_inputs_reach_the_production_backends(tmp_path, activation, split):
    case, expected, actual = capture(tmp_path, activation, split)
    inputs = torch.load(case["replay"]["manifest"], weights_only=True)
    assert all(value is None or torch.equal(inputs[name], value) for name, value in expected.items())
    prepared = []
    for adapter in (TorchExpertLoopAdapter(), TorchGroupedMMAdapter()):
        torch.manual_seed(999)
        prepared.append(adapter.prepare(case, "cpu", {"moe_inputs": inputs}))
    assert torch.equal(prepared[0].fn(), actual)
    matrices = [inputs["fc1"].shape[1:], inputs["fc2"].shape[1:]] + ([inputs["fc1_gate"].shape[1:]] if split else [])
    oracle = inputs["expert_ids"].numel() * sum(2 * rows * columns for rows, columns in matrices)
    core = json.loads((tmp_path / "stats/core_ops.jsonl").read_text())
    assert core["work"]["logical_flops"] == prepared[0].work == prepared[1].work == oracle
    assert prepared[0].extra_metrics["routing_distribution"] == "captured_inputs"
    assert core["input_artifact"] == case["replay"]


@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.float64])
def test_capture_uses_production_fp32_route_scales(tmp_path, scale_dtype):
    case, original, actual = capture(tmp_path, dtype=torch.float32, scale_dtype=scale_dtype)
    inputs = torch.load(case["replay"]["manifest"], weights_only=True)
    assert original["scales"].dtype == scale_dtype
    assert inputs["scales"].dtype == torch.float32
    assert torch.equal(inputs["scales"], original["scales"].float())
    adapter = TorchExpertLoopAdapter()
    assert adapter.support_error(case) is None
    assert torch.equal(adapter.prepare(case, "cpu", {"moe_inputs": inputs}).fn(), actual)


@pytest.mark.parametrize("buffer", ["activation", "activation_view", "scales"])
def test_output_buffer_alias_preserves_pre_call_operands(tmp_path, buffer):
    op = TorchExpertLoopFusedMoE(torch.eye(2).repeat(2, 1, 1), torch.eye(2).repeat(2, 1, 1), "gelu")
    x, ids, scales = torch.tensor([[1.0, 2.0]]), torch.tensor([[0, 1]]), torch.tensor([[0.2, 0.7]])
    original_x, original_scales = x.clone(), scales.clone()
    expected = op.apply(x.clone(), ids, scales.clone())
    output = {"activation": x, "activation_view": x.view_as(x), "scales": scales}[buffer]
    saved = []
    with workload_stats.WorkloadStats(tmp_path / "stats", families=["moe"], on_moe_inputs=lambda sample: saved.append(sample["inputs"])):
        assert torch.equal(op.apply(x, ids, scales, output=output), expected)
    inputs = saved[0]
    assert torch.equal(inputs["activation"], original_x) and torch.equal(inputs["scales"], original_scales)


def test_fixed_inputs_require_identity_and_never_fall_back_to_random(tmp_path):
    case, _, _ = capture(tmp_path)
    inputs = torch.load(case["replay"]["manifest"], weights_only=True)
    with pytest.raises(ValueError, match="task-local plugin"):
        TorchExpertLoopAdapter().prepare(case, "cpu", {})
    with pytest.raises(ValueError, match="replay identity"):
        TorchExpertLoopAdapter().prepare({**case, "replay": None}, "cpu", {"moe_inputs": inputs})


def test_unsupported_export_is_a_gap_not_a_changed_forward(tmp_path):
    op = TorchExpertLoopFusedMoE(torch.ones(1, 2, 3), torch.ones(1, 4, 2), "gelu")
    x, ids, scales = torch.ones(1, 3), torch.zeros(1, 1, dtype=torch.int64), torch.ones(1, 1)
    expected = op.apply(x, ids, scales)
    with workload_stats.WorkloadStats(tmp_path / "stats", families=["moe"], on_moe_inputs=lambda sample: pytest.fail("unsupported export")):
        assert torch.equal(op.apply(x, ids, scales), expected)
    row = json.loads((tmp_path / "stats/core_ops.jsonl").read_text())
    assert row["input_export_gap"] and row["work"]["logical_flops"] == 28


@pytest.mark.parametrize("tuning_fails", [False, True])
def test_flashinfer_preparation_tunes_once_outside_timed_callable(tmp_path, monkeypatch, tuning_fails):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from lightx2v.common.ops.moe import fused_moe
    from tools.benchmarks.operator_backends import FlashInferMoEAdapter

    case, expected, _ = capture(tmp_path)
    calls, active, weights = [], [], []

    def prepare():
        inputs = torch.load(case["replay"]["manifest"], weights_only=True)
        weights.extend(weakref.ref(inputs[name]) for name in ("fc1", "fc1_gate"))
        return FlashInferMoEAdapter().prepare(case, "cpu", {"moe_inputs": inputs})

    @contextmanager
    def autotune(tune_mode=True, **kwargs):
        assert kwargs == {"tuning_buckets": (case["shape"]["tokens"],)}
        active.append(tune_mode)
        try:
            if tuning_fails and tune_mode:
                raise RuntimeError("autotune fixture failure")
            yield
        finally:
            active.pop()

    def apply(op, x, ids, scales):
        calls.append(active[-1] if active else None)
        assert op.tune_max_num_tokens == x.shape[0]
        assert torch.equal(op.shards[0].fc1_weight, torch.cat((expected["fc1"], expected["fc1_gate"]), dim=1))
        return x

    monkeypatch.setattr(fused_moe.FlashInferFusedMoE, "apply", apply)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(autotune=autotune))
    if tuning_fails:
        with pytest.raises(RuntimeError, match="autotune fixture failure"):
            prepare()
        assert not calls and not active
    else:
        prepared = prepare()
        assert calls == [True] and not active
        assert len(weights) == 2 and all(weight() is None for weight in weights)
        prepared.fn()
        assert calls == [True, False]
        assert prepared.extra_metrics["autotune"]["tune_max_num_tokens"] == case["shape"]["tokens"]
        assert prepared.extra_metrics["autotune"]["inference_tuning_buckets"] == [case["shape"]["tokens"]]

        def reject(*args):
            raise RuntimeError("inference fixture failure")

        monkeypatch.setattr(fused_moe.FlashInferFusedMoE, "apply", reject)
        with pytest.raises(RuntimeError, match="inference fixture failure"):
            prepared.fn()
        assert not active
