"""真实同进程采集与CLI恢复；不依赖后续分析器。"""

import gzip
import json
import sys
import types

import pytest


def test_real_cpu_scheduler_ranges_and_output_preservation(tmp_path):
    import torch

    from lightx2v.common.ops.mm.mm_weight import MMWeight
    from tools.profile.capture import Capture

    mm = MMWeight("weight")
    mm.weight, mm.bias = torch.randn(4, 3), None
    x = torch.randn(2, 4)
    expected = mm.apply(x)
    calls = []

    def target():
        calls.append(len(calls) + 1)
        return mm.apply(x)

    output = tmp_path / "capture"
    with Capture(output, enabled=True, activities=("cpu",), warmup=2, count=2, workloads=True) as capture:
        for _ in range(4):
            assert torch.equal(capture.call(target), expected)
    assert calls == [1, 2, 3, 4]
    session = json.loads((output / "session.json").read_text())
    assert session["status"] == "complete" and session["captured"] == 2 and not session["record_shapes"]
    assert session["activities"] == ["cpu"] and not session["moe_distribution"]
    with gzip.open(output / session["trace_files"][0], "rt") as stream:
        trace = json.load(stream)
    markers = {e["name"] for e in trace["traceEvents"] if e.get("cat") == "user_annotation"}
    samples = [json.loads((path / "core_ops.jsonl").read_text()) for path in sorted(output.glob("call_*.stats"))]
    assert len(samples) == 2 and len({row["trace_marker"] for row in samples}) == 2
    assert all(row["trace_marker"] in markers for row in samples)
    assert all(row["work"]["logical_flops"] == 48 for row in samples)
    assert not any(e.get("args", {}).get("Input Dims") for e in trace["traceEvents"])
    assert not capture._entered


@pytest.mark.parametrize("failure", [False, True])
def test_cli_restores_inherited_method_argv_path_and_failure_state(tmp_path, monkeypatch, failure):
    from tools.profile.capture import main

    module = types.ModuleType("profile_test_target")

    class Parent:
        def forward(self):
            if failure:
                raise ValueError("target failure")
            return 7

    class Child(Parent):
        pass

    module.Child = Child
    monkeypatch.setitem(sys.modules, module.__name__, module)
    entry = tmp_path / "entry.py"
    entry.write_text("from profile_test_target import Child\nChild().forward()\n")
    previous_argv, previous_path = sys.argv, sys.path
    args = ["--target", "profile_test_target:Child.forward", "--output", str(tmp_path / "capture"), "--observe-only", str(entry)]
    if failure:
        with pytest.raises(ValueError, match="target failure"):
            main(args)
    else:
        assert main(args) == 0
    assert "forward" not in Child.__dict__ and Child.forward is Parent.forward
    assert sys.argv is previous_argv and sys.path is previous_path
    session = json.loads((tmp_path / "capture/session.json").read_text())
    assert session["status"] == ("failed" if failure else "complete")
    assert not session["trace_files"]


def test_compute_range_uses_existing_switch_cap_and_exception_cleanup(tmp_path):
    import torch

    from lightx2v.common.ops.moe.fused_moe import TorchExpertLoopFusedMoE
    from lightx2v.utils import workload_stats
    from tools.profile.capture import Capture

    op = TorchExpertLoopFusedMoE(torch.ones(2, 4, 3), torch.ones(2, 3, 4), activation="gelu")
    inputs = (torch.ones(2, 3), torch.tensor([[0, 1], [1, 0]]), torch.full((2, 2), 0.5))
    expected = op.apply(*inputs)
    with Capture(tmp_path / "disabled") as capture:
        assert torch.equal(capture.call(op.apply, *inputs), expected)
    assert not (tmp_path / "disabled").exists()
    with Capture(tmp_path / "stats", enabled=True, trace=False, activities=("cpu",), workloads=True) as capture:
        assert torch.equal(capture.call(op.apply, *inputs), expected)
    sample = json.loads((tmp_path / "stats/call_0001.stats/core_ops.jsonl").read_text())
    assert "trace_marker" not in sample and "compute_trace_marker" not in sample

    def twice():
        op.apply(*inputs)
        return op.apply(*inputs)

    with Capture(tmp_path / "capped", enabled=True, activities=("cpu",), warmup=0, workloads=True, stats_max_calls=1) as capture:
        assert torch.equal(capture.call(twice), expected)
    with gzip.open(next((tmp_path / "capped").glob("*.pt.trace.json.gz")), "rt") as stream:
        trace = json.load(stream)
    assert sum(e.get("name", "").endswith(".compute") for e in trace["traceEvents"]) == 1
    with Capture(tmp_path / "failed", enabled=True, activities=("cpu",), warmup=0, workloads=True) as capture:

        def fail(*args):
            raise ValueError("expert failure")

        op._run_experts = fail
        with pytest.raises(ValueError, match="expert failure"):
            capture.call(op.apply, *inputs)
    assert workload_stats._active is None
