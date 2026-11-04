"""CPU regressions for actual bases, exclusive sessions, sampling, errors, default-off behavior, and coverage gaps."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from textwrap import dedent

import pytest
import torch

from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight
from lightx2v.common.ops.conv.conv2d import Conv2dWeight
from lightx2v.common.ops.mm.mm_weight import MMWeight
from lightx2v.common.ops.moe.fused_moe import TorchExpertLoopFusedMoE
from lightx2v.utils import workload_stats
from lightx2v.utils.core_op_stats import CoreOpStatistics


def rows(path):
    return [json.loads(line) for line in (path / "core_ops.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("distribution", [False, True])
@pytest.mark.parametrize("export", [False, True])
def test_moe_distribution_and_export_are_independent_and_capped(tmp_path, monkeypatch, distribution, export):
    op = TorchExpertLoopFusedMoE(torch.ones(3, 4, 3), torch.ones(3, 3, 4), "gelu")
    x, ids, scales = torch.ones(2, 3), torch.tensor([[0, 1], [0, 1]]), torch.ones(2, 2)
    expected = op.apply(x, ids, scales)
    exported, prepared = [], []
    original = op.replay_inputs

    def prepare(arguments):
        prepared.append(1)
        return original(arguments)

    def save(sample):
        exported.append(sample)
        return {"recorded": True}

    monkeypatch.setattr(op, "replay_inputs", prepare)
    with workload_stats.WorkloadStats(tmp_path / "stats", moe_distribution=distribution, on_moe_inputs=save if export else None, max_calls=1):
        for _ in range(2):
            value = x.clone()
            assert torch.equal(op.apply(value, ids, scales, output=value), expected)
    (row,) = rows(tmp_path / "stats")
    assert row["work"]["logical_flops"] == 192
    assert ("routing" in row) is distribution
    assert ("input_artifact" in row) is export
    assert len(prepared) == len(exported) == int(export)
    if distribution:
        assert row["routing"]["expert_counts"] == [2, 2, 0]
    if export:
        assert torch.equal(exported[0]["inputs"]["activation"], x)
        assert torch.equal(exported[0]["inputs"]["expert_ids"], ids)
        assert torch.equal(exported[0]["output"], expected)
        assert exported[0]["inputs"]["activation"].untyped_storage().data_ptr() != exported[0]["output"].untyped_storage().data_ptr()


def test_moe_export_failure_marks_session_failed_and_allows_recovery(tmp_path):
    op = TorchExpertLoopFusedMoE(torch.ones(2, 4, 3), torch.ones(2, 3, 4), "gelu")
    x, ids, scales = torch.ones(2, 3), torch.zeros(2, 1, dtype=torch.int64), torch.ones(2, 1)
    expected = op.apply(x, ids, scales)

    def reject(sample):
        assert torch.equal(sample["output"], expected)
        raise OSError("export failure")

    with pytest.raises(OSError, match="export failure"):
        with workload_stats.WorkloadStats(tmp_path / "failed", on_moe_inputs=reject):
            op.apply(x, ids, scales)
    assert json.loads((tmp_path / "failed/session.json").read_text())["status"] == "failed"
    assert rows(tmp_path / "failed") == []
    with workload_stats.WorkloadStats(tmp_path / "recovered"):
        assert torch.equal(op.apply(x, ids, scales), expected)
    assert len(rows(tmp_path / "recovered")) == 1


def test_concurrent_session_creation_is_exclusive(tmp_path, monkeypatch):
    in_mkdir, release_mkdir, second_mkdir, second_done = (threading.Event() for _ in range(4))
    original = Path.mkdir
    errors, rejected = [], []

    def mkdir(path, *args, **kwargs):
        if path == tmp_path / "first":
            in_mkdir.set()
            assert release_mkdir.wait(5)
        elif path == tmp_path / "second":
            second_mkdir.set()
        return original(path, *args, **kwargs)

    def first():
        try:
            with workload_stats.WorkloadStats(tmp_path / "first"):
                assert second_done.wait(5)
                assert workload_stats.enabled("gemm")
        except BaseException as error:
            errors.append(error)

    def second():
        try:
            with workload_stats.WorkloadStats(tmp_path / "second"):
                errors.append(AssertionError("two sessions entered"))
        except RuntimeError as error:
            rejected.append(str(error))
        finally:
            second_done.set()

    monkeypatch.setattr(Path, "mkdir", mkdir)
    one, two = threading.Thread(target=first), threading.Thread(target=second)
    try:
        one.start()
        assert in_mkdir.wait(5)
        two.start()
        assert not second_mkdir.wait(0.05)
    finally:
        release_mkdir.set()
        for thread in (one, two):
            if thread.ident is not None:
                thread.join(timeout=6)
    assert not one.is_alive() and not two.is_alive() and not errors
    assert len(rejected) == 1 and "already active" in rejected[0]
    assert not (tmp_path / "second").exists()
    with workload_stats.WorkloadStats(tmp_path / "recovered"):
        assert workload_stats.enabled("gemm")


def test_packed_kv_keeps_output_and_reports_missing_shape_contract(tmp_path, monkeypatch):
    from lightx2v.common.ops.attn import sage_attn

    op = sage_attn.SageAttn2KInt8VFP8Weight()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(op, "quant_query_per_thread_int8", lambda q, **kwargs: (q.to(torch.int8), torch.ones(1)))

    def native(q, k, v, output, *args):
        output.copy_(q)

    monkeypatch.setattr(sage_attn, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf", native, raising=False)
    q = torch.ones(3, 2, 4, dtype=torch.bfloat16)
    kv = (q.to(torch.int8), torch.ones(1))
    expected = op.apply(q, kv, kv)
    with workload_stats.WorkloadStats(tmp_path / "stats"):
        assert torch.equal(op.apply(q, kv, kv), expected)
    (row,) = rows(tmp_path / "stats")
    assert row["status"] == "unsupported" and "packed/quantized KV" in row["unsupported_reason"]


def test_disabled_and_cap_do_not_read_region_and_failure_cleans_up(tmp_path):
    q = torch.ones(1, 1, 3, 4)
    workload_stats.record_sparse_attn(q, q, q, object(), operator="new_backend")
    target = tmp_path / "stats"
    with workload_stats.WorkloadStats(target, families=["attention"], max_calls=1) as session:
        workload_stats.record_sparse_attn(q, q, q, torch.ones(1, 1, 1, 1), 3, 3, operator="new_backend")
        workload_stats.record_sparse_attn(q, q, q, object(), operator="new_backend")
        assert session._core_stack == []
    assert len((target / "workloads.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="binary"):
        with workload_stats.WorkloadStats(tmp_path / "failed", families=["attention"]):
            workload_stats.record_sparse_attn(q, q, q, torch.full((1, 1, 1, 1), 2), 3, 3, operator="new_backend")
    assert not workload_stats.enabled("attention")
    assert json.loads((tmp_path / "failed/session.json").read_text())["status"] == "failed"


def test_bases_compute_from_actual_input_output_and_default_off(tmp_path):
    mm = MMWeight("w")
    mm.weight, mm.bias = torch.randn(3, 4), None
    x = torch.randn(2, 3)
    expected = mm.apply(x)
    conv = Conv2dWeight("w", None, groups=2)
    conv.weight, conv.bias = torch.ones(4, 1, 3, 3), None
    attention = TorchSDPAWeight()
    q = torch.randn(3, 2, 4)
    with workload_stats.WorkloadStats(
        tmp_path / "stats",
    ):
        assert torch.equal(mm.apply(input_tensor=x), expected)
        y = conv.apply(torch.ones(1, 2, 5, 5))
        attention.apply(q, q, q)
    recorded = rows(tmp_path / "stats")
    assert [row["work"].get("logical_flops", row["work"].get("dense_equivalent_flops")) for row in recorded] == [48, 2 * y.numel() * 9, 2 * 2 * 3 * 3 * 8]
    assert (tmp_path / "stats/core_ops.md").exists()
    assert not workload_stats.enabled("gemm")
    assert torch.equal(mm.apply(x), expected)


def test_first_production_import_and_call_need_no_tools(tmp_path):
    program = dedent("""
        import builtins, json, sys
        original = builtins.__import__
        def without_tools(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and (name == "tools" or name.startswith("tools.")):
                raise ModuleNotFoundError("wheel has no tools", name=name)
            return original(name, globals, locals, fromlist, level)
        builtins.__import__ = without_tools
        import torch
        from lightx2v.common.ops.mm.mm_weight import MMWeight
        from lightx2v.utils.workload_stats import WorkloadStats
        mm = MMWeight("w")
        mm.weight, mm.bias = torch.ones(3, 4), None
        with WorkloadStats(sys.argv[1]):
            assert torch.equal(mm.apply(torch.ones(2, 3)), torch.full((2, 4), 3.0))
        row = json.loads(open(sys.argv[1] + "/core_ops.jsonl").read())
        assert row["work"]["logical_flops"] == 48
    """)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": "", "PLATFORM": "cuda", "SKIP_PLATFORM_CHECK": "1"}
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path / "stats")], cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("activation,split", [("gelu", False), ("swiglu", False), ("swiglu", True)])
def test_moe_actual_ids_per_expert_and_zero_scales_do_not_skip(tmp_path, activation, split):
    fc1 = torch.randn(3, 4 if activation == "gelu" or split else 8, 3)
    fc2 = torch.randn(3, 3, 4)
    gate = torch.randn(3, 4, 3) if split else None
    op = TorchExpertLoopFusedMoE(fc1, fc2, activation, fc1_gate_weight=gate)
    x = torch.randn(4, 3)
    ids_a = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    ids_b = torch.tensor([[0, 2], [0, 2], [0, 2], [0, 2]])
    scales = torch.zeros(4, 2)
    baseline = op.apply(x, ids_a, scales)
    with workload_stats.WorkloadStats(tmp_path / "stats", max_calls=2, moe_distribution=True):
        assert torch.equal(op.apply(x, ids_a, scales), baseline)
        op.apply(x, ids_b, scales)
    a, b = rows(tmp_path / "stats")
    assert a["routing"]["expert_counts"] == [4, 4, 0]
    assert b["routing"]["expert_counts"] == [4, 0, 4]
    assert a["work"]["logical_flops"] == b["work"]["logical_flops"] == 8 * 2 * 3 * 4 * (2 if activation == "gelu" else 3)
    assert a["routing"]["source"] == b["routing"]["source"] == "production_expert_counts"
    assert a["work"]["routes"] == 8
    assert a["work"]["scheduled_tile_flops"] is None


def test_core_caps_super_nested_calls_exception_and_compile_disabled_path(tmp_path):
    class Parent(CoreOpStatistics):
        compute_family = "gemm"

        def apply(self, x):
            return x + 1

        def compute_stats(self, arguments, output):
            return {"work": {"logical_flops": output.numel()}}

    class Child(Parent):
        def apply(self, x):
            return super().apply(x)

    class Composite(Parent):
        def apply(self, x):
            return Child().apply(x)

    class Failed(Parent):
        def apply(self, x):
            raise ValueError("original error")

    with workload_stats.WorkloadStats(tmp_path / "stats", max_calls=1):
        Composite().apply(torch.ones(2))
        Child().apply(torch.ones(2))
        with pytest.raises(ValueError, match="original error"):
            Failed().apply(torch.ones(2))
    recorded = rows(tmp_path / "stats")
    assert len(recorded) == 2
    assert recorded[0]["parent_call"] == recorded[1]["call"]
    session = json.loads((tmp_path / "stats/session.json").read_text())
    assert sum(session["core_op_observed"].values()) == 4
    assert any("sample limit" in gap for gap in session["coverage_gaps"])
    x = torch.ones(2)
    assert torch.equal(torch.compile(Child().apply, backend="eager", fullgraph=True)(x), x + 1)
    with pytest.raises(ValueError, match="original error"):
        with workload_stats.WorkloadStats(
            tmp_path / "failed",
        ):
            Failed().apply(x)
    assert not workload_stats.enabled("gemm")


def test_meta_calls_and_core_only_sparse_sample_cap_do_not_copy_again(tmp_path, monkeypatch):
    mm = MMWeight("w")
    mm.weight, mm.bias = torch.empty(3, 4, device="meta"), None
    with workload_stats.WorkloadStats(
        tmp_path / "meta",
    ):
        assert mm.apply(torch.empty(2, 3, device="meta")).is_meta
    assert not rows(tmp_path / "meta")
    assert any("fake/meta" in key for key in json.loads((tmp_path / "meta/session.json").read_text())["coverage_gaps"])

    class Sparse(CoreOpStatistics):
        compute_family = "attention"

        def apply(self, q, mask):
            if workload_stats.enabled("attention"):
                workload_stats.record_sparse_attn(q, q, q, mask, 2, 2, 0.5, "fixture")
            return q

        def compute_stats(self, arguments, output):
            return {"work": {"dense_equivalent_flops": 144}}

    with workload_stats.WorkloadStats(tmp_path / "cap", max_calls=1, moe_distribution=True):
        op = Sparse()
        q = torch.randn(1, 1, 3, 4)
        op.apply(q, torch.ones(1, 1, 2, 2))
        op.apply(q, None)  # Beyond cap: no second mask read.
    recorded = rows(tmp_path / "cap")
    assert len(recorded) == 1 and len(recorded[0]["actual_backend_routes"]) == 1
    assert len((tmp_path / "cap/workloads.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("cap", [1, 2])
def test_same_backend_nested_calls_reserve_cap_and_release_after_failure(tmp_path, cap):
    class Node(CoreOpStatistics):
        compute_family = "gemm"

        def __init__(self, child=None, fail=False):
            self.child, self.fail = child, fail

        def apply(self, x):
            if self.fail:
                raise ValueError("node failure")
            return self.child.apply(x) if self.child else x + 1

        def compute_stats(self, arguments, output):
            return {"work": {"logical_flops": output.numel()}}

    with workload_stats.WorkloadStats(tmp_path / "stats", max_calls=cap):
        with pytest.raises(ValueError, match="node failure"):
            Node(Node(fail=True)).apply(torch.ones(2))
        Node(Node()).apply(torch.ones(2))
    recorded = rows(tmp_path / "stats")
    assert len(recorded) == cap
    session = json.loads((tmp_path / "stats/session.json").read_text())
    assert list(session["core_op_observed"].values()) == [4]
    if cap == 1:
        assert recorded[0]["parent_call"] is None
        assert any("sample limit" in gap for gap in session["coverage_gaps"])
    else:
        assert recorded[0]["parent_call"] == recorded[1]["call"]


def test_cuda_capture_guard_is_not_skipped_after_sample_cap(tmp_path, monkeypatch):
    class Node(CoreOpStatistics):
        compute_family = "gemm"

        def apply(self, x):
            return x + 1

    class CudaFlagTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    with workload_stats.WorkloadStats(tmp_path / "stats", max_calls=1):
        op = Node()
        op.apply(torch.ones(2))
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="CUDA graph capture"):
            op.apply(torch.ones(2).as_subclass(CudaFlagTensor))
    assert len(rows(tmp_path / "stats")) == 1


def test_uniform_moe_total_reads_shape_not_ids_and_default_histogram_is_off(tmp_path):
    op = TorchExpertLoopFusedMoE(torch.randn(3, 8, 3), torch.randn(3, 3, 4), "swiglu")

    x, ids = torch.randn(4, 3), torch.zeros(4, 2, dtype=torch.int64)
    # Meta preserves shape/numel but rejects value reads and CPU copies, independently of detach().
    shape_only_ids = torch.empty(4, 2, dtype=torch.int64, device="meta")
    facts = op.compute_stats({"input": x, "token_selected_experts": shape_only_ids}, None)
    assert facts["work"]["logical_flops"] == 8 * 2 * 3 * 4 * 3
    with workload_stats.WorkloadStats(
        tmp_path / "stats",
    ):
        op.apply(x, ids, torch.ones(4, 2))
    assert "routing" not in rows(tmp_path / "stats")[0]


def test_composite_attention_aux_boundary_uses_actual_child_not_outer_shape(tmp_path):
    import torch.distributed as dist

    from lightx2v.common.ops.attn.ring_attn import RingAttnWeight
    from lightx2v.common.ops.attn.ulysses_attn import UlyssesAttnWeight

    assert RingAttnWeight().compute_stats({}, None)["status"] == "composite"
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method="file://" + str(tmp_path / "gloo"), rank=0, world_size=1)
    try:
        q, aux = torch.randn(3, 2, 4), torch.randn(1, 2, 4)
        with workload_stats.WorkloadStats(
            tmp_path / "stats",
        ):
            UlyssesAttnWeight().apply(q, q, q, aux_q=aux, aux_k=aux, aux_v=aux, attention_module=TorchSDPAWeight())
        child, outer = rows(tmp_path / "stats")
        assert child["work"]["dense_equivalent_flops"] == 512
        assert outer["status"] == "composite" and "work" not in outer
        assert child["parent_call"] == outer["call"]
    finally:
        dist.destroy_process_group()
