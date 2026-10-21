"""区间与关联oracle、真实CPU采集/恢复；native CUDA另行验收。"""

import json

import pytest

from tools.profile import trace_report as report


def event(name, category, start, duration, *, pid=100, tid=1, **args):
    return {"ph": "X", "name": name, "cat": category, "ts": start, "dur": duration, "pid": pid, "tid": tid, "args": args}


def test_interval_union_and_nested_cpu_self_time():
    events = [event("A", "user_annotation", 0, 100), event("B", "cpu_op", 10, 40), event("C", "cpu_op", 40, 40), event("D", "cpu_op", 20, 10), event("E", "cpu_op", 30, 5)]
    assert report.union_us([(10, 50), (40, 80), (20, 30)]) == 70
    assert report.cpu_self_times(events, range(5)) == {0: 30, 1: 25, 2: 40, 3: 10, 4: 5}


def two_kernel_trace():
    events = [
        event("target", "user_annotation", 0, 10),
        event("lightx2v.core.unique.1", "user_annotation", 1, 8),
        event("aten::mm", "cpu_op", 2, 6, **{"External id": 3}),
        event("cudaLaunchKernel", "runtime", 3, 1, correlation=0, **{"External id": 3}),
        event("gemm_first", "kernel", 20, 20, pid=0, tid=2, correlation=0, device=0),
        event("gemm_second", "kernel", 30, 20, pid=0, tid=3, correlation=0, device=0),
    ]
    sample = {
        "kind": "core_op_compute_stats",
        "trace_marker": "lightx2v.core.unique.1",
        "operator_family": "gemm",
        "backend": "real.MM",
        "shape": {"m": 2, "n": 3, "k": 4},
        "work": {"logical_flops": 48, "definition": "2mnk"},
        "actual_input_dtypes": {"input_tensor": "bf16"},
        "actual_output_dtype": "bf16",
    }
    return {"traceEvents": events}, sample


def test_async_tail_multiple_kernels_and_no_shape_fallback():
    trace, sample = two_kernel_trace()
    actual = report.analyze(trace, scope="target", samples=[sample])
    assert actual["coverage"]["selected_gpu_events"] == 2
    assert actual["devices"]["0"]["span_ms"] == pytest.approx(0.030)
    assert actual["devices"]["0"]["activity_union_ms"] == pytest.approx(0.030)
    (row,) = actual["core_calls"]
    assert row["logical_flops"] == 48 and row["gpu_ms"] == pytest.approx(0.030)
    assert len(row["main_gpu_event_indices"]) == 2
    assert row["compute_tflops"] == pytest.approx(48 / 30 / 1e6)
    trace["traceEvents"][2]["args"]["Input Dims"] = [[999, 999], [999, 999]]
    assert report.analyze(trace, scope="target")["core_calls"] == []


def test_composite_parent_auxiliary_does_not_receive_inclusive_flops():
    trace, sample = two_kernel_trace()
    trace["traceEvents"].extend(
        [
            event("lightx2v.core.parent.1", "user_annotation", 0, 10),
            event("aten::cat", "cpu_op", 9, 1, **{"External id": 4}),
            event("at::native::CatArrayBatchedCopy", "kernel", 70, 5, pid=0, device=0, **{"External id": 4}),
        ]
    )
    parent = {**sample, "trace_marker": "lightx2v.core.parent.1", "work": {"logical_flops": 9999, "dense_equivalent_flops": 9999}}
    facts = report.analyze(trace, samples=[parent, sample])
    (row,) = facts["core_calls"]
    assert row["marker"] == sample["trace_marker"] and row["logical_flops"] == 48
    assert facts["coverage"]["selected_gpu_events"] == 3 and not facts["coverage"]["binding_gaps"]
    assert any(row["full_name"] == "at::native::CatArrayBatchedCopy" and row["gpu_ms"] == pytest.approx(0.005) for row in facts["auxiliary_groups"])


def test_clean_trace_classifies_native_gemm_without_workload_samples():
    trace, _ = two_kernel_trace()
    for entry in trace["traceEvents"]:
        if entry["cat"] == "kernel":
            entry["name"] = "nvjet_sm90_opaque_name"
    trace["traceEvents"].extend([event("elementwise_epilogue", "kernel", 50, 2, pid=0, correlation=0, device=0), event("opaque_other", "kernel", 53, 1, pid=0, device=0)])
    facts = report.analyze(trace)
    assert facts["core_calls"] == []
    assert sum(row["count"] for row in facts["gpu_groups"] if row["family"] == "gemm") == 2
    (unknown,) = [row for row in facts["gpu_groups"] if row["family"] == "unknown"]
    assert unknown["classification_gap"] == "unrecognized_kernel_and_native"
    assert any("类型待识别" in row["name"] for row in facts["auxiliary_groups"])


@pytest.mark.parametrize("scope_name", ["lightx2v::dynamic_sparse_sage2", "aten::scaled_dot_product_attention", "unverified_attention"])
@pytest.mark.parametrize("same_thread", [True, False])
def test_routing_gemm_role_requires_verified_native_scope(scope_name, same_thread):
    trace, _ = two_kernel_trace()
    trace["traceEvents"].append(event(scope_name, "cpu_op", 0, 10, tid=1 if same_thread else 9))
    facts = report.analyze(trace)
    is_routing = scope_name == "lightx2v::dynamic_sparse_sage2" and same_thread
    assert {row["family"] for row in facts["gpu_groups"]} == {"gemm"}
    assert {row["work_role"] for row in facts["gpu_groups"]} == {"auxiliary" if is_routing else "core"}
    assert bool(facts["unbound_core_groups"]) != is_routing
    assert facts["devices"]["0"]["activity_union_ms"] == pytest.approx(0.030)


def test_scope_selection_does_not_report_other_calls_missing():
    trace, sample = two_kernel_trace()
    second, other = two_kernel_trace()
    for entry in second["traceEvents"]:
        entry["ts"] += 100
        if entry["name"] == sample["trace_marker"]:
            entry["name"] += ".second"
        if "correlation" in entry["args"]:
            entry["args"]["correlation"] = 1
        if "External id" in entry["args"]:
            entry["args"]["External id"] = 4
    other["trace_marker"] += ".second"
    trace["traceEvents"].extend(second["traceEvents"])
    for occurrence in (1, 2):
        result = report.analyze(trace, scope="target", occurrence=occurrence, samples=[sample, other])
        assert len(result["core_calls"]) == 1
        assert result["core_calls"][0]["marker"] == (sample if occurrence == 1 else other)["trace_marker"]
        assert result["coverage"]["binding_gaps"] == []
    # 真正缺失和重复身份仍是缺口，不用scope过滤掩盖。
    missing = {**sample, "trace_marker": "absent"}
    result = report.analyze(trace, scope="target", samples=[sample, other, missing])
    assert result["coverage"]["binding_gaps"] == [{"marker": "absent", "reason": "marker_missing_or_not_unique"}]
    trace["traceEvents"].append(dict(trace["traceEvents"][1]))
    result = report.analyze(trace, scope="target", samples=[sample, other])
    assert result["coverage"]["binding_gaps"] == [{"marker": sample["trace_marker"], "reason": "marker_missing_or_not_unique"}]


def test_cross_process_correlation_ambiguity_is_not_guessed():
    trace, sample = two_kernel_trace()
    trace["traceEvents"].append(event("cudaLaunchKernel", "runtime", 3, 1, pid=101, correlation=0, **{"External id": 3}))
    actual = report.analyze(trace, scope="target", samples=[sample])
    assert actual["coverage"]["selected_gpu_events"] == 0
    assert len(actual["coverage"]["ambiguous_process_gpu_events"]) == 2
    assert not actual["core_calls"]


def test_copy_fill_and_visible_other_work_reduce_gaps():
    trace, _ = two_kernel_trace()
    trace["traceEvents"].extend(
        [
            event("copy", "gpu_memcpy", 70, 10, pid=0, correlation=0, device=0),
            event("fill", "gpu_memset", 75, 10, pid=0, correlation=0, device=0),
            event("unrelated", "kernel", 50, 10, pid=0, device=0),
        ]
    )
    facts = report.analyze(trace, scope="target")["devices"]["0"]
    assert facts["span_ms"] == pytest.approx(0.065)
    assert facts["activity_union_ms"] == pytest.approx(0.045)
    assert facts["observed_gap_ms"] == pytest.approx(0.010)
    assert facts["other_or_unattributed_active_ms"] == pytest.approx(0.010)
    assert {r["category"] for r in facts["activity"]} == {"core", "gpu_memcpy", "gpu_memset"}


def test_sync_wait_intersection_locates_without_claiming_cause():
    trace = {
        "traceEvents": [
            event("target", "user_annotation", 0, 100),
            event("cudaStreamSynchronize", "runtime", 20, 60, correlation=1),
            event("launch", "runtime", 1, 1, correlation=2),
            event("first", "kernel", 0, 20, pid=0, correlation=2, device=0),
            event("last", "kernel", 60, 20, pid=0, correlation=2, device=0),
        ]
    }
    facts = report.analyze(trace, scope="target")
    gap = facts["gap_diagnostics"]["0"]
    assert facts["devices"]["0"]["observed_gap_ms"] == pytest.approx(0.040)
    assert gap["synchronizations"][0]["gap_overlap_ms"] == pytest.approx(0.040)
    assert gap["synchronizations"][0]["callers"] == ["target"]
    text = report.markdown(facts)
    assert "不能自动认定为原因" in text and "小kernel" not in text


def test_sparse_unknown_actual_keeps_shape_and_dense_equivalent():
    trace, sample = two_kernel_trace()
    sample.update(operator_family="attention", shape={"q": [5, 2, 4], "k": [5, 2, 4]}, work={"dense_equivalent_flops": 800, "definition": "dense equivalent only"})
    for e in trace["traceEvents"]:
        if e["cat"] == "kernel":
            e["name"] = "sparse_attn"
    facts = report.analyze(trace, samples=[sample])
    (row,) = facts["core_calls"]
    assert row["shape"] == sample["shape"] and row["logical_flops"] is None
    assert row["dense_equivalent_tflops"] is not None
    assert "800" not in report.markdown(facts)  # 算量未知不把等价量冒充actual。


def test_custom_precision_reaches_actual_report_consumer():
    trace, sample = two_kernel_trace()
    trace["traceEvents"][2]["name"] = "lightx2v_kernel::cutlass_scaled_nvfp4_mm_sm120"
    for e in trace["traceEvents"]:
        if e["cat"] == "kernel":
            e["name"] = "cutlass_fp4_matmul"
    facts = report.analyze(trace, samples=[sample])
    (row,) = facts["core_calls"]
    assert row["precision"]["input_dtype"] == "fp4"
    assert row["precision"]["accum_dtype"] == "fp32"
    assert row["precision"]["actual_input_dtypes"] == {"input_tensor": "bf16"}
    assert row["ideal_gemm_bytes"] == 22 and row["ideal_arithmetic_intensity"] == pytest.approx(48 / 22)
    report.apply_peaks(facts, {"platforms": {"test": {"peaks": {"fp4": {"dense_rate": 100, "unit": "TFLOPS"}}, "memory_bandwidth_gbps": 1000}}}, "test")
    assert row["ideal_roofline_tflops"] == pytest.approx(48 / 22)


def test_lora_actual_producer_and_report_denominator(tmp_path):
    import torch

    from lightx2v.common.ops.mm.mm_weight import MMWeight
    from tools.profile.capture import Capture

    mm = MMWeight("weight")
    mm.weight, mm.bias = torch.randn(4, 3), None
    mm.register_lora({mm.lora_down_name: torch.randn(2, 4), mm.lora_up_name: torch.randn(3, 2)})
    x = torch.randn(2, 4)
    with Capture(tmp_path / "capture", enabled=True, activities=("cpu",), warmup=0, workloads=True) as capture:
        capture.call(mm.apply, x)
    sample = json.loads((tmp_path / "capture/call_0001.stats/core_ops.jsonl").read_text())
    assert sample["auxiliary_gemm_present"] and sample["work"]["logical_flops"] == 48
    trace, _ = two_kernel_trace()
    sample["trace_marker"] = "lightx2v.core.unique.1"
    (row,) = report.analyze(trace, samples=[sample])["core_calls"]
    assert row["time_basis"] == "whole_operator_including_lora"
    assert row["logical_flops"] == 48 and row["compute_tflops"] is None


def test_active_core_settings_follow_warmup_initialization(tmp_path):
    import torch

    from lightx2v.common.ops.mm.mm_weight import MMWeight
    from tools.profile.capture import Capture
    from tools.profile.operator_facts import call_precision

    mm = MMWeight("weight")
    mm.weight, mm.bias = torch.randn(4, 3), None
    old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction

    def target():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not old
        return mm.apply(torch.ones(2, 4))

    try:
        with Capture(tmp_path / "capture", enabled=True, activities=("cpu",), workloads=True) as capture:
            for _ in range(2):
                capture.call(target)
        sample = json.loads((tmp_path / "capture/call_0002.stats/core_ops.jsonl").read_text())
        context = sample["execution_context"]
        assert context["before"] == context["after"]
        assert context["before"]["allow_bf16_reduced_precision_reduction"] is not old
        # 后半段是显式的CUDA库合同fixture，不把CPU wheel伪装为实际CUDA执行。
        before = {**context["before"], "device_type": "cuda", "cuda_build": "13.0"}
        known = call_precision(["aten::mm"], [], "bf16", {"before": before, "after": before})
        assert known["accum_dtype"] == "fp32" and known["nominal_accumulation"]
        changed = call_precision(["aten::mm"], [], "bf16", {"before": before, "after": {**before, "allow_tf32": not before["allow_tf32"]}})
        assert changed["runtime_evidence"] == "missing_or_changed_at_core_entry" and changed.get("accum_dtype") is None
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old


def test_convolution_implicit_gemm_is_compute_and_layout_is_auxiliary():
    trace, sample = two_kernel_trace()
    sample.update(operator_family="convolution", shape={"input": [1, 16, 8, 32, 32]}, work={"logical_flops": 48000})
    trace["traceEvents"][2]["name"] = "aten::cudnn_convolution"
    trace["traceEvents"][4].update(name="cudnn::nchwToNhwcKernel<bf16>", ts=20, dur=2)
    trace["traceEvents"][5].update(name="sm80_xmma_fprop_implicit_gemm_indexed_bf16bf16_bf16f32_f32_kernel__cudnn", ts=23, dur=8)
    trace["traceEvents"].append(event("cudnn::nhwcToNchwKernel<bf16>", "kernel", 32, 2, pid=0, correlation=0, device=0))
    facts = report.analyze(trace, samples=[sample])
    (row,) = facts["core_calls"]
    assert row["gpu_ms"] == pytest.approx(0.008) and row["operator_gpu_ms"] == pytest.approx(0.012)
    assert row["main_gpu_event_indices"] == [5]
    assert {g["family"] for g in facts["gpu_groups"]} == {"convolution", "auxiliary"}
    assert sum(g["count"] for g in facts["auxiliary_groups"]) == 2


@pytest.mark.parametrize("lora", [False, True])
def test_whole_operator_timing_includes_copy_fill_union(lora):
    trace, sample = two_kernel_trace()
    sample["auxiliary_gemm_present"] = lora
    for entry in trace["traceEvents"]:
        if entry["cat"] == "kernel":
            entry["name"] = "opaque_compute"
    trace["traceEvents"][2]["name"] = "opaque_native"
    trace["traceEvents"].extend(
        [
            event("copy", "gpu_memcpy", 45, 20, pid=0, correlation=0, device=0),
            event("fill", "gpu_memset", 60, 10, pid=0, correlation=0, device=0),
        ]
    )
    (row,) = report.analyze(trace, samples=[sample])["core_calls"]
    assert row["gpu_ms"] == row["operator_gpu_ms"] == pytest.approx(0.050)
    assert row["time_basis"] == ("whole_operator_including_lora" if lora else "whole_operator_including_auxiliary")
    assert row["compute_tflops"] == (None if lora else pytest.approx(48 / 0.050 / 1e9))


def test_moe_compute_scope_uses_real_experts_and_keeps_combine_auxiliary(tmp_path):
    import torch

    from lightx2v.common.ops.moe.fused_moe import TorchExpertLoopFusedMoE
    from tools.profile.capture import Capture

    op = TorchExpertLoopFusedMoE(torch.ones(2, 4, 3, dtype=torch.bfloat16), torch.ones(2, 3, 4, dtype=torch.bfloat16), activation="gelu")
    inputs = (torch.ones(2, 3, dtype=torch.bfloat16), torch.tensor([[0, 1], [1, 0]]), torch.full((2, 2), 0.5))
    expected = op.apply(*inputs)
    directory = tmp_path / "moe"
    with Capture(directory, enabled=True, activities=("cpu",), warmup=0, workloads=True) as capture:
        assert torch.equal(capture.call(op.apply, *inputs), expected)
    samples, _ = report.read_statistics(directory)
    trace = report.read_trace(next(directory.glob("*.pt.trace.json.gz")))
    events = trace["traceEvents"]
    native = [e for e in events if e.get("cat") == "cpu_op" and e["name"] in {"aten::mm", "aten::bmm"}]
    assert [e["name"] for e in native].count("aten::mm") == 4
    assert [e["name"] for e in native].count("aten::bmm") == 1
    start = max(e.get("ts", 0) + e.get("dur", 0) for e in events) + 100
    # Only these GPU events/times are synthetic; markers, native IDs and work come from the real CPU producer.
    for e in native:
        duration = 20 if e["name"] == "aten::bmm" else 1
        events.append(event("route_combine" if duration == 20 else "expert_gemm", "kernel", start, duration, pid=0, device=0, **{"External id": e["args"]["External id"]}))
        start += duration
    facts = report.analyze(trace, samples=samples)
    (row,) = facts["core_calls"]
    assert row["logical_flops"] == 192 and row["gpu_ms"] == pytest.approx(0.004) and row["operator_gpu_ms"] == pytest.approx(0.024)
    assert row["time_basis"] == "compute_kernel_group" and row["precision"]["native_operators"] == ["aten::mm"]
    (combine,) = [g for g in facts["gpu_groups"] if g["name"] == "route_combine"]
    assert combine["family"] == "gemm" and combine["work_role"] == "auxiliary"
    assert not facts["unbound_core_groups"] and any("route_combine" in g["name"] for g in facts["auxiliary_groups"])
    (phase,) = [e for e in events if e["name"] == samples[0]["compute_trace_marker"]]
    for variant in ("missing", "duplicate", "outside_owner"):
        altered = json.loads(json.dumps(trace))
        sample = dict(samples[0])
        if variant == "missing":
            sample.pop("compute_trace_marker")
        elif variant == "duplicate":
            altered["traceEvents"].append(dict(phase))
        else:
            next(e for e in altered["traceEvents"] if e["name"] == phase["name"])["pid"] = -1
        fallback = report.analyze(altered, samples=[sample])
        (value,) = fallback["core_calls"]
        assert value["gpu_ms"] == pytest.approx(0.024) and value["logical_flops"] == 192, variant
        assert value["time_basis"] == "whole_operator_including_auxiliary" and value["compute_tflops"] is None, variant
        assert value["precision"]["input_dtype"] is None and not fallback["unbound_core_groups"], variant
        assert all(g["work_role"] == "unresolved" for g in fallback["gpu_groups"]), variant
    for native_name in ("aten::bmm", "aten::_grouped_mm"):
        altered = json.loads(json.dumps(trace))
        for e in altered["traceEvents"]:
            if e.get("cat") == "cpu_op" and e["name"] == "aten::mm":
                e["name"] = native_name
        # Consumer fixture: a bmm/grouped_mm inside the real expert range is still compute, not globally excluded.
        (value,) = report.analyze(altered, samples=samples)["core_calls"]
        assert value["gpu_ms"] == pytest.approx(0.004) and value["precision"]["native_operators"] == [native_name]
    altered = json.loads(json.dumps(trace))
    first = next(e for e in altered["traceEvents"] if e.get("cat") == "cpu_op" and e["name"] == "aten::mm")
    first["name"] = "unrecognized_expert_native"
    external_id = first["args"]["External id"]
    next(e for e in altered["traceEvents"] if e.get("cat") == "kernel" and e["args"]["External id"] == external_id)["name"] = "opaque_compute"
    (value,) = report.analyze(altered, samples=samples)["core_calls"]
    assert value["time_basis"] == "compute_range_including_auxiliary" and value["gpu_ms"] == pytest.approx(0.004)
    assert value["precision"]["input_dtype"] is None  # Do not divide all expert work by only the three recognized kernels.
