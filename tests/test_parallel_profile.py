import pytest

from tools.profile.parallel_report import collective_facts, wrapper_facts


def event(name, ts, dur, args=None, cat="cpu_op", pid=1):
    return {"name": name, "ts": ts, "dur": dur, "ph": "X", "cat": cat, "pid": pid, "tid": 1, "args": args or {}}


def comm_trace(dtype="BFloat16", kernel_name="ncclDevKernel_AllGather"):
    return [
        event(
            "record_param_comms",
            0,
            10,
            {"External id": 7, "Collective name": "all_gather", "Process Group Ranks": "[0, 2]", "Group size": 2, "In msg nelems": 4096, "Out msg nelems": 8192, "dtype": dtype},
        ),
        event(kernel_name, 10, 5, {"External id": 7, "device": 0}, cat="kernel", pid=0),
        event(kernel_name, 12, 5, {"External id": 7, "device": 0}, cat="kernel", pid=0),
    ]


@pytest.mark.parametrize("kernel_name", ["ncclDevKernel_AllGather", "unfamiliar_kernel"])
def test_collective_counts_payload_once_and_unions_multiple_gpu_kernels(kernel_name):
    rows, indices, missing = collective_facts(comm_trace(kernel_name=kernel_name), {1, 2})
    (row,) = rows
    assert indices == {1, 2} and not missing
    assert row["input_bytes"] == 8192 and row["output_bytes"] == 16384
    assert row["gpu_ms"] == pytest.approx(0.007)
    assert row["logical_input_gbps"] == pytest.approx(8192 / 7000)
    assert row["group_ranks"] == "[0, 2]"


def test_communication_without_kernel_hint_reports_ambiguous_owner():
    trace = comm_trace(kernel_name="unfamiliar_kernel")
    trace.append(event("record_param_comms", 0, 10, {"External id": 7}))
    rows, indices, missing = collective_facts(trace, {1, 2})
    assert not rows and indices == {1, 2}
    assert {row["gpu_event_index"] for row in missing} == {1, 2}


def test_wrapper_own_activity_excludes_child_and_communication_events():
    events = [
        event("parent", 0, 100, cat="user_annotation"),
        event("child", 10, 20, cat="user_annotation"),
        event("aten::cat", 4, 1, {"External id": 11}),
        event("aten::mm", 12, 2, {"External id": 12}),
        event("cat_kernel", 40, 10, {"External id": 11, "device": 0}, cat="kernel", pid=0),
        event("gemm_kernel", 60, 15, {"External id": 12, "device": 0}, cat="kernel", pid=0),
        event("record_param_comms", 50, 5, {"External id": 13}),
        event("communication_kernel", 80, 5, {"External id": 13, "device": 0}, cat="kernel", pid=0),
    ]
    samples = [{"kind": "core_op_compute_stats", "trace_marker": name, "backend": name} for name in ("parent", "child")]
    _, communication, missing = collective_facts(events, {4, 5, 7})
    assert communication == {7} and not missing
    (own,) = wrapper_facts(events, {4, 5, 7}, samples, communication)
    assert own["backend"] == "parent" and own["gpu_event_indices"] == [4]
    assert own["own_activity_ms"] == pytest.approx(0.01)
