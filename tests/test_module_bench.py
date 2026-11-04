from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tools.benchmarks import module_bench as bench


def identity():
    return {"target": {"name": "fixture"}, "input": {"kind": "synthetic", "sha256": "fixture-global-input"}, "source": {"fixture": "CPU protocol"}}


def test_reset_check_and_outputs_are_outside_primary_timing(monkeypatch):
    clock = [0.0]
    state = [0]
    calls = []

    def reset():
        clock[0] += 0.100
        state[0] = 0
        calls.append("reset")

    def synchronize():
        clock[0] += 0.007

    def call():
        assert state[0] == 0
        state[0] = 1
        clock[0] += 0.002
        calls.append("call")
        return 1

    def check(output):
        assert output == 1
        clock[0] += 0.200
        calls.append("check")
        return {"status": "pending_review", "max_abs": 0.01, "scope": "whole fixture"}

    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name, value in (("memory_allocated", 100), ("memory_reserved", 200), ("max_memory_allocated", 150), ("max_memory_reserved", 250)):
        monkeypatch.setattr(torch.cuda, name, lambda device, value=value: value)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: calls.append("reset_peak"))
    prepared = bench.PreparedModule(call, reset, synchronize, identity(), check, "cuda:0")
    report, raw = bench.benchmark(prepared, warmup=2, iterations=2, repeats=2)
    assert report["max_rank_latency_ms"] == pytest.approx([9.0] * 4)
    assert calls.count("call") == 7 and calls.count("check") == 1
    assert calls.index("reset_peak") > calls.index("check")
    assert state == [0]
    assert raw[0]["memory"] == {
        "warmed_baseline": {"allocated_bytes": 100, "reserved_bytes": 200},
        "end": {"allocated_bytes": 100, "reserved_bytes": 200},
        "measurement_peak": {"allocated_bytes": 150, "reserved_bytes": 250},
        "peak_increment": {"allocated_bytes": 50, "reserved_bytes": 50},
    }
    assert raw[0]["correctness"]["status"] == "pending_review"
    assert "winner" not in report and "quality_approval" not in report


def _gloo_worker(rank, rendezvous, output, fault):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=3, timeout=timedelta(seconds=20))
    group = dist.new_group([0, 2], backend="gloo")
    singleton0 = dist.new_group([0], backend="gloo")
    singleton2 = dist.new_group([2], backend="gloo")
    if rank in (0, 2):
        value = torch.zeros(1)
        count, failed = [0], [False]

        def reset():
            value.fill_(rank + 1)
            if fault == "mid_reset" and rank == 2 and count[0] and not failed[0]:
                failed[0] = True
                raise ValueError("rank 2 mid_reset fixture")

        def call():
            dist.all_reduce(value, group=group)
            count[0] += 1
            if rank == 2 and ((fault == "mid_call" and count[0] == 2) or (fault == "preflight_call" and count[0] == 1)):
                raise ValueError(f"rank 2 {fault} fixture")
            return value

        def check(actual):
            # The checker may gather SP outputs; coordinate call failures before entering it.
            dist.all_reduce(torch.ones(1), group=group)
            assert actual.item() == 4
            if fault == "check" and rank == 2:
                raise ValueError("rank 2 check fixture")
            return {"status": "strict_equal", "output": 4, "scope": "whole fixture"}

        durations = [0, 0.001, 0, 0.009] if rank == 0 else [0, 0.005, 0, 0.002]
        ticks = iter(durations)
        original = bench.time.perf_counter
        bench.time.perf_counter = lambda: next(ticks)
        try:
            prepared = bench.PreparedModule(call, reset, lambda: None, identity(), check, "cpu", group, group, singleton0 if rank == 0 else singleton2)
            try:
                report, raw = bench.benchmark(prepared, warmup=2, iterations=2, repeats=1)
                result = {"report": report, "raw": raw}
            except RuntimeError as exc:
                result = {"error": str(exc), "restored": value.item() == rank + 1}
            import json

            Path(output, f"rank{rank}.json").write_text(json.dumps(result))
        finally:
            bench.time.perf_counter = original
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("fault", [False, "check", "mid_call", "mid_reset", "preflight_call"])
def test_existing_gloo_subgroups_max_each_iteration_and_collective_check_failure(tmp_path, fault):
    import json

    mp.spawn(_gloo_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), fault), nprocs=3, join=True)
    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 2)]
    if fault:
        assert all(f"rank 2 {fault} fixture" in row["error"] and row["restored"] for row in results)
        return
    for result in results:
        report = result["report"]
        assert report["max_rank_latency_ms"] == pytest.approx([5.0, 9.0])
        assert report["latency_ms"]["mean"] == pytest.approx(7.0)
        assert [row["rank"] for row in report["ranks"]] == [0, 2]
        assert [row["groups"]["sp_ranks"] for row in report["ranks"]] == [[0], [2]]
        assert all(row["groups"]["measurement_ranks"] == row["groups"]["tp_ranks"] == [0, 2] for row in report["ranks"])
        assert all(row["memory"]["warmed_baseline"] is None for row in result["raw"])


def test_preflight_output_is_released_before_warmup_and_measurement():
    import weakref

    references = []

    def call():
        output = torch.ones(3)
        references.append(weakref.ref(output))
        return output

    def reset():
        assert all(ref() is None for ref in references)

    prepared = bench.PreparedModule(call, reset, lambda: None, identity(), lambda output: {"status": "strict_equal", "scope": "full fixture"}, "cpu")
    bench.benchmark(prepared, warmup=1, iterations=1, repeats=1)
    assert len(references) == 3 and all(ref() is None for ref in references)
