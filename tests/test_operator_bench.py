from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.benchmarks import operator_backends as backends
from tools.benchmarks import operator_bench as core
from tools.benchmarks import sp_bench
from tools.benchmarks.operator_backends import BackendDescriptor, BackendRegistry, load_registry

H100_PEAKS = {
    "platforms": {
        "h100_sxm_80gb": {
            "identification": {"cuda_capability": "9.0", "gpu_name_regex": "^NVIDIA H100 80GB HBM3$"},
            "peaks": {
                "bf16": {"dense_rate": 989.5, "unit": "TFLOPS"},
                "fp8": {"dense_rate": 1979.0, "unit": "TFLOPS"},
                "int8": {"dense_rate": 1979.0, "unit": "TOPS"},
            },
        }
    }
}


def gemm_case(**overrides: object) -> dict:
    value = {
        "case_id": "gemm.1",
        "operator_family": "gemm",
        "operator_name": "linear",
        "shape": {"m": 16, "n": 32, "k": 64, "bias": False},
        "precision": {"input_dtype": "bf16"},
        "source": {"kind": "test"},
        "tags": [],
    }
    value.update(overrides)
    return value


def shape_suite(*cases: dict) -> dict:
    return {
        "schema_version": 1,
        "kind": "operator_benchmark_shape_suite_v1",
        "suite_id": "test_suite",
        "source_kind": "test",
        "cases": list(cases),
    }


def sp_case(**overrides: object) -> dict:
    value = {
        "case_id": "sp_attention.1",
        "shape": {
            "sequence": 1024,
            "heads": 8,
            "kv_heads": 8,
            "head_dim": 128,
            "sp_size": 4,
            "aux_tokens": 0,
            "aux_q": False,
            "aux_first": False,
            "causal": False,
        },
        "precision": {"input_dtype": "bf16"},
    }
    value.update(overrides)
    return value


def sp_suite(*cases: dict) -> dict:
    return {
        "schema_version": 1,
        "kind": "sp_attention_benchmark_shape_suite_v1",
        "suite_id": "sp_test_suite",
        "cases": list(cases),
    }


def sp_catalog(case: dict, *candidates: dict) -> dict:
    return {
        "kind": "sp_attention_benchmark_candidate_catalog_v1",
        "catalog_fingerprint": "catalog-1",
        "scope_complete": True,
        "cases": [{"case_id": case["case_id"], "candidates": list(candidates), "exclusions": []}],
    }


def operator_catalog(family: str, *names: str) -> dict:
    return {
        "kind": "operator_benchmark_backend_catalog_v1",
        "catalog_fingerprint": "catalog-1",
        "unmapped_attention_backends": [],
        "unmapped_mm_backends": [],
        "candidates": [{"name": name, "family": family, "status": "eligible"} for name in names],
    }


def attention_case() -> dict:
    return {
        "case_id": "attention.1",
        "operator_family": "dense_attention",
        "shape": {"batch": 1, "seq_q": 64, "seq_kv": 64, "heads": 8, "kv_heads": 8, "head_dim": 64, "causal": False},
        "precision": {"input_dtype": "bf16"},
    }


def sparse_attention_case(**overrides: object) -> dict:
    value = {
        "case_id": "sparse_attention.1",
        "operator_family": "sparse_attention",
        "shape": {"batch": 1, "seq_q": 4, "seq_kv": 4, "heads": 2, "kv_heads": 2, "head_dim": 8, "causal": False},
        "precision": {"input_dtype": "bf16"},
        "sparse": {"keep_ratio": 0.25},
        "replay": {"manifest": "/tmp/replay.json", "sha256": "0" * 64},
    }
    value.update(overrides)
    return value


def moe_case() -> dict:
    return {
        "case_id": "moe.1",
        "operator_family": "moe",
        "shape": {"tokens": 16, "hidden_size": 64, "intermediate_size": 128, "num_experts": 4, "top_k": 2, "activation": "swiglu", "expert_bias": False},
        "precision": {"input_dtype": "bf16"},
    }


def raw_record(
    case: dict,
    backend: str,
    run: int,
    latency_ms: float,
    rate: float,
    *,
    gpu: str = "NVIDIA H100 80GB HBM3",
    precision: dict | None = None,
    rate_metric: str = "tflops",
    extra_metrics: dict | None = None,
) -> dict:
    runtime_case = {
        "case_id": f"{case['case_id']}.{backend}",
        "operator_family": case["operator_family"],
        "backend": backend,
        "shape": case["shape"],
        "precision": precision or {
            "input_dtype": case["precision"]["input_dtype"],
            "weight_dtype": case["precision"]["input_dtype"],
            "accum_dtype": "backend_default",
            "output_dtype": case["precision"]["input_dtype"],
            "quant": None,
        },
        "source": {"canonical_case_id": case["case_id"]},
        "run": {
            "warmup": 1,
            "iterations": 2,
            "device": "cuda:4",
            "timing_mode": core.TIMING_MODE,
            "rate_schema": core.RATE_SCHEMA,
            "catalog_fingerprint": "catalog-1",
        },
    }
    for field in ("replay", "sparse"):
        if field in case:
            runtime_case[field] = case[field]
    return {
        "run_id": f"repeat-{run:03d}",
        "case": runtime_case,
        "environment": {
            "gpu": {"name": gpu, "major": 9, "minor": 0, "total_memory_bytes": 80 * 1024**3}
        },
        "status": "ok",
        "metrics": {
            "latency_ms_mean": latency_ms,
            rate_metric: rate,
            "work_definition": "2*m*n*k",
            "measurement_scope": "kernel_cuda_event_list_single_sync",
            **(extra_metrics or {}),
        },
        "correctness": {"checked": False, "passed": None},
    }


def write_raw(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_combined_example_is_valid_and_inspectable() -> None:
    value = shape_suite(gemm_case(), attention_case(), moe_case())
    core.validate_suite(value)
    inspection = core.inspect_suite(value)
    assert inspection["case_count"] == 3
    assert inspection["family_case_counts"] == {"dense_attention": 1, "gemm": 1, "moe": 1}


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ({"m": 0, "n": 2, "k": 3, "bias": False}, "positive integer"),
        ({"m": 1, "n": 2, "k": 3, "bias": "maybe"}, "boolean"),
    ],
)
def test_suite_rejects_invalid_gemm_shapes(shape: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        core.validate_suite(shape_suite(gemm_case(shape=shape)))


def test_suite_rejects_unknown_case_fields() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        core.validate_suite(shape_suite(gemm_case(extra=True)))


def test_sweep_is_deterministic() -> None:
    axes = {"m": [1, 16], "n": [32], "k": [64], "bias": [False, True]}
    first = core.generate_sweep("gemm", axes, "grid", "bf16")
    second = core.generate_sweep("gemm", axes, "grid", "bf16")
    assert first == second
    assert len(first["cases"]) == 4


def test_observed_moe_routing_is_validated() -> None:
    case = {
        "case_id": "moe.1",
        "operator_family": "moe",
        "operator_name": "fused_moe",
        "shape": {
            "tokens": 4,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_experts": 4,
            "top_k": 2,
            "activation": "swiglu",
            "expert_bias": False,
        },
        "precision": {"input_dtype": "bf16"},
        "routing": {"expert_counts": [2, 2, 2, 1]},
        "source": {},
        "tags": [],
    }
    with pytest.raises(ValueError, match="expert_counts"):
        core.validate_suite(shape_suite(case))


def test_torch_grouped_mm_applies_expert_bias(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")

    def grouped_mm(inputs: object, weights: object, *, offs: object) -> object:
        outputs = []
        start = 0
        for expert, end in enumerate(offs.tolist()):
            outputs.append((inputs[start:end].float() @ weights[expert].float()).to(inputs.dtype))
            start = end
        return torch.cat(outputs)

    def ones(shape: object, *, device: object = None, dtype: object = None, **_: object) -> object:
        return torch.ones(shape, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "_grouped_mm", grouped_mm, raising=False)
    monkeypatch.setattr(torch, "randn", ones)
    adapter = backends.TorchGroupedMMAdapter()
    case = moe_case()
    case["shape"] = {
        "tokens": 4,
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_experts": 2,
        "top_k": 1,
        "activation": "gelu",
        "expert_bias": False,
    }
    torch.manual_seed(1)
    unbiased = adapter.prepare(case, "cpu", {"moe_routing": "balanced"}).fn()
    torch.manual_seed(1)
    biased = adapter.prepare({**case, "shape": {**case["shape"], "expert_bias": True}}, "cpu", {"moe_routing": "balanced"}).fn()
    assert unbiased.shape == biased.shape == torch.Size([4, 4])
    assert not torch.equal(unbiased, biased)


def test_sparse_attention_requires_real_replay() -> None:
    core.validate_suite(shape_suite(sparse_attention_case()))
    case = sparse_attention_case()
    del case["replay"]
    with pytest.raises(ValueError, match="sparse replay is required"):
        core.validate_suite(shape_suite(case))
    with pytest.raises(ValueError, match="cannot be swept"):
        core.generate_sweep(
            "sparse_attention",
            {"batch": [1], "seq_q": [4], "seq_kv": [4], "heads": [2], "kv_heads": [2], "head_dim": [8], "causal": [False]},
            "invalid",
            "bf16",
        )


def test_sparse_replay_manifest_rebuilds_qkv(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    tensors = {
        name: torch.arange(64, dtype=torch.float32).reshape(1, 4, 2, 8).to(torch.bfloat16) + offset
        for offset, name in enumerate(("q", "k", "v"))
    }
    specs = {}
    for name, tensor in tensors.items():
        path = tmp_path / f"{name}.pt"
        torch.save({name: tensor}, path)
        specs[name] = {
            "dtype": "bf16",
            "shape": [1, 4, 2, 8],
            "shards": [
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "tensor_key": name,
                    "sequence_start": 0,
                    "sequence_end": 4,
                }
            ],
        }
    manifest = {
        "schema_version": 1,
        "kind": "operator_benchmark_qkv_replay_v1",
        "layout": "BSHD",
        "provenance": {"kind": "test_capture"},
        "tensors": specs,
    }
    manifest_path = tmp_path / "replay.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    case = sparse_attention_case(
        replay={"manifest": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    )
    q, k, v, metadata = backends._load_qkv_replay(case, "cpu", torch)
    assert torch.equal(q, tensors["q"].squeeze(0))
    assert torch.equal(k, tensors["k"].squeeze(0))
    assert torch.equal(v, tensors["v"].squeeze(0))
    assert metadata["input_kind"] == "real_qkv_replay"


def test_sparse_work_exposes_actual_and_dense_equivalent_flops() -> None:
    case = sparse_attention_case()
    prepared = backends._sparse_replay_prepared(
        case,
        lambda: None,
        "test",
        {"block_sparsity_actual": 0.75},
    )
    dense_work = 4 * 1 * 2 * 4 * 4 * 8
    assert prepared.rate_metric == "actual_tflops"
    assert prepared.work == dense_work * 0.25
    assert prepared.dense_equivalent_work == dense_work


def test_builtin_and_optional_plugins_register_without_runtime_dependencies() -> None:
    registry = load_registry()
    names = {item["name"] for item in registry.descriptors()}
    assert {"torch_linear", "torch_sdpa", "torch_expert_loop"} <= names
    assert {"fp8-vllm", "fp8-q8f", "fp8-triton", "flash_attn3", "torch_grouped_mm", "Default"} <= names
    assert {"sparge_sage2_replay", "dynamic_sparse_sage2_replay", "flash_attn3_replay"} <= names
    assert {"flash_attn4", "sage_attn3", "dynamic_sparse_sage3_replay", "dynamic_sparse_fa4_replay"} <= names


def test_backend_catalog_tracks_architecture_and_registry_evolution() -> None:
    registry = load_registry()
    descriptors = registry.descriptors()
    production = {
        item["production_backend"]
        for item in descriptors
        if item["family"] in {"dense_attention", "sparse_attention"} and item["production_backend"]
    } | set(backends.PRODUCTION_ATTN_EXCLUSIONS)
    h100 = backends.build_backend_catalog_report(
        registry,
        cuda_capability="9.0",
        production_backends=production,
        symbol_probe=lambda value: None,
        dependency_probe=lambda value: None,
    )
    h100_status = {item["name"]: item["status"] for item in h100["candidates"]}
    assert h100["coverage_complete"]
    assert h100_status["flash_attn3"] == "eligible"
    assert h100_status["flash_attn4"] == "unsupported_arch"
    assert h100_status["sage_attn3"] == "unsupported_arch"

    blackwell = backends.build_backend_catalog_report(
        registry,
        cuda_capability="12.0",
        production_backends=production | {"future_attention"},
        symbol_probe=lambda value: None,
        dependency_probe=lambda value: None,
    )
    blackwell_status = {item["name"]: item["status"] for item in blackwell["candidates"]}
    assert blackwell_status["flash_attn3"] == "unsupported_arch"
    assert blackwell_status["flash_attn4"] == "eligible"
    assert blackwell_status["sage_attn3"] == "eligible"
    assert blackwell["unmapped_production_backends"] == ["future_attention"]
    assert not blackwell["coverage_complete"]

    thor = backends.build_backend_catalog_report(
        registry,
        cuda_capability="11.0",
        production_backends=production,
        symbol_probe=lambda value: None,
        dependency_probe=lambda value: None,
    )
    thor_status = {item["name"]: item["status"] for item in thor["candidates"]}
    assert thor_status["flash_attn4"] == "eligible"
    assert thor_status["sage_attn3"] == "eligible"

    future_blackwell = backends.build_backend_catalog_report(
        registry,
        cuda_capability="12.1",
        production_backends=production,
        symbol_probe=lambda value: None,
        dependency_probe=lambda value: None,
    )
    future_status = {item["name"]: item["status"] for item in future_blackwell["candidates"]}
    assert future_status["flash_attn4"] == "eligible"
    assert future_status["flash_attn3"] == "unsupported_arch"

    incomplete_environment = backends.build_backend_catalog_report(
        registry,
        cuda_capability="12.0",
        production_backends=production,
        symbol_probe=lambda value: None,
        dependency_probe=lambda value: "missing" if value == "sageattn3" else None,
    )
    incomplete_status = {
        item["name"]: item["status"] for item in incomplete_environment["candidates"]
    }
    assert incomplete_status["sage_attn3"] == "dependency_missing"
    assert not incomplete_environment["environment_complete"]

    sparse_scope_catalog = {
        "candidates": [
            {"name": "dynamic_sparse_sage3_replay", "family": "sparse_attention", "status": "eligible"},
            {"name": "spas_fa4_replay", "family": "sparse_attention", "status": "eligible"},
            {"name": "flash_attn4_replay", "family": "sparse_attention", "status": "eligible"},
            {"name": "future_sparse_replay", "family": "sparse_attention", "status": "eligible"},
            {"name": "dynamic_sparse_sage2_replay", "family": "sparse_attention", "status": "unsupported_arch"},
        ]
    }
    supported, exclusions, unmapped = sp_bench._sparse_sp_backend_scope(sparse_scope_catalog)
    assert supported == ["dynamic_sparse_sage3_replay", "spas_fa4_replay"]
    assert set(exclusions) == {"flash_attn4_replay"}
    assert unmapped == ["future_sparse_replay"]

    sparse_runtime = {
        "catalog": {**sparse_scope_catalog, "catalog_fingerprint": "catalog", "unmapped_attention_backends": []},
        "sparse_backends": supported,
        "sparse_sp_exclusions": exclusions,
        "unmapped_sparse_sp_backends": unmapped,
        "a2a_backends": ["torch"],
        "fp4_available": True,
    }
    sparse = sp_case(replay={"manifest": "replay.json", "sha256": "0" * 64}, sparse={"keep_ratio": 0.15})
    sparse_catalog = sp_bench.build_candidate_catalog(sp_suite(sparse), sparse_runtime)
    assert not sparse_catalog["scope_complete"]
    assert sparse_catalog["backend_scope_blockers"]["unmapped_sparse_sp_backends"] == ["future_sparse_replay"]

    mm_production = {item["production_backend"] for item in descriptors if item["family"] == "gemm" and item["production_backend"]}
    mm_production.update(backends.PRODUCTION_MM_EXCLUSIONS)
    mm_catalog = backends.build_backend_catalog_report(
        registry,
        cuda_capability="9.0",
        production_backends=production,
        production_mm_backends=mm_production | {"future_mm"},
        symbol_probe=lambda value: None,
        dependency_probe=lambda value: None,
    )
    assert mm_catalog["unmapped_mm_backends"] == ["future_mm"]
    assert not mm_catalog["catalog_complete"]


def test_backend_assignment_parser_is_explicit() -> None:
    assert core.parse_backend_assignments(["gemm=a,b", "moe=c"]) == {"gemm": ["a", "b"], "moe": ["c"]}
    with pytest.raises(ValueError, match="FAMILY"):
        core.parse_backend_assignments(["gemm"])


def test_measure_enqueues_all_iterations_before_single_event_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    device_syncs = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing
            self.syncs = 0
            events.append(self)

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            self.syncs += 1

        def elapsed_time(self, end: object) -> float:
            assert end in events
            return 1.0

    fake_cuda = SimpleNamespace(
        Event=FakeEvent,
        synchronize=lambda device: device_syncs.append(device),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda, device=lambda value: value))
    calls = []

    latencies, result = core._measure(
        lambda: calls.append(len(calls)) or len(calls),
        warmup=2,
        iterations=3,
        prewarm_seconds=0,
        device="cuda:4",
    )

    assert calls == [0, 1, 2, 3, 4]
    assert result == 5
    assert latencies == [1.0, 1.0, 1.0]
    assert device_syncs == ["cuda:4"]
    assert [event.syncs for event in events] == [0, 0, 0, 0, 0, 1]


class FakeAdapter:
    def __init__(self, name: str) -> None:
        self.descriptor = BackendDescriptor(name, "gemm", "test", ("bf16",), "fake")

    def support_error(self, case: dict) -> str | None:
        return None

    def precision(self, case: dict) -> dict:
        return {"input_dtype": "bf16", "weight_dtype": "bf16"}


def test_direct_runner_writes_repeats_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = BackendRegistry()
    registry.register(FakeAdapter("a"))
    registry.register(FakeAdapter("b"))
    monkeypatch.setattr(core, "load_registry", lambda modules: registry)
    monkeypatch.setattr(core, "environment", lambda device: {"device": device})

    def fake_run(case: dict, adapter: FakeAdapter, run_id: str, env: dict, options: dict) -> dict:
        return {
            "run_id": run_id,
            "case": {
                "backend": adapter.descriptor.name,
                "shape": case["shape"],
                "precision": {"input_dtype": case["precision"]["input_dtype"]},
                "source": {"canonical_case_id": case["case_id"]},
                "run": {
                    "warmup": options["warmup"],
                    "iterations": options["iterations"],
                    "prewarm_seconds": options["prewarm_seconds"],
                    "device": options["device"],
                    "moe_routing": options["moe_routing"],
                    "seed": options["seed"],
                    "timing_mode": options["timing_mode"],
                    "rate_schema": options["rate_schema"],
                    "catalog_fingerprint": options.get("catalog_fingerprint"),
                },
            },
            "environment": env,
            "status": "ok",
        }

    monkeypatch.setattr(core, "run_case", fake_run)
    output = tmp_path / "raw.jsonl"
    value = shape_suite(gemm_case())
    first = core.run_suite(value, output, {"gemm": ["a", "b"]}, [], repeat_runs=2)
    second = core.run_suite(value, output, {"gemm": ["a", "b"]}, [], repeat_runs=2, append=True)
    assert first["written"] == 4
    assert second["written"] == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4
    with pytest.raises(ValueError, match="measurement arguments"):
        core.run_suite(value, output, {"gemm": ["a", "b"]}, [], repeat_runs=2, warmup=11, append=True)
    with pytest.raises(ValueError, match="measurement arguments"):
        core.run_suite(value, output, {"gemm": ["a", "b"]}, [], repeat_runs=2, seed=43, append=True)
    monkeypatch.setattr(core, "environment", lambda device: {"device": device, "torch": "changed"})
    with pytest.raises(ValueError, match="environment differs"):
        core.run_suite(value, output, {"gemm": ["a", "b"]}, [], repeat_runs=2, append=True)


def test_sparse_append_rejects_replay_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = sparse_attention_case()
    output = tmp_path / "raw.jsonl"
    write_raw(output, [raw_record(case, "dynamic_sparse_sage2_replay", 0, 1.0, 100.0)])
    monkeypatch.setattr(core, "environment", lambda device: {"device": device})
    case["replay"] = {**case["replay"], "sha256": "1" * 64}
    with pytest.raises(ValueError, match="replay drift"):
        core.run_suite(shape_suite(case), output, {}, [], repeat_runs=1, append=True)


def test_report_selects_stable_winner_and_computes_h100_efficiency(tmp_path: Path) -> None:
    case = gemm_case(call_count=10, observed_backend="slow")
    records = []
    for repeat in range(3):
        records.append(raw_record(case, "fast", repeat, 1.0 + repeat * 0.001, 500.0))
        records.append(raw_record(case, "slow", repeat, 2.0 + repeat * 0.001, 250.0))
    raw = tmp_path / "raw.jsonl"
    write_raw(raw, records)
    value = core.build_recommendation_report(
        shape_suite(case),
        core.load_records([raw]),
        peaks=H100_PEAKS,
        platform_id="h100_sxm_80gb",
        candidate_catalog=operator_catalog("gemm", "fast", "slow"),
    )
    workload = value["workloads"][0]
    assert workload["winner"] == "fast"
    assert workload["observed_backend"]["speedup"] == pytest.approx(2.0, rel=0.01)
    assert workload["backends"]["fast"]["nominal_efficiency"] == pytest.approx(500 / 989.5)
    assert value["summary"]["weighted_per_shape_measured_winner_ms"] == pytest.approx(10.01)


def test_report_only_publishes_measured_winner_without_complete_catalog(tmp_path: Path) -> None:
    case = gemm_case()
    records = [raw_record(case, "fast", repeat, 1.0, 500.0) for repeat in range(3)]
    without_catalog = core.build_recommendation_report(shape_suite(case), records)
    workload = without_catalog["workloads"][0]
    assert workload["measured_winner"] == "fast"
    assert workload["winner"] is None
    assert workload["candidate_coverage_complete"] is False

    incomplete_catalog = operator_catalog("gemm", "fast", "not-measured")
    with_catalog = core.build_recommendation_report(
        shape_suite(case),
        records,
        candidate_catalog=incomplete_catalog,
    )
    workload = with_catalog["workloads"][0]
    assert workload["winner"] is None
    assert workload["coverage_blockers"]["not_measured"] == ["not-measured"]


def test_report_rejects_mixed_environments_and_downgrades_unmatched_raw() -> None:
    case = gemm_case()
    records = [raw_record(case, "fast", repeat, 1.0, 500.0) for repeat in range(3)]
    mixed = json.loads(json.dumps(records))
    mixed[-1]["environment"]["torch"] = "different"
    with pytest.raises(ValueError, match="mixed raw hardware or software environments"):
        core.build_recommendation_report(shape_suite(case), mixed)

    unrelated_case = gemm_case(case_id="gemm.unrelated")
    unrelated = raw_record(unrelated_case, "fast", 10, 1.0, 500.0)
    report = core.build_recommendation_report(
        shape_suite(case),
        [*records, unrelated],
        candidate_catalog=operator_catalog("gemm", "fast"),
    )
    workload = report["workloads"][0]
    assert workload["winner"] is None
    assert workload["coverage_blockers"]["unmatched_raw_records"] == 1


def test_report_does_not_treat_measurement_failures_as_terminal_unavailability() -> None:
    case = gemm_case()
    records = [raw_record(case, "broken", repeat, 1.0, 1.0) for repeat in range(3)]
    for record in records:
        record["status"] = "error"
        record.pop("metrics")
        record["error"] = {"type": "RuntimeError", "message": "kernel failed", "stage": "measure"}
    report = core.build_recommendation_report(
        shape_suite(case),
        records,
        candidate_catalog=operator_catalog("gemm", "broken"),
    )
    workload = report["workloads"][0]
    assert workload["backends"]["broken"]["status"] == "measurement_error"
    assert workload["candidate_coverage_complete"] is False

    for record in records:
        record["error"]["stage"] = "prepare"
    report = core.build_recommendation_report(
        shape_suite(case),
        records,
        candidate_catalog=operator_catalog("gemm", "broken"),
    )
    assert report["workloads"][0]["backends"]["broken"]["status"] == "unavailable"
    assert report["workloads"][0]["candidate_coverage_complete"] is True


@pytest.mark.parametrize(
    ("precision", "rate_metric", "peak_family", "rate_unit"),
    [
        (
            {"input_dtype": "bf16", "weight_dtype": "fp8_e4m3_per_channel", "quant": "fp8_e4m3_dynamic_activation"},
            "effective_tflops",
            "fp8",
            "TFLOPS",
        ),
        (
            {"input_dtype": "bf16", "weight_dtype": "int8_per_channel", "quant": "int8_dynamic_activation"},
            "effective_tops",
            "int8",
            "TOPS",
        ),
    ],
)
def test_report_uses_precision_specific_peak(
    tmp_path: Path,
    precision: dict,
    rate_metric: str,
    peak_family: str,
    rate_unit: str,
) -> None:
    case = gemm_case()
    raw = tmp_path / "raw.jsonl"
    records = [
        raw_record(case, "quantized", repeat, 1.0, 989.5, precision=precision, rate_metric=rate_metric)
        for repeat in range(3)
    ]
    write_raw(raw, records)
    value = core.build_recommendation_report(
        shape_suite(case),
        core.load_records([raw]),
        peaks=H100_PEAKS,
        platform_id="h100_sxm_80gb",
    )
    result = value["workloads"][0]["backends"]["quantized"]
    assert result["peak_family"] == peak_family
    assert result["peak_status"] == "available"
    assert result["rate_unit"] == rate_unit
    assert result["nominal_efficiency"] == pytest.approx(0.5)


def test_report_exposes_missing_peak(tmp_path: Path) -> None:
    case = gemm_case()
    raw = tmp_path / "raw.jsonl"
    precision = {"input_dtype": "bf16", "weight_dtype": "fp8_e4m3_per_channel", "quant": "fp8_e4m3"}
    write_raw(
        raw,
        [
            raw_record(case, "fp8", repeat, 1.0, 1000.0, precision=precision, rate_metric="effective_tflops")
            for repeat in range(3)
        ],
    )
    bf16_only = {
        "platforms": {
            "h100_sxm_80gb": {
                "identification": H100_PEAKS["platforms"]["h100_sxm_80gb"]["identification"],
                "peaks": {"bf16": {"dense_rate": 989.5, "unit": "TFLOPS"}},
            }
        }
    }
    value = core.build_recommendation_report(
        shape_suite(case),
        core.load_records([raw]),
        peaks=bf16_only,
        platform_id="h100_sxm_80gb",
    )
    result = value["workloads"][0]["backends"]["fp8"]
    assert result["peak_status"] == "peak_missing"
    assert result["nominal_efficiency"] is None
    assert "peak_missing" in core.markdown_report(value)


def test_sparse_report_exposes_actual_and_dense_equivalent_efficiency(tmp_path: Path) -> None:
    case = sparse_attention_case()
    raw = tmp_path / "raw.jsonl"
    write_raw(
        raw,
        [
            raw_record(
                case,
                "sparse",
                repeat,
                1.0,
                300.0,
                rate_metric="actual_tflops",
                extra_metrics={"dense_equivalent_tflops": 2000.0},
            )
            for repeat in range(3)
        ],
    )
    value = core.build_recommendation_report(
        shape_suite(case),
        core.load_records([raw]),
        peaks=H100_PEAKS,
        platform_id="h100_sxm_80gb",
    )
    result = value["workloads"][0]["backends"]["sparse"]
    assert result["peak_status"] == "available"
    assert result["actual_rate"] == pytest.approx(300.0)
    assert result["actual_peak_efficiency"] == pytest.approx(300.0 / 989.5)
    assert result["dense_equivalent_rate"] == pytest.approx(2000.0)
    assert result["dense_equivalent_peak_efficiency"] == pytest.approx(2000.0 / 989.5)
    markdown = core.markdown_report(value)
    assert "30.32%" in markdown
    assert "202.12%" in markdown


def test_report_selects_accumulator_specific_peak(tmp_path: Path) -> None:
    case = gemm_case()
    raw = tmp_path / "raw.jsonl"
    precisions = {
        "fp32_accum": {"input_dtype": "bf16", "weight_dtype": "fp8_e4m3", "accum_dtype": "fp32"},
        "fp16_accum": {"input_dtype": "bf16", "weight_dtype": "fp8_e4m3", "accum_dtype": "fp16"},
        "unknown_accum": {
            "input_dtype": "bf16",
            "weight_dtype": "fp8_e4m3",
            "accum_dtype": "production_wrapper_default",
        },
    }
    records = []
    for repeat in range(3):
        for backend, rate in (("fp32_accum", 209.5), ("fp16_accum", 419.0), ("unknown_accum", 419.0)):
            records.append(
                raw_record(
                    case,
                    backend,
                    repeat,
                    1.0,
                    rate,
                    precision=precisions[backend],
                    rate_metric="effective_tflops",
                )
            )
    write_raw(raw, records)
    variant_peaks = {
        "platforms": {
            "variant_gpu": {
                "peaks": {
                    "fp8": {
                        "variants": {
                            "fp32": {"dense_rate": 419.0, "unit": "TFLOPS"},
                            "fp16": {"dense_rate": 838.0, "unit": "TFLOPS"},
                        }
                    }
                }
            }
        }
    }
    value = core.build_recommendation_report(
        shape_suite(case),
        core.load_records([raw]),
        peaks=variant_peaks,
        platform_id="variant_gpu",
    )
    backends = value["workloads"][0]["backends"]
    assert backends["fp32_accum"]["peak_variant"] == "fp32"
    assert backends["fp32_accum"]["nominal_efficiency"] == pytest.approx(0.5)
    assert backends["fp16_accum"]["peak_variant"] == "fp16"
    assert backends["fp16_accum"]["nominal_efficiency"] == pytest.approx(0.5)
    assert backends["unknown_accum"]["peak_status"] == "precision_variant_unknown"
    assert backends["unknown_accum"]["nominal_efficiency"] is None


def test_report_rejects_wrong_hardware_profile(tmp_path: Path) -> None:
    case = gemm_case()
    raw = tmp_path / "raw.jsonl"
    write_raw(raw, [raw_record(case, "a", repeat, 1.0, 1.0, gpu="NVIDIA GeForce RTX 5090") for repeat in range(3)])
    with pytest.raises(ValueError, match="GPU name"):
        core.build_recommendation_report(
            shape_suite(case),
            core.load_records([raw]),
            peaks=H100_PEAKS,
            platform_id="h100_sxm_80gb",
        )


def test_cli_inspect_and_sweep(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_suite = tmp_path / "input.json"
    input_suite.write_text(json.dumps(shape_suite(gemm_case())), encoding="utf-8")
    assert core.main(["inspect", "--suite", str(input_suite)]) == 0
    assert json.loads(capsys.readouterr().out)["case_count"] == 1
    output = tmp_path / "sweep.json"
    assert (
        core.main(
            [
                "sweep",
                "--family",
                "gemm",
                "--axis",
                "m=1,2",
                "--axis",
                "n=4",
                "--axis",
                "k=8",
                "--axis",
                "bias=false",
                "--suite-id",
                "grid",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(core.load_suite(output)["cases"]) == 2


def test_cli_failed_run_does_not_overwrite_existing_backend_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(shape_suite(gemm_case())), encoding="utf-8")
    output_dir = tmp_path / "results"
    catalog_path = output_dir / "backend_catalog.json"
    core.write_json(catalog_path, {"kind": "previous_catalog"})
    monkeypatch.setattr(core, "load_registry", lambda _: object())
    monkeypatch.setattr(core, "probe_backend_catalog", lambda *_: {"kind": "new_catalog"})

    def fail_run(*_: object, **__: object) -> dict:
        raise ValueError("existing raw differs")

    monkeypatch.setattr(core, "run_suite", fail_run)
    with pytest.raises(SystemExit):
        core.main(["run", "--suite", str(suite_path), "--output-dir", str(output_dir)])

    assert json.loads(catalog_path.read_text(encoding="utf-8")) == {"kind": "previous_catalog"}


def test_sp_suite_contract_and_inspection() -> None:
    value = sp_suite(sp_case(call_count=12))
    sp_bench.validate_suite(value)
    inspection = sp_bench.inspect_suite(value)
    assert inspection["case_count"] == 1
    assert inspection["sp_sizes"] == {4: 1}
    assert inspection["missing_call_count"] == []

    invalid = sp_case()
    invalid["shape"] = {**invalid["shape"], "sequence": 1025}
    with pytest.raises(ValueError, match="divisible"):
        sp_bench.validate_suite(sp_suite(invalid))


def test_sp_replay_suite_derivation_uses_per_degree_aux_prefix(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "kind": "operator_benchmark_qkv_replay_v1",
        "layout": "BSHD",
        "provenance": {"model": "minimax_h3"},
        "tensors": {
            name: {
                "dtype": "bf16",
                "shape": [1, 109151, 56, 128],
                "shards": [{"path": f"{name}.pt", "sha256": "0" * 64, "tensor_key": name, "sequence_start": 0, "sequence_end": 109151}],
            }
            for name in ("q", "k", "v")
        },
    }
    manifest_path = tmp_path / "replay.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    outputs = sp_bench.derive_replay_suites(
        manifest_path,
        tmp_path / "suites",
        suite_id="minimax_h3_sparse",
        conditioner_tokens=89,
        sp_sizes=[2, 4, 8],
        keep_ratio=0.15,
    )
    shapes = [sp_bench.load_suite(path)["cases"][0]["shape"] for path in outputs]
    assert [(shape["sp_size"], shape["sequence"], shape["aux_tokens"]) for shape in shapes] == [
        (2, 109062, 89),
        (4, 109064, 87),
        (8, 109064, 87),
    ]


def test_sparse_sp_candidates_use_replay_leafs_and_exclude_ring() -> None:
    case = sp_case(
        shape={**sp_case()["shape"], "sequence": 1020, "aux_tokens": 4, "aux_q": True, "aux_first": True},
        replay={"manifest": "/tmp/replay.json", "sha256": "0" * 64},
        sparse={"keep_ratio": 0.15},
    )
    sp_bench.validate_suite(sp_suite(case))
    result = sp_bench.enumerate_sparse_candidates(
        case,
        ["dynamic_sparse_triton_replay", "dynamic_sparse_sage2_replay"],
        fp4_available=False,
    )
    assert result["candidates"]
    assert all(item["algorithm"] == "ulysses" and item["leaf_family"] == "sparse_attention" for item in result["candidates"])
    assert {item["attention_backend"] for item in result["candidates"]} == {
        "dynamic_sparse_triton_replay",
        "dynamic_sparse_sage2_replay",
    }
    assert any("Sparse Ring requires" in item["reason"] for item in result["exclusions"])


def test_sp_replay_loader_slices_aux_and_rank_main(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    specs = {}
    for offset, name in enumerate(("q", "k", "v")):
        tensor = torch.arange(8 * 4 * 2, dtype=torch.float32).reshape(1, 8, 4, 2).to(torch.bfloat16) + offset
        path = tmp_path / f"{name}.pt"
        torch.save({name: tensor}, path)
        specs[name] = {
            "dtype": "bf16",
            "shape": [1, 8, 4, 2],
            "shards": [
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "tensor_key": name,
                    "sequence_start": 0,
                    "sequence_end": 8,
                }
            ],
        }
    manifest = {"schema_version": 1, "kind": "operator_benchmark_qkv_replay_v1", "layout": "BSHD", "tensors": specs}
    manifest_path = tmp_path / "replay.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    case = sp_case(
        shape={"sequence": 6, "heads": 4, "kv_heads": 4, "head_dim": 2, "sp_size": 2, "aux_tokens": 2, "aux_q": True, "aux_first": True, "causal": False},
        replay={"manifest": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
        sparse={"keep_ratio": 0.5},
    )
    q, k, v, aux_q, aux_k, aux_v = sp_bench._load_sp_replay_inputs(case, torch.device("cpu"), rank=1)
    assert [tensor.shape for tensor in (q, k, v)] == [torch.Size([3, 4, 2])] * 3
    assert [tensor.shape for tensor in (aux_q, aux_k, aux_v)] == [torch.Size([2, 4, 2])] * 3
    full_q = torch.load(tmp_path / "q.pt", weights_only=True)["q"].squeeze(0)
    assert torch.equal(aux_q, full_q[:2])
    assert torch.equal(q, full_q[5:8])

    torch.save({"q": torch.load(tmp_path / "q.pt", weights_only=True)["q"].to(torch.float16)}, tmp_path / "q.pt")
    specs["q"]["shards"][0]["sha256"] = hashlib.sha256((tmp_path / "q.pt").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    case["replay"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="tensor mismatch"):
        sp_bench._load_sp_replay_inputs(case, torch.device("cpu"), rank=1)


def test_sp_candidates_encode_production_constraints() -> None:
    case = sp_case()
    result = sp_bench.enumerate_candidates(
        case,
        ["flash_attn3", "flash_attn4"],
        ring_lse_backends={"flash_attn3"},
        fp4_available=False,
    )
    candidates = result["candidates"]
    exclusions = result["exclusions"]
    assert any(item["algorithm"] == "ulysses" and item["prepost_backend"] == "triton" for item in candidates)
    assert any(item["algorithm"] == "ring" and item["dense_backend"] == "flash_attn3" for item in candidates)
    assert not any(item["algorithm"] == "ring" and item["dense_backend"] == "flash_attn4" for item in candidates)
    assert not any(item["quant_scheme"] == "fp4" for item in candidates)
    assert any("native apply_with_lse" in item["reason"] for item in exclusions)
    assert any("FP4 communication dependency" in item["reason"] for item in exclusions)
    assert not any(
        item["prepost_backend"] == "triton" and not item["tensor_fusion"]
        for item in candidates
        if item["algorithm"] == "ulysses"
    )


def test_sp_candidates_reject_gqa_fusion_and_ring() -> None:
    case = sp_case()
    case["shape"] = {**case["shape"], "heads": 8, "kv_heads": 4}
    result = sp_bench.enumerate_candidates(case, ["flash_attn3"], ring_lse_backends={"flash_attn3"})
    assert result["candidates"]
    assert all(item["algorithm"] == "ulysses" for item in result["candidates"])
    assert all(not item["tensor_fusion"] and not item["head_parallel"] for item in result["candidates"])


def test_sp_ring_aux_candidates_are_released_by_validated_matrix() -> None:
    case = sp_case()
    case["shape"] = {**case["shape"], "aux_tokens": 16, "aux_q": True}
    result = sp_bench.enumerate_candidates(case, ["flash_attn3"], ring_lse_backends={"flash_attn3"})
    ring = [item for item in result["candidates"] if item["algorithm"] == "ring"]
    assert {(item["quant_scheme"], item["tensor_fusion"]) for item in ring} == {
        (None, False),
        (None, True),
        ("fp8", False),
        ("fp8", True),
    }
    assert any("FP4 communication is not validated" in item["reason"] for item in result["exclusions"])

    fp16_case = {**case, "precision": {"input_dtype": "fp16"}}
    fp16 = sp_bench.enumerate_candidates(fp16_case, ["flash_attn3"], ring_lse_backends={"flash_attn3"})
    fp16_ring = [item for item in fp16["candidates"] if item["algorithm"] == "ring"]
    assert {(item["quant_scheme"], item["tensor_fusion"]) for item in fp16_ring} == {
        (None, False),
        (None, True),
        ("fp8", False),
        ("fp8", True),
    }
    assert sp_bench._result_correctness_tolerance(case, ring[0]) == 0.02
    assert sp_bench._result_correctness_tolerance(fp16_case, fp16_ring[0]) == 0.01
    assert sp_bench._result_correctness_tolerance(case, next(item for item in ring if item["quant_scheme"] == "fp8")) == 0.10
    assert sp_bench._result_correctness_tolerance(case, {**ring[0], "dense_backend": "sage_attn2"}) == 0.10
    assert sp_bench._result_correctness_policy(case, ring[0])["policy"].startswith("exact_dense")
    with pytest.raises(ValueError, match="missing SP correctness contract"):
        sp_bench._result_correctness_policy(case, {**ring[0], "dense_backend": "new_lse_backend"})
    assert sp_bench._result_correctness_policy(
        case,
        {**ring[0], "dense_backend": "new_lse_backend"},
        allow_missing=True,
    )["policy"] == "exploratory_unregistered_backend_v1"

    unavailable = sp_bench.enumerate_candidates(
        case,
        ["flash_attn3"],
        ring_lse_backends={"flash_attn3"},
        fp4_available=False,
    )
    assert any("FP4 communication dependency is unavailable" in item["reason"] for item in unavailable["exclusions"])

    new_leaf = sp_bench.enumerate_candidates(case, ["new_lse_backend"], ring_lse_backends={"new_lse_backend"})
    assert not any(item["algorithm"] == "ring" for item in new_leaf["candidates"])
    assert any("dense backend new_lse_backend is not validated" in item["reason"] for item in new_leaf["exclusions"])

    validation = sp_bench.enumerate_candidates(
        case,
        ["new_lse_backend"],
        ring_lse_backends={"new_lse_backend"},
        allow_unvalidated=True,
    )
    assert any(item["algorithm"] == "ring" for item in validation["candidates"])


def test_sp_reference_validation_cli_contract() -> None:
    args = sp_bench.build_parser().parse_args(
        [
            "validate-reference",
            "--suite",
            "suite.json",
            "--output",
            "reference.json",
            "--algorithm",
            "ulysses",
            "--dense-backend",
            "flash_attn3",
        ]
    )
    assert args.command == "validate-reference"
    assert args.algorithm == ["ulysses"]
    assert args.dense_backend == ["flash_attn3"]

    run_args = sp_bench.build_parser().parse_args(
        [
            "run",
            "--suite",
            "suite.json",
            "--output-dir",
            "results",
            "--sparse-backend",
            "dynamic_sparse_sage2_replay",
        ]
    )
    assert run_args.sparse_backend == ["dynamic_sparse_sage2_replay"]


def test_sp_aux_inputs_are_replicated_across_ranks() -> None:
    torch = pytest.importorskip("torch")
    case = sp_case()
    case["shape"] = {**case["shape"], "aux_tokens": 4, "aux_q": True}
    rank0 = sp_bench._make_inputs(case, torch.device("cpu"), seed=42, rank=0)
    rank1 = sp_bench._make_inputs(case, torch.device("cpu"), seed=42, rank=1)
    assert not torch.equal(rank0[0], rank1[0])
    for left, right in zip(rank0[3:], rank1[3:]):
        assert torch.equal(left, right)


def test_sp_report_requires_repeated_runs() -> None:
    case = sp_case()
    candidate = sp_bench.enumerate_candidates(case, ["flash_attn3"], ring_lse_backends={"flash_attn3"})["candidates"][0]
    records = []
    for repeat, latency in enumerate((2.0, 1.99, 2.01)):
        records.append(
            {
                "run_id": f"repeat-{repeat:03d}",
                "status": "ok",
                "case": case,
                "candidate": candidate,
                "metrics": {"latency_ms_mean": latency, "aggregate_effective_tflops": 100.0 / latency},
            }
        )
    for record in records:
        record["catalog_fingerprint"] = "catalog-1"
    report = sp_bench.build_report(sp_suite(case), records, candidate_catalog=sp_catalog(case, candidate))
    workload = report["workloads"][0]
    assert workload["winner"] == candidate["candidate_id"]
    assert workload["candidates"][candidate["candidate_id"]]["latency_ms"] == pytest.approx(2.0)
    recommendation = workload["recommendation"]
    assert recommendation["configuration"] == candidate
    assert recommendation["workload"] == case
    assert recommendation["latency_ms"] == pytest.approx(2.0)
    assert recommendation["logical_attention"]["q_shape"] == [1, 1024, 8, 128]
    assert recommendation["local_input"]["q_shape"] == [256, 8, 128]
    with pytest.raises(ValueError, match="duplicate raw identity"):
        sp_bench.build_report(sp_suite(case), [*records, records[0]])


def test_sp_shape_description_reports_fp8_fused_ulysses_communication() -> None:
    case = sp_case()
    case["shape"] = {
        **case["shape"],
        "sequence": 109064,
        "heads": 56,
        "kv_heads": 56,
        "head_dim": 128,
        "aux_tokens": 87,
        "aux_q": True,
        "aux_first": True,
    }
    candidate = {
        "algorithm": "ulysses",
        "quant_scheme": "fp8",
        "tensor_fusion": True,
        "head_parallel": False,
    }
    value = sp_bench._sp_shape_description(case, candidate)
    assert value["logical_attention"]["q_shape"] == [1, 109151, 56, 128]
    assert value["local_input"]["q_shape"] == [27266, 56, 128]
    assert value["per_rank_attention"]["q_shape"] == [109151, 14, 128]
    qkv = value["communication"]["qkv_all_to_all"]["packed_tensors"][0]
    assert qkv["payload_shape"] == [4, 27266, 3, 14, 128]
    assert qkv["scale_shape"] == [4, 27266, 3, 14, 1]
    output = value["communication"]["output_all_to_all"]["packed_tensors"][0]
    assert output["payload_shape"] == [4, 14, 27266, 128]
    assert output["scale_shape"] == [4, 14, 27266, 1]
    assert value["communication"]["auxiliary"]["output_all_gather_input_shape"] == [87, 14, 128]


def test_sp_candidate_catalog_fingerprint_is_stable() -> None:
    suite = sp_suite(sp_case())
    runtime = {
        "catalog": {
            "catalog_fingerprint": "backend-catalog",
            "unmapped_attention_backends": [],
            "candidates": [{"name": "flash_attn3", "family": "dense_attention", "status": "eligible"}],
        },
        "dense_backends": ["flash_attn3"],
        "ring_lse_backends": {"flash_attn3"},
        "a2a_backends": ["torch", "round_robin"],
        "fp4_available": False,
    }
    first = sp_bench.build_candidate_catalog(suite, runtime)
    second = sp_bench.build_candidate_catalog(suite, runtime)
    assert first == second
    assert len(first["catalog_fingerprint"]) == 64
    assert first["scope_complete"] is True

    narrowed = sp_bench.build_candidate_catalog(suite, {**runtime, "dense_backends": ["flash_attn3", "sage_attn2"]}, ["flash_attn3"])
    assert narrowed["scope_complete"] is False
    assert narrowed["catalog_fingerprint"] != first["catalog_fingerprint"]

    missing_dependency = {
        **runtime,
        "catalog": {
            **runtime["catalog"],
            "candidates": [
                *runtime["catalog"]["candidates"],
                {"name": "future_attn", "family": "dense_attention", "status": "dependency_missing"},
            ],
        },
    }
    blocked = sp_bench.build_candidate_catalog(suite, missing_dependency)
    assert blocked["backend_scope_complete"] is False
    assert blocked["scope_complete"] is False
    assert blocked["backend_scope_blockers"]["dependency_missing"] == ["future_attn"]


def test_sp_report_does_not_publish_partial_or_unstable_winner() -> None:
    case = sp_case()
    candidates = sp_bench.enumerate_candidates(case, ["flash_attn3"], ring_lse_backends={"flash_attn3"})["candidates"][:2]
    records = []
    for repeat, latency in enumerate((1.0, 1.2, 1.0)):
        records.append(
            {
                "run_id": f"repeat-{repeat:03d}",
                "status": "ok",
                "case": case,
                "candidate": candidates[0],
                "metrics": {"latency_ms_mean": latency, "aggregate_effective_tflops": 100.0},
                "correctness": {"passed": True},
                "catalog_fingerprint": "catalog-1",
            }
        )
    report = sp_bench.build_report(
        sp_suite(case),
        records,
        candidate_catalog=sp_catalog(case, *candidates),
    )
    workload = report["workloads"][0]
    assert workload["winner"] is None
    assert workload["measured_winner"] is None
    assert workload["candidate_coverage_complete"] is False
    assert workload["candidates"][candidates[0]["candidate_id"]]["status"] == "unstable"
    assert workload["candidates"][candidates[1]["candidate_id"]]["status"] == "not_measured"


def test_sp_report_requires_catalog_for_formal_winner_and_rejects_mixed_environment() -> None:
    case = sp_case()
    candidate = sp_bench.enumerate_candidates(case, ["flash_attn3"], ring_lse_backends={"flash_attn3"})["candidates"][0]
    records = [
        {
            "run_id": f"repeat-{repeat:03d}",
            "status": "ok",
            "case": case,
            "candidate": candidate,
            "metrics": {"latency_ms_mean": 1.0, "aggregate_effective_tflops": 100.0},
            "environment": {"ranks": [{"rank": 0, "torch": "test"}]},
            "run": {"iterations": 2},
        }
        for repeat in range(3)
    ]
    report = sp_bench.build_report(sp_suite(case), records)
    assert report["workloads"][0]["measured_winner"] == candidate["candidate_id"]
    assert report["workloads"][0]["winner"] is None

    records[-1]["environment"]["ranks"][0]["torch"] = "different"
    with pytest.raises(ValueError, match="mixed SP raw hardware or software environments"):
        sp_bench.build_report(sp_suite(case), records)


def test_sp_existing_raw_rejects_contract_drift() -> None:
    case = sp_case()
    candidate = sp_bench.enumerate_candidates(case, ["flash_attn3"], ring_lse_backends={"flash_attn3"})["candidates"][0]
    catalog = sp_catalog(case, candidate)
    environment = [
        {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "gpu": {"uuid": "gpu-0"},
            "torch": "test",
            "cuda_runtime": "test",
            "cuda_visible_devices": "0",
            "git_commit": "commit",
            "nccl_version": [1, 0, 0],
            "nccl_environment": {},
        }
    ]
    contract = {"warmup": 1, "iterations": 2, "seed": 42, "world_size": 1, "timing_mode": "max_rank_event_list_single_sync"}
    record = {
        "kind": "sp_attention_benchmark_raw_v1",
        "run_id": "repeat-000",
        "case": case,
        "candidate": candidate,
        "catalog_fingerprint": catalog["catalog_fingerprint"],
        "environment": {"ranks": environment},
        "run": contract,
    }
    keys = sp_bench.validate_existing_records(sp_suite(case), catalog, [record], contract, environment, repeat_runs=3)
    assert keys == {("repeat-000", case["case_id"], candidate["candidate_id"])}
    with pytest.raises(ValueError, match="measurement arguments"):
        sp_bench.validate_existing_records(sp_suite(case), catalog, [record], {**contract, "iterations": 3}, environment, repeat_runs=3)


def test_sp_communication_byte_and_bandwidth_metrics() -> None:
    torch = pytest.importorskip("torch")
    packed = ((torch.empty((4, 8), dtype=torch.bfloat16), None), (torch.empty((2, 4), dtype=torch.float32), torch.empty((2, 1), dtype=torch.float32)))
    assert sp_bench._tensor_bytes(packed) == 104
    rates = sp_bench._communication_rates(logical_bytes=4000, network_bytes=3000, latency_ms=2.0, world_size=4)
    assert rates["algorithmic_bandwidth_gbps"] == pytest.approx(0.002)
    assert rates["bus_bandwidth_gbps"] == pytest.approx(0.0015)
    assert rates["aggregate_network_bytes"] == 12000
    overlap = sp_bench._overlap_estimate(full_l1_ms=7.0, compute_only_ms=5.0, layer2_ms=4.0)
    assert overlap["estimate_status"] == "estimated"
    assert overlap["estimated_exposed_layout_communication_ms"] == pytest.approx(2.0)
    assert overlap["estimated_overlap_ratio"] == pytest.approx(0.5)
    unresolved = sp_bench._overlap_estimate(full_l1_ms=1.3, compute_only_ms=0.7, layer2_ms=0.2)
    assert unresolved["estimate_status"] == "unresolved_positive_residual"
    assert unresolved["estimated_overlap_ratio"] is None


def test_sp_diagnostic_report_aggregates_repeats() -> None:
    case = sp_case()
    candidate = sp_bench.enumerate_candidates(case, ["flash_attn3"], ring_lse_backends={"flash_attn3"})["candidates"][0]
    records = []
    for repeat, latency in enumerate((1.0, 1.1, 0.9)):
        records.append(
            {
                "run_id": f"repeat-{repeat:03d}",
                "status": "ok",
                "case": case,
                "candidate": candidate,
                "metrics": {
                    "layer2": {
                        "total": {"latency_ms_mean": latency},
                        "segments": {"pack_qkv": {"latency_ms_mean": 0.1, "calls_per_attention": 1}},
                    },
                    "layer3": {
                        "latency_ms_mean": 0.5,
                        "logical_payload_bytes_per_rank": 4000,
                        "network_bytes_per_rank": 3000,
                        "aggregate_network_bytes": 12000,
                        "algorithmic_bandwidth_gbps": 0.008,
                        "bus_bandwidth_gbps": 0.006,
                        "communication_pattern": "all_to_all_with_aux_all_gather",
                        "peak_profile_pattern": "all_to_all",
                        "primitive": "torch",
                        "communication_components": {
                            "main_qkv_all_to_all": {
                                "logical_payload_bytes_per_rank": 3000,
                                "network_bytes_per_rank": 2250,
                                "aggregate_network_bytes": 9000,
                                "latency_ms_mean": 0.25,
                                "algorithmic_bandwidth_gbps": 0.012,
                                "bus_bandwidth_gbps": 0.009,
                                "peak_profile_pattern": "all_to_all",
                            },
                            "aux_output_all_gather": {
                                "logical_payload_bytes_per_rank": 1000,
                                "network_bytes_per_rank": 750,
                                "aggregate_network_bytes": 3000,
                                "latency_ms_mean": 0.1,
                                "algorithmic_bandwidth_gbps": 0.01,
                                "bus_bandwidth_gbps": 0.0075,
                                "peak_profile_pattern": "all_gather",
                            },
                        },
                        "aux_qkv_input": "replicated_bypass_qkv_all_to_all",
                    },
                    "overlap": {
                        "full_l1_latency_ms": 1.5,
                        "compute_only_latency_ms": 0.8,
                        "layer2_layout_communication_latency_ms": latency,
                        "serial_reference_latency_ms": 0.8 + latency,
                        "full_minus_serial_reference_ms": 0.7 - latency,
                        "estimation_tolerance_ms": 0.05,
                        "estimate_status": "estimated",
                        "observed_full_minus_compute_ms": 0.7,
                        "estimated_exposed_layout_communication_ms": 0.7,
                        "estimated_hidden_layout_communication_ms": max(0.0, latency - 0.7),
                        "estimated_overlap_ratio": max(0.0, latency - 0.7) / latency,
                        "definition": "test",
                        "interpretation": "test",
                    },
                },
            }
        )
    strict_report = sp_bench.build_diagnostic_report(sp_suite(case), records)
    assert strict_report["results"][0]["status"] == "unstable"
    report = sp_bench.build_diagnostic_report(sp_suite(case), records, max_spread_pct=30.0)
    result = report["results"][0]
    assert result["status"] == "accepted"
    assert result["layer2_latency_ms"] == pytest.approx(1.0)
    assert result["layer2_spread_pct"] == pytest.approx(20.0)
    assert result["layer3"]["bus_bandwidth_gbps"] == pytest.approx(0.006)
    assert result["layer3"]["peak_profile_pattern"] == "all_to_all"
    assert result["layer3"]["communication_components"]["aux_output_all_gather"]["logical_payload_bytes_per_rank"] == 1000
    assert result["layer3"]["communication_components"]["aux_output_all_gather"]["latency_ms"] == pytest.approx(0.1)
    assert result["layer3"]["communication_components"]["aux_output_all_gather"]["bus_bandwidth_gbps"] == pytest.approx(0.0075)
    assert result["layer3"]["communication_components"]["aux_output_all_gather"]["status"] == "accepted"
    assert result["candidate"] == candidate
    assert result["workload"] == case
    assert result["overlap"]["estimate_status"] == "estimated"
    assert result["overlap"]["estimated_overlap_ratio"] == pytest.approx(0.3)
    markdown = sp_bench.render_diagnostic_markdown(report)
    assert "L2 分段" in markdown
    assert "aux_output_all_gather" in markdown
    assert "replicated_bypass_qkv_all_to_all" in markdown
    assert "输入精度：`bf16`" in markdown
    assert "communication=`none`" in markdown
    assert "峰值口径：`all_to_all`" in markdown

    wrong_case = sp_case(shape={**case["shape"], "head_dim": 64})
    with pytest.raises(ValueError, match="raw suite drift"):
        sp_bench.build_diagnostic_report(sp_suite(wrong_case), records)

    environment = [
        {
            "rank": 0,
            "local_rank": 0,
            "gpu": {"name": "NVIDIA H100 80GB HBM3", "major": 9, "minor": 0, "pci_domain_id": 0, "pci_bus_id": 1, "pci_device_id": 0},
            "nvidia_smi_topology": ["GPU0 GPU1", "GPU0 X NV18", "GPU1 NV18 X"],
        },
        {
            "rank": 1,
            "local_rank": 1,
            "gpu": {"name": "NVIDIA H100 80GB HBM3", "major": 9, "minor": 0, "pci_domain_id": 0, "pci_bus_id": 2, "pci_device_id": 0},
        },
    ]
    for record in records:
        record["environment"] = {"ranks": environment}
    fingerprint = sp_bench._topology_fingerprint(records)
    profiles = {
        "schema_version": 1,
        "platforms": {
            "h100_test": {
                "identification": {
                    "gpu_name_regex": "^NVIDIA H100 80GB HBM3$",
                    "cuda_capability": "9.0",
                    "world_size": 2,
                    "topology_fingerprint": fingerprint,
                },
                "interconnect_peaks": {
                    "all_to_all": {
                        "bus_bandwidth": 0.012,
                        "unit": "GB/s",
                        "kind": "empirical_envelope",
                        "source": "unit test",
                    },
                    "all_gather": {
                        "bus_bandwidth": 0.01,
                        "unit": "GB/s",
                        "kind": "empirical_envelope",
                        "source": "unit test",
                    }
                },
            }
        }
    }
    profiled = sp_bench.build_diagnostic_report(
        sp_suite(case),
        records,
        max_spread_pct=30.0,
        interconnect_profiles=profiles,
        platform_id="h100_test",
    )
    assert profiled["hardware"]["topology_fingerprint"] == fingerprint
    components = profiled["results"][0]["layer3"]["communication_components"]
    assert components["main_qkv_all_to_all"]["bus_peak_efficiency"] == pytest.approx(0.75)
    assert components["aux_output_all_gather"]["bus_peak_efficiency"] == pytest.approx(0.75)
    assert components["aux_output_all_gather"]["observed_bus_peak_efficiency"] == pytest.approx(0.75)
    assert "75.00% (`all_gather`)" in sp_bench.render_diagnostic_markdown(profiled)
    assert profiled["results"][0]["layer3"]["peak_status"] == "available"
    assert profiled["results"][0]["layer3"]["bus_peak_efficiency"] == pytest.approx(0.5)
    no_topology = [{"environment": {"ranks": [{**environment[0], "nvidia_smi_topology": []}, environment[1]]}}]
    assert sp_bench._topology_fingerprint(no_topology) is None
    profiles["platforms"]["h100_test"]["identification"]["topology_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="topology"):
        sp_bench.build_diagnostic_report(
            sp_suite(case),
            records,
            max_spread_pct=30.0,
            interconnect_profiles=profiles,
            platform_id="h100_test",
        )


def test_sp_diagnostics_can_load_formal_winners(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "kind": "sp_attention_benchmark_report_v1",
                "suite_id": "diagnostic-suite",
                "workloads": [
                    {"case_id": "case-a", "recommendation": {"candidate_id": "winner-a"}},
                    {"case_id": "case-b", "recommendation": {"candidate_id": "winner-b"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    suite = {"suite_id": "diagnostic-suite", "cases": [{"case_id": "case-a"}, {"case_id": "case-b"}]}
    assert sp_bench.diagnostic_candidates_from_report(report_path, suite) == {"case-a": "winner-a", "case-b": "winner-b"}
    args = sp_bench.build_parser().parse_args(
        [
            "diagnose",
            "--suite",
            "suite.json",
            "--output-dir",
            "results",
            "--recommendation-report",
            str(report_path),
        ]
    )
    assert args.candidate == []
    assert args.recommendation_report == report_path
