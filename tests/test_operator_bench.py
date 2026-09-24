from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.benchmarks import operator_backends as backends
from tools.benchmarks import operator_bench as core
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
