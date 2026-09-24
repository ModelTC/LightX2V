from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def attention_case() -> dict:
    return {
        "case_id": "attention.1",
        "operator_family": "dense_attention",
        "shape": {"batch": 1, "seq_q": 64, "seq_kv": 64, "heads": 8, "kv_heads": 8, "head_dim": 64, "causal": False},
        "precision": {"input_dtype": "bf16"},
    }


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
) -> dict:
    return {
        "run_id": f"repeat-{run:03d}",
        "case": {
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
            },
        },
        "environment": {
            "gpu": {"name": gpu, "major": 9, "minor": 0, "total_memory_bytes": 80 * 1024**3}
        },
        "status": "ok",
        "metrics": {
            "latency_ms_mean": latency_ms,
            rate_metric: rate,
            "work_definition": "2*m*n*k",
            "measurement_scope": "kernel_cuda_event_list_single_sync",
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


def test_builtin_and_optional_plugins_register_without_runtime_dependencies() -> None:
    registry = load_registry()
    names = {item["name"] for item in registry.descriptors()}
    assert {"torch_linear", "torch_sdpa", "torch_expert_loop"} <= names
    assert {"fp8-vllm", "flash_attn3", "torch_grouped_mm", "Default"} <= names


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
                    "timing_mode": options["timing_mode"],
                    "rate_schema": options["rate_schema"],
                },
            },
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
    )
    workload = value["workloads"][0]
    assert workload["winner"] == "fast"
    assert workload["observed_backend"]["speedup"] == pytest.approx(2.0, rel=0.01)
    assert workload["backends"]["fast"]["nominal_efficiency"] == pytest.approx(500 / 989.5)
    assert value["summary"]["weighted_per_shape_winner_ms"] == pytest.approx(10.01)


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
