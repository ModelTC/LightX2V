"""Backend contracts, registry, and portable Torch baselines."""

from __future__ import annotations

import hashlib
import heapq
import importlib
import io
import json
import math
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    family: str
    plugin: str
    input_dtypes: tuple[str, ...]
    description: str
    optional_dependencies: tuple[str, ...] = ()
    production_backend: str | None = None
    production_variant: str | None = None
    cuda_capabilities: tuple[str, ...] = ()
    probe_symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["input_dtypes"] = list(self.input_dtypes)
        value["optional_dependencies"] = list(self.optional_dependencies)
        value["cuda_capabilities"] = list(self.cuda_capabilities)
        value["probe_symbols"] = list(self.probe_symbols)
        return value


@dataclass
class PreparedOperation:
    fn: Callable[[], Any]
    work: int
    rate_metric: str
    work_definition: str
    dense_equivalent_work: int | None = None
    dense_equivalent_work_definition: str | None = None
    extra_metrics: dict[str, Any] = field(default_factory=dict)
    correctness: dict[str, Any] | None = None


class BackendAdapter(Protocol):
    descriptor: BackendDescriptor

    def support_error(self, case: dict[str, Any]) -> str | None: ...

    def precision(self, case: dict[str, Any]) -> dict[str, Any]: ...

    def prepare(self, case: dict[str, Any], device: str, options: dict[str, Any]) -> PreparedOperation: ...


class BackendRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, BackendAdapter] = {}

    def register(self, adapter: BackendAdapter) -> None:
        name = adapter.descriptor.name
        if name in self._adapters:
            raise ValueError(f"duplicate backend registration: {name}")
        self._adapters[name] = adapter

    def get(self, name: str) -> BackendAdapter | None:
        return self._adapters.get(name)

    def require(self, name: str) -> BackendAdapter:
        adapter = self.get(name)
        if adapter is None:
            raise ValueError(f"backend is not registered: {name}")
        return adapter

    def descriptors(self) -> list[dict[str, Any]]:
        return [self._adapters[name].descriptor.to_dict() for name in sorted(self._adapters)]


TORCH_DTYPES = ("bf16", "fp16", "fp32")

PRODUCTION_ATTN_EXCLUSIONS = {
    "draft_attn": "model-specific draft attention",
    "general_sparse_attn": "composite sparse policy; benchmark its operator variants directly",
    "nbhd_attn": "model-specific sparse layout",
    "nbhd_attn_flashinfer": "model-specific sparse layout",
    "radial_attn": "model-specific sparse layout",
    "rainfusion_attn": "composite sparse policy",
    "ring": "distributed attention is outside the single-card scope",
    "sage_attn2_k_int8_v_fp8": "mixed-precision specialized path requires a separate precision contract",
    "sol_attn": "composite runtime selector; benchmark its leaf backends directly",
    "svg2_attn": "model-specific sparse layout",
    "svg_attn": "model-specific sparse layout",
    "ulysses": "distributed attention is outside the single-card scope",
    "ulysses-4090": "distributed attention is outside the single-card scope",
}

PRODUCTION_MM_EXCLUSIONS = {
    "Calib": "calibration wrapper rather than an inference GEMM backend",
    "CalibMax": "calibration wrapper rather than an inference GEMM backend",
    "Default-ForceFp32": "FP32-sensitive model path requires a separate input precision contract",
    "TensorParallel": "distributed tensor-parallel wrapper is outside the single-card scope",
    "fp8-b128-deepgemm": "block-scaled weight layout requires a separate synthetic weight contract",
    "fp8-f16-accum": "requires a model-provided activation qmax before the FP16-accumulation path is enabled",
    "fp8-pertensor": "checkpoint scale granularity requires a separate synthetic weight contract",
    "int4-g128-marlin": "groupwise INT4 packing requires a separate synthetic weight contract",
    "int8-convrot": "ConvRot requires checkpoint-provided rotation metadata",
    "mxfp4": "microscaling weight layout requires a separate synthetic weight contract",
    "mxfp6-mxfp8": "microscaling weight layout requires a separate synthetic weight contract",
    "mxfp8": "microscaling weight layout requires a separate synthetic weight contract",
    "nvfp4": "NVFP4 weight layout requires a separate synthetic weight contract",
    "nvfp4-split-n-workaround": "NVFP4 weight layout requires a separate synthetic weight contract",
}
PRODUCTION_MM_EXCLUSIONS.update(
    {
        name: "GGUF checkpoint format requires model weights rather than synthetic dense tensors"
        for name in (
            "gguf-BF16",
            "gguf-Q8_0",
            "gguf-Q6_K",
            "gguf-Q5_K_S",
            "gguf-Q5_K_M",
            "gguf-Q5_1",
            "gguf-Q5_0",
            "gguf-Q4_K_M",
            "gguf-Q4_K_S",
            "gguf-Q4_1",
            "gguf-Q4_0",
            "gguf-Q3_K_M",
            "gguf-Q3_K_S",
        )
    }
)


def _dtype(torch: Any, name: str) -> Any:
    values = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if name not in values:
        raise ValueError(f"unsupported Torch dtype: {name}")
    return values[name]


def _torch_precision(case: dict[str, Any]) -> dict[str, Any]:
    dtype = str(case["precision"]["input_dtype"])
    return {
        "input_dtype": dtype,
        "weight_dtype": dtype,
        "accum_dtype": "backend_default",
        "output_dtype": dtype,
        "quant": None,
    }


class TorchLinearAdapter:
    descriptor = BackendDescriptor(
        name="torch_linear",
        family="gemm",
        plugin="builtin_torch",
        input_dtypes=TORCH_DTYPES,
        description="torch.mm for bias=false and torch.addmm for bias=true",
    )

    def support_error(self, case: dict[str, Any]) -> str | None:
        dtype = str(case["precision"]["input_dtype"])
        return None if dtype in self.descriptor.input_dtypes else f"torch_linear does not support dtype {dtype}"

    def precision(self, case: dict[str, Any]) -> dict[str, Any]:
        return _torch_precision(case)

    def prepare(self, case: dict[str, Any], device: str, options: dict[str, Any]) -> PreparedOperation:
        import torch

        shape = case["shape"]
        dtype = _dtype(torch, str(case["precision"]["input_dtype"]))
        m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
        a = torch.randn((m, k), device=device, dtype=dtype)
        b = torch.randn((k, n), device=device, dtype=dtype)
        if shape["bias"]:
            bias = torch.randn((n,), device=device, dtype=dtype)

            def fn() -> Any:
                return torch.addmm(bias, a, b)
        else:

            def fn() -> Any:
                return torch.mm(a, b)
        return PreparedOperation(fn=fn, work=2 * m * n * k, rate_metric="tflops", work_definition="2*m*n*k")


class TorchSDPAAdapter:
    descriptor = BackendDescriptor(
        name="torch_sdpa",
        family="dense_attention",
        plugin="builtin_torch",
        input_dtypes=("bf16", "fp16"),
        description="torch.nn.functional.scaled_dot_product_attention",
        production_backend="torch_sdpa",
    )

    def support_error(self, case: dict[str, Any]) -> str | None:
        dtype = str(case["precision"]["input_dtype"])
        return None if dtype in self.descriptor.input_dtypes else f"torch_sdpa does not support dtype {dtype}"

    def precision(self, case: dict[str, Any]) -> dict[str, Any]:
        return {**_torch_precision(case), "weight_dtype": None}

    def prepare(self, case: dict[str, Any], device: str, options: dict[str, Any]) -> PreparedOperation:
        import torch
        import torch.nn.functional as functional

        shape = case["shape"]
        dtype = _dtype(torch, str(case["precision"]["input_dtype"]))
        batch = int(shape["batch"])
        seq_q, seq_kv = int(shape["seq_q"]), int(shape["seq_kv"])
        heads, kv_heads, head_dim = int(shape["heads"]), int(shape["kv_heads"]), int(shape["head_dim"])
        q = torch.randn((batch, heads, seq_q, head_dim), device=device, dtype=dtype)
        k = torch.randn((batch, kv_heads, seq_kv, head_dim), device=device, dtype=dtype)
        v = torch.randn((batch, kv_heads, seq_kv, head_dim), device=device, dtype=dtype)
        kwargs = {"enable_gqa": True} if heads != kv_heads else {}

        def fn() -> Any:
            return functional.scaled_dot_product_attention(q, k, v, is_causal=bool(shape["causal"]), **kwargs)
        return PreparedOperation(
            fn=fn,
            work=4 * batch * heads * seq_q * seq_kv * head_dim,
            rate_metric="effective_tflops",
            work_definition="4*batch*heads*seq_q*seq_kv*head_dim",
        )


def _observed_routes(expert_counts: list[int], tokens: int, top_k: int, device: str, torch: Any) -> Any:
    heap = [(-count, expert) for expert, count in enumerate(expert_counts) if count]
    heapq.heapify(heap)
    rows = []
    for _ in range(tokens):
        if len(heap) < top_k:
            raise ValueError("observed expert_counts cannot form unique top-k routes")
        row = [heapq.heappop(heap) for _ in range(top_k)]
        rows.append([expert for _, expert in row])
        for negative_count, expert in row:
            if negative_count < -1:
                heapq.heappush(heap, (negative_count + 1, expert))
    if heap:
        raise ValueError("observed expert_counts were not fully consumed")
    return torch.tensor(rows, device=device, dtype=torch.int64)


def _routing(case: dict[str, Any], device: str, distribution: str, torch: Any) -> tuple[Any, Any, dict[str, Any]]:
    shape = case["shape"]
    tokens, num_experts, top_k = int(shape["tokens"]), int(shape["num_experts"]), int(shape["top_k"])
    token_ids = torch.arange(tokens, device=device, dtype=torch.int64).unsqueeze(1)
    route_ids = torch.arange(top_k, device=device, dtype=torch.int64).unsqueeze(0)
    if distribution == "observed":
        counts = [int(value) for value in (case.get("routing") or {}).get("expert_counts") or []]
        if len(counts) != num_experts or sum(counts) != tokens * top_k:
            raise ValueError("observed routing requires num_experts counts summing to tokens * top_k")
        selected = _observed_routes(counts, tokens, top_k, device, torch)
    elif distribution == "balanced":
        selected = (token_ids * top_k + route_ids) % num_experts
    elif distribution == "uniform":
        selected = torch.rand((tokens, num_experts), device=device).topk(top_k, dim=1, sorted=False).indices
    elif distribution == "skewed":
        selected = (token_ids * top_k + route_ids) % num_experts
        hot_tokens = int(tokens * 0.8)
        selected[:hot_tokens] = route_ids.expand(hot_tokens, -1)
    else:
        raise ValueError(f"unsupported MoE routing distribution: {distribution}")
    scales = torch.rand((tokens, top_k), device=device, dtype=torch.float32)
    scales = scales / scales.sum(dim=1, keepdim=True)
    counts = torch.bincount(selected.reshape(-1), minlength=num_experts).float()
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    normalized_entropy = entropy / math.log(num_experts) if num_experts > 1 else entropy.new_tensor(1.0)
    metrics = {
        "expert_counts": [int(value) for value in counts.cpu().tolist()],
        "active_experts": int((counts > 0).sum().item()),
        "max_expert_load": int(counts.max().item()),
        "min_expert_load": int(counts.min().item()),
        "load_cv": float((counts.std(unbiased=False) / counts.mean()).item()),
        "normalized_entropy": float(normalized_entropy.item()),
        "routing_distribution": distribution,
    }
    return selected, scales, metrics


class TorchExpertLoopAdapter:
    descriptor = BackendDescriptor(
        name="torch_expert_loop",
        family="moe",
        plugin="builtin_torch",
        input_dtypes=("bf16", "fp16"),
        description="portable per-expert Torch linear baseline",
    )

    def support_error(self, case: dict[str, Any]) -> str | None:
        dtype = str(case["precision"]["input_dtype"])
        return None if dtype in self.descriptor.input_dtypes else f"torch_expert_loop does not support dtype {dtype}"

    def precision(self, case: dict[str, Any]) -> dict[str, Any]:
        return _torch_precision(case)

    def prepare(self, case: dict[str, Any], device: str, options: dict[str, Any]) -> PreparedOperation:
        import torch
        import torch.nn.functional as functional

        shape = case["shape"]
        dtype = _dtype(torch, str(case["precision"]["input_dtype"]))
        tokens = int(shape["tokens"])
        hidden = int(shape["hidden_size"])
        intermediate = int(shape["intermediate_size"])
        experts, top_k = int(shape["num_experts"]), int(shape["top_k"])
        fc1_size = intermediate if shape["activation"] == "gelu" else 2 * intermediate
        inputs = torch.randn((tokens, hidden), device=device, dtype=dtype)
        fc1 = torch.randn((experts, fc1_size, hidden), device=device, dtype=dtype) * 0.02
        fc2 = torch.randn((experts, hidden, intermediate), device=device, dtype=dtype) * 0.02
        if shape["expert_bias"]:
            fc1_bias = torch.randn((experts, fc1_size), device=device, dtype=dtype) * 0.02
            fc2_bias = torch.randn((experts, hidden), device=device, dtype=dtype) * 0.02
        else:
            fc1_bias = fc2_bias = None
        selected, scales, routing_metrics = _routing(
            case, device, str(options.get("moe_routing") or "balanced"), torch
        )
        output = torch.empty_like(inputs)

        def fn() -> Any:
            output.zero_()
            for expert in range(experts):
                positions = (selected == expert).nonzero(as_tuple=False)
                if not positions.numel():
                    continue
                token_indexes, route_indexes = positions[:, 0], positions[:, 1]
                value = functional.linear(
                    inputs[token_indexes], fc1[expert], None if fc1_bias is None else fc1_bias[expert]
                )
                if shape["activation"] == "gelu":
                    value = functional.gelu(value)
                else:
                    value, gate = value.chunk(2, dim=-1)
                    value = value * functional.silu(gate)
                value = functional.linear(value, fc2[expert], None if fc2_bias is None else fc2_bias[expert])
                value = value * scales[token_indexes, route_indexes].to(dtype).unsqueeze(1)
                output.index_add_(0, token_indexes, value)
            return output

        routes = tokens * top_k
        fc1_factor = 2 if shape["activation"] == "swiglu" else 1
        return PreparedOperation(
            fn=fn,
            work=routes * 2 * hidden * intermediate * (fc1_factor + 1),
            rate_metric="effective_tflops",
            work_definition="routed_fc1_and_fc2_matmul_only",
            extra_metrics=routing_metrics,
        )


@dataclass(frozen=True)
class MMRegistryContract:
    key: str
    weight_dtype: str
    quant_dtype: str | None
    scale_layout: str | None
    provider: str
    dependencies: tuple[str, ...]
    probe_symbols: tuple[str, ...] = ()


MM_REGISTRY_CONTRACTS = (
    MMRegistryContract("Default", "bf16", None, None, "torch", ("lightx2v",)),
    MMRegistryContract("fp8-vllm", "fp8_e4m3_per_channel", "fp8_e4m3", "n", "vllm", ("lightx2v", "vllm")),
    MMRegistryContract("int8-vllm", "int8_per_channel", "int8", "n", "vllm", ("lightx2v", "vllm")),
    MMRegistryContract("fp8-sgl", "fp8_e4m3_per_channel", "fp8_e4m3", "n", "sgl_kernel", ("lightx2v", "sgl_kernel")),
    MMRegistryContract("int8-sgl", "int8_per_channel", "int8", "n", "sgl_kernel", ("lightx2v", "sgl_kernel", "vllm")),
    MMRegistryContract("fp8-torchao", "fp8_e4m3_per_channel", "fp8_e4m3", "n_by_1", "torchao", ("lightx2v", "torchao")),
    MMRegistryContract("int8-torchao", "int8_per_channel", "int8", "n_by_1", "torchao", ("lightx2v", "torchao")),
    MMRegistryContract(
        "fp8-q8f",
        "fp8_e4m3_per_channel",
        "fp8_e4m3",
        "n",
        "q8_kernels",
        ("lightx2v", "q8_kernels"),
        ("lightx2v.common.ops.mm.mm_weight:fp8_linear",),
    ),
    MMRegistryContract(
        "int8-q8f",
        "int8_per_channel",
        "int8",
        "n",
        "q8_kernels",
        ("lightx2v", "q8_kernels"),
        ("lightx2v.common.ops.mm.mm_weight:q8_linear",),
    ),
    MMRegistryContract("fp8-triton", "fp8_e4m3_per_channel", "fp8_e4m3", "n", "triton", ("lightx2v", "triton")),
    MMRegistryContract("int8-triton", "int8_per_channel", "int8", "n", "triton", ("lightx2v", "triton")),
)


def _quantize_mm_weight(raw_weight: Any, contract: MMRegistryContract, torch: Any) -> tuple[Any, Any]:
    limit = 448.0 if contract.quant_dtype == "fp8_e4m3" else 127.0
    scale = (raw_weight.float().abs().amax(dim=1) / limit).clamp_min(1e-8)
    normalized = raw_weight.float() / scale[:, None]
    if contract.quant_dtype == "fp8_e4m3":
        quantized = normalized.clamp(-limit, limit).to(torch.float8_e4m3fn)
    else:
        quantized = normalized.round().clamp(-limit, limit).to(torch.int8)
    return quantized, scale[:, None].float() if contract.scale_layout == "n_by_1" else scale.float()


class LightX2VMMAdapter:
    def __init__(self, contract: MMRegistryContract) -> None:
        self.contract = contract
        self.descriptor = BackendDescriptor(
            name=contract.key,
            family="gemm",
            plugin="lightx2v_mm_registry",
            input_dtypes=("bf16",),
            description=f"LightX2V production MM wrapper {contract.key}",
            optional_dependencies=contract.dependencies,
            production_backend=contract.key,
            probe_symbols=contract.probe_symbols,
        )

    def support_error(self, case: dict[str, Any]) -> str | None:
        dtype = str(case["precision"]["input_dtype"])
        if dtype != "bf16":
            return f"{self.contract.key} requires BF16 input, got {dtype}"
        shape = case["shape"]
        if self.contract.quant_dtype and (int(shape["n"]) % 16 or int(shape["k"]) % 16):
            return f"{self.contract.key} requires n and k divisible by 16"
        return None

    def precision(self, case: dict[str, Any]) -> dict[str, Any]:
        quant = None
        if self.contract.quant_dtype:
            quant = f"{self.contract.quant_dtype}_dynamic_activation_per_channel_weight"
        return {
            "input_dtype": "bf16",
            "weight_dtype": self.contract.weight_dtype,
            "accum_dtype": "production_wrapper_default",
            "output_dtype": "bf16",
            "quant": quant,
            "implementation": f"lightx2v_mm_registry:{self.contract.key}",
        }

    def prepare(self, case: dict[str, Any], device: str, options: dict[str, Any]) -> PreparedOperation:
        import torch

        from lightx2v.common.ops.mm import mm_weight as _mm_weight  # noqa: F401
        from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER

        wrapper_class = MM_WEIGHT_REGISTER.get(self.contract.key)
        if wrapper_class is None:
            raise RuntimeError(f"LightX2V MM registry key is unavailable: {self.contract.key}")
        shape = case["shape"]
        m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
        activation = torch.randn((m, k), device=device, dtype=torch.bfloat16)
        raw_weight = torch.randn((n, k), device=device, dtype=torch.bfloat16)
        bias = torch.randn((n,), device=device, dtype=torch.bfloat16) if shape["bias"] else None
        weight_name = "operator_bench.weight"
        bias_name = "operator_bench.bias" if bias is not None else None
        wrapper = wrapper_class(weight_name, bias_name)
        values = {weight_name: raw_weight}
        if self.contract.quant_dtype:
            weight, scale = _quantize_mm_weight(raw_weight, self.contract, torch)
            values = {weight_name: weight, "operator_bench.weight_scale": scale}
        if bias is not None:
            values[bias_name] = bias
        wrapper.load(values)

        def fn() -> Any:
            return wrapper.apply(activation)

        if self.contract.quant_dtype == "int8":
            rate_metric = "effective_tops"
        elif self.contract.quant_dtype:
            rate_metric = "effective_tflops"
        else:
            rate_metric = "tflops"
        return PreparedOperation(
            fn=fn,
            work=2 * m * n * k,
            rate_metric=rate_metric,
            work_definition="2*m*n*k; LightX2V production wrapper apply",
            extra_metrics={"registry_key": self.contract.key},
        )


def _attention_support_error(case: dict[str, Any], backend: str) -> str | None:
    dtype = str(case["precision"]["input_dtype"])
    if dtype not in {"bf16", "fp16"}:
        return f"{backend} supports BF16/FP16, got {dtype}"
    head_dim = int(case["shape"]["head_dim"])
    if backend in {"sage_attn2", "sage_attn3"} and head_dim not in {64, 128}:
        return f"{backend} supports head_dim 64 or 128, got {head_dim}"
    if backend not in {"sage_attn2", "sage_attn3"} and (head_dim > 256 or head_dim % 8):
        return f"{backend} requires head_dim <= 256 and divisible by 8"
    if backend in {"flash_attn4", "sage_attn3"} and case["shape"]["causal"]:
        return f"{backend} production wrapper does not provide causal attention"
    return None


def _attention_tensors(case: dict[str, Any], device: str, torch: Any) -> tuple[Any, Any, Any]:
    shape = case["shape"]
    dtype = _dtype(torch, str(case["precision"]["input_dtype"]))
    batch, seq_q, seq_kv = int(shape["batch"]), int(shape["seq_q"]), int(shape["seq_kv"])
    heads, kv_heads, head_dim = int(shape["heads"]), int(shape["kv_heads"]), int(shape["head_dim"])
    q = torch.randn((batch, seq_q, heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch, seq_kv, kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch, seq_kv, kv_heads, head_dim), device=device, dtype=dtype)
    return q, k, v


def _attention_prepared(case: dict[str, Any], fn: Callable[[], Any], backend: str) -> PreparedOperation:
    shape = case["shape"]
    work = 4 * int(shape["batch"]) * int(shape["heads"]) * int(shape["seq_q"]) * int(shape["seq_kv"]) * int(shape["head_dim"])
    return PreparedOperation(
        fn=fn,
        work=work,
        rate_metric="effective_tflops",
        work_definition="4*batch*heads*seq_q*seq_kv*head_dim",
        extra_metrics={"implementation": backend},
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_qkv_replay(case: dict[str, Any], device: str, torch: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    replay = case["replay"]
    manifest_path = Path(replay["manifest"]).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"sparse replay manifest does not exist: {manifest_path}")
    manifest_sha256 = _file_sha256(manifest_path)
    if manifest_sha256 != replay["sha256"]:
        raise ValueError(f"sparse replay manifest sha256 mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "operator_benchmark_qkv_replay_v1":
        raise ValueError(f"unsupported sparse replay manifest: {manifest_path}")
    if manifest.get("layout") != "BSHD":
        raise ValueError("sparse replay layout must be BSHD")
    specs = manifest.get("tensors")
    if not isinstance(specs, dict) or set(specs) != {"q", "k", "v"}:
        raise ValueError("sparse replay manifest requires exactly q/k/v tensors")

    shape = case["shape"]
    expected_shapes = {
        "q": [shape["batch"], shape["seq_q"], shape["heads"], shape["head_dim"]],
        "k": [shape["batch"], shape["seq_kv"], shape["kv_heads"], shape["head_dim"]],
        "v": [shape["batch"], shape["seq_kv"], shape["kv_heads"], shape["head_dim"]],
    }
    expected_dtype = str(case["precision"]["input_dtype"])
    tensors = []
    for name in ("q", "k", "v"):
        spec = specs[name]
        if not isinstance(spec, dict) or spec.get("shape") != expected_shapes[name] or spec.get("dtype") != expected_dtype:
            raise ValueError(f"sparse replay {name} contract does not match the case")
        shards = spec.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"sparse replay {name} requires shards")
        chunks = []
        cursor = 0
        for index, shard in enumerate(shards):
            start, end = shard.get("sequence_start"), shard.get("sequence_end")
            if type(start) is not int or type(end) is not int or start != cursor or end <= start:
                raise ValueError(f"sparse replay {name} shard coverage drift at {index}")
            shard_path = Path(str(shard.get("path")))
            if not shard_path.is_absolute():
                shard_path = manifest_path.parent / shard_path
            if not shard_path.is_file() or _file_sha256(shard_path) != shard.get("sha256"):
                raise ValueError(f"sparse replay {name} shard file mismatch at {index}: {shard_path}")
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            tensor = payload.get(shard.get("tensor_key")) if isinstance(payload, dict) else None
            expected_shard_shape = [expected_shapes[name][0], end - start, *expected_shapes[name][2:]]
            actual_dtype = str(getattr(tensor, "dtype", "")).removeprefix("torch.")
            dtype_alias = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(actual_dtype)
            if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != expected_shard_shape or dtype_alias != expected_dtype:
                raise ValueError(f"sparse replay {name} shard tensor mismatch at {index}")
            chunks.append(tensor)
            cursor = end
        if cursor != expected_shapes[name][1]:
            raise ValueError(f"sparse replay {name} shards do not cover the sequence")
        tensors.append(torch.cat(chunks, dim=1).squeeze(0).to(device=device).contiguous())
    metadata = {
        "input_kind": "real_qkv_replay",
        "replay_manifest": str(manifest_path),
        "replay_manifest_sha256": manifest_sha256,
        "replay_provenance": manifest.get("provenance") or {},
    }
    return tensors[0], tensors[1], tensors[2], metadata


def _sparse_replay_prepared(
    case: dict[str, Any],
    fn: Callable[[], Any],
    backend: str,
    metadata: dict[str, Any],
) -> PreparedOperation:
    shape = case["shape"]
    dense_work = 4 * int(shape["batch"]) * int(shape["heads"]) * int(shape["seq_q"]) * int(shape["seq_kv"]) * int(shape["head_dim"])
    block_density = 1.0 - float(metadata["block_sparsity_actual"])
    actual_work = round(dense_work * block_density)
    return PreparedOperation(
        fn=fn,
        work=actual_work,
        rate_metric="actual_tflops",
        work_definition="selected_block_qk_and_pv_flops_estimated_from_block_density; routing_flops_excluded",
        dense_equivalent_work=dense_work,
        dense_equivalent_work_definition="4*batch*heads*seq_q*seq_kv*head_dim",
        extra_metrics={
            "implementation": backend,
            "keep_ratio": float(case["sparse"]["keep_ratio"]),
            "block_density_actual": block_density,
            **metadata,
        },
    )


def _prepare_sparge_replay(case: dict[str, Any], device: str) -> PreparedOperation:
    import spas_sage_attn
    import torch

    from lightx2v.common.ops.attn.sparge_attn import SpargeAttnWeight

    q, k, v, metadata = _load_qkv_replay(case, device, torch)
    keep_ratio = float(case["sparse"]["keep_ratio"])
    operator = SpargeAttnWeight()
    operator.topk = keep_ratio

    def fn() -> Any:
        return operator.apply(q, k, v, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0])

    diagnostic_output, sparsity = spas_sage_attn.core.spas_sage2_attn_meansim_topk_cuda(
        q.unsqueeze(0).transpose(1, 2).contiguous(),
        k.unsqueeze(0).transpose(1, 2).contiguous(),
        v.unsqueeze(0).transpose(1, 2).contiguous(),
        topk=keep_ratio,
        return_sparsity=True,
    )
    del diagnostic_output
    metadata.update({"routing_scope": "included", "block_sparsity_actual": float(sparsity)})
    return _sparse_replay_prepared(case, fn, "sparge_sage2_replay", metadata)


def _prepare_dynamic_sparse_replay(
    case: dict[str, Any],
    device: str,
    operator_name: str,
    backend_name: str,
) -> PreparedOperation:
    import torch

    from lightx2v.common.ops.attn.dynamic_sparse_attn import DynamicSparseAttnWeight
    from lightx2v.common.ops.attn.utils.sla_util import get_block_map

    q, k, v, metadata = _load_qkv_replay(case, device, torch)
    keep_ratio = float(case["sparse"]["keep_ratio"])
    operator = DynamicSparseAttnWeight({"sparsity_ratio": 1.0 - keep_ratio, "operator": operator_name})

    def fn() -> Any:
        return operator.apply(q, k, v, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0])

    q_hnd = q.unsqueeze(0).transpose(1, 2).contiguous()
    k_hnd = k.unsqueeze(0).transpose(1, 2).contiguous()
    sparse_map, _lut, real_topk = get_block_map(
        q_hnd,
        k_hnd,
        topk_ratio=keep_ratio,
        BLKQ=operator.BLKQ,
        BLKK=operator.BLKK,
    )
    metadata.update(
        {
            "routing_scope": "included",
            "q_block_size": operator.BLKQ,
            "k_block_size": operator.BLKK,
            "real_topk_blocks_per_row": int(real_topk),
            "block_sparsity_actual": 1.0 - float(sparse_map.float().mean().item()),
        }
    )
    return _sparse_replay_prepared(case, fn, backend_name, metadata)


def _prepare_production_sparse_replay(
    case: dict[str, Any],
    device: str,
    class_name: str,
    backend_name: str,
) -> PreparedOperation:
    import torch

    from lightx2v.common.ops.attn import flash_attn, sage_attn
    from lightx2v.common.ops.attn.utils.sla_util import get_block_map

    classes = {
        "SparseFlashAttn4Weight": flash_attn.SparseFlashAttn4Weight,
        "SparseSageAttn2Weight": sage_attn.SparseSageAttn2Weight,
        "SparseSageAttn3Weight": sage_attn.SparseSageAttn3Weight,
    }
    q, k, v, metadata = _load_qkv_replay(case, device, torch)
    keep_ratio = float(case["sparse"]["keep_ratio"])
    operator = classes[class_name]()
    operator.topk = keep_ratio

    def fn() -> Any:
        return operator.apply(q, k, v, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0])

    q_hnd = q.unsqueeze(0).transpose(1, 2).contiguous()
    k_hnd = k.unsqueeze(0).transpose(1, 2).contiguous()
    sparse_map, _lut, real_topk = get_block_map(
        q_hnd,
        k_hnd,
        topk_ratio=keep_ratio,
        BLKQ=operator.BLKQ,
        BLKK=operator.BLKK,
    )
    metadata.update(
        {
            "routing_scope": "included",
            "q_block_size": operator.BLKQ,
            "k_block_size": operator.BLKK,
            "real_topk_blocks_per_row": int(real_topk),
            "block_sparsity_actual": 1.0 - float(sparse_map.float().mean().item()),
        }
    )
    return _sparse_replay_prepared(case, fn, backend_name, metadata)


def _prepare_flash3_replay(case: dict[str, Any], device: str) -> PreparedOperation:
    import torch

    from lightx2v.common.ops.attn.flash_attn import FlashAttn3Weight, flash_attn_func_v3

    if flash_attn_func_v3 is None:
        raise RuntimeError("FlashAttention3 is unavailable")
    q, k, v, metadata = _load_qkv_replay(case, device, torch)
    operator = FlashAttn3Weight()

    def fn() -> Any:
        return operator.apply(q, k, v, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0], causal=bool(case["shape"]["causal"]))

    metadata.update({"routing_scope": "dense_baseline", "block_sparsity_actual": 0.0})
    return _sparse_replay_prepared(case, fn, "flash_attn3_replay", metadata)


def _prepare_production_dense_replay(
    case: dict[str, Any],
    device: str,
    class_name: str,
    backend_name: str,
) -> PreparedOperation:
    import torch

    from lightx2v.common.ops.attn import flash_attn, sage_attn

    classes = {
        "FlashAttn4Weight": flash_attn.FlashAttn4Weight,
        "SageAttn3Weight": sage_attn.SageAttn3Weight,
    }
    q, k, v, metadata = _load_qkv_replay(case, device, torch)
    operator = classes[class_name]()

    def fn() -> Any:
        return operator.apply(q, k, v, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0], causal=bool(case["shape"]["causal"]))

    metadata.update({"routing_scope": "dense_baseline", "block_sparsity_actual": 0.0})
    return _sparse_replay_prepared(case, fn, backend_name, metadata)


def _prepare_flash2(case: dict[str, Any], device: str) -> PreparedOperation:
    import torch
    from flash_attn import flash_attn_func

    q, k, v = _attention_tensors(case, device, torch)

    def fn() -> Any:
        return flash_attn_func(q, k, v, causal=bool(case["shape"]["causal"]))

    return _attention_prepared(case, fn, "flash_attn2")


def _prepare_flash3(case: dict[str, Any], device: str) -> PreparedOperation:
    import torch
    from flash_attn_interface import flash_attn_func

    q, k, v = _attention_tensors(case, device, torch)

    def fn() -> Any:
        return flash_attn_func(q, k, v, causal=bool(case["shape"]["causal"]))

    return _attention_prepared(case, fn, "flash_attn3")


def _prepare_sage2(case: dict[str, Any], device: str) -> PreparedOperation:
    import torch
    from sageattention import sageattn

    q, k, v = _attention_tensors(case, device, torch)

    def fn() -> Any:
        return sageattn(q, k, v, tensor_layout="NHD", is_causal=bool(case["shape"]["causal"]))

    return _attention_prepared(case, fn, "sage_attn2")


def _prepare_production_dense(
    case: dict[str, Any],
    device: str,
    class_name: str,
    backend_name: str,
) -> PreparedOperation:
    import torch

    from lightx2v.common.ops.attn import flash_attn, sage_attn

    classes = {
        "FlashAttn4Weight": flash_attn.FlashAttn4Weight,
        "SageAttn3Weight": sage_attn.SageAttn3Weight,
    }
    q, k, v = _attention_tensors(case, device, torch)
    operator = classes[class_name]()

    def fn() -> Any:
        return operator.apply(q, k, v, causal=bool(case["shape"]["causal"]))

    return _attention_prepared(case, fn, backend_name)


class DenseAttentionAdapter:
    def __init__(
        self,
        name: str,
        plugin: str,
        dependencies: tuple[str, ...],
        prepare_fn: Callable[..., PreparedOperation],
        *,
        production_backend: str,
        cuda_capabilities: tuple[str, ...] = (),
        probe_symbols: tuple[str, ...] = (),
    ) -> None:
        self.descriptor = BackendDescriptor(
            name=name,
            family="dense_attention",
            plugin=plugin,
            input_dtypes=("bf16", "fp16"),
            description=f"Direct dense attention through {plugin}",
            optional_dependencies=dependencies,
            production_backend=production_backend,
            cuda_capabilities=cuda_capabilities,
            probe_symbols=probe_symbols,
        )
        self.prepare_fn = prepare_fn

    def support_error(self, case: dict[str, Any]) -> str | None:
        return _attention_support_error(case, self.descriptor.name)

    def precision(self, case: dict[str, Any]) -> dict[str, Any]:
        dtype = str(case["precision"]["input_dtype"])
        return {"input_dtype": dtype, "weight_dtype": None, "accum_dtype": "backend_default", "output_dtype": dtype, "quant": None}

    def prepare(self, case: dict[str, Any], device: str, options: dict[str, Any]) -> PreparedOperation:
        return self.prepare_fn(case, device)


class SparseReplayAttentionAdapter:
    def __init__(
        self,
        name: str,
        plugin: str,
        dependencies: tuple[str, ...],
        prepare_fn: Callable[..., PreparedOperation],
        *,
        production_backend: str,
        production_variant: str | None = None,
        cuda_capabilities: tuple[str, ...] = (),
        probe_symbols: tuple[str, ...] = (),
    ) -> None:
        self.descriptor = BackendDescriptor(
            name=name,
            family="sparse_attention",
            plugin=plugin,
            input_dtypes=("bf16", "fp16"),
            description=f"Real-QKV sparse attention replay through {plugin}",
            optional_dependencies=dependencies,
            production_backend=production_backend,
            production_variant=production_variant,
            cuda_capabilities=cuda_capabilities,
            probe_symbols=probe_symbols,
        )
        self.prepare_fn = prepare_fn

    def support_error(self, case: dict[str, Any]) -> str | None:
        dtype = str(case["precision"]["input_dtype"])
        if dtype not in self.descriptor.input_dtypes:
            return f"{self.descriptor.name} does not support dtype {dtype}"
        if self.descriptor.name != "flash_attn3_replay" and case["shape"]["causal"]:
            return f"{self.descriptor.name} does not provide causal replay"
        if self.descriptor.name not in {"flash_attn3_replay", "flash_attn4_replay"} and int(case["shape"]["head_dim"]) not in {64, 128}:
            return f"{self.descriptor.name} requires head_dim 64 or 128"
        return None

    def precision(self, case: dict[str, Any]) -> dict[str, Any]:
        dtype = str(case["precision"]["input_dtype"])
        return {"input_dtype": dtype, "weight_dtype": None, "accum_dtype": "backend_default", "output_dtype": dtype, "quant": None}

    def prepare(self, case: dict[str, Any], device: str, options: dict[str, Any]) -> PreparedOperation:
        return self.prepare_fn(case, device)


class TorchGroupedMMAdapter:
    descriptor = BackendDescriptor(
        name="torch_grouped_mm",
        family="moe",
        plugin="torch._grouped_mm",
        input_dtypes=("bf16", "fp16"),
        description="single-GPU routed MoE using torch._grouped_mm",
    )

    def support_error(self, case: dict[str, Any]) -> str | None:
        dtype = str(case["precision"]["input_dtype"])
        return None if dtype in self.descriptor.input_dtypes else f"torch_grouped_mm does not support {dtype}"

    def precision(self, case: dict[str, Any]) -> dict[str, Any]:
        return _torch_precision(case)

    def prepare(self, case: dict[str, Any], device: str, options: dict[str, Any]) -> PreparedOperation:
        import torch

        if not hasattr(torch, "_grouped_mm"):
            raise RuntimeError("this PyTorch build does not provide torch._grouped_mm")
        shape = case["shape"]
        dtype = _dtype(torch, str(case["precision"]["input_dtype"]))
        tokens, hidden = int(shape["tokens"]), int(shape["hidden_size"])
        intermediate, experts, top_k = int(shape["intermediate_size"]), int(shape["num_experts"]), int(shape["top_k"])
        fc1_size = intermediate if shape["activation"] == "gelu" else 2 * intermediate
        inputs = torch.randn((tokens, hidden), device=device, dtype=dtype)
        fc1 = torch.randn((experts, fc1_size, hidden), device=device, dtype=dtype).contiguous()
        fc2 = torch.randn((experts, hidden, intermediate), device=device, dtype=dtype).contiguous()
        if shape["expert_bias"]:
            fc1_bias = torch.randn((experts, fc1_size), device=device, dtype=dtype)
            fc2_bias = torch.randn((experts, hidden), device=device, dtype=dtype)
        else:
            fc1_bias = fc2_bias = None
        selected, scales, metrics = _routing(case, device, str(options.get("moe_routing") or "balanced"), torch)
        output = torch.empty_like(inputs)

        def fn() -> Any:
            flat = selected.reshape(-1)
            counts = torch.bincount(flat, minlength=experts)
            order = torch.argsort(flat)
            token_indexes = torch.div(order, top_k, rounding_mode="floor")
            expert_indexes = flat.index_select(0, order)
            offsets = counts.cumsum(0, dtype=torch.int32)
            projected = torch._grouped_mm(inputs.index_select(0, token_indexes), fc1.transpose(1, 2), offs=offsets)
            if fc1_bias is not None:
                projected.add_(fc1_bias.index_select(0, expert_indexes))
            if shape["activation"] == "gelu":
                projected = torch.nn.functional.gelu(projected)
            else:
                value, gate = projected.chunk(2, dim=-1)
                projected = value * torch.nn.functional.silu(gate)
            expert_output = torch._grouped_mm(projected, fc2.transpose(1, 2), offs=offsets)
            if fc2_bias is not None:
                expert_output.add_(fc2_bias.index_select(0, expert_indexes))
            expanded = torch.empty_like(expert_output)
            expanded.index_copy_(0, order, expert_output)
            routed = expanded.view(tokens, top_k, hidden)
            output.copy_(torch.bmm(scales.unsqueeze(1), routed.float()).squeeze(1))
            return output

        fc1_factor = 2 if shape["activation"] == "swiglu" else 1
        return PreparedOperation(
            fn=fn,
            work=tokens * top_k * 2 * hidden * intermediate * (fc1_factor + 1),
            rate_metric="effective_tflops",
            work_definition="routed_fc1_and_fc2_matmul_only",
            extra_metrics=metrics,
        )


def _probe_symbol(value: str) -> str | None:
    module_name, separator, attribute = value.partition(":")
    if not separator:
        return f"invalid probe symbol: {value}"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"{value}: {type(exc).__name__}: {exc}"
    if getattr(module, attribute, None) is None:
        return f"{value}: unavailable"
    return None


def _probe_dependency(value: str) -> str | None:
    try:
        available = importlib.util.find_spec(value) is not None
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        return f"{value}: {type(exc).__name__}: {exc}"
    return None if available else f"{value}: unavailable"


def build_backend_catalog_report(
    registry: BackendRegistry,
    *,
    cuda_capability: str | None,
    production_backends: Iterable[str],
    production_mm_backends: Iterable[str] = (),
    symbol_probe: Callable[[str], str | None] = _probe_symbol,
    dependency_probe: Callable[[str], str | None] = _probe_dependency,
) -> dict[str, Any]:
    production = sorted(set(production_backends))
    production_mm = sorted(set(production_mm_backends))
    descriptors = [
        item
        for item in registry.descriptors()
        if item["family"] in {"gemm", "dense_attention", "sparse_attention", "moe"}
    ]
    mapped_attention = {
        item["production_backend"]
        for item in descriptors
        if item["family"] in {"dense_attention", "sparse_attention"} and item["production_backend"]
    }
    mapped_mm = {
        item["production_backend"]
        for item in descriptors
        if item["family"] == "gemm" and item["production_backend"]
    }
    attention_exclusions = {name: reason for name, reason in PRODUCTION_ATTN_EXCLUSIONS.items() if name in production}
    mm_exclusions = {name: reason for name, reason in PRODUCTION_MM_EXCLUSIONS.items() if name in production_mm}
    unmapped_attention = sorted(set(production) - mapped_attention - set(attention_exclusions))
    unmapped_mm = sorted(set(production_mm) - mapped_mm - set(mm_exclusions))
    exclusions = {**attention_exclusions, **mm_exclusions}
    unmapped = sorted((*unmapped_attention, *unmapped_mm))
    candidates = []
    for descriptor in descriptors:
        capabilities = descriptor["cuda_capabilities"]
        capability_supported = cuda_capability in capabilities or any(
            value.endswith(".x")
            and cuda_capability is not None
            and cuda_capability.split(".", 1)[0] == value[:-2]
            for value in capabilities
        )
        if cuda_capability is None:
            status, reasons = "cuda_unavailable", ["CUDA capability is unavailable"]
        elif capabilities and not capability_supported:
            status = "unsupported_arch"
            reasons = [f"requires CUDA capability in {capabilities}, got {cuda_capability}"]
        else:
            reasons = [reason for dependency in descriptor["optional_dependencies"] if (reason := dependency_probe(dependency))]
            reasons.extend(reason for symbol in descriptor["probe_symbols"] if (reason := symbol_probe(symbol)))
            status = "dependency_missing" if reasons else "eligible"
        candidates.append({**descriptor, "status": status, "reasons": reasons})
    catalog_complete = not unmapped
    environment_complete = bool(cuda_capability) and not any(item["status"] in {"cuda_unavailable", "dependency_missing"} for item in candidates)
    fingerprint_payload = {
        "cuda_capability": cuda_capability,
        "production_backends": production,
        "production_mm_backends": production_mm,
        "candidates": [
            {
                "name": item["name"],
                "family": item["family"],
                "plugin": item["plugin"],
                "production_backend": item["production_backend"],
                "production_variant": item["production_variant"],
                "optional_dependencies": item["optional_dependencies"],
                "cuda_capabilities": item["cuda_capabilities"],
                "probe_symbols": item["probe_symbols"],
                "status": item["status"],
            }
            for item in candidates
        ],
        "exclusions": exclusions,
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "kind": "operator_benchmark_backend_catalog_v1",
        "cuda_capability": cuda_capability,
        "catalog_fingerprint": fingerprint,
        "catalog_complete": catalog_complete,
        "environment_complete": environment_complete,
        "coverage_complete": catalog_complete and environment_complete,
        "production_backends": production,
        "production_mm_backends": production_mm,
        "excluded_production_backends": exclusions,
        "excluded_attention_backends": attention_exclusions,
        "excluded_mm_backends": mm_exclusions,
        "unmapped_production_backends": unmapped,
        "unmapped_attention_backends": unmapped_attention,
        "unmapped_mm_backends": unmapped_mm,
        "candidates": candidates,
    }


def probe_backend_catalog(registry: BackendRegistry, device: str) -> dict[str, Any]:
    import torch

    discovery_output = io.StringIO()
    with redirect_stdout(discovery_output):
        import lightx2v.common.ops.attn  # noqa: F401
        from lightx2v.common.ops.mm import mm_weight as _mm_weight  # noqa: F401
        from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER

    capability = None
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(torch.device(device))
        capability = f"{major}.{minor}"
    report = build_backend_catalog_report(
        registry,
        cuda_capability=capability,
        production_backends=ATTN_WEIGHT_REGISTER.keys(),
        production_mm_backends=MM_WEIGHT_REGISTER.keys(),
    )
    report["discovery_warnings"] = [line for line in discovery_output.getvalue().splitlines() if line.strip()]
    return report


@dataclass(frozen=True)
class AttentionBackendContract:
    name: str
    family: str
    plugin: str
    dependencies: tuple[str, ...]
    prepare: Callable[..., PreparedOperation]
    production_backend: str
    production_variant: str | None = None
    cuda_capabilities: tuple[str, ...] = ()
    probe_symbols: tuple[str, ...] = ()


BLACKWELL_CAPABILITIES = ("10.x", "11.x", "12.x")
ATTENTION_BACKEND_CONTRACTS = (
    AttentionBackendContract(
        "flash_attn2", "dense_attention", "flash_attn", ("flash_attn",), _prepare_flash2, "flash_attn2", probe_symbols=("lightx2v.common.ops.attn.flash_attn:flash_attn_func_v2",)
    ),
    AttentionBackendContract(
        "flash_attn3",
        "dense_attention",
        "flash_attn_interface",
        ("flash_attn_interface",),
        _prepare_flash3,
        "flash_attn3",
        cuda_capabilities=("9.0",),
        probe_symbols=("lightx2v.common.ops.attn.flash_attn:flash_attn_func_v3",),
    ),
    AttentionBackendContract(
        "flash_attn4",
        "dense_attention",
        "flash_attn.cute",
        ("flash_attn.cute",),
        partial(_prepare_production_dense, class_name="FlashAttn4Weight", backend_name="flash_attn4"),
        "flash_attn4",
        cuda_capabilities=BLACKWELL_CAPABILITIES,
        probe_symbols=("lightx2v.common.ops.attn.flash_attn:flash_attn_func_v4",),
    ),
    AttentionBackendContract("sage_attn2", "dense_attention", "sageattention", ("sageattention",), _prepare_sage2, "sage_attn2", probe_symbols=("lightx2v.common.ops.attn.sage_attn:sageattn",)),
    AttentionBackendContract(
        "sage_attn3",
        "dense_attention",
        "sageattn3",
        ("sageattn3",),
        partial(_prepare_production_dense, class_name="SageAttn3Weight", backend_name="sage_attn3"),
        "sage_attn3",
        cuda_capabilities=BLACKWELL_CAPABILITIES,
        probe_symbols=("lightx2v.common.ops.attn.sage_attn:sageattn3_blackwell",),
    ),
    AttentionBackendContract(
        "sparge_sage2_replay", "sparse_attention", "spas_sage_attn", ("spas_sage_attn",), _prepare_sparge_replay, "sparge_attn", probe_symbols=("lightx2v.common.ops.attn.sparge_attn:spas_sage_attn",)
    ),
    AttentionBackendContract(
        "dynamic_sparse_triton_replay",
        "sparse_attention",
        "lightx2v_dynamic_sparse_triton",
        ("lightx2v", "triton"),
        partial(_prepare_dynamic_sparse_replay, operator_name="triton", backend_name="dynamic_sparse_triton_replay"),
        "dynamic_sparse_attn",
        "operator=triton",
    ),
    AttentionBackendContract(
        "dynamic_sparse_sage2_replay",
        "sparse_attention",
        "lightx2v_dynamic_sparse_sage2",
        ("lightx2v", "spas_sage_attn"),
        partial(_prepare_dynamic_sparse_replay, operator_name="sage2", backend_name="dynamic_sparse_sage2_replay"),
        "dynamic_sparse_attn",
        "operator=sage2",
        probe_symbols=("lightx2v.common.ops.attn.dynamic_sparse_attn:sage2_block_sparse_attn",),
    ),
    AttentionBackendContract(
        "dynamic_sparse_sage3_replay",
        "sparse_attention",
        "lightx2v_dynamic_sparse_sage3",
        ("lightx2v", "sageattn3_sparse"),
        partial(_prepare_dynamic_sparse_replay, operator_name="sage3", backend_name="dynamic_sparse_sage3_replay"),
        "dynamic_sparse_attn",
        "operator=sage3",
        BLACKWELL_CAPABILITIES,
        ("lightx2v.common.ops.attn.dynamic_sparse_attn:sage3_block_sparse_attn",),
    ),
    AttentionBackendContract(
        "dynamic_sparse_fa4_replay",
        "sparse_attention",
        "lightx2v_dynamic_sparse_fa4",
        ("lightx2v", "flash_attn.cute"),
        partial(_prepare_dynamic_sparse_replay, operator_name="fa4", backend_name="dynamic_sparse_fa4_replay"),
        "dynamic_sparse_attn",
        "operator=fa4",
        BLACKWELL_CAPABILITIES,
        ("lightx2v.common.ops.attn.dynamic_sparse_attn:flash_attn_func_v4",),
    ),
    AttentionBackendContract(
        "spas_sage2_replay",
        "sparse_attention",
        "lightx2v_spas_sage2",
        ("lightx2v", "spas_sage_attn"),
        partial(_prepare_production_sparse_replay, class_name="SparseSageAttn2Weight", backend_name="spas_sage2_replay"),
        "spas_sage_attn2",
        probe_symbols=("lightx2v.common.ops.attn.sage_attn:sage2_block_sparse_attn",),
    ),
    AttentionBackendContract(
        "spas_sage3_replay",
        "sparse_attention",
        "lightx2v_spas_sage3",
        ("lightx2v", "sageattn3_sparse"),
        partial(_prepare_production_sparse_replay, class_name="SparseSageAttn3Weight", backend_name="spas_sage3_replay"),
        "spas_sage_attn3",
        cuda_capabilities=BLACKWELL_CAPABILITIES,
        probe_symbols=("lightx2v.common.ops.attn.sage_attn:sage3_block_sparse_attn",),
    ),
    AttentionBackendContract(
        "spas_fa4_replay",
        "sparse_attention",
        "lightx2v_spas_fa4",
        ("lightx2v", "flash_attn.cute"),
        partial(_prepare_production_sparse_replay, class_name="SparseFlashAttn4Weight", backend_name="spas_fa4_replay"),
        "spas_flash_attn4",
        cuda_capabilities=BLACKWELL_CAPABILITIES,
        probe_symbols=("lightx2v.common.ops.attn.flash_attn:flash_attn_func_v4",),
    ),
    AttentionBackendContract(
        "flash_attn3_replay",
        "sparse_attention",
        "flash_attn_interface",
        ("flash_attn_interface",),
        _prepare_flash3_replay,
        "flash_attn3",
        "dense_replay_baseline",
        ("9.0",),
        ("lightx2v.common.ops.attn.flash_attn:flash_attn_func_v3",),
    ),
    AttentionBackendContract(
        "flash_attn4_replay",
        "sparse_attention",
        "flash_attn.cute",
        ("flash_attn.cute",),
        partial(_prepare_production_dense_replay, class_name="FlashAttn4Weight", backend_name="flash_attn4_replay"),
        "flash_attn4",
        "dense_replay_baseline",
        BLACKWELL_CAPABILITIES,
        ("lightx2v.common.ops.attn.flash_attn:flash_attn_func_v4",),
    ),
    AttentionBackendContract(
        "sage_attn3_replay",
        "sparse_attention",
        "sageattn3",
        ("sageattn3",),
        partial(_prepare_production_dense_replay, class_name="SageAttn3Weight", backend_name="sage_attn3_replay"),
        "sage_attn3",
        "dense_replay_baseline",
        BLACKWELL_CAPABILITIES,
        ("lightx2v.common.ops.attn.sage_attn:sageattn3_blackwell",),
    ),
)


def _attention_adapter(contract: AttentionBackendContract) -> BackendAdapter:
    kwargs = {
        "production_backend": contract.production_backend,
        "cuda_capabilities": contract.cuda_capabilities,
        "probe_symbols": contract.probe_symbols,
    }
    if contract.family == "dense_attention":
        return DenseAttentionAdapter(
            contract.name,
            contract.plugin,
            contract.dependencies,
            contract.prepare,
            **kwargs,
        )
    return SparseReplayAttentionAdapter(
        contract.name,
        contract.plugin,
        contract.dependencies,
        contract.prepare,
        production_variant=contract.production_variant,
        **kwargs,
    )


def load_registry(plugin_modules: Iterable[str] = ()) -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(TorchLinearAdapter())
    registry.register(TorchSDPAAdapter())
    registry.register(TorchExpertLoopAdapter())
    for contract in MM_REGISTRY_CONTRACTS:
        registry.register(LightX2VMMAdapter(contract))
    for contract in ATTENTION_BACKEND_CONTRACTS:
        registry.register(_attention_adapter(contract))
    registry.register(TorchGroupedMMAdapter())
    for module_name in plugin_modules:
        module = importlib.import_module(module_name)
        register = getattr(module, "register_backends", None)
        if not callable(register):
            raise ValueError(f"backend plugin must define register_backends(registry): {module_name}")
        register(registry)
    return registry
