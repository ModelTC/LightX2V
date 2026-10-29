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

from tools.compute_stats import attention_flops, convolution_flops, gemm_flops, routed_matmul_flops


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
    expected_output_shape: tuple[int, ...] | None = None


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

# Classification hints for entries without adapters; descriptors take precedence.
ATTENTION_SCOPE_HINTS = {
    "dense_attention": ("torch_sdpa_mps", "sage_attn2_k_int8_v_fp8", "flash_attn3_paged"),
    "sparse_attention": (
        "draft_attn",
        "flex_attn",
        "nbhd_attn",
        "nbhd_attn_flashinfer",
        "radial_attn",
        "rainfusion_attn",
        "svg_attn",
        "svg2_attn",
        "sol_attn",
    ),
    "sp_attention": ("ring", "ulysses", "ulysses-4090"),
}

PRODUCTION_MM_EXCLUSIONS = {
    "Calib": "calibration wrapper rather than an inference GEMM backend",
    "CalibMax": "calibration wrapper rather than an inference GEMM backend",
    "Default-ForceFp32": "FP32-sensitive model path requires a separate input precision contract",
    "TensorParallel": "distributed tensor-parallel wrapper is outside the single-card scope",
    "fp8-b128-deepgemm": "block-scaled weight layout requires a separate synthetic weight contract",
    "fp8-f16-accum": "requires a model-provided activation qmax before the FP16-accumulation path is enabled",
    "fp8-f32-accum": "requires a model-provided activation qmax and matching checkpoint weight/scale; use a model-specific adapter to enable the FP32-accumulation path",
    "fp8-pertensor": "checkpoint scale granularity requires a separate synthetic weight contract",
    "int4-g128-marlin": "groupwise INT4 packing requires a separate synthetic weight contract",
    "int8-convrot": "ConvRot requires checkpoint-provided rotation metadata",
    "int8-metal": "Apple MPS backend is outside this CUDA benchmark runner",
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
        weight = torch.randn((n, k), device=device, dtype=dtype)
        b = weight.t()
        if shape["bias"]:
            bias = torch.randn((n,), device=device, dtype=dtype)

            def fn() -> Any:
                return torch.addmm(bias, a, b)
        else:

            def fn() -> Any:
                return torch.mm(a, b)

        return PreparedOperation(fn=fn, work=gemm_flops(m, n, k), rate_metric="tflops", work_definition="2*m*n*k", expected_output_shape=(m, n))


class TorchConvolutionAdapter:
    descriptor = BackendDescriptor("torch_convolution", "convolution", "builtin_torch", TORCH_DTYPES, "torch conv2d/conv3d; same F.conv path as production Default")

    def support_error(self, case):
        return None if case["precision"]["input_dtype"] in self.descriptor.input_dtypes else "unsupported convolution dtype"

    def precision(self, case):
        return _torch_precision(case)

    def prepare(self, case, device, options):
        import torch
        import torch.nn.functional as functional

        shape = case["shape"]
        dtype = _dtype(torch, case["precision"]["input_dtype"])
        x = torch.randn(shape["input"], device=device, dtype=dtype)
        weight = torch.randn(shape["weight"], device=device, dtype=dtype)
        bias = torch.randn(shape["weight"][0], device=device, dtype=dtype) if shape["bias"] else None
        conv = functional.conv2d if x.ndim == 4 else functional.conv3d
        kwargs = {key: shape[key] for key in ("stride", "padding", "dilation", "groups")}

        def fn():
            return conv(x, weight, bias, **kwargs)

        output = fn()
        return PreparedOperation(
            fn=fn,
            work=convolution_flops(output.numel(), weight.shape),
            expected_output_shape=tuple(output.shape),
            rate_metric="tflops",
            work_definition="2*output_elements*weight_channels_per_group*kernel_volume; includes padded MACs; excludes bias",
        )


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

        q, k, v = _attention_tensors(case, device, torch)
        kwargs = {"enable_gqa": True} if q.shape[2] != k.shape[2] else {}

        def fn() -> Any:
            return functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=bool(case["shape"]["causal"]), **kwargs).transpose(1, 2)

        return _attention_prepared(case, fn, "torch_sdpa")


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


def _moe_operands(case, device, options, torch):
    """Use task-provided operands, or explicitly prepare a synthetic workload."""
    inputs = options.get("moe_inputs")
    if inputs is not None:
        if not case.get("replay"):
            raise ValueError("Fixed MoE inputs require a replay identity in the suite")
        counts = torch.bincount(inputs["expert_ids"].reshape(-1).long(), minlength=case["shape"]["num_experts"])
        metrics = {"expert_counts": counts.cpu().tolist(), "active_experts": int((counts > 0).sum()), "routing_distribution": "captured_inputs", "capture_provenance": case.get("source", {})}
        return inputs, metrics
    if case.get("replay"):
        raise ValueError("MoE replay requires a task-local plugin supplying options['moe_inputs']")
    shape = case["shape"]
    dtype = _dtype(torch, case["precision"]["input_dtype"])
    tokens, hidden = shape["tokens"], shape["hidden_size"]
    experts, intermediate = shape["num_experts"], shape["intermediate_size"]
    fc1_size = intermediate if shape["activation"] == "gelu" else 2 * intermediate
    x = torch.randn((tokens, hidden), device=device, dtype=dtype)
    fc1 = torch.randn((experts, fc1_size, hidden), device=device, dtype=dtype) * 0.02
    fc2 = torch.randn((experts, hidden, intermediate), device=device, dtype=dtype) * 0.02
    b1 = torch.randn((experts, fc1_size), device=device, dtype=dtype) * 0.02 if shape["expert_bias"] else None
    b2 = torch.randn((experts, hidden), device=device, dtype=dtype) * 0.02 if shape["expert_bias"] else None
    selected, scales, metrics = _routing(case, device, options.get("moe_routing") or "balanced", torch)
    return {"activation": x, "fc1": fc1, "fc2": fc2, "fc1_bias": b1, "fc2_bias": b2, "expert_ids": selected, "scales": scales}, metrics


def _moe_prepared(case, device, options, backend):
    """Prepare the production MoE on fixed operands; tune only before timing."""
    import torch

    from lightx2v.common.ops.moe.fused_moe import create_local_fused_moe

    inputs, metrics = _moe_operands(case, device, options, torch)
    activation = case["shape"]["activation"]
    fc1, gate = inputs["fc1"], inputs.get("fc1_gate")
    if backend == "flashinfer" and gate is not None:
        # FlashInfer生产factory接受packed SwiGLU：value在前，SiLU gate在后。
        fc1, gate = torch.cat((fc1, gate), dim=1), None
    op = create_local_fused_moe(backend, fc1, inputs["fc2"], activation, fc1_bias=inputs.get("fc1_bias"), fc2_bias=inputs.get("fc2_bias"), fc1_gate_weight=gate)
    x, expert_ids, scales = inputs["activation"], inputs["expert_ids"], inputs["scales"]

    def fn():
        return op.apply(x, expert_ids, scales)

    if backend == "flashinfer":
        from flashinfer import autotune

        # 调优和调用均固定当前bucket，避免库的共享配置被其他token测例改变。
        tokens = x.shape[0]
        op.tune_max_num_tokens = tokens
        with autotune(tuning_buckets=(tokens,)):
            op.apply(x, expert_ids, scales)

        def fn():
            with autotune(False, tuning_buckets=(tokens,)):
                return op.apply(x, expert_ids, scales)

        metrics["autotune"] = {"mode": "current_shape", "tune_max_num_tokens": tokens, "tuning_buckets": [tokens], "inference_tuning_buckets": [tokens]}
    return PreparedOperation(
        fn=fn,
        work=routed_matmul_flops(expert_ids.numel(), op.route_matmul_shapes()),
        rate_metric="effective_tflops",
        work_definition="actual_routes_fc1_fc2_gate_matmul_only",
        extra_metrics=metrics,
        expected_output_shape=tuple(x.shape),
    )


class TorchExpertLoopAdapter:
    descriptor = BackendDescriptor(
        name="torch_expert_loop",
        family="moe",
        plugin="builtin_torch",
        input_dtypes=TORCH_DTYPES,
        description="production per-expert Torch MoE",
        production_backend="torch_expert_loop",
    )

    def support_error(self, case: dict[str, Any]) -> str | None:
        dtype = str(case["precision"]["input_dtype"])
        return None if dtype in self.descriptor.input_dtypes else f"torch_expert_loop does not support dtype {dtype}"

    def precision(self, case: dict[str, Any]) -> dict[str, Any]:
        return _torch_precision(case)

    def prepare(self, case: dict[str, Any], device: str, options: dict[str, Any]) -> PreparedOperation:
        return _moe_prepared(case, device, options, "torch_expert_loop")


class FlashInferMoEAdapter:
    descriptor = BackendDescriptor(
        name="flashinfer",
        family="moe",
        plugin="lightx2v_fused_moe_registry",
        input_dtypes=("bf16", "fp16"),
        description="production FlashInfer CUTLASS MoE",
        optional_dependencies=("flashinfer",),
        production_backend="flashinfer",
        probe_symbols=("lightx2v.common.ops.moe.fused_moe:_flashinfer_cutlass_fused_moe", "lightx2v.common.ops.moe.fused_moe:_FlashInferActivationType"),
    )

    def support_error(self, case):
        dtype = case["precision"]["input_dtype"]
        return None if dtype in self.descriptor.input_dtypes else f"FlashInfer MoE requires BF16/FP16 input, got {dtype}"

    def precision(self, case):
        return _torch_precision(case)

    def prepare(self, case, device, options):
        return _moe_prepared(case, device, options, "flashinfer")


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
            work=gemm_flops(m, n, k),
            rate_metric=rate_metric,
            work_definition="2*m*n*k; LightX2V production wrapper apply",
            extra_metrics={"registry_key": self.contract.key},
            expected_output_shape=(m, n),
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
    if "replay" in case:
        q, k, v, _ = _load_qkv_replay(case, device, torch, keep_batch=True)
        return q, k, v
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
    batch, sequence, heads, dim = (shape[key] for key in ("batch", "seq_q", "heads", "head_dim"))
    output_shape = (batch * sequence, heads * dim) if backend in {"flash_attn4", "sage_attn3"} else (batch, sequence, heads, dim)
    work = attention_flops(shape["batch"] * shape["heads"] * shape["seq_q"] * shape["seq_kv"], shape["head_dim"], shape["head_dim"])
    return PreparedOperation(
        fn=fn,
        work=work,
        rate_metric="effective_tflops",
        work_definition="4*batch*heads*seq_q*seq_kv*head_dim",
        extra_metrics={"implementation": backend},
        expected_output_shape=output_shape,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_qkv_replay(case: dict[str, Any], device: str, torch: Any, *, keep_batch: bool = False) -> tuple[Any, Any, Any, dict[str, Any]]:
    replay = case["replay"]
    manifest_path = Path(replay["manifest"]).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"QKV replay manifest does not exist: {manifest_path}")
    manifest_sha256 = _file_sha256(manifest_path)
    if manifest_sha256 != replay["sha256"]:
        raise ValueError(f"QKV replay manifest sha256 mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "operator_benchmark_qkv_replay_v1":
        raise ValueError(f"unsupported QKV replay manifest: {manifest_path}")
    if manifest.get("layout") != "BSHD":
        raise ValueError("QKV replay layout must be BSHD")
    specs = manifest.get("tensors")
    if not isinstance(specs, dict) or set(specs) != {"q", "k", "v"}:
        raise ValueError("QKV replay manifest requires exactly q/k/v tensors")

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
            raise ValueError(f"QKV replay {name} contract does not match the case")
        shards = spec.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"QKV replay {name} requires shards")
        chunks = []
        cursor = 0
        for index, shard in enumerate(shards):
            start, end = shard.get("sequence_start"), shard.get("sequence_end")
            if type(start) is not int or type(end) is not int or start != cursor or end <= start:
                raise ValueError(f"QKV replay {name} shard coverage drift at {index}")
            shard_path = Path(str(shard.get("path")))
            if not shard_path.is_absolute():
                shard_path = manifest_path.parent / shard_path
            if not shard_path.is_file() or _file_sha256(shard_path) != shard.get("sha256"):
                raise ValueError(f"QKV replay {name} shard file mismatch at {index}: {shard_path}")
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            tensor = payload.get(shard.get("tensor_key")) if isinstance(payload, dict) else None
            expected_shard_shape = [expected_shapes[name][0], end - start, *expected_shapes[name][2:]]
            actual_dtype = str(getattr(tensor, "dtype", "")).removeprefix("torch.")
            dtype_alias = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(actual_dtype)
            if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != expected_shard_shape or dtype_alias != expected_dtype:
                raise ValueError(f"QKV replay {name} shard tensor mismatch at {index}")
            chunks.append(tensor)
            cursor = end
        if cursor != expected_shapes[name][1]:
            raise ValueError(f"QKV replay {name} shards do not cover the sequence")
        tensor = torch.cat(chunks, dim=1)
        tensors.append((tensor if keep_batch else tensor.squeeze(0)).to(device=device).contiguous())
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
    dense_work = attention_flops(int(shape["batch"]) * int(shape["heads"]) * int(shape["seq_q"]) * int(shape["seq_kv"]), int(shape["head_dim"]), int(shape["head_dim"]))
    # A separate diagnostic reads final production regions; never estimate from block density.
    return PreparedOperation(
        fn=fn,
        work=0,
        rate_metric="actual_tflops",
        work_definition="pending_actual_production_regions",
        dense_equivalent_work=dense_work,
        dense_equivalent_work_definition="4*batch*heads*seq_q*seq_kv*head_dim",
        expected_output_shape=(int(shape["seq_q"]), int(shape["heads"]) * int(shape["head_dim"])),
        extra_metrics={
            "implementation": backend,
            "keep_ratio": float(case["sparse"]["keep_ratio"]),
            "block_density_actual": None,
            **metadata,
            "causal": bool(shape["causal"]),
        },
    )


def _prepare_sparge_replay(case: dict[str, Any], device: str) -> PreparedOperation:
    import torch

    from lightx2v.common.ops.attn.sparge_attn import SpargeAttnWeight

    q, k, v, metadata = _load_qkv_replay(case, device, torch)
    keep_ratio = float(case["sparse"]["keep_ratio"])
    operator = SpargeAttnWeight()
    operator.topk = keep_ratio

    def fn() -> Any:
        return operator.apply(q, k, v, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0])

    metadata.update({"routing_scope": "included", "sparse_algorithm": "sparge"})
    return _sparse_replay_prepared(case, fn, "sparge.sage2.library", metadata)


def _prepare_dynamic_sparse_replay(
    case: dict[str, Any],
    device: str,
    operator_name: str,
    backend_name: str,
) -> PreparedOperation:
    import torch

    from lightx2v.common.ops.attn.dynamic_sparse_attn import DynamicSparseAttnWeight

    q, k, v, metadata = _load_qkv_replay(case, device, torch)
    keep_ratio = float(case["sparse"]["keep_ratio"])
    operator = DynamicSparseAttnWeight({"sparsity_ratio": 1.0 - keep_ratio, "operator": operator_name})

    def fn() -> Any:
        return operator.apply(q, k, v, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0])

    metadata.update(
        {
            "routing_scope": "included",
            "sparse_algorithm": "sla",
            "q_block_size": operator.BLKQ,
            "k_block_size": operator.BLKK,
        }
    )
    return _sparse_replay_prepared(case, fn, backend_name, metadata)


def _prepare_production_sparse_replay(
    case: dict[str, Any],
    device: str,
    class_name: str,
    backend_name: str,
    sparse_mode: str = "sla_mode",
) -> PreparedOperation:
    import torch

    from lightx2v.common.ops.attn import flash_attn, sage_attn

    classes = {
        "SparseFlashAttn4Weight": flash_attn.SparseFlashAttn4Weight,
        "SparseSageAttn2Weight": sage_attn.SparseSageAttn2Weight,
        "SparseSageAttn3Weight": sage_attn.SparseSageAttn3Weight,
    }
    q, k, v, metadata = _load_qkv_replay(case, device, torch)
    keep_ratio = float(case["sparse"]["keep_ratio"])
    operator = classes[class_name]()
    operator.topk = keep_ratio
    operator.sparse_mode = sparse_mode

    def fn() -> Any:
        return operator.apply(q, k, v, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0])

    metadata.update(
        {
            "routing_scope": "included",
            "sparse_algorithm": sparse_mode.removesuffix("_mode"),
            "q_block_size": operator.BLKQ,
            "k_block_size": operator.BLKK,
        }
    )
    return _sparse_replay_prepared(case, fn, backend_name, metadata)


def make_general_sparse_operator(method, operator_name, keep_ratio):
    from lightx2v.common.ops.attn.general_sparse_attn import GeneralSparseAttnWeight

    return GeneralSparseAttnWeight(
        sparse_mask_generator=f"{method}_mask_generator",
        sparse_operator=operator_name,
        sparse_setting={"sparsity_ratio": 1.0 - keep_ratio},
    )


def _prepare_general_sparse_replay(case, device, method, operator_name, backend_name):
    import torch

    q, k, v, metadata = _load_qkv_replay(case, device, torch)
    operator = make_general_sparse_operator(method, operator_name, float(case["sparse"]["keep_ratio"]))

    def fn():
        return operator.apply(q, k, v, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0])

    metadata.update(routing_scope="included", sparse_algorithm=method, q_block_size=operator.operator.q_block_size, k_block_size=operator.operator.k_block_size)
    return _sparse_replay_prepared(case, fn, backend_name, metadata)


def _prepare_flash2(case: dict[str, Any], device: str) -> PreparedOperation:
    import torch

    from lightx2v.common.ops.attn.flash_attn import FlashAttn2Weight

    q, k, v = _attention_tensors(case, device, torch)
    operator = FlashAttn2Weight()
    cu_q = torch.arange(q.shape[0] + 1, device=device, dtype=torch.int32) * q.shape[1]
    cu_kv = torch.arange(k.shape[0] + 1, device=device, dtype=torch.int32) * k.shape[1]

    def fn() -> Any:
        return operator.apply(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv, max_seqlen_q=q.shape[1], max_seqlen_kv=k.shape[1], causal=bool(case["shape"]["causal"])).reshape(q.shape)

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

    from lightx2v.common.ops.attn.sage_attn import sageattn

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
        # These prepare functions do not pass a causal mask to production.
        if case["shape"]["causal"]:
            return f"{self.descriptor.name} does not provide causal replay"
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
        production_backend="torch_grouped_mm",
        probe_symbols=("torch:_grouped_mm",),
    )

    def support_error(self, case: dict[str, Any]) -> str | None:
        dtype = str(case["precision"]["input_dtype"])
        return None if dtype in self.descriptor.input_dtypes else f"torch_grouped_mm does not support {dtype}"

    def precision(self, case: dict[str, Any]) -> dict[str, Any]:
        return _torch_precision(case)

    def prepare(self, case: dict[str, Any], device: str, options: dict[str, Any]) -> PreparedOperation:
        return _moe_prepared(case, device, options, "torch_grouped_mm")


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
    production_moe_backends: Iterable[str] = (),
    production_conv_backends: dict[str, Iterable[str]] | None = None,
    production_sparse_operators: Iterable[str] = (),
    production_mask_generators: Iterable[str] = (),
    symbol_probe: Callable[[str], str | None] = _probe_symbol,
    dependency_probe: Callable[[str], str | None] = _probe_dependency,
) -> dict[str, Any]:
    production = sorted(set(production_backends))
    production_mm = sorted(set(production_mm_backends))
    production_moe = sorted(set(production_moe_backends))
    production_sparse_operators = sorted(set(production_sparse_operators))
    production_mask_generators = sorted(set(production_mask_generators))
    descriptors = [item for item in registry.descriptors() if item["family"] in {"gemm", "dense_attention", "sparse_attention", "moe", "convolution"}]
    mapped_attention = {item["production_backend"] for item in descriptors if item["family"] in {"dense_attention", "sparse_attention"} and item["production_backend"]}
    mapped_mm = {item["production_backend"] for item in descriptors if item["family"] == "gemm" and item["production_backend"]}
    mm_exclusions = {name: reason for name, reason in PRODUCTION_MM_EXCLUSIONS.items() if name in production_mm and name not in mapped_mm}
    general_variants = {part for item in descriptors if item["production_backend"] == "general_sparse_attn" for part in (item["production_variant"] or "").split(";")}
    sparse_gaps = [f"general_sparse_attn/operator:{name}" for name in production_sparse_operators if f"operator={name}" not in general_variants]
    sparse_gaps += [f"general_sparse_attn/mask:{name}" for name in production_mask_generators if f"mask={name.removesuffix('_mask_generator')}" not in general_variants]
    unmapped_attention = sorted(set(production) - mapped_attention - set(ATTENTION_SCOPE_HINTS["sp_attention"])) + sparse_gaps
    attention_families = {name: family for family, names in ATTENTION_SCOPE_HINTS.items() for name in names}
    attention_families.update({item["production_backend"]: item["family"] for item in descriptors if item["family"] in {"dense_attention", "sparse_attention"} and item["production_backend"]})
    attention_families.update({name: "sparse_attention" for name in sparse_gaps})
    attention_scopes = {}
    for family in ("dense_attention", "sparse_attention", "sp_attention"):
        attention_scopes[family] = {
            "production_backends": [name for name in production if attention_families.get(name) == family],
            "excluded_production_backends": {},
            "unmapped_production_backends": [name for name in unmapped_attention if attention_families.get(name) in {None, family}] if family != "sp_attention" else [],
        }
    unmapped_mm = sorted(set(production_mm) - mapped_mm - set(mm_exclusions))
    exclusions = mm_exclusions
    mapped_moe = {item["production_backend"] for item in descriptors if item["family"] == "moe" and item["production_backend"]}
    unmapped_moe = sorted(set(production_moe) - mapped_moe)
    convolution_scopes = {}
    for subtype, names in (production_conv_backends or {}).items():
        names = sorted(set(names))
        mapped = {item["production_backend"]: item["name"] for item in descriptors if item["family"] == "convolution" and item["production_backend"]}
        if "Default" in names:
            mapped.setdefault("Default", "torch_convolution")
        excluded = (
            {"Default-ForceFp32": "forced FP32 conversion is outside the direct Torch adapter"} if subtype == "conv2d" and "Default-ForceFp32" in names and "Default-ForceFp32" not in mapped else {}
        )
        gaps = dict.fromkeys(
            sorted((set(names) | {"lightx2v_kernel.fp8_conv3d"}) - set(mapped) - set(excluded)), "production convolution has no benchmark adapter; inspect its input/precision contract"
        )
        convolution_scopes[subtype] = {
            "production_backends": names,
            "mapped_backends": mapped,
            "excluded_production_backends": excluded,
            "unmapped_production_backends": sorted(gaps),
            "integration_gaps": gaps,
        }
    unmapped = sorted((*unmapped_attention, *unmapped_mm, *unmapped_moe, *(f"{subtype}/{name}" for subtype, scope in convolution_scopes.items() for name in scope["unmapped_production_backends"])))
    candidates = []
    for descriptor in descriptors:
        capabilities = descriptor["cuda_capabilities"]
        capability_supported = cuda_capability in capabilities or any(value.endswith(".x") and cuda_capability is not None and cuda_capability.split(".", 1)[0] == value[:-2] for value in capabilities)
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
        "production_moe_backends": production_moe,
        "production_sparse_operators": sorted(production_sparse_operators),
        "production_mask_generators": sorted(production_mask_generators),
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
        "attention_scopes": attention_scopes,
        "convolution_scopes": convolution_scopes,
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
        "production_moe_backends": production_moe,
        "excluded_production_backends": exclusions,
        "production_sparse_operators": production_sparse_operators,
        "production_mask_generators": production_mask_generators,
        "excluded_attention_backends": {},
        "attention_scopes": attention_scopes,
        "convolution_scopes": convolution_scopes,
        "excluded_mm_backends": mm_exclusions,
        "unmapped_production_backends": unmapped,
        "unmapped_attention_backends": unmapped_attention,
        "unmapped_mm_backends": unmapped_mm,
        "unmapped_moe_backends": unmapped_moe,
        "attention_integration_gaps": dict.fromkeys(unmapped_attention, "production entry has no benchmark adapter; inspect its current input contract"),
        "candidates": candidates,
    }


def probe_backend_catalog(registry: BackendRegistry, device: str) -> dict[str, Any]:
    import torch

    discovery_output = io.StringIO()
    unresolved_imports = set()
    with redirect_stdout(discovery_output):
        import lightx2v.common.ops.attn  # noqa: F401

        for module in ("flex_attn", "paged_flash_attn"):
            try:
                importlib.import_module("lightx2v.common.ops.attn." + module)
            except ImportError as exc:
                unresolved_imports.add("unresolved_import:" + module)
                print(f"{module}: {type(exc).__name__}: {exc}")
        import lightx2v.common.ops.conv.conv2d  # noqa: F401
        import lightx2v.common.ops.conv.conv3d  # noqa: F401
        import lightx2v.common.ops.moe.fused_moe  # noqa: F401
        from lightx2v.common.ops.mm import mm_weight as _mm_weight  # noqa: F401
        from lightx2v.utils.registry_factory import (
            ATTN_WEIGHT_REGISTER,
            CONV2D_WEIGHT_REGISTER,
            CONV3D_WEIGHT_REGISTER,
            FUSED_MOE_REGISTER,
            MM_WEIGHT_REGISTER,
            SPARSE_MASK_GENERATOR_REGISTER,
            SPARSE_OPERATOR_REGISTER,
        )

    capability = None
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(torch.device(device))
        capability = f"{major}.{minor}"
    report = build_backend_catalog_report(
        registry,
        cuda_capability=capability,
        production_backends=ATTN_WEIGHT_REGISTER.keys() | unresolved_imports,
        production_mm_backends=MM_WEIGHT_REGISTER.keys(),
        production_moe_backends=FUSED_MOE_REGISTER.keys(),
        production_conv_backends={"conv2d": CONV2D_WEIGHT_REGISTER.keys(), "conv3d": CONV3D_WEIGHT_REGISTER.keys()},
        production_sparse_operators=SPARSE_OPERATOR_REGISTER.keys(),
        production_mask_generators=SPARSE_MASK_GENERATOR_REGISTER.keys(),
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
SAGE2_SPARSE_PROBES = ("lightx2v.common.ops.attn.utils.sparge_util:fused", "lightx2v.common.ops.attn.utils.sparge_util:qattn")
# GeneralSparse将mask策略与计算实现分离；配置只描述现有生产入口。
GENERAL_SPARSE_OPERATORS = (
    ("triton", "sla_triton_operator", ("triton",), (), ()),
    ("sage2", "spas_sage2_operator", ("spas_sage_attn",), (), ("lightx2v.common.ops.attn.utils.sparge_util:sage2_block_sparse_attn", *SAGE2_SPARSE_PROBES)),
    ("sage3", "spas_sage3_operator", ("sageattn3_sparse",), BLACKWELL_CAPABILITIES, ("lightx2v.common.ops.attn.sparse_operator:sage3_block_sparse_attn",)),
    ("fa4", "spas_fa4_operator", ("flash_attn.cute",), BLACKWELL_CAPABILITIES, ("lightx2v.common.ops.attn.sparse_operator:flash_attn_func_v4",)),
    ("magi", "magi_operator", ("magi_attention",), (), ("lightx2v.common.ops.attn.sparse_operator:magi_ffa_func",)),
    ("flex_block", "flex_block_operator", ("flex_block_attn",), (), ("lightx2v.common.ops.attn.sparse_operator:flex_block_attn_func",)),
    ("flashinfer", "flashinfer_operator", ("flashinfer",), (), ("flashinfer.sparse:VariableBlockSparseAttentionWrapper",)),
)

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
        "sparge.sage2.library", "sparse_attention", "spas_sage_attn", ("spas_sage_attn",), _prepare_sparge_replay, "sparge_attn", probe_symbols=("lightx2v.common.ops.attn.sparge_attn:spas_sage_attn",)
    ),
    AttentionBackendContract(
        "sla.triton.dynamic",
        "sparse_attention",
        "lightx2v_dynamic_sparse_triton",
        ("lightx2v", "triton"),
        partial(_prepare_dynamic_sparse_replay, operator_name="triton", backend_name="sla.triton.dynamic"),
        "dynamic_sparse_attn",
        "operator=triton",
    ),
    AttentionBackendContract(
        "sla.sage2.dynamic",
        "sparse_attention",
        "lightx2v_dynamic_sparse_sage2",
        ("lightx2v", "spas_sage_attn"),
        partial(_prepare_dynamic_sparse_replay, operator_name="sage2", backend_name="sla.sage2.dynamic"),
        "dynamic_sparse_attn",
        "operator=sage2",
        probe_symbols=("lightx2v.common.ops.attn.dynamic_sparse_attn:sage2_block_sparse_attn", *SAGE2_SPARSE_PROBES),
    ),
    AttentionBackendContract(
        "sla.sage3.dynamic",
        "sparse_attention",
        "lightx2v_dynamic_sparse_sage3",
        ("lightx2v", "sageattn3_sparse"),
        partial(_prepare_dynamic_sparse_replay, operator_name="sage3", backend_name="sla.sage3.dynamic"),
        "dynamic_sparse_attn",
        "operator=sage3",
        BLACKWELL_CAPABILITIES,
        ("lightx2v.common.ops.attn.dynamic_sparse_attn:sage3_block_sparse_attn",),
    ),
    AttentionBackendContract(
        "sla.fa4.dynamic",
        "sparse_attention",
        "lightx2v_dynamic_sparse_fa4",
        ("lightx2v", "flash_attn.cute"),
        partial(_prepare_dynamic_sparse_replay, operator_name="fa4", backend_name="sla.fa4.dynamic"),
        "dynamic_sparse_attn",
        "operator=fa4",
        BLACKWELL_CAPABILITIES,
        ("lightx2v.common.ops.attn.dynamic_sparse_attn:flash_attn_func_v4",),
    ),
    AttentionBackendContract(
        "sla.sage2.specialized",
        "sparse_attention",
        "lightx2v_spas_sage2",
        ("lightx2v", "spas_sage_attn"),
        partial(_prepare_production_sparse_replay, class_name="SparseSageAttn2Weight", backend_name="sla.sage2.specialized"),
        "spas_sage_attn2",
        probe_symbols=("lightx2v.common.ops.attn.utils.sparge_util:sage2_block_sparse_attn", *SAGE2_SPARSE_PROBES),
    ),
    AttentionBackendContract(
        "sla.sage3.specialized",
        "sparse_attention",
        "lightx2v_spas_sage3",
        ("lightx2v", "sageattn3_sparse"),
        partial(_prepare_production_sparse_replay, class_name="SparseSageAttn3Weight", backend_name="sla.sage3.specialized"),
        "spas_sage_attn3",
        cuda_capabilities=BLACKWELL_CAPABILITIES,
        probe_symbols=("lightx2v.common.ops.attn.sage_attn:sage3_block_sparse_attn",),
    ),
    AttentionBackendContract(
        "sla.fa4.specialized",
        "sparse_attention",
        "lightx2v_spas_fa4",
        ("lightx2v", "flash_attn.cute"),
        partial(_prepare_production_sparse_replay, class_name="SparseFlashAttn4Weight", backend_name="sla.fa4.specialized"),
        "spas_flash_attn4",
        cuda_capabilities=BLACKWELL_CAPABILITIES,
        probe_symbols=("lightx2v.common.ops.attn.flash_attn:flash_attn_func_v4",),
    ),
    AttentionBackendContract(
        "sparge.sage2.specialized",
        "sparse_attention",
        "lightx2v_sparge_sage2",
        ("lightx2v", "spas_sage_attn"),
        partial(_prepare_production_sparse_replay, class_name="SparseSageAttn2Weight", backend_name="sparge.sage2.specialized", sparse_mode="sparge_mode"),
        "spas_sage_attn2",
        "sparse_mode=sparge_mode",
        probe_symbols=("lightx2v.common.ops.attn.utils.sparge_util:sage2_block_sparse_attn", *SAGE2_SPARSE_PROBES),
    ),
    AttentionBackendContract(
        "sparge.sage3.specialized",
        "sparse_attention",
        "lightx2v_sparge_sage3",
        ("lightx2v", "sageattn3_sparse"),
        partial(_prepare_production_sparse_replay, class_name="SparseSageAttn3Weight", backend_name="sparge.sage3.specialized", sparse_mode="sparge_mode"),
        "spas_sage_attn3",
        "sparse_mode=sparge_mode",
        BLACKWELL_CAPABILITIES,
        ("lightx2v.common.ops.attn.sage_attn:sage3_block_sparse_attn",),
    ),
    AttentionBackendContract(
        "sparge.fa4.specialized",
        "sparse_attention",
        "lightx2v_sparge_fa4",
        ("lightx2v", "flash_attn.cute"),
        partial(_prepare_production_sparse_replay, class_name="SparseFlashAttn4Weight", backend_name="sparge.fa4.specialized", sparse_mode="sparge_mode"),
        "spas_flash_attn4",
        "sparse_mode=sparge_mode",
        BLACKWELL_CAPABILITIES,
        ("lightx2v.common.ops.attn.flash_attn:flash_attn_func_v4",),
    ),
)


ATTENTION_BACKEND_CONTRACTS += tuple(
    AttentionBackendContract(
        f"sla.{operator}.dynamic",
        "sparse_attention",
        "lightx2v_dynamic_sparse_" + operator,
        dependencies,
        partial(_prepare_dynamic_sparse_replay, operator_name=operator, backend_name=f"sla.{operator}.dynamic"),
        "dynamic_sparse_attn",
        "operator=" + operator,
        probe_symbols=symbols,
    )
    for operator, dependencies, symbols in (
        ("triton_ar", ("lightx2v", "triton"), ()),
        ("magi", ("lightx2v", "triton", "magi_attention"), ("lightx2v.common.ops.attn.dynamic_sparse_attn:magi_ffa_func",)),
    )
)
ATTENTION_BACKEND_CONTRACTS += tuple(
    AttentionBackendContract(
        f"{method}.{label}.general",
        "sparse_attention",
        f"lightx2v_general_{method}_{label}",
        ("lightx2v", "triton", *dependencies),
        partial(_prepare_general_sparse_replay, method=method, operator_name=operator, backend_name=f"{method}.{label}.general"),
        "general_sparse_attn",
        f"mask={method};operator={operator}",
        capabilities,
        symbols,
    )
    for method in ("sla", "sparge")
    for label, operator, dependencies, capabilities, symbols in GENERAL_SPARSE_OPERATORS
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
    registry.register(TorchConvolutionAdapter())
    registry.register(TorchSDPAAdapter())
    registry.register(TorchExpertLoopAdapter())
    registry.register(FlashInferMoEAdapter())
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
