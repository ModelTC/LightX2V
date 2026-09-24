"""Backend contracts, registry, and portable Torch baselines."""

from __future__ import annotations

import heapq
import importlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Protocol


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    family: str
    plugin: str
    input_dtypes: tuple[str, ...]
    description: str
    optional_dependencies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["input_dtypes"] = list(self.input_dtypes)
        value["optional_dependencies"] = list(self.optional_dependencies)
        return value


@dataclass
class PreparedOperation:
    fn: Callable[[], Any]
    work: int
    rate_metric: str
    work_definition: str
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


MM_REGISTRY_CONTRACTS = (
    MMRegistryContract("Default", "bf16", None, None, "torch", ("lightx2v",)),
    MMRegistryContract("fp8-vllm", "fp8_e4m3_per_channel", "fp8_e4m3", "n", "vllm", ("lightx2v", "vllm")),
    MMRegistryContract("int8-vllm", "int8_per_channel", "int8", "n", "vllm", ("lightx2v", "vllm")),
    MMRegistryContract("fp8-sgl", "fp8_e4m3_per_channel", "fp8_e4m3", "n", "sgl_kernel", ("lightx2v", "sgl_kernel")),
    MMRegistryContract("int8-sgl", "int8_per_channel", "int8", "n", "sgl_kernel", ("lightx2v", "sgl_kernel", "vllm")),
    MMRegistryContract("fp8-torchao", "fp8_e4m3_per_channel", "fp8_e4m3", "n_by_1", "torchao", ("lightx2v", "torchao")),
    MMRegistryContract("int8-torchao", "int8_per_channel", "int8", "n_by_1", "torchao", ("lightx2v", "torchao")),
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
    if backend == "sage_attn2" and head_dim not in {64, 128}:
        return f"sage_attn2 supports head_dim 64 or 128, got {head_dim}"
    if backend != "sage_attn2" and (head_dim > 256 or head_dim % 8):
        return f"{backend} requires head_dim <= 256 and divisible by 8"
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


class DenseAttentionAdapter:
    def __init__(self, name: str, plugin: str, dependencies: tuple[str, ...], prepare_fn: Callable[..., PreparedOperation]) -> None:
        self.descriptor = BackendDescriptor(
            name=name,
            family="dense_attention",
            plugin=plugin,
            input_dtypes=("bf16", "fp16"),
            description=f"Direct dense attention through {plugin}",
            optional_dependencies=dependencies,
        )
        self.prepare_fn = prepare_fn

    def support_error(self, case: dict[str, Any]) -> str | None:
        return _attention_support_error(case, self.descriptor.name)

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


def load_registry(plugin_modules: Iterable[str] = ()) -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(TorchLinearAdapter())
    registry.register(TorchSDPAAdapter())
    registry.register(TorchExpertLoopAdapter())
    for contract in MM_REGISTRY_CONTRACTS:
        registry.register(LightX2VMMAdapter(contract))
    registry.register(DenseAttentionAdapter("flash_attn2", "flash_attn", ("flash_attn",), _prepare_flash2))
    registry.register(DenseAttentionAdapter("flash_attn3", "flash_attn_interface", ("flash_attn_interface",), _prepare_flash3))
    registry.register(DenseAttentionAdapter("sage_attn2", "sageattention", ("sageattention",), _prepare_sage2))
    registry.register(TorchGroupedMMAdapter())
    for module_name in plugin_modules:
        module = importlib.import_module(module_name)
        register = getattr(module, "register_backends", None)
        if not callable(register):
            raise ValueError(f"backend plugin must define register_backends(registry): {module_name}")
        register(registry)
    return registry
