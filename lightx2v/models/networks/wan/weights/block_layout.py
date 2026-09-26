"""Loading contract for persistent Wan2.1 block storage."""

import re
from pathlib import Path

import torch
from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.offload.block_layout import BlockBuffer, BlockLayout, BlockLoadContext
from lightx2v.common.ops.mm.mm_weight import MMWeight, MMWeightWfp8channelAfp8channeldynamicVllm
from lightx2v.common.ops.norm.layer_norm_weight import LNWeightTemplate
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTemplate
from lightx2v.common.ops.tensor.tensor import DefaultTensor
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

NPU_WEIGHT_TYPES = ()
if AI_DEVICE == "npu":
    from lightx2v_platform.ops.mm.ascend_npu.mm_weight import MMWeightWint8channelAint8channeldynamicNpu
    from lightx2v_platform.ops.norm.ascend_npu.npu_layer_norm import NpuLayerNormWeight
    from lightx2v_platform.ops.norm.ascend_npu.npu_rms_norm import NpuRmsNormWeight

    NPU_WEIGHT_TYPES = (MMWeightWint8channelAint8channeldynamicNpu, NpuLayerNormWeight, NpuRmsNormWeight)


def validate_npu_checkpoint(path, scheme):
    """Reject incompatible matrix weights before the loader converts their dtype."""
    path = Path(path)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    expected = ("I8",) if scheme == "int8-npu" else ("BF16", "F16", "F32")
    names = set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if not re.fullmatch(r"blocks\.\d+\..+", name):
                    continue
                if name in names:
                    raise ValueError(f"Duplicate checkpoint weight: {name}")
                names.add(name)
                tensor = handle.get_slice(name)
                if name.endswith(".weight") and len(tensor.get_shape()) == 2 and tensor.get_dtype() not in expected:
                    raise ValueError(f"{scheme} requires {expected} matrix weights; {name} in {file} is {tensor.get_dtype()}. No implicit FP8 conversion is performed.")
    if not names:
        raise ValueError(f"No blocks.<index> weights found in {path}")


def validate_contiguous_config(config, lora_path=None):
    layout = config.get("cpu_offload_layout", "per_tensor")
    if layout not in ("per_tensor", "contiguous"):
        raise ValueError(f"Unsupported cpu_offload_layout: {layout!r}")
    if layout != "contiguous":
        return
    if AI_DEVICE not in ("cuda", "npu") or config.get("model_cls") != "wan2.1" or config.get("task") != "i2v":
        raise ValueError("contiguous layout supports Wan2.1 I2V on CUDA or NPU")
    scheme = config.get("dit_quant_scheme", "Default")
    supported = {"cuda": ("Default", "fp8-vllm"), "npu": ("Default", "int8-npu")}
    if scheme not in supported[AI_DEVICE] or bool(config.get("dit_quantized")) != (scheme != "Default"):
        raise ValueError(f"Unsupported contiguous precision on {AI_DEVICE}: {scheme}")
    if not config.get("cpu_offload") or config.get("offload_granularity") != "block":
        raise ValueError("contiguous layout requires CPU block offload")
    incompatible = (
        "shared_cpu_weights",
        "lazy_load",
        "parallel",
        "tensor_parallel",
        "seq_parallel",
        "cfg_parallel",
        "lora_configs",
        "lora_path",
        "lora_dynamic_apply",
        "use_compile",
        "enable_cuda_graph",
        "dummy_model",
        "do_mm_calib",
        "weight_auto_quant",
        "quant_method",
        "adapter_model_path",
    )
    if any(config.get(key) for key in incompatible) or lora_path or config.get("feature_caching", "NoCaching") != "NoCaching":
        raise ValueError("contiguous layout requires static rank-local weights, single-GPU eager inference, and NoCaching")
    if GET_SENSITIVE_DTYPE() != GET_DTYPE():
        raise ValueError("contiguous layout requires the default sensitive-layer dtype")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("contiguous layout currently supports a single rank")


def _bindings(block):
    weights, auxiliary = {}, {}
    for path, leaf in block.named_weight_leaves():
        if getattr(leaf, "is_post_adapter", False):
            raise ValueError("contiguous layout does not support adapter blocks")
        attrs = list(getattr(leaf, "base_attrs", ()))
        if isinstance(leaf, DefaultTensor):
            attrs.append((leaf.tensor_name, "tensor", False))
        if attrs and not isinstance(leaf, (MMWeight, MMWeightWfp8channelAfp8channeldynamicVllm, LNWeightTemplate, RMSWeightTemplate, DefaultTensor) + NPU_WEIGHT_TYPES):
            raise ValueError(f"Unsupported contiguous weight operator: {type(leaf).__name__}")
        for name, attr, transpose in attrs:
            weights[path, attr] = (leaf, name, transpose)
        for attr, name_attr in getattr(leaf, "lora_attrs", {}).items():
            tensor = getattr(leaf, attr, None)
            if tensor is not None:
                # Common RMSNorm creates device diff placeholders even without LoRA.
                if not isinstance(leaf, RMSWeightTemplate) or attr != "weight_diff" or tensor.device.type != AI_DEVICE:
                    raise ValueError(f"Unsupported contiguous auxiliary weight: {path}.{attr}")
                auxiliary[path, attr] = (leaf, getattr(leaf, name_attr))
    return weights, auxiliary


def load_contiguous_block(block, weight_dict):
    """Load final block views; WanModel validates the configuration once."""
    weights, auxiliary = _bindings(block)
    entries = []
    for key, (leaf, name, transpose) in weights.items():
        _, attr = key
        source = weight_dict[name]
        if source.device.type != "cpu":
            raise ValueError(f"Expected a CPU checkpoint tensor: {name}")
        dtype = source.dtype
        if isinstance(leaf, MMWeightWfp8channelAfp8channeldynamicVllm):
            if attr == "weight":
                if dtype != torch.float8_e4m3fn:
                    raise ValueError(f"Expected FP8 E4M3 weight: {name}")
                transpose = leaf.weight_need_transpose
            elif attr == "weight_scale":
                dtype = torch.float32 if leaf.scale_force_fp32 else dtype
            elif attr == "bias":
                dtype = torch.float32 if leaf.bias_force_fp32 else leaf.infer_dtype
        elif isinstance(leaf, NPU_WEIGHT_TYPES):
            dtype = leaf.contiguous_dtype(attr, dtype)
        elif isinstance(leaf, (MMWeight, LNWeightTemplate, RMSWeightTemplate)):
            # Floating-point operators keep the configured inference dtype.
            if dtype != GET_DTYPE():
                raise ValueError(f"Expected the inference dtype for weight: {name}")
        entries.append((key, name, tuple(source.shape), dtype, transpose))
    layout = BlockLayout.build(entries)
    expected_names = {spec.name for spec in layout.tensors}
    prefixes = {".".join(name.split(".")[:2]) + "." for name in expected_names}
    extra = {name for name in weight_dict if any(name.startswith(prefix) for prefix in prefixes)} - expected_names
    if extra:
        raise ValueError(f"Unmapped contiguous block weights: {sorted(extra)}")
    buffer = BlockBuffer(layout, AI_DEVICE if block.create_cuda_buffer else "cpu")
    context = BlockLoadContext(weight_dict, buffer)
    WeightModule.load(block, context)
    context.finish()

    # Operator post-processing must preserve the planned views.
    for spec in buffer.layout.tensors:
        leaf, _, _ = weights[spec.key]
        _, attr = spec.key
        buffer_attr = f"{attr}_cuda_buffer" if block.create_cuda_buffer else f"pin_{attr}"
        actual = getattr(leaf, buffer_attr)
        expected = spec.view(buffer.storage, operator=True)
        if actual.data_ptr() != expected.data_ptr() or actual.dtype != expected.dtype or actual.shape != expected.shape or actual.stride() != expected.stride():
            raise ValueError(f"Contiguous weight escaped its planned view: {spec.name}")
        if actual.device.type == "cpu" and not actual.is_pinned():
            raise ValueError(f"Contiguous weight is not pinned: {spec.name}")
        if block.create_cuda_buffer:
            setattr(leaf, attr, actual)
    # Include leaves that expose state outside base_attrs/tensor_name.
    state = block.state_dict()
    auxiliary_names = {name for _, name in auxiliary.values()}
    extra = {name for name, tensor in state.items() if tensor is not None} - expected_names - auxiliary_names
    if extra:
        raise ValueError(f"Undeclared contiguous block state: {sorted(extra)}")
    block.block_buffer = buffer
    block.block_auxiliary = {key: getattr(leaf, key[1]) for key, (leaf, _) in auxiliary.items()}
