"""Rank-local NUMA placement using the existing shared pinned arena backend."""

from dataclasses import dataclass

import torch
from loguru import logger

from lightx2v.common.offload.shared_pinned_arena import CudaRuntime, NumaBindingError, SharedPinnedArena, SharedPinnedArenaError, SharedWeightManifest, discover_gpu_numa_node
from lightx2v.common.offload.shared_weight_map import PinnedWeightViewMap, validate_shared_operator_views


def validate_numa_config(config, lora_path=None):
    enabled = config.get("cpu_offload_numa", False)
    if type(enabled) is not bool and enabled != "auto":
        raise ValueError('cpu_offload_numa must be true, false, or "auto"')
    if not enabled:
        return
    if not config.get("cpu_offload", False) or config.get("offload_granularity", "block") != "block":
        raise ValueError("cpu_offload_numa requires CPU block offload")
    incompatible = ("shared_cpu_weights", "cpu_offload_block_contiguous", "lazy_load", "tensor_parallel", "weight_auto_quant")
    if config.get("lora_mode", "merge") != "merge" or any(config.get(key) for key in incompatible) or lora_path:
        raise ValueError("cpu_offload_numa requires static rank-local weights without lazy loading, TP, auto-quantization, dynamic LoRA, shared or contiguous offload")
    # Qwen merges adapters into CPU weights before creating the NUMA arena.
    if config.get("lora_configs"):
        qwen_image_task = config.get("model_cls") == "qwen_image" and config.get("task") in ("t2i", "i2i")
        if not qwen_image_task or config.get("layered") or config.get("dit_quantized"):
            raise ValueError("cpu_offload_numa supports startup LoRA merging only for non-layered Qwen Image T2I/I2I with an unquantized DiT")
    for key in ("dit_quant_scheme", "adapter_quant_scheme"):
        scheme = config.get(key, "Default")
        # These loaders allocate private storage or perform extra conversions
        # that the ordinary base_attrs/cpu_storage_dtype contract cannot adopt.
        if scheme in ("fp8-pertensor", "int4-g128-marlin") or (isinstance(scheme, str) and scheme.startswith(("nvfp4", "gguf-"))):
            raise ValueError(f"cpu_offload_numa requires an adapter for the specialized loading/storage path of {key}={scheme!r}")


def cpu_weight_leaves(roots):
    """Exclude GPU staging slots, which reuse the CPU blocks' checkpoint keys."""
    for root in roots:
        for _, leaf in root.named_weight_leaves():
            if not getattr(leaf, "create_cuda_buffer", False) and not getattr(leaf, "create_cpu_buffer", False):
                yield leaf


def describe_cpu_weights(weights, roots):
    """Describe final CPU dtypes after the ordinary checkpoint conversions.

    Operators declare their source names through the same base_attrs contract
    used by create_default_tensors. Quantized operators additionally declare
    post-processing dtypes, so a later cast cannot escape the pinned arena.
    """
    metadata = {}
    for leaf in cpu_weight_leaves(roots):
        attrs = getattr(leaf, "base_attrs", ())
        if hasattr(leaf, "tensor_name"):
            attrs = ((leaf.tensor_name, "tensor", False),)
        if not attrs:
            try:
                if not leaf.state_dict():
                    continue
            except AttributeError:
                pass
            raise ValueError(f"Missing NUMA storage description for {type(leaf).__name__}")
        for name, attr, _ in attrs:
            if name not in weights:
                continue
            source = weights[name]
            if source.device.type != "cpu":
                raise ValueError(f"NUMA offload requires CPU checkpoint tensors: {name}")
            dtype = source.dtype
            storage_dtype = getattr(leaf, "cpu_storage_dtype", None)
            if storage_dtype is not None:
                dtype = storage_dtype(attr, dtype)
            tensor = torch.empty(source.shape, dtype=dtype, device="meta")
            previous = metadata.get(name)
            if previous is not None and (previous.dtype != dtype or previous.shape != tensor.shape):
                raise ValueError(f"Conflicting NUMA storage requirements for {name}")
            metadata[name] = tensor
    if not metadata:
        raise ValueError("No CPU offload weights were described")
    return metadata


@dataclass
class LocalPinnedWeights:
    arena: SharedPinnedArena
    numa_node: int

    def close(self):
        self.arena.close()


def materialize_numa_weights(weights, roots, *, required=True):
    """Move rank-local CPU weights into pinned memory on the current GPU's NUMA node."""
    if not torch.cuda.is_available():
        if required:
            raise ValueError("cpu_offload_numa requires CUDA")
        return None
    cuda_device = torch.cuda.current_device()
    try:
        runtime = CudaRuntime.load()
        pci_bus_id = runtime.device_pci_bus_id(cuda_device)
        numa_node = discover_gpu_numa_node(pci_bus_id)
    except SharedPinnedArenaError as exc:
        if required:
            raise
        logger.warning("GPU NUMA topology unavailable; using ordinary CPU block offload: {}", exc)
        return None
    if numa_node is None:
        if required:
            raise ValueError("cpu_offload_numa requires a known GPU NUMA node")
        logger.info("GPU has no known NUMA node; using ordinary CPU block offload")
        return None
    metadata = describe_cpu_weights(weights, roots)
    manifest = SharedWeightManifest.from_tensors(metadata, weight_signature="rank-local-pinned-v1", alignment=4096)
    try:
        arena = SharedPinnedArena.create(manifest=manifest, numa_node=numa_node, cuda_device=cuda_device, cuda_runtime=runtime, auto_remove=True)
    except NumaBindingError as exc:
        if required:
            raise
        logger.warning("Cannot bind weights to NUMA node {}; using ordinary CPU block offload: {}", numa_node, exc)
        return None
    owner = LocalPinnedWeights(arena, numa_node)
    try:
        views = arena.tensor_views()
        for name, destination in views.items():
            # copy_ performs the final dtype conversion from the already loaded
            # tensor, preserving the legacy inference-dtype rounding of scales.
            destination.copy_(weights.pop(name))
        result = PinnedWeightViewMap(weights, views, owner=owner)
    except BaseException as error:
        try:
            owner.close()
        except BaseException as cleanup_error:
            error.add_note(f"NUMA weight cleanup also failed: {cleanup_error!r}")
        raise
    logger.info(
        "[NUMAWeights] GPU={} PCI={} node={} tensors={} arena={:.3f} GiB address={:#x}",
        cuda_device,
        pci_bus_id,
        numa_node,
        len(views),
        arena.nbytes / 1024**3,
        arena.address,
    )
    return result


def validate_numa_operator_views(weight_map, roots):
    state = {}
    for leaf in cpu_weight_leaves(roots):
        if hasattr(leaf, "state_dict"):
            leaf.state_dict(state)
    expected = set(weight_map.owner.arena.manifest.by_name)
    unmapped = {name for name, tensor in state.items() if tensor is not None and tensor.device.type == "cpu"} - expected
    if unmapped:
        raise ValueError(f"CPU weights missing from the NUMA storage description: {sorted(unmapped)}")
    validate_shared_operator_views(weight_map, state)
    logger.info("[NUMAWeights] Validated {} pinned operator views", len(weight_map.owner.arena.manifest.tensors))
