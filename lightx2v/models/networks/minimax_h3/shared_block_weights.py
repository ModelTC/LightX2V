"""Shared checkpoint loading for H3 components; model-specific selection stays explicit."""

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.common.offload.checkpoint_metadata import checkpoint_content_digest, read_checkpoint_json, read_safetensors_header
from lightx2v.common.offload.shared_pinned_arena import SharedWeightManifest, TensorSpec
from lightx2v.common.offload.shared_weight_coordinator import coordinate_rank_local_error, materialize_shared_weight_arena, validate_shared_weight_config
from lightx2v.common.offload.shared_weight_map import SharedWeightViewMap

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}
_QKV = re.compile(r"^(.*\.attn)\.to_([qkv])\.(weight(?:_scale)?)$")


def load_h3_shared_weights(path, config, component, *, expected=None, runtime_dtypes=None, include=None, shared=None, validate=None, transform=None, fused_qkv=False):
    """All ranks preflight; only replica leaders materialize shared tensors.

    Content fingerprinting can scan shard bytes when the cached HF digest is
    unavailable or stale; this does not allocate another tensor payload copy.

    expected maps exact required names to (shape, checkpoint dtype). include
    filters unused checkpoint tensors; shared separates shared/private weights.
    runtime_dtypes controls conversion directly into the final shared storage.
    transform maps one source tensor to native named tensors, on both meta and
    CPU devices. Include/expected refer to source names; shared refers to native
    names. Fused QKV projections keep adjacent arena storage without a copy.
    """
    error = None
    try:
        logger.info("H3 {}: inspecting shared checkpoint metadata at {}", component, path)
        if validate is not None:
            validate()
        validate_shared_weight_config(config)
        chunk_mb = config.get("shared_cpu_weight_register_chunk_mb", 128)
        root = Path(path).resolve()
        indexes = [] if root.is_file() else sorted(root.glob("*.safetensors.index.json"))
        if len(indexes) > 1:
            raise ValueError(f"Ambiguous H3 checkpoint indexes under {root}")
        index_map = read_checkpoint_json(indexes[0])["weight_map"] if indexes else None
        files = [root] if root.is_file() else [root / name for name in sorted(set(index_map.values()))] if index_map else sorted(root.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No H3 checkpoint shards under {root}")
        entries, signatures, indexed = {}, [], {}
        for file in files:
            header = read_safetensors_header(file)
            signatures.append((file.name, checkpoint_content_digest(file)))
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                if name in indexed:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                indexed[name] = file.name
                if expected is not None and name not in expected:
                    continue
                if include is not None and not include(name):
                    continue
                dtype = _DTYPES.get(spec["dtype"])
                shape = tuple(spec["shape"])
                if dtype is None or not shape or any(type(n) is not int or n <= 0 for n in shape):
                    raise ValueError(f"Invalid H3 tensor specification: {name}")
                if expected is not None and (shape, dtype) != expected[name]:
                    raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
                if spec["data_offsets"][1] - spec["data_offsets"][0] != math.prod(shape) * dtype.itemsize:
                    raise ValueError(f"Invalid checkpoint tensor byte extent: {name}")
                entries[name] = (file, shape, dtype)
        if index_map is not None and index_map != indexed:
            raise ValueError(f"Checkpoint index/header mismatch: {indexes[0]}")
        if expected is not None and entries.keys() != expected.keys():
            raise ValueError(f"Missing H3 checkpoint tensors: {sorted(expected.keys() - entries.keys())}")
        convert = transform or (lambda name, tensor: {name: tensor})
        native_metadata = {}
        for name, (_, shape, dtype) in entries.items():
            for native_name, tensor in convert(name, torch.empty(shape, dtype=dtype, device="meta")).items():
                if native_name in native_metadata:
                    raise ValueError(f"Duplicate native H3 tensor: {native_name}")
                native_metadata[native_name] = tensor
        shared_names = {name for name in native_metadata if shared is None or shared(name)}
        if not shared_names:
            raise ValueError("Empty H3 shared checkpoint selection")
        runtime_dtypes = runtime_dtypes or {}
        metadata = {n: torch.empty(native_metadata[n].shape, device="meta", dtype=runtime_dtypes.get(n, native_metadata[n].dtype)) for n in shared_names}
        signature = hashlib.sha256(
            json.dumps(
                {"format": "h3-shared-v1", "component": component, "files": signatures, "tensors": [(n, list(t.shape), str(t.dtype)) for n, t in sorted(metadata.items())]}, sort_keys=True
            ).encode()
        ).hexdigest()
        manifest = _qkv_manifest(metadata, signature) if fused_qkv else SharedWeightManifest.from_tensors(metadata, weight_signature=signature, alignment=4096)
        private = {}
        private_sources = {n: e for n, e in entries.items() if any(name not in shared_names for name in convert(n, torch.empty(e[1], dtype=e[2], device="meta")))}
        for name, tensor in _read_entries(private_sources):
            for native_name, native_tensor in convert(name, tensor).items():
                if native_name not in shared_names:
                    private[native_name] = native_tensor.clone()
    except Exception as exc:
        error = exc
    coordinate_rank_local_error(f"H3 {component} checkpoint preflight", error)
    logger.info("H3 {}: materializing {:.3f} GiB shared CPU weights", component, manifest.nbytes / 1024**3)

    def populate(views):
        shared_sources = {n: e for n, e in entries.items() if any(name in shared_names for name in convert(n, torch.empty(e[1], dtype=e[2], device="meta")))}
        for name, tensor in _read_entries(shared_sources):
            for native_name, native_tensor in convert(name, tensor).items():
                if native_name in shared_names:
                    views[native_name].copy_(native_tensor)

    allocation = materialize_shared_weight_arena(
        manifest,
        populate,
        scope=config.get("shared_cpu_weight_scope", "auto"),
        strict_numa=config.get("shared_cpu_weight_strict_numa", True),
        register_chunk_bytes=chunk_mb * 1024 * 1024,
    )
    return SharedWeightViewMap(private, allocation.tensor_views(), owner=allocation)


def _qkv_manifest(tensors, signature):
    """Make Q/K/V groups adjacent even when scale tensors have page padding.

    FusedQKVStorage recognizes views of one allocation only in Q,K,V order.
    Page alignment applies to the group; individual projections need only
    dtype alignment. All names remain ordinary independently validated views.
    """

    def key(name):
        match = _QKV.fullmatch(name)
        return (match[1] + ".to_qkv", match[3], "qkv".index(match[2])) if match else (name, "", 0)

    offset, specs, previous_group = 0, [], None
    for name in sorted(tensors, key=key):
        tensor = tensors[name]
        match = _QKV.fullmatch(name)
        group = (match[1], match[3]) if match else None
        alignment = tensor.element_size() if group is not None and group == previous_group else 4096
        offset = (offset + alignment - 1) // alignment * alignment
        spec = SharedWeightManifest.from_tensors({name: tensor}, signature, alignment=1).tensors[0]
        specs.append(TensorSpec(name=name, offset=offset, nbytes=spec.nbytes, dtype=spec.dtype, storage_shape=spec.storage_shape))
        offset += spec.nbytes
        previous_group = group
    return SharedWeightManifest(nbytes=(offset + 4095) // 4096 * 4096, tensors=tuple(specs), weight_signature=signature, alignment=4096)


def _read_entries(entries):
    by_file = defaultdict(list)
    for name, (file, shape, dtype) in entries.items():
        by_file[file].append((name, shape, dtype))
    for file, specs in sorted(by_file.items()):
        logger.info("H3 shared loader reading {} selected tensors from {}", len(specs), file)
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for name, shape, dtype in specs:
                tensor = checkpoint.get_tensor(name)
                if tensor.shape != shape or tensor.dtype != dtype:
                    raise ValueError(f"H3 checkpoint changed after preflight: {name}")
                yield name, tensor


def load_shared_dit(model):
    config = model.config

    def validate():
        if not model.block_offload or not model.use_adaln_cache:
            raise ValueError("H3 shared DiT weights require block offload and AdaLN cache")
        if (
            any(config.get(k, False) for k in ("tensor_parallel", "dit_quantized", "weight_auto_quant", "lazy_load", "unload_modules", "dummy_model", "lora_dynamic_apply", "use_compile"))
            or model.lora_path
            or config.get("lora_configs")
        ):
            raise ValueError("H3 shared DiT supports original weights without TP, quantization, LoRA or lazy loading")

    weights = load_h3_shared_weights(
        model.model_path,
        config,
        "dit",
        include=lambda name: not any(part in name for part in model.remove_keys),
        shared=lambda name: name.startswith("transformer_blocks."),
        validate=validate,
    )
    model._record_checkpoint_metadata(weights)
    return weights
