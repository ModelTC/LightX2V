"""Share distilled causal H3 blocks without converting its original checkpoint."""

import hashlib
import json
from pathlib import Path

import torch

from lightx2v.common.offload.shared_weight_coordinator import coordinate_rank_local_error, materialize_shared_weight_arena, validate_shared_weight_config
from lightx2v.common.offload.shared_weight_map import SharedWeightViewMap
from lightx2v.models.networks.minimax_h3.shared_block_weights import _qkv_manifest, load_h3_shared_weights


def load_shared_causal_weights(model, sensitive_layer):
    """Cast original FP32 blocks directly into one registered CPU allocation.

    The PT file remains mmap-backed on each rank. Only replica leaders touch
    block payloads; private input/output projections retain original FP32.
    FP8 uses the same shared safetensors adapter as the other H3 models.
    """
    config = model.config

    def validate():
        validate_shared_weight_config(config)
        if not model.block_offload or not model.use_adaln_cache:
            raise ValueError("Causal H3 shared weights require block CPU offload and AdaLN cache")
        if (
            any(config.get(key, False) for key in ("tensor_parallel", "weight_auto_quant", "lazy_load", "unload_modules", "dummy_model", "lora_dynamic_apply"))
            or model.lora_path
            or config.get("lora_configs")
        ):
            raise ValueError("Causal H3 shared weights do not support TP, LoRA or lazy loading")
        if model.dit_quantized and config.get("dit_quant_scheme") not in ("fp8-f16-accum", "fp8-f32-accum"):
            raise ValueError("Causal H3 shared quantized weights require fp8-f16-accum or fp8-f32-accum")

    def include(name):
        return not any(remove in name for remove in model.remove_keys)

    if model.dit_quantized:
        weights = load_h3_shared_weights(
            config["dit_quantized_ckpt"],
            config,
            "causal-dit-fp8-f16",
            include=include,
            shared=lambda name: name.startswith("transformer_blocks."),
            validate=validate,
            transform=lambda name, tensor: {name: model._fp8_f16_checkpoint_tensor(name, tensor, sensitive_layer)},
            fused_qkv=bool(config.get("use_fused_qkv", False)),
        )
    else:
        error = None
        try:
            validate()
            path = Path(config["dit_original_ckpt"]).resolve()
            if path.suffix != ".pt":
                raise ValueError("Causal H3 shared BF16 loading expects the original distilled .pt checkpoint")
            source = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
            selected = {name: tensor for name, tensor in source.items() if include(name)}
            if not selected or any(not isinstance(tensor, torch.Tensor) for tensor in selected.values()):
                raise ValueError("Causal H3 checkpoint must contain a flat tensor mapping")
            shared = {name: tensor for name, tensor in selected.items() if name.startswith("transformer_blocks.")}
            if not shared:
                raise ValueError("Causal H3 checkpoint contains no transformer blocks")
            metadata = {name: torch.empty(tensor.shape, dtype=torch.bfloat16, device="meta") for name, tensor in shared.items()}
            identity = dict(
                path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, tensors=[(name, list(tensor.shape), str(tensor.dtype)) for name, tensor in sorted(metadata.items())]
            )
            signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            manifest = _qkv_manifest(metadata, signature)
            private = {name: tensor.to(torch.float32 if name.split(".", 1)[0] in sensitive_layer else torch.bfloat16) for name, tensor in selected.items() if name not in shared}
        except Exception as exc:
            error = exc
        coordinate_rank_local_error("causal H3 PT checkpoint preflight", error)

        def populate(views):
            for name, tensor in shared.items():
                views[name].copy_(tensor)

        allocation = materialize_shared_weight_arena(
            manifest,
            populate,
            scope=config.get("shared_cpu_weight_scope", "auto"),
            strict_numa=config.get("shared_cpu_weight_strict_numa", True),
            register_chunk_bytes=config.get("shared_cpu_weight_register_chunk_mb", 128) * 1024**2,
        )
        weights = SharedWeightViewMap(private, allocation.tensor_views(), owner=allocation)
    model._record_checkpoint_metadata(weights)
    return weights
