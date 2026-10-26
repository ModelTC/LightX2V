"""Borrow immutable resident Stage3 base tensors without borrowing its LoRA."""

from pathlib import Path

import torch

from lightx2v.common.ops.mm.mm_weight import unwrap_tp_weight


def validate_shared_base(owner, config):
    """Both views must address the same local shard of the same checkpoint."""
    if owner.cpu_offload or config.get("cpu_offload", False):
        raise ValueError("TaoMate shared Base10 weights require both DiTs to stay resident")
    if not owner.use_adaln_cache or not config.get("use_adaln_cache", False):
        raise ValueError("TaoMate shared Base10 weights require separate Stage3 and Base10 AdaLN caches")
    defaults = dict(dit_quantized=False, dit_quant_scheme="Default", tensor_parallel=False, hidden_size=5376, num_layers=50, num_attention_heads=56, text_refiner_tensor_parallel=False)
    defaults["use_fused_qkv"] = bool(config.get("dit_quantized", False))
    for name, default in defaults.items():
        if owner.config.get(name, default) != config.get(name, default):
            raise ValueError(f"TaoMate shared Base10 weight configuration differs at {name}")
    checkpoint_key = "dit_quantized_ckpt" if config.get("dit_quantized", False) else "dit_original_ckpt"
    if Path(owner.config[checkpoint_key]).resolve() != Path(config[checkpoint_key]).resolve():
        raise ValueError("TaoMate Base10 and Stage3 must borrow the same checkpoint")
    if config.get("tensor_parallel", False):
        source_group = owner.config["device_mesh"].get_group(mesh_dim="tensor_p")
        target_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
        if source_group is not target_group:
            raise ValueError("TaoMate shared Base10 weights require the same TP process group")


def bind_base_tensors(target, source):
    """Bind matching leaf storage directly, preserving target operators/adapters."""
    source_leaves = dict(source.named_weight_leaves())
    borrowed = 0
    for name, target_leaf in target.named_weight_leaves():
        if name not in source_leaves:
            raise ValueError(f"TaoMate shared Base10 has no source operator for {name}")
        source_leaf = source_leaves[name]
        source_base, target_base = unwrap_tp_weight(source_leaf), unwrap_tp_weight(target_leaf)
        if getattr(source_base, "has_diff", False):
            raise ValueError(f"TaoMate Base10 cannot borrow a merged adapter at {name}")
        source_attrs = {key: (attr, transpose) for key, attr, transpose in getattr(source_base, "base_attrs", ())}
        for key, attr, transpose in getattr(target_base, "base_attrs", ()):
            if key not in source_attrs or source_attrs[key][1] != transpose:
                raise ValueError(f"TaoMate shared Base10 storage contract differs for {key}")
            tensor = getattr(source_base, source_attrs[key][0], None)
            if tensor is None and attr != "bias":
                raise ValueError(f"TaoMate shared Base10 source tensor is not resident: {key}")
            setattr(target_base, attr, tensor)
            borrowed += isinstance(tensor, torch.Tensor)
        # Row-parallel bias lives outside the inner MM and is added after
        # all-reduce. Its ownership must stay with the source wrapper too.
        if hasattr(target_leaf, "_row_split_bias"):
            target_leaf._row_split_bias = source_leaf._row_split_bias
        if getattr(target_base, "has_lora_branch", False) or getattr(target_base, "has_diff", False):
            raise ValueError(f"TaoMate Base10 unexpectedly contains an adapter at {name}")
    # The QKV projections already reference the owner's packed allocation.
    # Refresh only creates the corresponding target view; it does not copy.
    blocks = getattr(target, "blocks", ())
    for block in blocks:
        block.attn._build_fused_qkv()
    return borrowed
