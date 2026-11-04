"""Mixed-precision selection and checkpoint contract for H3 Apple INT8."""

import json
from pathlib import Path

MLP_SUFFIXES = (".ff.net.0.proj", ".ff.net.2")
QUANT_SCOPES = {
    "fc1": ("fc1",),
    "fc2": ("fc2",),
    "mlp": ("fc1", "fc2"),
    "mlp_out": ("fc1", "fc2", "out"),
    "mlp_qkv": ("fc1", "fc2", "qkv"),
    "dit_proj": ("fc1", "fc2", "qkv", "out"),
}


def linear_component(name):
    if not name.startswith("transformer_blocks."):
        return None
    if name.endswith(".ff.net.0.proj"):
        return "fc1"
    if name.endswith(".ff.net.2"):
        return "fc2"
    if name.endswith((".attn.to_q", ".attn.to_k", ".attn.to_v")):
        return "qkv"
    if name.endswith(".attn.to_out.0"):
        return "out"
    return None


def is_mlp_linear(name):
    return name.startswith("transformer_blocks.") and name.endswith(MLP_SUFFIXES)


def linear_quant_scheme(config, name):
    scheme = config.get("dit_quant_scheme", "Default")
    scope = config.get("dit_quant_scope", "all")
    if scope == "all":
        return scheme
    if scope not in QUANT_SCOPES or scheme != "int8-metal":
        raise ValueError("H3 dit_quant_scope supports all, or an explicit int8-metal projection profile")
    return scheme if linear_component(name) in QUANT_SCOPES[scope] else "Default"


def validate_metal_checkpoint(config):
    path = Path(config["dit_quantized_ckpt"])
    indices = list(path.glob("*.safetensors.index.json")) if path.is_dir() else []
    if len(indices) != 1:
        raise ValueError("H3 int8-metal requires one indexed safetensors checkpoint directory")
    index = json.loads(indices[0].read_text())
    metadata = index.get("metadata", {})
    if metadata.get("lightx2v_quant_scheme") != "int8-metal":
        raise ValueError("H3 checkpoint does not declare lightx2v_quant_scheme=int8-metal")
    if metadata.get("lightx2v_quant_scope") != config.get("dit_quant_scope", "all"):
        raise ValueError("H3 INT8 checkpoint scope does not match dit_quant_scope")
    import torch

    from lightx2v.common.offload.safetensors_checkpoint import SafetensorsCheckpoint

    checkpoint = SafetensorsCheckpoint(path, weight_map=index["weight_map"])
    scope = config.get("dit_quant_scope", "all")
    selected = set()
    for name in checkpoint.weight_map:
        dtype, shape, _, _ = checkpoint.tensor_metadata(name)
        if name.endswith(".weight") and linear_quant_scheme(dict(config, dit_quant_scheme="int8-metal"), name[:-7]) == "int8-metal":
            # Explicit scopes select only their declared projections.
            if scope == "all" and linear_component(name[:-7]) is None:
                continue
            if dtype != torch.int8 or len(shape) != 2 or any(n <= 0 or n % 128 for n in shape):
                raise ValueError("Invalid INT8 Metal weight shape/dtype: " + name)
            scale = name + "_scale"
            if scale not in checkpoint.weight_map:
                raise ValueError("Missing INT8 Metal scale: " + scale)
            sdtype, sshape, _, _ = checkpoint.tensor_metadata(scale)
            if sdtype != torch.float32 or sshape != (shape[0], 1):
                raise ValueError("Invalid INT8 Metal scale shape/dtype: " + scale)
            selected.add(name)
        elif dtype == torch.int8:
            raise ValueError("INT8 tensor outside declared projection scope: " + name)
    if not selected:
        raise ValueError("No INT8 weights in declared H3 projection scope")
    if metadata.get("quantized_linear_count", len(selected)) != len(selected):
        raise ValueError("Declared INT8 projection count does not match checkpoint")
    return index


def load_metal_checkpoint(config, device, remove_keys=()):
    """Honor the authoritative index, including selective checkpoint views."""
    from collections import defaultdict

    from safetensors import safe_open

    index = validate_metal_checkpoint(config)
    root = Path(config["dit_quantized_ckpt"])
    by_file = defaultdict(list)
    for name, file in index["weight_map"].items():
        if not any(part in name for part in remove_keys):
            by_file[file].append(name)
    tensors = {}
    for file, names in sorted(by_file.items()):
        with safe_open(str(root / file), framework="pt", device="cpu") as source:
            for name in names:
                tensors[name] = source.get_tensor(name).to(device)
    return tensors
