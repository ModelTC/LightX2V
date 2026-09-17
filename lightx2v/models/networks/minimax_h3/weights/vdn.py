"""Released VDN-H3 branch weights and ordered, inference-time LoRA merging."""

import json
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.ops.tensor.tensor import DefaultTensor
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


def configure_vdn(config):
    checkpoint = Path(config["vdn_checkpoint"]).expanduser().resolve()
    with (checkpoint / "model_spec.json").open() as handle:
        spec = json.load(handle)
    config["vdn_checkpoint"] = str(checkpoint)
    config["vdn_attention"] = spec["transforms"][0]["config"]


def _branch_shapes(config):
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    dim = int(config["vdn_attention"]["linear_attention"]["linear_head_dim"])
    channels = heads * dim
    shapes = {
        "linear_attention.alpha.A_log": (heads,),
        "linear_attention.alpha.down.weight": (dim, hidden),
        "linear_attention.alpha.dt_bias": (channels,),
        "linear_attention.alpha.up.weight": (channels, dim),
        "linear_attention.beta_proj.weight": (heads, hidden),
        "linear_attention.norm.weight": (dim,),
        "linear_attention.output_gate.down.weight": (dim, hidden),
        "linear_attention.output_gate.up.bias": (channels,),
        "linear_attention.output_gate.up.weight": (channels, dim),
        "linear_attention.short_conv.k_sp.weight": (channels, 1, 5, 5),
        "linear_attention.short_conv.k_tm.weight": (channels, 1, 5),
        "linear_attention.short_conv.v_sp.weight": (channels, 1, 5, 5),
        "linear_attention.short_conv.v_tm.weight": (channels, 1, 5),
        "softmax_gate.up.bias": (heads,),
        "softmax_gate.up.weight": (heads, hidden),
        "to_out_linear.weight": (hidden, channels),
    }
    return {f"transformer_blocks.{index}.attn.{suffix}": shape for index in range(int(config["num_layers"])) for suffix, shape in shapes.items()}


@torch.no_grad()
def _merge_adapter(weight_dict, path, name, *, skip_cached_modulation=False):
    """Match VDN: CPU FP32 B@A, cast delta, then add in the base dtype."""
    from lightx2v.models.networks.minimax_h3.adaln_cache import is_adaln_cache_key

    suffix = f".lora_A.{name}.weight"
    with safe_open(path, framework="pt", device="cpu") as source:
        keys = set(source.keys())
        a_keys = sorted(key for key in keys if key.endswith(suffix))
        b_keys = {key.replace(".lora_A.", ".lora_B.") for key in a_keys}
        if not a_keys or keys != set(a_keys) | b_keys:
            raise ValueError(f"Incomplete or unsupported VDN {name} adapter tensors in {path}")
        pairs = []
        for a_key in a_keys:
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            target = a_key[: -len(suffix)].replace(".attn.orig.", ".attn.") + ".weight"
            if skip_cached_modulation and is_adaln_cache_key(target):
                continue
            if target not in weight_dict:
                raise KeyError(f"VDN {name} LoRA target is missing from the base checkpoint: {target}")
            pairs.append((a_key, b_key, target))
        for a_key, b_key, target in pairs:
            _add_vdn_delta(weight_dict[target], source.get_tensor(a_key), source.get_tensor(b_key))
    logger.info("Merged VDN {} adapter: {} pairs, CPU FP32 delta and original base dtype", name, len(pairs))
    return len(pairs)


@torch.no_grad()
def _add_vdn_delta(weight, a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or tuple(weight.shape) != (b.shape[0], a.shape[1]):
        raise ValueError(f"VDN LoRA shape mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}, base={tuple(weight.shape)}")
    delta = b.to(device="cpu", dtype=torch.float32) @ a.to(device="cpu", dtype=torch.float32)
    weight.add_(delta.to(device=weight.device, dtype=weight.dtype))
    return weight


def vdn_adapter_paths(config):
    checkpoint = Path(config["vdn_checkpoint"])
    with (checkpoint / "model_spec.json").open() as handle:
        spec = json.load(handle)
    return [(adapter["config"].get("name", "default"), checkpoint / "adapters" / adapter["config"].get("name", "default") / "adapter_model.safetensors") for adapter in spec["adapters"]]


def merge_vdn_tensor(weight, target, adapters):
    """Merge one cache tensor using ordered, already-open adapter readers."""
    stem = target.removesuffix(".weight")
    if stem.startswith("transformer_blocks."):
        stem = stem.replace(".attn.", ".attn.orig.")
    for name, source in adapters:
        a_key = f"{stem}.lora_A.{name}.weight"
        b_key = f"{stem}.lora_B.{name}.weight"
        keys = source.keys()
        if a_key not in keys and b_key not in keys:
            continue
        if a_key not in keys or b_key not in keys:
            raise ValueError(f"Incomplete VDN {name} LoRA pair for {target}")
        _add_vdn_delta(weight, source.get_tensor(a_key), source.get_tensor(b_key))
    return weight


def load_vdn_weights(weight_dict, config):
    """Augment native H3 checkpoint tensors before WeightModule consumes them."""
    checkpoint = Path(config["vdn_checkpoint"])
    branch_path = checkpoint / "linear_branch" / "model.safetensors"
    expected = _branch_shapes(config)
    with safe_open(branch_path, framework="pt", device="cpu") as source:
        keys = set(source.keys())
        if keys != expected.keys():
            missing = sorted(expected.keys() - keys)
            extra = sorted(keys - expected.keys())
            raise ValueError(f"VDN branch tensor mismatch: missing={missing[:4]}, extra={extra[:4]}")
        for key, shape in expected.items():
            tensor = source.get_slice(key)
            if tuple(tensor.get_shape()) != shape or tensor.get_dtype() != "BF16":
                raise ValueError(f"VDN branch {key} must be BF16 with shape {shape}")
            if key in weight_dict:
                raise ValueError(f"VDN branch key already exists in the base checkpoint: {key}")
        # Keep the full branch on each SP rank for native block offload.
        device = next(iter(weight_dict.values())).device
        for key in expected:
            weight_dict[key] = source.get_tensor(key).to(device)
    for name, path in vdn_adapter_paths(config):
        _merge_adapter(weight_dict, path, name, skip_cached_modulation=config.get("use_adaln_cache", False))
    logger.info("Loaded VDN-H3 branch: {} tensors from {}", len(expected), checkpoint)
    return weight_dict


class _VDNTensor(DefaultTensor):
    """Use common tensor offload storage with the inference weight spelling."""

    @property
    def weight(self):
        return self.tensor


class MiniMaxH3VDNWeights(WeightModule):
    """The 16 extra tensors for one DiT attention block; no upstream modules."""

    def __init__(self, prefix, create_cuda_buffer=False):
        super().__init__()
        linears = {
            "alpha_down": ("linear_attention.alpha.down", False),
            "alpha_up": ("linear_attention.alpha.up", False),
            "beta_proj": ("linear_attention.beta_proj", False),
            "output_gate_down": ("linear_attention.output_gate.down", False),
            "output_gate_up": ("linear_attention.output_gate.up", True),
            "softmax_gate": ("softmax_gate.up", True),
            "to_out_linear": ("to_out_linear", False),
        }
        for name, (suffix, bias) in linears.items():
            self.add_module(
                name,
                MM_WEIGHT_REGISTER["Default"](
                    f"{prefix}.{suffix}.weight",
                    f"{prefix}.{suffix}.bias" if bias else None,
                    create_cuda_buffer=create_cuda_buffer,
                    lora_prefix="transformer_blocks",
                ),
            )
        tensors = {
            "alpha_a_log": "linear_attention.alpha.A_log",
            "alpha_dt_bias": "linear_attention.alpha.dt_bias",
            "norm": "linear_attention.norm.weight",
            "k_sp": "linear_attention.short_conv.k_sp.weight",
            "k_tm": "linear_attention.short_conv.k_tm.weight",
            "v_sp": "linear_attention.short_conv.v_sp.weight",
            "v_tm": "linear_attention.short_conv.v_tm.weight",
        }
        for name, suffix in tensors.items():
            self.add_module(name, _VDNTensor(f"{prefix}.{suffix}", create_cuda_buffer=create_cuda_buffer))
