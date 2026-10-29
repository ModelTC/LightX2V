"""Map the released StreamRender-H3 DiT checkpoint to native H3 weights.

The released fused QKV rows are contiguous [all Q, all K, all V].
LightX2V's H3 weight wrappers store separate projections. The runtime loads
these rows directly into its fused projection without a head-wise reorder.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

_BLOCK = re.compile(r"^dit\.blocks\.(\d+)\.(.+)$")
_REFINER_BLOCK = re.compile(r"^dit\.token_refiner\.blocks\.(\d+)\.(.+)$")

_GLOBAL_NAMES = {
    "dit.video_patch_proj.weight": "proj_in.weight",
    "dit.video_patch_proj.bias": "proj_in.bias",
    "dit.audio_patch_proj.weight": "audio_proj_in.weight",
    "dit.audio_patch_proj.bias": "audio_proj_in.bias",
    "dit.condition_proj.weight": "context_embedder.weight",
    "dit.condition_proj.bias": "context_embedder.bias",
    "dit.time_embedder.proj_in.weight": "time_embedder.linear_1.weight",
    "dit.time_embedder.proj_in.bias": "time_embedder.linear_1.bias",
    "dit.time_embedder.proj_out.weight": "time_embedder.linear_2.weight",
    "dit.time_embedder.proj_out.bias": "time_embedder.linear_2.bias",
    "dit.final_layer.video_out.weight": "proj_out.weight",
    "dit.final_layer.video_out.bias": "proj_out.bias",
    "dit.final_layer.audio_out.weight": "audio_proj_out.weight",
    "dit.final_layer.audio_out.bias": "audio_proj_out.bias",
    "dit.final_layer.adaln_proj.linear.weight": "norm_out.linear.weight",
    "dit.final_layer.adaln_proj.linear.bias": "norm_out.linear.bias",
    "dit.final_layer.norm.weight": "norm_out.norm.weight",
    "dit.token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
}

_BLOCK_SUFFIXES = {
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "adaln_proj.linear.weight": "adaln_proj.linear.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
    "attn.q_norm.weight": "attn.norm_q.weight",
    "attn.k_norm.weight": "attn.norm_k.weight",
    "attn.out_proj.weight": "attn.to_out.0.weight",
    "mlp.fc1.weight": "ff.net.0.proj.weight",
    "mlp.fc2.weight": "ff.net.2.weight",
}
_QKV_SUFFIX = "attn.qkv_proj.weight"
_QKV_PROJECTIONS = ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight")


def native_weight_names(source_name):
    """Return the native keys for one released checkpoint key.

    ``dit.rope.inv_freq`` is derived from the configured RoPE frequencies and
    has no trainable counterpart in LightX2V. Unknown names fail explicitly so
    a later upstream checkpoint revision cannot be partially loaded silently.
    """
    if source_name == "dit.rope.inv_freq":
        return ()
    native_name = _GLOBAL_NAMES.get(source_name)
    if native_name is not None:
        return (native_name,)

    block = _BLOCK.fullmatch(source_name)
    if block is not None:
        prefix = f"transformer_blocks.{block.group(1)}."
    else:
        block = _REFINER_BLOCK.fullmatch(source_name)
        if block is None:
            raise KeyError(f"Unrecognized StreamRender-H3 checkpoint key: {source_name}")
        prefix = f"token_refiner.refiner_blocks.{block.group(1)}."
    suffix = block.group(2)
    if suffix == _QKV_SUFFIX:
        return tuple(prefix + projection for projection in _QKV_PROJECTIONS)
    mapped = _BLOCK_SUFFIXES.get(suffix)
    if mapped is None:
        raise KeyError(f"Unrecognized StreamRender-H3 checkpoint key: {source_name}")
    return (prefix + mapped,)


def remap_streamrender_weight(source_name, tensor, *, num_heads=56, head_dim=128):
    """Convert one released tensor without materializing a full state dict."""
    native_names = native_weight_names(source_name)
    if not native_names:
        return {}
    if len(native_names) == 1:
        # StreamRender fc1 is [gate, up], whereas native H3's SwiGLU
        # consumes [value/up, gate]. Swap the two row halves once on load.
        if source_name.endswith(".mlp.fc1.weight"):
            if tensor.ndim != 2 or tensor.shape[0] % 2:
                raise ValueError(f"Invalid fused MLP shape for {source_name}: {tuple(tensor.shape)}")
            gate, up = tensor.chunk(2, dim=0)
            tensor = torch.cat((up, gate), dim=0)
        return {native_names[0]: tensor}
    if tensor.ndim != 2 or tensor.shape[0] != 3 * num_heads * head_dim:
        raise ValueError(f"Invalid fused QKV shape for {source_name}: {tuple(tensor.shape)}")

    return {name: projection.contiguous() for name, projection in zip(native_names, tensor.chunk(3, dim=0))}


def load_streamrender_block_weights(block_weights, index_path, block_index):
    """Load one real StreamRender block into native H3 weight wrappers.

    Safetensors shards are opened one at a time, and only this block's tensors
    are materialized. The full 66 GB checkpoint is never built as a state dict.
    """
    index_path = Path(index_path)
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = f"dit.blocks.{block_index}."
    selected = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
    if len(selected) != 10:
        raise ValueError(f"StreamRender-H3 block {block_index} has {len(selected)} checkpoint tensors; expected 10")
    mapped = {}
    for shard in sorted(set(selected.values())):
        with safe_open(index_path.parent / shard, framework="pt", device="cpu") as source:
            for name, filename in selected.items():
                if filename == shard:
                    mapped.update(remap_streamrender_weight(name, source.get_tensor(name)))
    if len(mapped) != 12:
        raise ValueError(f"StreamRender-H3 block {block_index} maps to {len(mapped)} native tensors; expected 12")
    block_weights.load(mapped)
    return block_weights


def load_streamrender_rope_frequencies(index_path):
    """Preserve the release's rounded RoPE buffer for native pre-inference.

    The safetensors file stores BF16 frequencies. Loading them into the
    upstream FP32 RoPE buffer keeps those rounded values, which differ from
    recomputing ideal base-10000 frequencies in FP32.
    """
    index_path = Path(index_path)
    name = "dit.rope.inv_freq"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    with safe_open(index_path.parent / weight_map[name], framework="pt", device="cpu") as source:
        frequencies = source.get_tensor(name).to(torch.float32)
    if frequencies.shape != (16,):
        raise ValueError(f"Invalid StreamRender-H3 RoPE frequency shape: {tuple(frequencies.shape)}")
    return frequencies
