# Copyright 2026 The LightX2V Team.
# SPDX-License-Identifier: Apache-2.0

"""Load original or Diffusers H3 image VAE tensors into the local modules."""

import torch
from safetensors import safe_open

from lightx2v.models.video_encoders.hf.minimax_h3.weights import _assign_tensor


def convert_original_vae_tensor(key, tensor, heads, dim_head):
    if key == "decoder.mask_token":
        return []
    if key.startswith("encoder.down."):
        level, rest = key.removeprefix("encoder.down.").split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.")
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        key = f"encoder.down_blocks.{level}.{rest}"
    key = key.replace("decoder.x_embedder.", "decoder.proj_in.")
    if ".attn.to_qkv." in key:
        prefix, suffix = key.split(".attn.to_qkv.")
        # Original shards interleave Q/K/V within each head, including bias.
        grouped = tensor.reshape(heads, 3 * dim_head, *tensor.shape[1:])
        tensors = grouped.split(dim_head, dim=1)
        return [(f"{prefix}.attn.to_{name}.{suffix}", part.reshape(heads * dim_head, *tensor.shape[1:]).contiguous()) for name, part in zip(("q", "k", "v"), tensors)]
    key = key.replace(".attn.to_out.", ".attn.to_out.0.")
    if ".ff.w1." in key:
        gate, value = tensor.chunk(2, dim=0)
        return [(key.replace(".ff.w1.", ".ff.net.0.proj."), torch.cat((value, gate), dim=0))]
    return [(key.replace(".ff.w2.", ".ff.net.2."), tensor)]


def load_image_vae_weights(model, component, original):
    filename = component / "diffusion_pytorch_model.safetensors"
    expected = model.state_dict()
    loaded = set()
    with safe_open(filename, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            tensor = checkpoint.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise TypeError(f"H3 image VAE checkpoint {key!r} must be float32, got {tensor.dtype}")
            tensors = convert_original_vae_tensor(key, tensor, model.config["decoder_num_attention_heads"], model.config["decoder_attention_head_dim"]) if original else [(key, tensor)]
            for target, value in tensors:
                if target not in expected or target in loaded:
                    raise KeyError(f"Unexpected H3 image VAE tensor {key!r} -> {target!r}")
                if value.shape != expected[target].shape:
                    raise ValueError(f"Shape mismatch for {key!r}: expected {tuple(expected[target].shape)}, got {tuple(value.shape)}")
                _assign_tensor(model, target, value)
                loaded.add(target)
    missing = sorted(set(expected) - loaded)
    if missing:
        raise RuntimeError(f"H3 image VAE checkpoint is missing tensors: {missing[:20]}")
