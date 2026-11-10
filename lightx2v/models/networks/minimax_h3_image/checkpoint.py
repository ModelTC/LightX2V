"""Map the released, head-interleaved DiT checkpoint onto native H3 ops."""

import math

import torch


def convert_transformer_tensor(name, tensor, config):
    if name == "rope.inv_freq" or name.startswith("final_layer.video_out."):
        return {}
    if name.startswith("token_refiner.blocks."):
        name = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif name.startswith("blocks."):
        name = name.replace("blocks.", "transformer_blocks.", 1)
    replacements = {
        "video_patch_proj.": "proj_in.",
        "condition_proj.": "context_embedder.",
        "time_embedder.proj_in.": "time_embedder.linear_1.",
        "time_embedder.proj_out.": "time_embedder.linear_2.",
        "final_layer.norm.": "norm_out.norm.",
        "final_layer.adaln_proj.linear.": "norm_out.linear.",
        ".attn.q_norm.": ".attn.norm_q.",
        ".attn.k_norm.": ".attn.norm_k.",
        ".attn.out_proj.": ".attn.to_out.0.",
        ".mlp.fc2.": ".ff.net.2.",
    }
    for source, target in replacements.items():
        name = name.replace(source, target)
    if name.endswith(".attn.qkv_proj.weight"):
        heads = int(config["num_attention_heads"])
        head_dim = int(config["attention_head_dim"])
        grouped = tensor.reshape(heads, 3, head_dim, *tensor.shape[1:])
        prefix = name.removesuffix("qkv_proj.weight")
        return {f"{prefix}to_{kind}.weight": grouped[:, index].reshape(heads * head_dim, *tensor.shape[1:]).contiguous() for index, kind in enumerate("qkv")}
    if name.endswith(".mlp.fc1.weight"):
        gate, value = tensor.chunk(2, dim=0)
        return {name.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight"): torch.cat((value, gate), dim=0)}
    return {name: tensor}


def convert_pdd_heads(tensors, config):
    weight, bias = tensors["video_out.weight"], tensors["video_out.bias"]
    patch_dim = int(config["in_channels"]) * math.prod(config.get("patch_size", (1, 2, 2)))
    expected = (6, patch_dim, int(config["hidden_size"]))
    if tuple(weight.shape) != expected or tuple(bias.shape) != expected[:2]:
        raise ValueError(f"MiniMax-H3-Image requires six PDD heads with shapes {expected} and {expected[:2]}")
    return {f"pdd_proj_out.{index}.{kind}": tensor[index].float().contiguous() for index in range(6) for kind, tensor in (("weight", weight), ("bias", bias))}
