"""Convert EdenLive Hugging Face encoder weights to native Wan tensor names."""

import torch


def remap_hf_umt5_weights(state):
    replacements = (
        ("encoder.block.", "blocks."),
        (".layer.0.SelfAttention.relative_attention_bias.", ".pos_embedding.embedding."),
        (".layer.0.SelfAttention.", ".attn."),
        (".layer.0.layer_norm.", ".norm1."),
        (".layer.1.layer_norm.", ".norm2."),
        (".layer.1.DenseReluDense.wi_0.", ".ffn.gate.0."),
        (".layer.1.DenseReluDense.wi_1.", ".ffn.fc1."),
        (".layer.1.DenseReluDense.wo.", ".ffn.fc2."),
        ("encoder.final_layer_norm.", "norm."),
        ("shared.weight", "token_embedding.weight"),
    )
    mapped = {}
    for key, value in state.items():
        for original, native in replacements:
            key = key.replace(original, native)
        mapped[key] = value
    return mapped


def remap_hf_clip_vision_weights(state, num_layers):
    mapped = {
        "cls_embedding": state["vision_model.embeddings.class_embedding"].view(1, 1, -1),
        "pos_embedding": state["vision_model.embeddings.position_embedding.weight"].unsqueeze(0),
        "patch_embedding.weight": state["vision_model.embeddings.patch_embedding.weight"],
        "head": state["visual_projection.weight"].t().contiguous(),
    }
    for suffix in ("weight", "bias"):
        mapped[f"pre_norm.{suffix}"] = state[f"vision_model.pre_layrnorm.{suffix}"]
        mapped[f"post_norm.{suffix}"] = state[f"vision_model.post_layernorm.{suffix}"]
        for index in range(num_layers):
            source = f"vision_model.encoder.layers.{index}."
            target = f"transformer.{index}."
            mapped[f"{target}attn.to_qkv.{suffix}"] = torch.cat([state[f"{source}self_attn.{projection}_proj.{suffix}"] for projection in ("q", "k", "v")])
            mapped[f"{target}attn.proj.{suffix}"] = state[f"{source}self_attn.out_proj.{suffix}"]
            for original, native in (("layer_norm1", "norm1"), ("layer_norm2", "norm2"), ("mlp.fc1", "mlp.0"), ("mlp.fc2", "mlp.2")):
                mapped[f"{target}{native}.{suffix}"] = state[f"{source}{original}.{suffix}"]
    return mapped
