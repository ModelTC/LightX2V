def remap_hf_umt5_weights(state, num_layers):
    """Translate Hugging Face UMT5 encoder weights to native Wan names."""
    block_weights = (
        ("layer.0.SelfAttention.q", "attn.q"),
        ("layer.0.SelfAttention.k", "attn.k"),
        ("layer.0.SelfAttention.v", "attn.v"),
        ("layer.0.SelfAttention.o", "attn.o"),
        ("layer.0.SelfAttention.relative_attention_bias", "pos_embedding.embedding"),
        ("layer.0.layer_norm", "norm1"),
        ("layer.1.layer_norm", "norm2"),
        ("layer.1.DenseReluDense.wi_0", "ffn.gate.0"),
        ("layer.1.DenseReluDense.wi_1", "ffn.fc1"),
        ("layer.1.DenseReluDense.wo", "ffn.fc2"),
    )
    names = {"shared.weight": "token_embedding.weight", "encoder.final_layer_norm.weight": "norm.weight"}
    for index in range(num_layers):
        for source, target in block_weights:
            names[f"encoder.block.{index}.{source}.weight"] = f"blocks.{index}.{target}.weight"
    if unexpected := state.keys() - names.keys():
        raise ValueError(f"Unexpected UMT5 encoder weights: {sorted(unexpected)}")
    return {target: state[source] for source, target in names.items()}
