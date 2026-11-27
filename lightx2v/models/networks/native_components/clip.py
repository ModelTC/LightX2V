"""Checkpoint-backed CLIP text inference.

Tokenization is handled by the runner. No transformers model is instantiated.
"""

import torch
import torch.nn.functional as F

from .ops import Attention


class CLIPTextInfer:
    def __init__(self, config):
        self.config = config
        self.attention = Attention()

    def infer(self, w, ids):
        cfg = self.config
        x = w.embedding("text_model.embeddings.token_embedding").apply(ids)
        x = x + w.tensor("text_model.embeddings.position_embedding.weight")[: ids.shape[1]]
        penultimate = None
        heads = cfg["num_attention_heads"]
        b, n, c = x.shape
        for i in range(cfg["num_hidden_layers"]):
            if i == cfg["num_hidden_layers"] - 1:
                penultimate = x
            p = f"text_model.encoder.layers.{i}"
            y = w.norm(p + ".layer_norm1", cfg["layer_norm_eps"])(x)
            q, k, v = [w.linear(p + ".self_attn." + t + "_proj")(y).reshape(b, n, heads, c // heads).transpose(1, 2) for t in ("q", "k", "v")]
            y = self.attention(q, k, v, causal=True).transpose(1, 2).reshape(b, n, c)
            x = x + w.linear(p + ".self_attn.out_proj")(y)
            y = w.linear(p + ".mlp.fc1")(w.norm(p + ".layer_norm2", cfg["layer_norm_eps"])(x))
            y = y * torch.sigmoid(1.702 * y) if cfg["hidden_act"] == "quick_gelu" else F.gelu(y)
            x = x + w.linear(p + ".mlp.fc2")(y)
        x = w.norm("text_model.final_layer_norm", cfg["layer_norm_eps"])(x)
        # Legacy CLIP configs use eos_token_id=2 but tokenizer EOS=49407.
        positions = ids.argmax(-1) if cfg["eos_token_id"] == 2 else (ids == cfg["eos_token_id"]).int().argmax(-1)
        pooled = x[torch.arange(b, device=x.device), positions]
        if "text_projection.weight" in w.tensors:
            pooled = w.linear("text_projection")(pooled)
        return penultimate, pooled
