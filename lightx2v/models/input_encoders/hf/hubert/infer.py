"""Evaluation-only HuBERT math over native LightX2V weight operators."""

import torch
import torch.nn.functional as F

from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight


class HubertInfer:
    def __init__(self, config, layer_indices):
        self.config = config
        self.layer_indices = tuple(layer_indices)
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        self.attention = TorchSDPAWeight()

    def infer_features(self, values, weights):
        hidden = values.reshape(1, 1, -1)
        for conv, norm in weights.conv_layers:
            hidden = conv.apply(hidden)
            hidden = norm.apply(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = F.gelu(hidden)
        hidden = hidden.squeeze(0).transpose(0, 1)
        if weights.projection_norm is not None:
            hidden = weights.projection_norm.apply(hidden)
        hidden = weights.projection.apply(hidden)
        positions = weights.position_conv.apply(hidden.t().unsqueeze(0))
        if self.config["num_conv_pos_embeddings"] % 2 == 0:
            positions = positions[..., :-1]
        positions = F.gelu(positions).squeeze(0).t()
        return hidden + positions

    def infer_block(self, hidden, block):
        residual = hidden
        hidden = block.layer_norm.apply(hidden)
        query = block.q_proj.apply(hidden).reshape(-1, self.num_heads, self.head_dim)
        key = block.k_proj.apply(hidden).reshape(-1, self.num_heads, self.head_dim)
        value = block.v_proj.apply(hidden).reshape(-1, self.num_heads, self.head_dim)
        hidden = residual + block.out_proj.apply(self.attention.apply(query, key, value))
        residual = hidden
        hidden = block.final_layer_norm.apply(hidden)
        hidden = F.gelu(block.intermediate_dense.apply(hidden))
        return residual + block.output_dense.apply(hidden)

    @torch.no_grad()
    def infer(self, values, weights):
        hidden = self.infer_features(values, weights)
        selected = {0: hidden} if 0 in self.layer_indices else {}
        for index, block in enumerate(weights.blocks, start=1):
            hidden = self.infer_block(hidden, block)
            # Stable HuBERT normalizes only the final (24th) hidden state;
            # intermediate states 9--12 must remain before this final norm.
            if index == self.config["num_hidden_layers"]:
                hidden = weights.final_norm.apply(hidden)
            if index in self.layer_indices:
                selected[index] = hidden
        return torch.stack([selected[index] for index in self.layer_indices], dim=1)
