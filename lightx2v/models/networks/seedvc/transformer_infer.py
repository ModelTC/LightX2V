# SPDX-License-Identifier: GPL-3.0-only
# Seed-VC numerical compatibility; upstream: https://github.com/Plachtaa/seed-vc
"""Seed-VC adaptive-normalized UViT blocks using LightX2V operators."""

import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer

from .weights import project


def rotary(x, frequencies):
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cos, sin = frequencies[None, :, None, :, 0], frequencies[None, :, None, :, 1]
    real, imag = pairs[..., 0], pairs[..., 1]
    return torch.stack((real * cos - imag * sin, imag * cos + real * sin), -1).flatten(3).to(x.dtype)


def adaptive_norm(x, embedding, norm, projection):
    value = norm.apply(x)
    if embedding is None:
        return value
    weight, bias = project(projection, embedding).chunk(2, -1)
    return weight * value + bias


class SeedVCTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.cpu_offload = config["cpu_offload"]
        self.init_compile(config)
        self.blocks_executed = 0

    def get_compile_block_key(self, block_idx, block):
        return id(block) if self.cpu_offload else block_idx

    def infer(self, index, block, *args):
        self.blocks_executed += 1
        return self.run_block(index, block, *args)

    def infer_block(self, block, x, embedding, frequencies, mask, skip):
        if skip is not None:
            x = project(block.skip, torch.cat((x, skip), -1))
        embedding = None if block.time_as_token else embedding
        h = adaptive_norm(x, embedding, block.attention_norm, block.attention_adapt)
        shape = (*h.shape[:2], block.heads, block.head_dim)
        q = rotary(project(block.q, h).view(shape), frequencies)
        k = rotary(project(block.k, h).view(shape), frequencies)
        v = project(block.v, h).view(shape)
        x = x + project(block.o, block.attention.apply(q, k, v, mask))
        h = adaptive_norm(x, embedding, block.ffn_norm, block.ffn_adapt)
        return x + project(block.down, F.silu(project(block.gate, h)) * project(block.up, h))
