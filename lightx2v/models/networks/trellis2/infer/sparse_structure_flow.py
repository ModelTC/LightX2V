# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import numpy as np
import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class TimestepEmbedderInfer(ModuleInfer):
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def infer(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModelInfer(ModuleInfer):
    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    def infer(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        if self.pe_mode == "ape":
            h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.to(dtype=self.dtype)
        h = h.to(dtype=self.dtype)
        cond = cond.to(dtype=self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond, self.rope_phases)
        h = h.to(dtype=x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3).contiguous()
        return h
