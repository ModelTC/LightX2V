# Copyright (c) Meta Platforms, Inc. and affiliates.
# See lightx2v/models/networks/vggt/LICENSE.txt.

import torch

from lightx2v.common.transformer_infer.module_infer import ModuleInfer


def _expand_tokens(tokens, batch, views):
    first = tokens[:, :1].expand(batch, 1, *tokens.shape[2:])
    others = tokens[:, 1:].expand(batch, views - 1, *tokens.shape[2:])
    return torch.cat((first, others), dim=1).reshape(batch * views, *tokens.shape[2:])


class AggregatorInfer(ModuleInfer):
    def infer(self, images):
        (batch, views, channels, height, width) = images.shape
        images = ((images - self._resnet_mean) / self._resnet_std).view(batch * views, channels, height, width)
        patches = self.patch_embed(images)["x_norm_patchtokens"]
        camera = _expand_tokens(self.camera_token, batch, views)
        registers = _expand_tokens(self.register_token, batch, views)
        tokens = torch.cat((camera, registers, patches), dim=1)
        (count, dim) = tokens.shape[1:]
        positions = self.position_getter(batch * views, height // self.patch_size, width // self.patch_size, images.device) + 1
        special = positions.new_zeros((batch * views, self.patch_start_idx, 2))
        frame_positions = torch.cat((special, positions), dim=1)
        global_positions = frame_positions.view(batch, views * count, 2)
        output = []
        for index, (frame, global_block) in enumerate(zip(self.frame_blocks, self.global_blocks)):
            tokens = frame(tokens.reshape(batch * views, count, dim), pos=frame_positions)
            frame_features = tokens.view(batch, views, count, dim)
            tokens = global_block(tokens.reshape(batch, views * count, dim), pos=global_positions)
            output.append(torch.cat((frame_features, tokens.view(batch, views, count, dim)), dim=-1) if index in self.cached_layers else None)
        return (output, self.patch_start_idx)
