# Copyright (c) Meta Platforms, Inc. and affiliates.
# See LICENSE.txt in this directory.

from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class VGGTInfer(ModuleInfer):
    def infer(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        (tokens, patch_start) = self.aggregator(images)
        poses = self.camera_head(tokens)
        (depth, depth_conf) = self.depth_head(tokens, images=images, patch_start_idx=patch_start)
        (points, points_conf) = self.point_head(tokens, images=images, patch_start_idx=patch_start)
        return {"pose_enc": poses[-1], "depth": depth, "depth_conf": depth_conf, "world_points": points, "world_points_conf": points_conf}
