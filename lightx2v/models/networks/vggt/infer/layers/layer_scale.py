# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# See the Apache-2.0 LICENSE in the repository root.
# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from torch import Tensor

from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class LayerScaleInfer(ModuleInfer):
    def infer(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
