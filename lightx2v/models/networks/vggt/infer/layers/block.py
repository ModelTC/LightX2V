# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class BlockInfer(ModuleInfer):
    def infer(self, x, pos=None):
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos))
        return x + self.ls2(self.mlp(self.norm2(x)))
