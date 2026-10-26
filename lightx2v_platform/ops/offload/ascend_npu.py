"""Ascend storage preparation for contiguous block transfers."""

import torch

from lightx2v_platform.ops.offload.template import TorchBlockOffload


class NpuBlockOffload(TorchBlockOffload):
    validate_checkpoint = True

    @staticmethod
    def prepare():
        # Byte views require ND storage, before any block buffers are created.
        torch.npu.config.allow_internal_format = False
