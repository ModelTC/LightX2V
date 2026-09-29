"""Block allocation and asynchronous copies using standard PyTorch operations."""

import torch


class TorchBlockOffload:
    validate_checkpoint = False

    @staticmethod
    def prepare():
        pass

    @staticmethod
    def allocate(nbytes, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device, pin_memory=device.type == "cpu")

    @staticmethod
    def copy(destination, source, stream):
        destination.copy_(source, non_blocking=True)
        destination.record_stream(stream)
        if source.device.type != "cpu":
            source.record_stream(stream)

    @staticmethod
    def device_module(device):
        return getattr(torch, device.type)
