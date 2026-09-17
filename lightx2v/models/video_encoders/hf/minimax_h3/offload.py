"""Two-slot offload for the native H3 video decoder, reused across tiles."""

import torch
from loguru import logger

from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.offload.module_adapter import NativeModuleBlockSlot, NativeModuleBlockSource, assign_module_tensor, module_tensors


class VideoVAEDecoderOffload:
    def __init__(self, blocks, device, *, shared):
        self.blocks = blocks
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("H3 video decoder block offload currently requires CUDA")
        if not blocks:
            raise ValueError("Video decoder must have at least one block")
        for block in blocks:
            for name, tensor in module_tensors(block).items():
                if tensor.device.type != "cpu":
                    raise ValueError("Video decoder block sources must remain on CPU")
                if shared and not tensor.is_pinned():
                    raise ValueError(f"Unregistered shared VAE source: {name}")
                if not shared and not tensor.is_pinned():
                    assign_module_tensor(block, name, tensor.detach().pin_memory())
        self.sources = [NativeModuleBlockSource(block) for block in blocks]
        schema = {n: (t.shape, t.dtype, t.stride()) for n, t in self.sources[0].state_dict().items()}
        for source in self.sources[1:]:
            if {n: (t.shape, t.dtype, t.stride()) for n, t in source.state_dict().items()} != schema:
                raise ValueError("Video decoder block layouts must match")
        self.manager = None
        self.completion = None

    def activate(self):
        if self.manager is None:
            with torch.cuda.device(self.device):
                slots = [NativeModuleBlockSlot(self.blocks[0], self.device) for _ in range(2)]
                self.manager = EventSlotWeightAsyncStreamManager("block")
                self.manager.init_cuda_buffer(blocks_cuda_buffer=slots)
            logger.info("H3 video VAE allocated two decoder slots: {:.2f} MiB", sum(s.nbytes for s in slots) / 1024**2)

    def infer(self, hidden_states, rotary_emb):
        self.activate()
        manager = self.manager
        stream = torch.cuda.current_stream(self.device)
        if self.completion is not None:
            manager.cuda_load_stream.wait_event(self.completion)
            stream.wait_event(self.completion)
        manager.reset_slots()
        try:
            manager.prefetch_to_slot(0, 0, self.sources)
            for index in range(len(self.sources)):
                slot = index % 2
                manager.wait_ready(slot, stream)
                if index + 1 < len(self.sources):
                    manager.prefetch_to_slot(1 - slot, index + 1, self.sources)
                hidden_states = manager.cuda_buffers[slot].module(hidden_states, rotary_emb)
                manager.record_free(slot, stream)
            self.completion = torch.cuda.Event()
            self.completion.record(stream)
            return hidden_states
        except BaseException:
            self.release()
            raise

    def release(self):
        if self.manager is not None:
            torch.cuda.synchronize(self.device)
            self.manager = None
            self.completion = None
