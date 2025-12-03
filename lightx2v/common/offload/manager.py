import torch
from packaging.version import parse

from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WeightAsyncStreamManager(object):
    def __init__(self, offload_granularity):
        self.offload_granularity = offload_granularity
        self.init_stream = torch_device_module.Stream(priority=0)
        torch_version = parse(torch.__version__.split("+")[0])
        if torch_version >= parse("2.7"):
            self.cuda_load_stream = torch_device_module.Stream(priority=1)
            self.compute_stream = torch_device_module.Stream(priority=1)
        else:
            self.cuda_load_stream = torch_device_module.Stream(priority=0)
            self.compute_stream = torch_device_module.Stream(priority=-1)

    def init_cpu_buffer(self, blocks_cpu_buffer=None, phases_cpu_buffer=None):
        if self.offload_granularity == "block":
            assert blocks_cpu_buffer is not None
            self.cpu_buffers = [blocks_cpu_buffer[i] for i in range(len(blocks_cpu_buffer))]
        elif self.offload_granularity == "phase":
            assert phases_cpu_buffer is not None
            self.cpu_buffers = [phases_cpu_buffer[i] for i in range(len(phases_cpu_buffer))]
        else:
            raise NotImplementedError

    def init_cuda_buffer(self, blocks_cuda_buffer=None, phases_cuda_buffer=None):
        if self.offload_granularity == "block":
            assert blocks_cuda_buffer is not None
            self.cuda_buffers = [blocks_cuda_buffer[i] for i in range(len(blocks_cuda_buffer))]
        elif self.offload_granularity == "phase":
            assert phases_cuda_buffer is not None
            self.cuda_buffers = [phases_cuda_buffer[i] for i in range(len(phases_cuda_buffer))]
        else:
            raise NotImplementedError

    def init_first_buffer(self, blocks, adapter_block_idx=None):
        with torch_device_module.stream(self.init_stream):
            if hasattr(self, "cpu_buffers"):
                self.cuda_buffers[0].load_state_dict(self.cpu_buffers[0].state_dict(), 0, adapter_block_idx)
            else:
                if self.offload_granularity == "block":
                    self.cuda_buffers[0].load_state_dict(blocks[0].state_dict(), 0, adapter_block_idx)
                else:
                    self.cuda_buffers[0].load_state_dict(blocks[0].compute_phases[0].state_dict(), 0, adapter_block_idx)
        self.init_stream.synchronize()

    def prefetch_weights(self, block_idx, blocks, adapter_block_idx=None):
        with torch_device_module.stream(self.cuda_load_stream):
            if hasattr(self, "cpu_buffers"):
                self.cpu_buffers[1].load_state_dict_from_disk(block_idx, adapter_block_idx)
                self.cuda_buffers[1].load_state_dict(self.cpu_buffers[1].state_dict(), block_idx, adapter_block_idx)
            else:
                self.cuda_buffers[1].load_state_dict(blocks[block_idx].state_dict(), block_idx, adapter_block_idx)

    def prefetch_phase(self, block_idx, phase_idx, blocks, adapter_block_idx=None):
        with torch_device_module.stream(self.cuda_load_stream):
            if hasattr(self, "cpu_buffers"):
                self.cpu_buffers[phase_idx].load_state_dict_from_disk(block_idx, adapter_block_idx)
                self.cuda_buffers[phase_idx].load_state_dict(self.cpu_buffers[phase_idx].state_dict(), block_idx, adapter_block_idx)
            else:
                self.cuda_buffers[phase_idx].load_state_dict(blocks[block_idx].compute_phases[phase_idx].state_dict(), block_idx, adapter_block_idx)

    def swap_blocks(self):
        self.cuda_load_stream.synchronize()
        self.compute_stream.synchronize()
        self.cuda_buffers[0], self.cuda_buffers[1] = (
            self.cuda_buffers[1],
            self.cuda_buffers[0],
        )

    def swap_phases(self):
        self.cuda_load_stream.synchronize()
        self.compute_stream.synchronize()
