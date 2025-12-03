import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.hunyuan_video.infer.transformer_infer import HunyuanVideo15TransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE
torch_device_module = getattr(torch, AI_DEVICE)

class HunyuanVideo15OffloadTransformerInfer(HunyuanVideo15TransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                self.infer_func = self.infer_with_blocks_offload
            elif offload_granularity == "model":
                self.infer_func = self.infer_without_offload
            else:
                raise NotImplementedError
            if offload_granularity != "model":
                self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)

    @torch.no_grad()
    def infer_with_blocks_offload(self, weights, infer_module_out):
        for block_idx in range(self.double_blocks_num):
            self.block_idx = block_idx
            if block_idx == 0:
                self.offload_manager.init_first_buffer(weights.double_blocks)
            if block_idx < self.double_blocks_num - 1:
                self.offload_manager.prefetch_weights(block_idx + 1, weights.double_blocks)
            with torch_device_module.stream(self.offload_manager.compute_stream):
                infer_module_out.img, infer_module_out.txt = self.infer_double_block(self.offload_manager.cuda_buffers[0], infer_module_out)
            self.offload_manager.swap_blocks()
