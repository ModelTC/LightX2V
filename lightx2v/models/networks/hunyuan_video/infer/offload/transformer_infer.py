import torch

from lightx2v.common.offload.config import get_offload_granularity
from lightx2v.models.networks.hunyuan_video.infer.transformer_infer import HunyuanVideo15TransformerInfer


class HunyuanVideo15OffloadTransformerInfer(HunyuanVideo15TransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            offload_granularity = get_offload_granularity(self.config)
            if offload_granularity == "block":
                self.infer_func = self.infer_with_blocks_offload
            elif offload_granularity == "model":
                self.infer_func = self.infer_without_offload
            else:
                raise NotImplementedError

    @torch.no_grad()
    def infer_with_blocks_offload(self, weights, infer_module_out):
        def run_hunyuan_block(block_idx, block):
            self.block_idx = block_idx
            infer_module_out.img, infer_module_out.txt = self.infer_double_block(block, infer_module_out)
            return infer_module_out.img, infer_module_out.txt

        self.run_blocks_with_offload(weights.double_blocks, run_hunyuan_block)
