import torch

from lightx2v.common.offload.config import get_offload_granularity
from lightx2v.models.networks.cosmos3.infer.transformer_infer import Cosmos3TransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class Cosmos3OffloadTransformerInfer(Cosmos3TransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if not self.config.get("cpu_offload", False):
            return
        offload_granularity = get_offload_granularity(self.config)
        if offload_granularity != "block":
            raise NotImplementedError("Cosmos3 transformer supports only block-level cpu_offload.")
        self.lazy_load = self.config.get("lazy_load", False)

    def infer_layers(self, layers, und_seq, gen_seq, rotary_emb):
        if self.lazy_load:
            return self.infer_layers_with_lazy_offload(layers, und_seq, gen_seq, rotary_emb)

        def run_cosmos_block(_block_idx, block):
            nonlocal und_seq, gen_seq
            und_seq, gen_seq = self._infer_block(block, und_seq, gen_seq, rotary_emb)
            return und_seq, gen_seq

        self.run_blocks_with_offload(layers, run_cosmos_block)
        return und_seq, gen_seq

    def infer_layers_with_lazy_offload(self, layers, und_seq, gen_seq, rotary_emb):
        manager = self.get_block_offload_manager(layers)
        current_stream = torch_device_module.current_stream()
        manager.compute_stream.wait_stream(current_stream)
        for block_idx in range(len(layers)):
            if self.lazy_load:
                next_prefetch = (block_idx + 1) % len(layers)
                manager.start_prefetch_block(next_prefetch)

            if manager.need_init_first_buffer:
                manager.init_first_buffer(layers)

            if self.lazy_load:
                manager.swap_cpu_buffers()

            manager.prefetch_weights((block_idx + 1) % len(layers), layers)
            if AI_DEVICE == "xpu":
                und_seq, gen_seq = self._infer_block(
                    manager.cuda_buffers[0],
                    und_seq,
                    gen_seq,
                    rotary_emb,
                )
            else:
                with torch_device_module.stream(manager.compute_stream):
                    und_seq, gen_seq = self._infer_block(
                        manager.cuda_buffers[0],
                        und_seq,
                        gen_seq,
                        rotary_emb,
                    )
            manager.swap_blocks()
        return und_seq, gen_seq
