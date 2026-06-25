import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.cosmos3.infer.transformer_infer import Cosmos3TransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class Cosmos3OffloadTransformerInfer(Cosmos3TransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if not self.config.get("cpu_offload", False):
            return
        offload_granularity = self.config.get("offload_granularity", "block")
        if offload_granularity != "block":
            raise NotImplementedError("Cosmos3 transformer supports only block-level cpu_offload.")
        self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)
        self.lazy_load = self.config.get("lazy_load", False)
        if self.lazy_load:
            self.offload_manager.init_lazy_load(num_workers=self.config.get("num_disk_workers", 4))

    def infer_layers(self, layers, und_seq, gen_seq, rotary_emb):
        current_stream = torch_device_module.current_stream()
        self.offload_manager.compute_stream.wait_stream(current_stream)
        for block_idx in range(len(layers)):
            if self.lazy_load:
                next_prefetch = (block_idx + 1) % len(layers)
                self.offload_manager.start_prefetch_block(next_prefetch)

            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(layers)

            if self.lazy_load:
                self.offload_manager.swap_cpu_buffers()

            self.offload_manager.prefetch_weights((block_idx + 1) % len(layers), layers)
            if AI_DEVICE == "xpu":
                und_seq, gen_seq = self._infer_block(
                    self.offload_manager.cuda_buffers[0],
                    und_seq,
                    gen_seq,
                    rotary_emb,
                )
            else:
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    und_seq, gen_seq = self._infer_block(
                        self.offload_manager.cuda_buffers[0],
                        und_seq,
                        gen_seq,
                        rotary_emb,
                    )
            self.offload_manager.swap_blocks()
        return und_seq, gen_seq
