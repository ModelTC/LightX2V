import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.seedvr.infer.transformer_infer import SeedVRTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class SeedVROffloadTransformerInfer(SeedVRTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.offload_manager = WeightAsyncStreamManager(offload_granularity="block")

    @torch.no_grad()
    def infer(self, block_weights, pre_infer_out):
        vid = pre_infer_out.vid
        txt = pre_infer_out.txt
        vid_shape = pre_infer_out.vid_shape
        txt_shape = pre_infer_out.txt_shape
        emb = pre_infer_out.emb
        cache = pre_infer_out.cache

        # Pre-infer and segment input preparation run on the caller's current
        # stream. Make the offload compute stream wait before consuming them.
        current_stream = torch_device_module.current_stream()
        self.offload_manager.compute_stream.wait_stream(current_stream)

        for block_idx in range(len(block_weights)):
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(block_weights)

            next_block_idx = (block_idx + 1) % len(block_weights)
            self.offload_manager.prefetch_weights(next_block_idx, block_weights)

            if AI_DEVICE == "xpu":
                vid, txt, vid_shape, txt_shape = self._infer_block(
                    self.offload_manager.cuda_buffers[0],
                    vid,
                    txt,
                    vid_shape,
                    txt_shape,
                    emb,
                    cache,
                )
            else:
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    vid, txt, vid_shape, txt_shape = self._infer_block(
                        self.offload_manager.cuda_buffers[0],
                        vid,
                        txt,
                        vid_shape,
                        txt_shape,
                        emb,
                        cache,
                    )

            self.offload_manager.swap_blocks()

        return vid, txt, vid_shape, txt_shape
