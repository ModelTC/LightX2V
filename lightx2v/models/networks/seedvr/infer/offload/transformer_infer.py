import torch

from lightx2v.models.networks.seedvr.infer.transformer_infer import SeedVRTransformerInfer


class SeedVROffloadTransformerInfer(SeedVRTransformerInfer):
    @torch.no_grad()
    def infer(self, block_weights, pre_infer_out):
        vid = pre_infer_out.vid
        txt = pre_infer_out.txt
        vid_shape = pre_infer_out.vid_shape
        txt_shape = pre_infer_out.txt_shape
        emb = pre_infer_out.emb
        cache = pre_infer_out.cache

        def infer_block(block_idx, block_weight):
            nonlocal vid, txt, vid_shape, txt_shape
            self.block_idx = block_idx
            vid, txt, vid_shape, txt_shape = self._infer_block(
                block_weight,
                vid,
                txt,
                vid_shape,
                txt_shape,
                emb,
                cache,
            )
            return vid, txt, vid_shape, txt_shape

        self.run_blocks_with_offload(block_weights, infer_block)

        return vid, txt, vid_shape, txt_shape
