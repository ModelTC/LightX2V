import torch

from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer


class WanMtxg3PostInfer(WanPostInfer):
    """Matrix-Game-3 post-processing.

    The official MG3 model prepends memory latents before patch embedding, then
    drops those memory frames from the final model output. Keep that behavior
    local to the MG3 adapter instead of changing the shared Wan post-infer path.
    """

    @torch.no_grad()
    def infer(self, x, pre_infer_out):
        x = self.unpatchify(x, pre_infer_out.grid_sizes.tuple)

        memory_length = getattr(pre_infer_out, "memory_length", 0)
        if memory_length > 0:
            x = [u[:, memory_length:] for u in x]

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        return [u.float() for u in x]
