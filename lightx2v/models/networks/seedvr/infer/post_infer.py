import torch
from einops import rearrange

from lightx2v.models.networks.seedvr.dit_v2 import na
from lightx2v.models.networks.seedvr.dit_v2.ops import gather_outputs

from .utils import apply_adaln_single


class SeedVRPostInfer:
    def __init__(self, config):
        self.config = config
        self.patch_size = config.get("patch_size", (1, 2, 2))

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _patch_out(self, weights, vid, vid_shape, cache):
        cache = cache.namespace("patch")
        vid_shape_before_patchify = cache.get("vid_shape_before_patchify")

        t, h, w = self.patch_size if isinstance(self.patch_size, (list, tuple)) else (self.patch_size, self.patch_size, self.patch_size)
        vid = weights.vid_out_proj.apply(vid)
        vid = gather_outputs(vid, gather_dim=0, padding_dim=0, unpad_shape=vid_shape, cache=cache.namespace("vid"))

        if not (t == h == w == 1):
            vid = na.unflatten(vid, vid_shape)
            for i in range(len(vid)):
                vid[i] = rearrange(vid[i], "T H W (t h w c) -> (T t) (H h) (W w) c", t=t, h=h, w=w)
                if t > 1 and vid_shape_before_patchify[i, 0] % t != 0:
                    vid[i] = vid[i][(t - vid_shape_before_patchify[i, 0] % t) :]
            vid, vid_shape = na.flatten(vid)

        return vid, vid_shape

    def infer(self, weights, vid, pre_infer_out):
        cache = pre_infer_out.cache
        vid_shape = pre_infer_out.vid_shape
        emb = pre_infer_out.emb

        if hasattr(weights, "vid_out_norm"):
            vid = weights.vid_out_norm.apply(vid)
            vid_len = cache("vid_len", lambda: vid_shape.prod(-1))
            vid = apply_adaln_single(
                vid,
                emb,
                layer_idx=0,
                num_layers=1,
                mode="in",
                cache=cache,
                hid_len=vid_len,
                branch_tag="vid",
                shift=weights.vid_out_ada_out_shift.tensor,
                scale=weights.vid_out_ada_out_scale.tensor,
                gate=weights.vid_out_ada_out_scale.tensor,
            )

        vid, _ = self._patch_out(weights, vid, vid_shape, cache)
        return vid
