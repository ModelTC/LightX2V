import torch
import torch.nn.functional as F
from einops import rearrange
from diffusers.models.embeddings import get_timestep_embedding

from lightx2v.models.networks.seedvr.dit_v2 import na
from lightx2v.models.networks.seedvr.dit_v2.cache import Cache
from lightx2v.models.networks.seedvr.dit_v2.ops import slice_inputs

from .module_io import SeedVRPreInferOutput


class SeedVRPreInfer:
    def __init__(self, config):
        self.config = config
        self.patch_size = config.get("patch_size", (1, 2, 2))
        self.sinusoidal_dim = config.get("timestep_sinusoidal_dim", 256)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _time_embedding(self, weights, timestep, device, dtype):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(device=device, dtype=dtype)

        emb = get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=self.sinusoidal_dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        )
        emb = emb.to(dtype)
        emb = weights.emb_in_proj_in.apply(emb)
        emb = F.silu(emb)
        emb = weights.emb_in_proj_hid.apply(emb)
        emb = F.silu(emb)
        emb = weights.emb_in_proj_out.apply(emb)
        return emb

    def _patch_in(self, weights, vid, vid_shape, cache: Cache):
        if not hasattr(weights, "vid_in_proj"):
            raise RuntimeError("SeedVR pre-weights missing vid_in_proj. Check ckpt keys for vid_in.proj.*")

        cache = cache.namespace("patch")
        vid_shape_before_patchify = cache("vid_shape_before_patchify", lambda: vid_shape)
        t, h, w = self.patch_size if isinstance(self.patch_size, (list, tuple)) else (self.patch_size, self.patch_size, self.patch_size)

        if not (t == h == w == 1):
            vid = na.unflatten(vid, vid_shape)
            for i in range(len(vid)):
                if t > 1 and vid_shape_before_patchify[i, 0] % t != 0:
                    vid[i] = torch.cat([vid[i][:1]] * (t - vid[i].size(0) % t) + [vid[i]], dim=0)
                vid[i] = rearrange(vid[i], "(T t) (H h) (W w) c -> T H W (t h w c)", t=t, h=h, w=w)
            vid, vid_shape = na.flatten(vid)

        vid = slice_inputs(vid, dim=0)
        vid = weights.vid_in_proj.apply(vid)
        return vid, vid_shape

    def infer(self, weights, vid, txt, vid_shape, txt_shape, timestep, disable_cache=False):
        cache = Cache(disable=disable_cache)

        # Text input
        txt = slice_inputs(txt, dim=0)
        if hasattr(weights, "txt_in"):
            txt = weights.txt_in.apply(txt)

        # Video input (patch + linear projection)
        vid, vid_shape = self._patch_in(weights, vid, vid_shape, cache)

        # Time embedding
        emb = self._time_embedding(weights, timestep, device=vid.device, dtype=vid.dtype)

        return SeedVRPreInferOutput(
            vid=vid,
            txt=txt,
            vid_shape=vid_shape,
            txt_shape=txt_shape,
            emb=emb,
            cache=cache,
        )
