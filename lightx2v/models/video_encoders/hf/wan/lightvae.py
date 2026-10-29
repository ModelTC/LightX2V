import json

import torch
from safetensors import safe_open


class WanLightVAEDecoder(torch.nn.Module):
    """Adapt the pip-installed LightVAE TLR decoder to normalized Wan latents."""

    def __init__(self, vae_path, scale=None, family="wan22", dtype=torch.bfloat16, device="cuda", cpu_offload=False):
        super().__init__()
        from lightvae.modules import Decoder

        with safe_open(str(vae_path), framework="pt", device="cpu") as handle:
            config = json.loads(handle.metadata()["checkpoint_metadata"])
        if config["family"] != family:
            raise ValueError(f"{family} latents cannot use a {config['family']} LightVAE checkpoint")
        self.family = family
        self.model = Decoder(config=config, checkpoint=vae_path)
        if family == "wan22":
            self.register_buffer("mean", scale[0].detach().clone().view(1, -1, 1, 1, 1))
            self.register_buffer("inv_std", scale[1].detach().clone().view(1, -1, 1, 1, 1))
        self.dtype = dtype
        self.device = torch.device(device)
        self.cpu_offload = cpu_offload
        self.to(device="cpu" if cpu_offload else self.device, dtype=dtype)
        self.eval().requires_grad_(False)

    def compile_decoder(self):
        if not getattr(self, "_decoder_compiled", False):
            self.model = torch.compile(self.model, dynamic=None)
            self._decoder_compiled = True

    def to_cpu(self):
        self.to("cpu")

    def to_cuda(self):
        self.to(self.device)

    @torch.no_grad()
    def decode(self, latents, *, cache=None):
        if self.cpu_offload:
            self.to_cuda()
        latents = latents.to(device=self.device, dtype=self.dtype).unsqueeze(0)
        if self.family == "wan22":
            latents = latents.div(self.inv_std).add_(self.mean)
        images = self.model(latents, cache=cache).clamp_(-1, 1)
        if self.cpu_offload:
            images = images.cpu()
            self.to_cpu()
        return images

    @torch.no_grad()
    def cached_decode_withflag(self, latents, is_first_chunk, is_last_chunk):
        if is_first_chunk:
            self._decode_cache = self.model.new_cache()
        images = self.decode(latents, cache=self._decode_cache)
        if is_last_chunk:
            self._decode_cache = None
        return images

    def get_decode_state(self):
        return torch.utils._pytree.tree_map_only(torch.Tensor, torch.clone, self._decode_cache)

    def set_decode_state(self, state):
        self._decode_cache = torch.utils._pytree.tree_map_only(torch.Tensor, torch.clone, state)

    def reset_decode_state(self):
        self._decode_cache = None
