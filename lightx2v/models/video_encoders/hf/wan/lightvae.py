import json

import torch
from safetensors import safe_open


class Wan22LightVAEDecoder(torch.nn.Module):
    """Adapt the pip-installed LightVAE TLR decoder to normalized Wan latents."""

    def __init__(self, vae_path, scale, dtype=torch.bfloat16, device="cuda", cpu_offload=False):
        super().__init__()
        from lightvae.wan22_lightvae_decoder import Decoder

        with safe_open(str(vae_path), framework="pt", device="cpu") as handle:
            config = json.loads(handle.metadata()["checkpoint_metadata"])
        self.model = Decoder(config=config, checkpoint=vae_path)
        self.register_buffer("mean", scale[0].detach().clone().view(1, -1, 1, 1, 1))
        self.register_buffer("inv_std", scale[1].detach().clone().view(1, -1, 1, 1, 1))
        self.dtype = dtype
        self.device = torch.device(device)
        self.cpu_offload = cpu_offload
        self.to(device="cpu" if cpu_offload else self.device, dtype=dtype)
        self.eval().requires_grad_(False)

    def to_cpu(self):
        self.to("cpu")

    def to_cuda(self):
        self.to(self.device)

    @torch.no_grad()
    def decode(self, latents, *, cache=None):
        if self.cpu_offload:
            self.to_cuda()
        latents = latents.to(device=self.device, dtype=self.dtype).unsqueeze(0).div(self.inv_std).add_(self.mean)
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
