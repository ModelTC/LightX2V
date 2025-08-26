import os

import torch

from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D, DiagonalGaussianDistribution


class HunyuanVAE:
    def __init__(self, model_path, dtype, device, config):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.config = config
        self.load()

    def load(self):
        if self.config.task == "t2v":
            self.vae_path = os.path.join(self.model_path, "hunyuan-video-t2v-720p/vae")
        else:
            self.vae_path = os.path.join(self.model_path, "hunyuan-video-i2v-720p/vae")
        config = AutoencoderKLCausal3D.load_config(self.vae_path)
        self.model = AutoencoderKLCausal3D.from_config(config)
        ckpt = torch.load(os.path.join(self.vae_path, "pytorch_model.pt"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(dtype=self.dtype, device=self.device)
        self.model.requires_grad_(False)
        self.model.eval()

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to("cuda")

    def decode(self, latents):
        if self.config.cpu_offload:
            self.to_cuda()
        latents = latents / self.model.config.scaling_factor
        latents = latents.to(dtype=self.dtype, device=torch.device("cuda"))
        self.model.enable_tiling()
        image = self.model.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float()
        if self.config.cpu_offload:
            self.to_cpu()
        return image

    def encode(self, x):
        h = self.model.encoder(x)
        moments = self.model.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior


if __name__ == "__main__":
    model_path = ""
    vae_model = HunyuanVAE(model_path, dtype=torch.float16, device=torch.device("cuda"))
