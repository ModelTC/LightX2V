import os

import torch

from lightx2v.models.video_encoders.hf.autoencoder_kl_causal_3d.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from lightx2v.models.video_encoders.trt.autoencoder_kl_causal_3d import trt_vae_infer


class VideoEncoderKLCausal3DModel:
    def __init__(self, model_path, dtype, device):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.load()

    def load(self):
        self.vae_path = os.path.join(self.model_path, "hunyuan-video-t2v-720p/vae")
        config = AutoencoderKLCausal3D.load_config(self.vae_path)
        self.model = AutoencoderKLCausal3D.from_config(config)
        ckpt = torch.load(os.path.join(self.vae_path, "pytorch_model.pt"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(dtype=self.dtype, device=self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        trt_decoder = trt_vae_infer.HyVaeTrtModelInfer(engine_path=os.path.join(self.vae_path, "vae_decoder.engine"))
        self.model.decoder = trt_decoder

    def decode(self, latents, generator):
        latents = latents / self.model.config.scaling_factor
        latents = latents.to(dtype=self.dtype, device=self.device)
        self.model.enable_tiling()
        image = self.model.decode(latents, return_dict=False, generator=generator)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float()
        return image


if __name__ == "__main__":
    model_path = ""
    vae_model = VideoEncoderKLCausal3DModel(model_path, dtype=torch.float16, device=torch.device("cuda"))
