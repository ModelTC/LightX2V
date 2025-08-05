import glob
import os

import torch  # type: ignore
from diffusers.video_processor import VideoProcessor  # type: ignore
from safetensors import safe_open  # type: ignore

from lightx2v.models.video_encoders.hf.cogvideox.autoencoder_ks_cogvidex import AutoencoderKLCogVideoX


class CogvideoxVAE:
    def __init__(self, config):
        self.config = config
        self.load()

    def _load_safetensor_to_dict(self, file_path):
        with safe_open(file_path, framework="pt") as f:
            tensor_dict = {key: f.get_tensor(key).to(torch.bfloat16).cuda() for key in f.keys()}
        return tensor_dict

    def _load_ckpt(self, model_path):
        safetensors_pattern = os.path.join(model_path, "*.safetensors")
        safetensors_files = glob.glob(safetensors_pattern)

        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in directory: {model_path}")
        weight_dict = {}
        for file_path in safetensors_files:
            file_weights = self._load_safetensor_to_dict(file_path)
            weight_dict.update(file_weights)
        return weight_dict

    def load(self):
        vae_path = os.path.join(self.config.model_path, "vae")
        self.vae_config = AutoencoderKLCogVideoX.load_config(vae_path)
        self.model = AutoencoderKLCogVideoX.from_config(self.vae_config)
        vae_ckpt = self._load_ckpt(vae_path)
        self.vae_scale_factor_spatial = 2 ** (len(self.vae_config["block_out_channels"]) - 1)  # 8
        self.vae_scale_factor_temporal = self.vae_config["temporal_compression_ratio"]  # 4
        self.vae_scaling_factor_image = self.vae_config["scaling_factor"]  # 0.7
        self.model.load_state_dict(vae_ckpt)
        self.model.to(torch.bfloat16).to(torch.device("cuda"))
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @torch.no_grad()
    def decode(self, latents, generator, config):
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = 1 / self.config.vae_scaling_factor_image * latents
        frames = self.model.decode(latents).sample
        images = self.video_processor.postprocess_video(video=frames, output_type="pil")[0]
        return images
