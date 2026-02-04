import torch
import numpy as np
from typing import Dict, Any, Optional

from lightx2v.disagg.services.base import BaseService
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import seed_all
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v.disagg.utils import (
    load_wan_text_encoder,
    load_wan_image_encoder,
    load_wan_vae_encoder,
    read_image_input,
)

class EncoderService(BaseService):
    def __init__(self, config):
        super().__init__(config)
        self.text_encoder = None
        self.image_encoder = None
        self.vae_encoder = None
        
        # Load models based on config
        self.load_models()
        
        # Seed everything if seed is in config
        if "seed" in self.config:
            seed_all(self.config["seed"])

    def load_models(self):
        self.logger.info("Loading Encoder Models...")
        
        # T5 Text Encoder
        text_encoders = load_wan_text_encoder(self.config)
        self.text_encoder = text_encoders[0] if text_encoders else None
        
        # CLIP Image Encoder (Optional per usage in wan_i2v.py)
        if self.config.get("use_image_encoder", False):
           self.image_encoder = load_wan_image_encoder(self.config)
        
        # VAE Encoder (Required for I2V)
        # Note: wan_i2v.py logic: if vae_encoder is None: raise RuntimeError
        # But we only load if needed or always? Let's check the config flags.
        # It seems always loaded for I2V task, but might be offloaded.
        # For simplicity of this service, we load it if the task implies it or just try to load.
        # But `load_wan_vae_encoder` will look at the config.
        self.vae_encoder = load_wan_vae_encoder(self.config)
        
        self.logger.info("Encoder Models loaded successfully.")

    def _get_latent_shape_with_lat_hw(self, latent_h, latent_w):
        return [
            self.config.get("num_channels_latents", 16),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            latent_h,
            latent_w,
        ]

    def _compute_latent_shape_from_image(self, image_tensor: torch.Tensor):
        h, w = image_tensor.shape[2:]
        aspect_ratio = h / w
        max_area = self.config["target_height"] * self.config["target_width"]

        latent_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.config["vae_stride"][1]
            // self.config["patch_size"][1]
            * self.config["patch_size"][1]
        )
        latent_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.config["vae_stride"][2]
            // self.config["patch_size"][2]
            * self.config["patch_size"][2]
        )
        latent_shape = self._get_latent_shape_with_lat_hw(latent_h, latent_w)
        return latent_shape, latent_h, latent_w

    def _get_vae_encoder_output(self, first_frame: torch.Tensor, latent_h: int, latent_w: int):
        h = latent_h * self.config["vae_stride"][1]
        w = latent_w * self.config["vae_stride"][2]

        msk = torch.ones(
            1,
            self.config["target_video_length"],
            latent_h,
            latent_w,
            device=torch.device(AI_DEVICE),
        )
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_h, latent_w)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [
                torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                torch.zeros(3, self.config["target_video_length"] - 1, h, w),
            ],
            dim=1,
        ).to(AI_DEVICE)

        vae_encoder_out = self.vae_encoder.encode(vae_input.unsqueeze(0).to(GET_DTYPE()))
        vae_encoder_out = torch.concat([msk, vae_encoder_out]).to(GET_DTYPE())
        return vae_encoder_out

    def process(self) -> Dict[str, Any]:
        """
        Generates encoder outputs from prompt and image input.
        """
        self.logger.info("Starting processing in EncoderService...")
        
        prompt = self.config.get("prompt")
        negative_prompt = self.config.get("negative_prompt")
        if prompt is None:
            raise ValueError("prompt is required in config.")

        # 1. Text Encoding
        text_len = self.config.get("text_len", 512)
        
        context = self.text_encoder.infer([prompt])
        context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])

        if self.config.get("enable_cfg", False):
            if negative_prompt is None:
                raise ValueError("negative_prompt is required in config when enable_cfg is True.")
            context_null = self.text_encoder.infer([negative_prompt])
            context_null = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context_null])
        else:
            context_null = None

        text_encoder_output = {
            "context": context,
            "context_null": context_null,
        }
        
        task = self.config.get("task")

        if task == "t2v":
            latent_h = self.config["target_height"] // self.config["vae_stride"][1]
            latent_w = self.config["target_width"] // self.config["vae_stride"][2]
            latent_shape = [
                self.config.get("num_channels_latents", 16),
                (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
                latent_h,
                latent_w,
            ]
            image_encoder_output = None
        elif task == "i2v":
            image_path = self.config.get("image_path")
            if image_path is None:
                raise ValueError("image_path is required for i2v task.")

            # 2. Image Encoding + VAE Encoding
            img, _ = read_image_input(image_path)

            clip_encoder_out = None
            if self.image_encoder is not None:
                # Assuming image_encoder.visual handles list of images
                clip_encoder_out = self.image_encoder.visual([img]).squeeze(0).to(GET_DTYPE())

            if self.vae_encoder is None:
                raise RuntimeError("VAE encoder is required but was not loaded.")

            latent_shape, latent_h, latent_w = self._compute_latent_shape_from_image(img)
            vae_encoder_out = self._get_vae_encoder_output(img, latent_h, latent_w)

            image_encoder_output = {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encoder_out,
            }
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        # Return both outputs and potentially latent_shape if needed downstream
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
            "latent_shape": latent_shape # Often needed by scheduler downstream
        }
