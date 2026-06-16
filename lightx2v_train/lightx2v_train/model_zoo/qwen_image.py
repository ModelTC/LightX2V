from dataclasses import dataclass

import torch
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline, QwenImageTransformer2DModel
from diffusers.image_processor import VaeImageProcessor

from lightx2v_train.utils.registry import MODEL_REGISTER

from .base import BaseModel


@dataclass
class QwenImageDenoiserInput:
    hidden_states: torch.Tensor
    img_shapes: list
    height: int
    width: int


@MODEL_REGISTER("qwen_image")
class QwenImageModel(BaseModel):
    """Supports weights from these Hugging Face repos:
    - https://huggingface.co/Qwen/Qwen-Image
    - https://huggingface.co/Qwen/Qwen-Image-2512
    """

    pipeline_cls = QwenImagePipeline

    def load_components(self):
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        self.text_pipeline = QwenImagePipeline.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.running_dtype,
        ).to(self.device)
        self.vae = AutoencoderKLQwenImage.from_pretrained(model_path, subfolder="vae").to(self.device, dtype=self.running_dtype)
        self.transformer = QwenImageTransformer2DModel.from_pretrained(model_path, subfolder="transformer").to(self.device, dtype=self.running_dtype)

        self.text_pipeline.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    def denoiser_module(self):
        return self.transformer

    def fsdp2_shard_plan(self, fsdp_config):
        reshard_config = fsdp_config["reshard_after_forward"]
        return [
            {
                "modules": self.transformer.transformer_blocks,
                "reshard_after_forward": reshard_config["block_reshard"],
            },
            {
                "module": self.transformer,
                "reshard_after_forward": reshard_config["root_reshard"],
            },
        ]

    def encode_to_latent(self, sample):
        image = sample["target_image"].to(device=self.device, dtype=self.running_dtype)
        pixel_values = image.unsqueeze(2)
        latent = self.vae.encode(pixel_values).latent_dist.sample()  # (B, C, T, H, W)

        latent_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.running_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latent_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.running_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latent - latent_mean) * latent_std

    def encode_condition(self, sample):
        prompt = sample["prompt"]
        prompt_embed, prompt_embed_mask = self.text_pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=self.config["model"].get("max_sequence_length", 1024),
        )
        return {
            "prompt_embed": prompt_embed,
            "prompt_embed_mask": prompt_embed_mask,
        }

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        # noisy_latent: (B, C, T, H, W)
        n = noisy_latent.shape[0]
        h, w = noisy_latent.shape[3], noisy_latent.shape[4]
        packed = QwenImagePipeline._pack_latents(noisy_latent, n, noisy_latent.shape[1], h, w)
        return QwenImageDenoiserInput(
            hidden_states=packed,
            img_shapes=[(1, h // 2, w // 2)] * n,
            height=h,
            width=w,
        )

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        return self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timestep_or_sigma,  # timestep_or_sigma is in [0, 1] not [0, 1000]
            guidance=None,
            encoder_hidden_states_mask=condition["prompt_embed_mask"],
            encoder_hidden_states=condition["prompt_embed"],
            img_shapes=denoiser_input.img_shapes,
            return_dict=False,
        )[0]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return QwenImagePipeline._unpack_latents(
            prediction,
            height=denoiser_input.height * self.vae_scale_factor,
            width=denoiser_input.width * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

    def prepare_infer_latents(self, height, width, generator=None):
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        shape = (1, self.vae.config.z_dim, 1, latent_h, latent_w)
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.running_dtype)

    def decode_latent(self, latent):
        # Reverse the normalization from encode_to_latent:
        # encode: normalized = (raw - mean) / latents_std
        # decode: raw = normalized * latents_std + mean
        latent_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.running_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latent_std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.running_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latent = latent * latent_std + latent_mean  # (B, C, T, H, W), C == z_dim

        image = self.vae.decode(latent).sample  # (B, C, T, H, W)
        image = image[:, :, 0, :, :]  # drop temporal dim -> (B, C, H, W), T == 1

        return self.image_processor.postprocess(image, output_type="pil")

    def assemble_pipeline(self, scheduler=None):
        return QwenImagePipeline(
            tokenizer=self.text_pipeline.tokenizer,
            text_encoder=self.text_pipeline.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=scheduler or self.text_pipeline.scheduler,  # use the original scheduler for bit-exact alignment with diffusers
        ).to(self.device)

    def get_pipeline_infer_kwargs(self, infer_config):
        # QwenImagePipeline uses `true_cfg_scale` instead of the standard `guidance_scale`
        return {
            "height": infer_config.get("height", 1024),
            "width": infer_config.get("width", 1024),
            "num_inference_steps": infer_config.get("num_inference_steps", 50),
            "true_cfg_scale": infer_config.get("cfg_guidance_scale", 4.0),
        }
