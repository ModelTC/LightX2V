import torch
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline, QwenImageTransformer2DModel

from lightx2v_train.utils.registry import MODEL_REGISTER

from .base import BaseModel, DenoiserInput


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
        self.transformer = self.load_transformer()
        self.vae.requires_grad_(False)

    def load_transformer(self):
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        return QwenImageTransformer2DModel.from_pretrained(model_path, subfolder="transformer").to(self.device, dtype=self.running_dtype)

    def build_pipeline(self):
        pipe = QwenImagePipeline(
            scheduler=self.flow_matching,
            vae=self.vae,
            text_encoder=self.text_pipeline.text_encoder,
            tokenizer=self.text_pipeline.tokenizer,
            transformer=self.transformer,
        )
        return pipe

    @property
    def vae_scale_factor(self):
        return 2 ** len(self.vae.temperal_downsample)

    def encode_to_latent(self, sample):
        image = sample["target_image"].to(device=self.device, dtype=self.running_dtype)
        pixel_values = image.unsqueeze(2)
        latent = self.vae.encode(pixel_values).latent_dist.sample()
        latent = latent.permute(0, 2, 1, 3, 4)

        latent_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.running_dtype).view(1, 1, self.vae.config.z_dim, 1, 1)
        latent_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.running_dtype).view(1, 1, self.vae.config.z_dim, 1, 1)
        return (latent - latent_mean) * latent_std

    def encode_condition(self, sample):
        prompt = sample["prompt"]
        return self.encode_prompt_condition(prompt)

    def encode_prompt_condition(self, prompt):
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

    def prepare_denoiser_input(self, noisy_latent, sample, condition):
        n = noisy_latent.shape[0]
        packed = QwenImagePipeline._pack_latents(
            noisy_latent,
            n,
            noisy_latent.shape[2],
            noisy_latent.shape[3],
            noisy_latent.shape[4],
        )
        img_shapes = [(1, noisy_latent.shape[3] // 2, noisy_latent.shape[4] // 2)] * n
        return DenoiserInput(
            hidden_states=packed,
            extra={
                "img_shapes": img_shapes,
                "height": noisy_latent.shape[3],
                "width": noisy_latent.shape[4],
            },
        )

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        return self.denoise_with_transformer(self.transformer, denoiser_input, timestep_or_sigma, condition)

    def denoise_with_transformer(self, transformer, denoiser_input, timestep_or_sigma, condition):
        return transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timestep_or_sigma,  # timestep_or_sigma is in [0, 1] not [0, 1000]
            guidance=None,
            encoder_hidden_states_mask=condition["prompt_embed_mask"],
            encoder_hidden_states=condition["prompt_embed"],
            img_shapes=denoiser_input.extra["img_shapes"],
            return_dict=False,
        )[0]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return QwenImagePipeline._unpack_latents(
            prediction,
            height=denoiser_input.extra["height"] * self.vae_scale_factor,
            width=denoiser_input.extra["width"] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

    def prepare_flow_matching_target(self, velocity):
        return velocity.permute(0, 2, 1, 3, 4)
