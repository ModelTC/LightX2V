import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImagePipeline, QwenImageTransformer2DModel

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
            torch_dtype=self.dtype,
        ).to(self.device)
        self.vae = AutoencoderKLQwenImage.from_pretrained(model_path, subfolder="vae").to(self.device, dtype=self.dtype)
        self.transformer = QwenImageTransformer2DModel.from_pretrained(model_path, subfolder="transformer").to(self.device, dtype=self.dtype)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.vae.requires_grad_(False)
        self.init_training_scheduler()

    def build_pipeline(self):
        pipe = QwenImagePipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_pipeline.text_encoder,
            tokenizer=self.text_pipeline.tokenizer,
            transformer=self.transformer,
        )
        return pipe

    @property
    def vae_scale_factor(self):
        return 2 ** len(self.vae.temperal_downsample)

    def encode_media(self, batch):
        images = batch["target_image"].to(device=self.device, dtype=self.dtype)
        pixel_values = images.unsqueeze(2)
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents.permute(0, 2, 1, 3, 4)

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.dtype).view(1, 1, self.vae.config.z_dim, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.dtype).view(1, 1, self.vae.config.z_dim, 1, 1)
        return (latents - latents_mean) * latents_std

    def encode_conditions(self, batch):
        prompts = batch["prompt"]
        prompt_embeds, prompt_embeds_mask = self.text_pipeline.encode_prompt(
            prompt=prompts,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=self.config["model"].get("max_sequence_length", 1024),
        )
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
        }

    def prepare_denoiser_input(self, noisy_latents, batch, conditions):
        batch_size = noisy_latents.shape[0]
        packed = QwenImagePipeline._pack_latents(
            noisy_latents,
            batch_size,
            noisy_latents.shape[2],
            noisy_latents.shape[3],
            noisy_latents.shape[4],
        )
        img_shapes = [(1, noisy_latents.shape[3] // 2, noisy_latents.shape[4] // 2)] * batch_size
        return DenoiserInput(
            hidden_states=packed,
            extra={
                "img_shapes": img_shapes,
                "height": noisy_latents.shape[3],
                "width": noisy_latents.shape[4],
            },
        )

    def denoise(self, denoiser_input, timesteps, conditions):
        return self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=conditions["prompt_embeds_mask"],
            encoder_hidden_states=conditions["prompt_embeds"],
            img_shapes=denoiser_input.extra["img_shapes"],
            return_dict=False,
        )[0]

    def unpack_prediction(self, prediction, denoiser_input):
        return QwenImagePipeline._unpack_latents(
            prediction,
            height=denoiser_input.extra["height"] * self.vae_scale_factor,
            width=denoiser_input.extra["width"] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

    def build_target(self, latents, noise):
        return (noise - latents).permute(0, 2, 1, 3, 4)
