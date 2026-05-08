from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, LongCatImagePipeline
from diffusers.models.transformers import LongCatImageTransformer2DModel
from diffusers.pipelines.longcat_image.pipeline_longcat_image import prepare_pos_ids

from lightx2v_train.utils.registry import MODEL_REGISTER

from .base import BaseModel, DenoiserInput


@MODEL_REGISTER("longcat_image")
class LongCatImageModel(BaseModel):
    pipeline_cls = LongCatImagePipeline

    def load_components(self):
        model_path = self.config["pretrained_model_name_or_path"]
        self.text_pipeline = LongCatImagePipeline.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(self.device, dtype=self.dtype)
        self.transformer = LongCatImageTransformer2DModel.from_pretrained(model_path, subfolder="transformer").to(self.device, dtype=self.dtype)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.vae.requires_grad_(False)
        self.init_training_scheduler()

    def build_pipeline(self):
        pipe = LongCatImagePipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_pipeline.text_encoder,
            tokenizer=self.text_pipeline.tokenizer,
            text_processor=self.text_pipeline.text_processor,
            transformer=self.transformer,
        )
        return pipe

    @property
    def vae_scale_factor(self):
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    def encode_to_latent(self, sample):
        image = sample["target_image"].to(device=self.device, dtype=self.dtype)
        latent = self.vae.encode(image).latent_dist.sample()
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scale = getattr(self.vae.config, "scaling_factor", 1.0)
        return (latent - shift) * scale

    def encode_condition(self, sample):
        prompt = sample["prompt"]
        if self.config.get("enable_prompt_rewrite_training", False):
            prompt = self.text_pipeline.rewire_prompt(prompt, self.device)
        prompt_embed, text_ids = self.text_pipeline.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=1,
        )
        return {"prompt_embed": prompt_embed, "text_ids": text_ids}

    def prepare_denoiser_input(self, noisy_latent, sample, condition):
        n = noisy_latent.shape[0]
        packed = LongCatImagePipeline._pack_latents(
            noisy_latent,
            n,
            noisy_latent.shape[1],
            noisy_latent.shape[2],
            noisy_latent.shape[3],
        )
        latent_image_ids = prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(
                self.text_pipeline.tokenizer_max_length,
                self.text_pipeline.tokenizer_max_length,
            ),
            height=noisy_latent.shape[2] // 2,
            width=noisy_latent.shape[3] // 2,
        ).to(self.device)
        return DenoiserInput(
            hidden_states=packed,
            extra={
                "img_ids": latent_image_ids,
                "height": noisy_latent.shape[2],
                "width": noisy_latent.shape[3],
            },
        )

    def denoise(self, denoiser_input, timesteps, condition):
        return self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states=condition["prompt_embed"],
            txt_ids=condition["text_ids"],
            img_ids=denoiser_input.extra["img_ids"],
            return_dict=False,
        )[0]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return LongCatImagePipeline._unpack_latents(
            prediction,
            height=denoiser_input.extra["height"] * self.vae_scale_factor,
            width=denoiser_input.extra["width"] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )
