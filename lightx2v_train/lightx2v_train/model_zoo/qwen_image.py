import copy
from dataclasses import dataclass

import torch
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline, QwenImageTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from torch import nn

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

    def load_components(self, transformer_only=False, reference_model=None):
        if transformer_only:
            if reference_model is not None:
                self.text_pipeline = reference_model.text_pipeline
                self.vae = reference_model.vae
                self.vae_scale_factor = reference_model.vae_scale_factor
                self.image_processor = reference_model.image_processor
            self.transformer = self.load_transformer()
            return
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        self.text_pipeline = QwenImagePipeline.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.running_dtype,
        ).to(self.device)
        self.vae = AutoencoderKLQwenImage.from_pretrained(model_path, subfolder="vae").to(self.device, dtype=self.running_dtype)
        self.transformer = self.load_transformer()

        self.text_pipeline.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    def load_transformer(self):
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        return QwenImageTransformer2DModel.from_pretrained(model_path, subfolder="transformer").to(self.device, dtype=self.running_dtype)

    def denoiser_module(self):
        return self.transformer

    def denoiser_prediction_type(self):
        # Qwen-Image follows x_t = (1 - t) * x_0 + t * noise and predicts
        # the corresponding velocity noise - x_0.
        return "velocity"

    def configure_consistency_model(self, capabilities):
        capabilities = frozenset(capabilities)
        unsupported = capabilities - {"endpoint_time", "log_variance"}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(f"QwenImageModel does not support consistency capabilities: {names}.")

        transformer = self.transformer
        if "log_variance" in capabilities and not hasattr(transformer, "logvar_linear"):
            transformer.logvar_linear = nn.Linear(transformer.inner_dim, 1).to(
                device=self.device,
                dtype=self.running_dtype,
            )

        if "endpoint_time" in capabilities:
            if not hasattr(transformer, "r_timestep_embedder"):
                transformer.r_timestep_embedder = copy.deepcopy(transformer.time_text_embed.timestep_embedder)
            self._consistency_endpoint_time = None
        if "endpoint_time" in capabilities and not hasattr(self, "_consistency_endpoint_hook"):
            self._consistency_endpoint_hook = transformer.time_text_embed.register_forward_hook(self._add_endpoint_time_embedding)
        self._consistency_capabilities = getattr(self, "_consistency_capabilities", frozenset()) | capabilities

    def _add_endpoint_time_embedding(self, time_embedder, inputs, output):
        endpoint_time = getattr(self, "_consistency_endpoint_time", None)
        if endpoint_time is None:
            return output
        projected = time_embedder.time_proj(endpoint_time)
        projected = projected.to(device=output.device, dtype=output.dtype)
        endpoint_embedding = self.transformer.r_timestep_embedder(projected)
        return output + endpoint_embedding.to(dtype=output.dtype)

    def set_consistency_modules_trainable(self):
        transformer = self.transformer
        for name in ("logvar_linear", "r_timestep_embedder"):
            module = getattr(transformer, name, None)
            if module is not None:
                module.requires_grad_(True)

    def consistency_auxiliary_parameter_names(self):
        prefixes = ("logvar_linear.", "r_timestep_embedder.")
        return tuple(name for name, _ in self.transformer.named_parameters() if name.startswith(prefixes))

    def predict_consistency_log_variance(self, time):
        if not hasattr(self.transformer, "logvar_linear"):
            raise RuntimeError("The Qwen consistency log-variance head has not been configured.")
        time_embedder = self.transformer.time_text_embed
        projected = time_embedder.time_proj(time)
        dtype = self.transformer.logvar_linear.weight.dtype
        embedding = time_embedder.timestep_embedder(projected.to(device=self.device, dtype=dtype))
        return self.transformer.logvar_linear(embedding)

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
        image = sample["inputs"]["target_image"].to(device=self.device, dtype=self.running_dtype)
        pixel_values = image.unsqueeze(2)
        latent = self.vae.encode(pixel_values).latent_dist.sample()  # (B, C, T, H, W)

        latent_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.running_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latent_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.running_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latent - latent_mean) * latent_std

    def encode_condition(self, sample):
        prompt = sample["conditioning"]["prompt"]
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

    def denoise(self, denoiser_input, timestep_or_sigma, condition, endpoint_time=None):
        if endpoint_time is not None and not hasattr(self.transformer, "r_timestep_embedder"):
            raise RuntimeError("endpoint_time was provided before Qwen endpoint conditioning was configured.")
        previous_endpoint = getattr(self, "_consistency_endpoint_time", None)
        self._consistency_endpoint_time = endpoint_time
        try:
            return self.transformer(
                hidden_states=denoiser_input.hidden_states,
                timestep=timestep_or_sigma,  # timestep_or_sigma is in [0, 1] not [0, 1000]
                guidance=None,
                encoder_hidden_states_mask=condition["prompt_embed_mask"],
                encoder_hidden_states=condition["prompt_embed"],
                img_shapes=denoiser_input.img_shapes,
                return_dict=False,
            )[0]
        finally:
            self._consistency_endpoint_time = previous_endpoint

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

    def dmd_latent_shape(self, batch_size, height, width):
        return (
            int(batch_size),
            int(self.vae.config.z_dim),
            1,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

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
        enable_cfg = infer_config.get("enable_cfg", True)
        kwargs = {
            "height": infer_config.get("height", infer_config.get("default_height", 1024)),
            "width": infer_config.get("width", infer_config.get("default_width", 1024)),
            "num_inference_steps": infer_config.get("num_inference_steps", 50),
            "true_cfg_scale": infer_config.get("cfg_guidance_scale", 4.0) if enable_cfg else 1.0,
        }
        sigmas = infer_config.get("sigmas")
        pcm_solver_steps = infer_config.get("pcm_solver_steps")
        if sigmas is not None and pcm_solver_steps is not None:
            raise ValueError("Set only one of inference.sigmas and inference.pcm_solver_steps.")
        if pcm_solver_steps is not None:
            from lightx2v_train.trainers.consistency.pcm import pcm_inference_sigmas

            sigmas = pcm_inference_sigmas(kwargs["num_inference_steps"], int(pcm_solver_steps))
        if sigmas is not None:
            kwargs["sigmas"] = [float(value) for value in sigmas]
        return kwargs
