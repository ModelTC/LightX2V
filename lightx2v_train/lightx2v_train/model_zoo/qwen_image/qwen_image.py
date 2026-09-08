from dataclasses import dataclass

import torch
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline, QwenImageTransformer2DModel
from diffusers.image_processor import VaeImageProcessor

from lightx2v_train.model_capabilities import ConsistencyModelCapability, DistributionMatchingCapability, FlowMatchingSFTCapability
from lightx2v_train.model_zoo.capability_adapters import SpatialLatentGeometry
from lightx2v_train.model_zoo.capability_adapters.common import GenericDistributionMatchingCapability, GenericFlowMatchingCapability
from lightx2v_train.model_zoo.qwen_image.capability_adapters import QwenImageConsistencyModelCapability
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import is_cache_build, is_train_cache_dataset

from ..base import BaseModel


@dataclass
class QwenImageDenoiserInput:
    hidden_states: torch.Tensor
    target_token_length: int
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
    distribution_matching_capability_cls = GenericDistributionMatchingCapability

    def register_capabilities(self):
        super().register_capabilities()
        self.capabilities.register(
            FlowMatchingSFTCapability,
            GenericFlowMatchingCapability(self),
        )
        self.capabilities.register(
            DistributionMatchingCapability,
            self.distribution_matching_capability_cls(
                self,
                latent_geometry=SpatialLatentGeometry(
                    channels_path="latent_channels",
                    temporal_size=1,
                ),
                guidance_in_denoiser_space=True,
            ),
        )
        self.capabilities.register(
            ConsistencyModelCapability,
            QwenImageConsistencyModelCapability(self),
        )

    def load_components(
        self,
        *,
        load_transformer,
        load_vae,
        load_condition_encoder,
    ):
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        if load_condition_encoder:
            self._load_condition_encoder(model_path)
        if load_vae:
            self._load_vae(model_path)
        else:
            self._load_vae_config(model_path)
        if load_transformer:
            self.transformer = self.load_transformer()

    def _load_condition_encoder(self, model_path):
        self.text_pipeline = self.pipeline_cls.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.running_dtype,
        ).to(self.device)
        self.text_pipeline.text_encoder.requires_grad_(False)
        self.text_pipeline.text_encoder.eval()

    def _load_vae(self, model_path):
        use_cpu = not is_cache_build(self.config) and is_train_cache_dataset(self.config) and self.config.get("inference", {}).get("vae_cpu_offload", False)
        device = torch.device("cpu") if use_cpu else self.device
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.running_dtype,
        ).to(device)
        self.vae_config = self.vae.config
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    def _load_vae_config(self, model_path):
        self.vae_config = AutoencoderKLQwenImage.load_config(model_path, subfolder="vae")
        self.vae_scale_factor = 2 ** len(self.vae_config["temperal_downsample"])

    def reuse_frozen_components_from(self, source):
        super().reuse_frozen_components_from(source)
        self.vae_scale_factor = source.vae_scale_factor

    @property
    def latent_channels(self):
        return int(self.vae_config["z_dim"] if isinstance(self.vae_config, dict) else self.vae_config.z_dim)

    def load_transformer(self):
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        return QwenImageTransformer2DModel.from_pretrained(model_path, subfolder="transformer").to(self.device, dtype=self.running_dtype)

    def load_full_weights_for_resume(self, resume_ckpt_path):
        self.transformer = QwenImageTransformer2DModel.from_pretrained(resume_ckpt_path, subfolder="transformer").to(self.device, dtype=self.running_dtype)

    def denoiser_module(self):
        return self.transformer

    def denoiser_prediction_type(self):
        # Qwen-Image follows x_t = (1 - t) * x_0 + t * noise and predicts
        # the corresponding velocity noise - x_0.
        return "velocity"

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

    def _latent_statistics(self, latents):
        config = self.vae.config
        shape = (1, config.z_dim, 1, 1, 1)
        mean = latents.new_tensor(config.latents_mean).view(shape)
        std = latents.new_tensor(config.latents_std).view(shape)
        return mean, std

    def _normalize_latents(self, latents):
        mean, std = self._latent_statistics(latents)
        return (latents - mean) / std

    def _denormalize_latents(self, latents):
        mean, std = self._latent_statistics(latents)
        return latents * std + mean

    def encode_to_latent(self, sample):
        return self._encode_target_latent(sample, mode="sample")

    def encode_to_cache_latent(self, sample):
        return self._encode_target_latent(sample, mode="mode")

    def _encode_target_latent(self, sample, *, mode):
        image = sample["inputs"]["target_pixel_values"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected target_pixel_values with shape [B, C, H, W], got {tuple(image.shape)}")
        image = image.to(device=self.device, dtype=self.running_dtype)
        pixel_values = image.unsqueeze(2)
        latent = getattr(self.vae.encode(pixel_values).latent_dist, mode)()  # (B, C, T, H, W)
        return self._normalize_latents(latent)

    def encode_condition(self, sample):
        prompt = sample["conditioning"]["prompt"]
        return self.encode_prompt_condition(prompt)

    def encode_prompt_condition(self, prompt, **kwargs):
        prompt_embed, prompt_embed_mask = self.text_pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=self.config["model"].get("max_sequence_length", 1024),
            **kwargs,
        )
        return {
            "prompt_embed": prompt_embed,
            "prompt_embed_mask": prompt_embed_mask,
        }

    def _get_additional_image_tokens(self, condition):
        return None, []

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        # noisy_latent: (B, C, T, H, W)
        n = noisy_latent.shape[0]
        h, w = noisy_latent.shape[3], noisy_latent.shape[4]
        packed = self.pipeline_cls._pack_latents(noisy_latent, n, noisy_latent.shape[1], h, w)
        additional_tokens, additional_shapes = self._get_additional_image_tokens(condition)
        hidden_states = packed if additional_tokens is None else torch.cat([packed, additional_tokens], dim=1)
        return QwenImageDenoiserInput(
            hidden_states=hidden_states,
            target_token_length=packed.shape[1],
            img_shapes=[[(1, h // 2, w // 2), *additional_shapes] for _ in range(n)],
            height=h,
            width=w,
        )

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        prediction = self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timestep_or_sigma,  # timestep_or_sigma is in [0, 1] not [0, 1000]
            guidance=None,
            encoder_hidden_states_mask=condition["prompt_embed_mask"],
            encoder_hidden_states=condition["prompt_embed"],
            img_shapes=denoiser_input.img_shapes,
            attention_kwargs={},
            return_dict=False,
        )[0]
        return prediction[:, : denoiser_input.target_token_length]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return self.pipeline_cls._unpack_latents(
            prediction,
            height=denoiser_input.height * self.vae_scale_factor,
            width=denoiser_input.width * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

    def prepare_infer_latents(self, height, width, generator=None):
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        shape = (1, self.latent_channels, 1, latent_h, latent_w)
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.running_dtype)

    def decode_latent(self, latent):
        latent = self._denormalize_latents(latent)
        image = self.vae.decode(latent).sample  # (B, C, T, H, W)
        image = image[:, :, 0, :, :]  # drop temporal dim -> (B, C, H, W), T == 1
        return self.image_processor.postprocess(image, output_type="pil")
