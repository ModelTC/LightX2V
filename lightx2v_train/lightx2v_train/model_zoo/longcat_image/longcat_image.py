from dataclasses import dataclass

import torch
from diffusers import AutoencoderKL, LongCatImagePipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.transformers import LongCatImageTransformer2DModel
from diffusers.pipelines.longcat_image.pipeline_longcat_image import prepare_pos_ids

from lightx2v_train.model_capabilities import ConsistencyModelCapability, DistributionMatchingCapability, FlowMatchingSFTCapability
from lightx2v_train.model_zoo.capability_adapters import SpatialLatentGeometry
from lightx2v_train.model_zoo.capability_adapters.common import GenericDistributionMatchingCapability, GenericFlowMatchingCapability
from lightx2v_train.model_zoo.longcat_image.capability_adapters import LongCatImageConsistencyModelCapability
from lightx2v_train.utils.registry import MODEL_REGISTER

from ..base import BaseModel


@dataclass
class LongCatImageDenoiserInput:
    hidden_states: torch.Tensor
    img_ids: torch.Tensor
    target_token_length: int
    height: int
    width: int


@MODEL_REGISTER("longcat_image")
class LongCatImageModel(BaseModel):
    """Supports weights from these Hugging Face repos:
    - https://huggingface.co/meituan-longcat/LongCat-Image
    - https://huggingface.co/meituan-longcat/LongCat-Image-Dev
    """

    pipeline_cls = LongCatImagePipeline
    distribution_matching_capability_cls = GenericDistributionMatchingCapability
    supports_cfg_renorm = True
    supports_prompt_rewrite = True

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
                ),
                guidance_in_denoiser_space=True,
            ),
        )
        self.capabilities.register(
            ConsistencyModelCapability,
            LongCatImageConsistencyModelCapability(self),
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
            self._maybe_set_attention_backend()

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
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.running_dtype,
        ).to(self.device)
        self.vae_config = self.vae.config
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    def _load_vae_config(self, model_path):
        self.vae_config = AutoencoderKL.load_config(model_path, subfolder="vae")

    @property
    def latent_channels(self):
        return int(self.vae_config["latent_channels"] if isinstance(self.vae_config, dict) else self.vae_config.latent_channels)

    def load_transformer(self, model_path=None):
        model_path = model_path or self.config["model"]["pretrained_model_name_or_path"]
        return LongCatImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.running_dtype,
        ).to(self.device)

    def _maybe_set_attention_backend(self):
        attention_backend = self.config["model"].get("attention_backend")
        if attention_backend is not None:
            self.transformer.set_attention_backend(attention_backend)

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
                "modules": self.transformer.single_transformer_blocks,
                "reshard_after_forward": reshard_config["block_reshard"],
            },
            {
                "module": self.transformer,
                "reshard_after_forward": reshard_config["root_reshard"],
            },
        ]

    @property
    def vae_scale_factor(self):
        config = self.vae_config
        block_out_channels = config["block_out_channels"] if isinstance(config, dict) else config.block_out_channels
        return 2 ** (len(block_out_channels) - 1)

    def _normalize_latents(self, latents):
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scale = getattr(self.vae.config, "scaling_factor", 1.0)
        return (latents - shift) * scale

    def _denormalize_latents(self, latents):
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scale = getattr(self.vae.config, "scaling_factor", 1.0)
        return latents / scale + shift

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
        latent = getattr(self.vae.encode(image).latent_dist, mode)()
        return self._normalize_latents(latent)

    def encode_condition(self, sample):
        prompt = sample["conditioning"]["prompt"]
        if self.config["model"].get("enable_prompt_rewrite_training", False):
            prompt = self.text_pipeline.rewire_prompt(prompt, self.device)
        return self.encode_prompt_condition(prompt)

    def encode_inference_condition(self, sample, *, is_negative=False):
        prompt = sample["conditioning"]["prompt"]
        rewrite = self.config.get("inference", {}).get("enable_prompt_rewrite", True)
        if self.supports_prompt_rewrite and rewrite and not is_negative:
            prompt = self.text_pipeline.rewire_prompt(prompt, self.device)
        return self.encode_prompt_condition(prompt)

    def encode_prompt_condition(self, prompt):
        prompt_embed, text_ids = self.text_pipeline.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=1,
        )
        return {"prompt_embed": prompt_embed, "text_ids": text_ids}

    def _get_additional_image_condition(self, condition, position_start):
        del condition, position_start
        return None, None

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        if condition is None:
            raise ValueError(f"{type(self).__name__}.prepare_denoiser_input requires condition")

        batch_size, channels, height, width = noisy_latent.shape
        target_tokens = self.pipeline_cls._pack_latents(noisy_latent, batch_size, channels, height, width)
        position_start = int(condition["prompt_embed"].shape[1])
        target_ids = prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(position_start, position_start),
            height=height // 2,
            width=width // 2,
        ).to(self.device)

        additional_tokens, additional_ids = self._get_additional_image_condition(condition, position_start)
        hidden_states = target_tokens
        img_ids = target_ids
        if additional_tokens is not None:
            if additional_ids is None:
                raise ValueError("Additional LongCat image tokens require matching image IDs")
            if additional_tokens.shape[0] != batch_size or additional_tokens.shape[2] != target_tokens.shape[2]:
                raise ValueError(f"Additional LongCat image tokens must match the target batch and channel dimensions, got {tuple(additional_tokens.shape)} and {tuple(target_tokens.shape)}")
            hidden_states = torch.cat([target_tokens, additional_tokens], dim=1)
            img_ids = torch.cat([target_ids, additional_ids], dim=0)

        return LongCatImageDenoiserInput(
            hidden_states=hidden_states,
            img_ids=img_ids,
            target_token_length=target_tokens.shape[1],
            height=height,
            width=width,
        )

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        prediction = self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timestep_or_sigma,
            guidance=None,
            encoder_hidden_states=condition["prompt_embed"],
            txt_ids=condition["text_ids"],
            img_ids=denoiser_input.img_ids,
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

    def apply_cfg(self, positive, negative, guidance_scale):
        prediction = super().apply_cfg(positive, negative, guidance_scale)
        infer_config = self.config.get("inference", {})
        if not self.supports_cfg_renorm or not infer_config.get("enable_cfg_renorm", True):
            return prediction

        positive_norm = torch.norm(positive, dim=-1, keepdim=True)
        prediction_norm = torch.norm(prediction, dim=-1, keepdim=True)
        minimum = float(infer_config.get("cfg_renorm_min", 0.0))
        scale = (positive_norm / prediction_norm.clamp_min(1e-8)).clamp(min=minimum, max=1.0)
        return prediction * scale

    def prepare_infer_latents(self, height, width, generator=None):
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (1, self.latent_channels, latent_height, latent_width)
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.running_dtype)

    def decode_latent(self, latent):
        image = self.vae.decode(self._denormalize_latents(latent)).sample
        return self.image_processor.postprocess(image, output_type="pil")
