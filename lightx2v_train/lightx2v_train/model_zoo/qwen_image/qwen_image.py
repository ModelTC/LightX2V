import os
from dataclasses import dataclass

import torch
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline, QwenImageTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from loguru import logger
from torch.utils.checkpoint import checkpoint

from lightx2v_train.model_capabilities import ConsistencyModelCapability, DistributionMatchingCapability, FlowMatchingSFTCapability
from lightx2v_train.model_zoo.capability_adapters import SpatialLatentGeometry
from lightx2v_train.model_zoo.capability_adapters.common import GenericDistributionMatchingCapability, GenericFlowMatchingCapability
from lightx2v_train.model_zoo.qwen_image.capability_adapters import QwenImageConsistencyModelCapability
from lightx2v_train.runtime.backend import get_backend
from lightx2v_train.runtime.fsdp import fsdp2_enabled
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import is_cache_build, is_train_cache_dataset

from ..base import BaseModel, validate_block_indices


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
    distribution_matching_uses_sample_context = False
    consistency_model_capability_cls = QwenImageConsistencyModelCapability

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
                uses_sample_context=self.distribution_matching_uses_sample_context,
            ),
        )
        self.capabilities.register(
            ConsistencyModelCapability,
            self.consistency_model_capability_cls(self),
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
            torch_dtype=self.text_encoder_dtype,
        ).to(self.device)
        self.text_pipeline.text_encoder.requires_grad_(False)
        self.text_pipeline.text_encoder.eval()

    def _load_vae(self, model_path):
        use_cpu = not is_cache_build(self.config) and is_train_cache_dataset(self.config) and self.config["inference"].get("vae_cpu_offload", False)
        device = torch.device("cpu") if use_cpu else self.device
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.vae_dtype,
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

        fsdp_config = self.config.get("distributed", {}).get("fsdp2", {})
        if fsdp_config.get("sharded_load", False):
            transformer_config = QwenImageTransformer2DModel.load_config(model_path, subfolder="transformer")
            with torch.device("meta"):
                # FSDP2 must wrap this skeleton before rank-local weights can load.
                transformer = QwenImageTransformer2DModel.from_config(transformer_config)
                self._configure_attention_backend(transformer)
                return transformer

        device = get_backend().host_device if fsdp2_enabled(self.config) else self.device
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.transformer_param_dtype,
        ).to(device)
        self._configure_attention_backend(transformer)
        return transformer

    def _configure_attention_backend(self, transformer):
        """Select the benchmark attention implementation before FSDP wrapping.

        Diffusers' Qwen processor dispatches through its attention registry.  The
        stock FlashAttention 2 entry does not expose the deterministic backward
        flag, so the benchmark installs a small wrapper that passes
        ``deterministic=True`` explicitly.  The cache used for this benchmark has
        no padding mask; an all-true mask is also treated as no mask.
        """
        backend = self.config.get("training", {}).get("attention_backend")
        if backend != "flash_deterministic":
            return

        try:
            from diffusers.models.attention_dispatch import (
                AttentionBackendName,
                _AttentionBackendRegistry,
            )
            from flash_attn import flash_attn_func
        except Exception as exc:  # pragma: no cover - exercised on GPU hosts
            raise RuntimeError("training.attention_backend=flash_deterministic requires flash-attn and Diffusers' attention dispatch registry.") from exc

        backend_name = AttentionBackendName.FLASH

        def deterministic_flash_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            window_size=(-1, -1),
            return_lse=False,
            _parallel_config=None,
        ):
            if _parallel_config is not None:
                raise RuntimeError("flash_deterministic benchmark does not support context parallelism")
            if attn_mask is not None:
                if bool(torch.all(attn_mask)):
                    attn_mask = None
                else:
                    raise RuntimeError("flash_deterministic requires an unmasked or all-true Qwen attention mask")
            output = flash_attn_func(
                q=query,
                k=key,
                v=value,
                dropout_p=dropout_p,
                softmax_scale=scale,
                causal=is_causal,
                window_size=window_size,
                deterministic=True,
                return_attn_probs=return_lse,
            )
            if return_lse:
                output, lse, *_ = output
                return output, lse
            return output

        _AttentionBackendRegistry._backends[backend_name] = deterministic_flash_attention
        for block in transformer.transformer_blocks:
            block.attn.processor._attention_backend = backend_name
        logger.info("[attention] backend=flash deterministic_backward=true")

    def load_deferred_transformer_weights(self):
        """Load Qwen weights after FSDP2 has defined the local parameter shards."""
        fsdp_config = self.config.get("distributed", {}).get("fsdp2", {})
        if not fsdp_config.get("sharded_load", False):
            return
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        self.load_sharded_transformer_weights(os.path.join(model_path, "transformer"))
        self._materialize_qwen_rope()

    def _materialize_qwen_rope(self):
        # QwenEmbedRope keeps these tables as plain tensor attributes instead
        # of registered buffers, so meta construction does not materialize them.
        rope = self.transformer.pos_embed
        if getattr(rope.pos_freqs, "is_meta", False):

            def build_freqs(index):
                values = [rope.rope_params(index, axis, rope.theta) for axis in rope.axes_dim]
                return torch.cat(values, dim=1).to(self.device)

            rope.pos_freqs = build_freqs(torch.arange(4096))
            rope.neg_freqs = build_freqs(torch.arange(4096).flip(0) * -1 - 1)

    def denoiser_module(self):
        return self.transformer

    def denoiser_prediction_type(self):
        # Qwen-Image follows x_t = (1 - t) * x_0 + t * noise and predicts
        # the corresponding velocity noise - x_0.
        return "velocity"

    def set_lora_trainable(self):
        super().set_lora_trainable()
        ops = self.config.get("runtime", {}).get("ops", {})
        attention = ops.get("attention")
        deterministic = isinstance(attention, dict) and attention.get("deterministic", False)
        fusions = ops.get("qwen_image_fusions", [])
        if (fusions or deterministic) and self.config["model"]["name"] == "qwen_image" and get_backend().name == "metax_cuda":
            from .training_fusions import configure_training_fusions

            deterministic = deterministic or self.config.get("training", {}).get("attention_backend") == "flash_deterministic"
            configure_training_fusions(self.transformer, fusions, deterministic)

    def enable_gradient_checkpointing(self):
        """Use a list of block indices or a half-open start/end range."""
        option = "gradient_checkpointing_blocks"
        indices = self.config.get("training", {}).get(option)
        blocks = self.transformer.transformer_blocks
        if isinstance(indices, dict):
            start = int(indices.get("start", 0))
            end = int(indices.get("end", len(blocks)))
            if not 0 <= start <= end <= len(blocks):
                raise ValueError(f"{option} range must satisfy 0 <= start <= end <= {len(blocks)}.")
            indices = list(range(start, end))

        if indices is None:
            selected_indices = tuple(range(len(blocks)))
            super().enable_gradient_checkpointing()
        else:
            selected_indices = tuple(sorted(validate_block_indices(indices, len(blocks), option)))
            selected = {blocks[i] for i in selected_indices}

            def checkpoint_selected(block, *args, **kwargs):
                # Both paths run Module.__call__, including FSDP's unshard hooks.
                if block in selected:
                    return checkpoint(block, *args, use_reentrant=False, **kwargs)
                return block(*args, **kwargs)

            if selected:
                self.transformer.enable_gradient_checkpointing(gradient_checkpointing_func=checkpoint_selected)
            else:
                self.transformer.disable_gradient_checkpointing()

        self.transformer._lightx2v_gradient_checkpointing_blocks = selected_indices
        logger.info("[train] gradient_checkpointing blocks={} count={} use_reentrant=false", selected_indices, len(selected_indices))

    def fsdp2_shard_plan(self, fsdp_config):
        reshard_config = fsdp_config["reshard_after_forward"]
        blocks = self.transformer.transformer_blocks
        keep = validate_block_indices(reshard_config.get("keep_unsharded_blocks", []), len(blocks), "keep_unsharded_blocks")
        plan = [{"module": block, "reshard_after_forward": False if index in keep else reshard_config["block_reshard"]} for index, block in enumerate(blocks)]
        plan.append({"module": self.transformer, "reshard_after_forward": reshard_config["root_reshard"]})
        return plan

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
        image = image.to(device=self.device, dtype=self.vae_dtype)
        pixel_values = image.unsqueeze(2)
        latent = getattr(self.vae.encode(pixel_values).latent_dist, mode)()  # (B, C, T, H, W)
        return self._normalize_latents(latent.to(dtype=self.latent_dtype))

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
        with self.transformer_forward_context():
            prediction = self.transformer(
                hidden_states=denoiser_input.hidden_states.to(dtype=self.running_dtype),
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
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.latent_dtype)

    def decode_latent(self, latent):
        latent = self._denormalize_latents(latent.to(dtype=self.latent_dtype)).to(dtype=self.vae_dtype)
        image = self.vae.decode(latent).sample  # (B, C, T, H, W)
        image = image[:, :, 0, :, :]  # drop temporal dim -> (B, C, H, W), T == 1
        return self.image_processor.postprocess(image, output_type="pil")
