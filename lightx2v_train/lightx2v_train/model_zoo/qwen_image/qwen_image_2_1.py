import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.peft_utils import get_adapter_name

from lightx2v_train.model_zoo.qwen_image.capability_adapters import QwenImage21ConsistencyModelCapability
from lightx2v_train.utils.image_ops import image_tensor_to_pil
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import is_cache_build, is_train_cache_dataset

from .qwen_image import QwenImageDenoiserInput, QwenImageModel


@MODEL_REGISTER("qwen_image_2_1")
class QwenImage21Model(QwenImageModel):
    """Qwen-Image 2.1's single-stream backbone with Qwen's shared training capabilities."""

    pipeline_cls = QwenImage21Pipeline
    distribution_matching_uses_sample_context = True
    consistency_model_capability_cls = QwenImage21ConsistencyModelCapability
    shared_condition_keys = ("source_latents", "source_img_shapes")

    def _load_vae_config(self, model_path):
        self.vae_config = AutoencoderKLQwenImage21.load_config(model_path, subfolder="vae")
        self.vae_scale_factor = int(self.vae_config["scale_factor_spatial"])

    def _load_vae(self, model_path):
        use_cpu = not is_cache_build(self.config) and is_train_cache_dataset(self.config) and self.config.get("inference", {}).get("vae_cpu_offload", False)
        device = torch.device("cpu") if use_cpu else self.device
        self.vae = AutoencoderKLQwenImage21.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.running_dtype,
        ).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_config = self.vae.config
        self.vae_scale_factor = int(self.vae_config.scale_factor_spatial)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=self.latent_channels,
        )

    def load_transformer(self):
        return QwenImage21Transformer2DModel.from_pretrained(
            self.config["model"]["pretrained_model_name_or_path"],
            subfolder="transformer",
            torch_dtype=self.running_dtype,
        ).to(self.device)

    def load_full_weights_for_resume(self, resume_ckpt_path):
        self.transformer = QwenImage21Transformer2DModel.from_pretrained(
            resume_ckpt_path,
            subfolder="transformer",
            torch_dtype=self.running_dtype,
        ).to(self.device)

    def add_lora(self, rank, alpha, target_modules):
        super().add_lora(rank, alpha, target_modules or ["to_q", "to_k", "to_v", "to_out.0"])

    def load_lora_for_infer(self, lora_path, adapter_name=None):
        denoiser = self.denoiser_module()
        adapter_name = adapter_name or get_adapter_name(denoiser)
        state_dict, metadata = self.pipeline_cls.lora_state_dict(lora_path, use_safetensors=True, return_lora_metadata=True)
        denoiser.load_lora_adapter(state_dict, adapter_name=adapter_name, metadata=metadata)
        self._infer_lora_adapter_name = adapter_name

    def _encode_target_latent(self, sample, *, mode):
        self._validate_image_size(*sample["inputs"]["target_pixel_values"].shape[-2:])
        return super()._encode_target_latent(sample, mode=mode)

    def encode_condition(self, sample):
        return self.encode_conditions_with_context(sample, [sample["conditioning"]["prompt"]])[0]

    def encode_conditions_with_context(self, sample, prompts):
        inputs = sample["inputs"]
        condition_images = [image_tensor_to_pil(image) for image in inputs.get("source_condition_images", [])]
        vae_images = inputs.get("source_vae_pixel_values", [])
        role_prompts = []
        for prompt in prompts:
            if not isinstance(prompt, str):
                (prompt,) = prompt
            role_prompts.append(prompt)
        # Share references and padding across prompt branches.
        encoded = self.encode_prompt_condition(role_prompts, image=condition_images or None)
        conditions = [{key: value[index : index + 1] for key, value in encoded.items()} for index in range(len(role_prompts))]
        if vae_images:
            source_latents, source_img_shapes = self._encode_source_image_latents(vae_images)
            for condition in conditions:
                condition["source_latents"] = source_latents
                condition["source_img_shapes"] = source_img_shapes
                self._reference_image_mask(condition, source_latents.shape[1])
        return conditions

    def _encode_source_image_latents(self, vae_images):
        packed_latents, img_shapes = [], []
        for image in vae_images:
            if image.ndim == 4:
                image = image.unsqueeze(0)
            self._validate_image_size(*image.shape[-2:])
            image = image.to(device=self.device, dtype=self.running_dtype)
            latent = self._normalize_latents(self.vae.encode(image).latent_dist.mode())
            batch, channels, _, height, width = latent.shape
            packed_latents.append(latent.reshape(batch, channels, height * width).transpose(1, 2))
            img_shapes.append((1, height, width))
        return torch.cat(packed_latents, dim=1), img_shapes

    def encode_prompt_condition(self, prompt, **kwargs):
        # The 2.1 pipeline takes Qwen3-VL features before its final RMSNorm.
        prompt_embed, prompt_embed_mask, image_pad_mask = self.text_pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            **kwargs,
        )
        # Keep cached conditions tensor-valued even when there is no padding.
        if prompt_embed_mask is None:
            prompt_embed_mask = torch.ones(prompt_embed.shape[:2], dtype=torch.bool, device=prompt_embed.device)
        return {"prompt_embed": prompt_embed, "prompt_embed_mask": prompt_embed_mask, "image_pad_mask": image_pad_mask}

    def _reference_image_mask(self, condition, source_token_length):
        embeds = condition["prompt_embed"]
        mask = condition.get("image_pad_mask")
        if mask is None:
            if source_token_length:
                raise ValueError("Editing caches require image_pad_mask; rebuild the cache.")
            # Previously built T2I caches contain only text embeddings.
            mask = torch.zeros(embeds.shape[:2], dtype=torch.bool, device=embeds.device)
        if not torch.all(mask.sum(dim=1) * 4 == source_token_length):
            raise ValueError("Reference image slots must expand to the VAE token count; check vision/VAE image sizes.")
        if not torch.equal(mask, mask[:1].expand_as(mask)):
            raise ValueError("Image-token layouts must match within a batch; use batch_size=1 for mixed tasks.")
        return mask

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        batch, channels, _, height, width = noisy_latent.shape
        if height % 2 or width % 2:
            raise ValueError("Qwen-Image 2.1 requires even latent height and width (image dimensions divisible by 32).")
        # Each 64-channel spatial position is one token.
        packed = noisy_latent.reshape(batch, channels, height * width).transpose(1, 2)
        source_latents = None if condition is None else condition.get("source_latents")
        source_shapes = [] if condition is None else condition.get("source_img_shapes", [])
        if source_latents is not None:
            if any(t != 1 or h <= 0 or w <= 0 or h % 2 or w % 2 for t, h, w in source_shapes):
                raise ValueError("Qwen-Image 2.1 reference shapes must be (1, even_height, even_width).")
            if sum(t * h * w for t, h, w in source_shapes) != source_latents.shape[1]:
                raise ValueError("Qwen-Image 2.1 reference latent tokens do not match source_img_shapes.")
            packed = torch.cat([source_latents.to(packed), packed], dim=1)
        elif source_shapes:
            raise ValueError("Qwen-Image 2.1 source_img_shapes requires source_latents.")
        return QwenImageDenoiserInput(
            hidden_states=packed,
            target_token_length=height * width,
            img_shapes=[[*source_shapes, (1, height, width)] for _ in range(batch)],
            height=height,
            width=width,
        )

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        embeds = condition["prompt_embed"]
        batch = embeds.shape[0]
        source_token_length = denoiser_input.hidden_states.shape[1] - denoiser_input.target_token_length
        image_pad_mask = self._reference_image_mask(condition, source_token_length)
        # Each image slot expands to four latent tokens inside the transformer.
        img_mask = torch.cat(
            [
                image_pad_mask,
                torch.ones((batch, denoiser_input.target_token_length // 4), dtype=torch.bool, device=embeds.device),
            ],
            dim=1,
        )
        prediction = self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timestep_or_sigma.reshape(-1).expand(batch),
            encoder_hidden_states=embeds,
            encoder_hidden_states_mask=condition.get("prompt_embed_mask"),
            img_shapes=denoiser_input.img_shapes,
            img_mask=img_mask,
            attention_kwargs={},
            # Recompute the trainable prefix each step to preserve its gradients.
            kv_cache=None,
            kv_cache_mode=None,
            return_dict=False,
        )[0]
        # 2.1 returns the joint text/image sequence, with target image tokens last.
        return prediction[:, -denoiser_input.target_token_length :]

    def _validate_image_size(self, height, width):
        multiple = self.vae_scale_factor * 2
        if height <= 0 or width <= 0 or height % multiple or width % multiple:
            raise ValueError(f"Qwen-Image 2.1 image dimensions must be positive multiples of {multiple}, got {height}x{width}.")

    def prepare_infer_latents(self, height, width, generator=None):
        self._validate_image_size(height, width)
        return super().prepare_infer_latents(height, width, generator)
