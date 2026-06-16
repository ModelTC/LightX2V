import math
from dataclasses import dataclass

import torch
from PIL import Image
from diffusers import AutoencoderKLQwenImage, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from diffusers.image_processor import VaeImageProcessor

from lightx2v_train.utils.registry import MODEL_REGISTER

from .base import BaseModel

CONDITION_IMAGE_AREA = 384 * 384


@dataclass
class QwenImageEditDenoiserInput:
    hidden_states: torch.Tensor
    target_token_length: int
    img_shapes: list
    height: int
    width: int


def _calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


@MODEL_REGISTER("qwen_image_edit")
class QwenImageEditModel(BaseModel):
    """Qwen-Image-Edit 2511 LoRA training.

    This follows the local Qwen-Image-Edit-2511 model's
    QwenImageEditPlusPipeline conditioning path: source images are used by the
    Qwen2.5-VL text encoder and are also encoded as additional VAE latent
    tokens for the transformer.
    """

    pipeline_cls = QwenImageEditPlusPipeline

    def load_components(self):
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        self.text_pipeline = QwenImageEditPlusPipeline.from_pretrained(
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
        self.latent_channels = self.vae.config.z_dim
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

    def _normalize_latents(self, latents):
        latent_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.running_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latent_std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.running_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latents - latent_mean) / latent_std

    def _denormalize_latents(self, latents):
        latent_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.running_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latent_std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.running_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return latents * latent_std + latent_mean

    def encode_to_latent(self, sample):
        image = sample["target_image"].to(device=self.device, dtype=self.running_dtype)
        pixel_values = image.unsqueeze(2)
        latent = self.vae.encode(pixel_values).latent_dist.sample()
        return self._normalize_latents(latent)

    def encode_condition(self, sample):
        source_images = self._source_images_from_sample(sample)
        condition_images = self._condition_images_from_source_tensors(source_images)
        prompt_embed, prompt_embed_mask = self.text_pipeline.encode_prompt(
            prompt=sample["prompt"],
            image=condition_images,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=self.config["model"].get("max_sequence_length", 1024),
        )
        condition = {
            "prompt_embed": prompt_embed,
            "prompt_embed_mask": prompt_embed_mask,
        }
        if source_images:
            source_latents, source_img_shapes = self._encode_source_image_latents(source_images)
            condition["source_latents"] = source_latents
            condition["source_img_shapes"] = source_img_shapes
        return condition

    def _source_images_from_sample(self, sample):
        source_images = sample.get("source_images")
        if source_images is None:
            return []
        tensors = []
        for image in source_images:
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"source_images must contain tensors after collation, got {type(image)}")
            if image.ndim == 3:
                image = image.unsqueeze(0)
            tensors.append(image)
        return tensors

    def _condition_images_from_source_tensors(self, source_images):
        if not source_images:
            return None
        batch_size = source_images[0].shape[0]
        if batch_size != 1:
            raise NotImplementedError("QwenImageEditModel currently expects batch_size=1 for source image prompt encoding.")

        condition_images = []
        for image in source_images:
            pil_image = self._tensor_to_pil(image[0])
            width, height = _calculate_dimensions(CONDITION_IMAGE_AREA, pil_image.size[0] / pil_image.size[1])
            condition_images.append(self.image_processor.resize(pil_image, height, width))
        return condition_images

    def _tensor_to_pil(self, image):
        image = ((image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte()
        array = image.permute(1, 2, 0).numpy()
        return Image.fromarray(array, mode="RGB")

    def _encode_source_image_latents(self, source_images):
        packed_latents = []
        img_shapes = []
        for image in source_images:
            image = image.to(device=self.device, dtype=self.running_dtype)
            pixel_values = image.unsqueeze(2)
            latent = self.vae.encode(pixel_values).latent_dist.mode()
            latent = self._normalize_latents(latent)

            n, c, _, h, w = latent.shape
            packed_latents.append(QwenImageEditPlusPipeline._pack_latents(latent, n, c, h, w))
            img_shapes.append((1, h // 2, w // 2))

        return torch.cat(packed_latents, dim=1), img_shapes

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        if condition is None:
            raise ValueError("QwenImageEditModel.prepare_denoiser_input requires condition.")

        n = noisy_latent.shape[0]
        h, w = noisy_latent.shape[3], noisy_latent.shape[4]
        packed = QwenImageEditPlusPipeline._pack_latents(noisy_latent, n, noisy_latent.shape[1], h, w)
        hidden_states = torch.cat([packed, condition["source_latents"]], dim=1)
        img_shapes = [[(1, h // 2, w // 2), *condition["source_img_shapes"]]] * n
        return QwenImageEditDenoiserInput(
            hidden_states=hidden_states,
            target_token_length=packed.shape[1],
            img_shapes=img_shapes,
            height=h,
            width=w,
        )

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        prompt_embed_mask = condition["prompt_embed_mask"]
        if prompt_embed_mask is None:
            txt_seq_lens = [condition["prompt_embed"].shape[1]] * condition["prompt_embed"].shape[0]
        else:
            txt_seq_lens = prompt_embed_mask.sum(dim=1).tolist()
        prediction = self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timestep_or_sigma,
            guidance=None,
            encoder_hidden_states_mask=prompt_embed_mask,
            encoder_hidden_states=condition["prompt_embed"],
            img_shapes=denoiser_input.img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs={},
            return_dict=False,
        )[0]
        return prediction[:, : denoiser_input.target_token_length]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return QwenImageEditPlusPipeline._unpack_latents(
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
        latent = self._denormalize_latents(latent)
        image = self.vae.decode(latent).sample
        image = image[:, :, 0, :, :]
        return self.image_processor.postprocess(image, output_type="pil")

    def assemble_pipeline(self, scheduler=None):
        return QwenImageEditPlusPipeline(
            tokenizer=self.text_pipeline.tokenizer,
            processor=self.text_pipeline.processor,
            text_encoder=self.text_pipeline.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=scheduler or self.text_pipeline.scheduler,
        ).to(self.device)

    def get_pipeline_infer_kwargs(self, infer_config):
        kwargs = {
            "num_inference_steps": infer_config.get("num_inference_steps", 50),
            "true_cfg_scale": infer_config.get("cfg_guidance_scale", 4.0),
        }
        if infer_config.get("height") is not None:
            kwargs["height"] = infer_config["height"]
        if infer_config.get("width") is not None:
            kwargs["width"] = infer_config["width"]
        return kwargs

    def get_pipeline_sample_kwargs(self, sample):
        source_images = sample.get("source_images", [])
        if not source_images:
            return {}

        images = []
        for path in source_images:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        return {"image": images[0] if len(images) == 1 else images}
