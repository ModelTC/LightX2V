import math

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

from lightx2v_train.model_zoo.qwen_image.capability_adapters import QwenImageEditDistributionMatchingCapability
from lightx2v_train.utils.registry import MODEL_REGISTER

from .qwen_image import QwenImageModel

CONDITION_IMAGE_AREA = 384 * 384
VAE_IMAGE_AREA = 1024 * 1024


def _calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


@MODEL_REGISTER("qwen_image_edit")
class QwenImageEditModel(QwenImageModel):
    """Qwen-Image-Edit 2511 LoRA training.

    This follows the local Qwen-Image-Edit-2511 model's
    QwenImageEditPlusPipeline conditioning path: source images are used by the
    Qwen2.5-VL text encoder and are also encoded as additional VAE latent
    tokens for the transformer.
    """

    pipeline_cls = QwenImageEditPlusPipeline
    distribution_matching_capability_cls = QwenImageEditDistributionMatchingCapability

    def encode_condition(self, sample):
        prompt = sample["conditioning"]["prompt"]
        return self.encode_conditions_with_source(sample, [prompt])[0]

    def encode_conditions_with_source(self, sample, prompts):
        source_images = self._source_images_from_sample(sample)
        condition_images, vae_images = self._prepare_source_images(source_images)
        conditions = [self.encode_prompt_condition(prompt, image=condition_images) for prompt in prompts]
        if vae_images:
            source_latents, source_img_shapes = self._encode_source_image_latents(vae_images)
            for condition in conditions:
                condition["source_latents"] = source_latents
                condition["source_img_shapes"] = source_img_shapes
        return conditions

    def _source_images_from_sample(self, sample):
        source_images = sample["inputs"].get("source_images")
        if source_images is None:
            return []
        tensors = []
        for image in source_images:
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"source_images must contain tensors after collation, got {type(image)}")
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4 or image.shape[0] != 1:
                raise ValueError(f"QwenImageEditPlusPipeline requires source images with batch_size=1, got {tuple(image.shape)}")
            tensors.append(image)
        return tensors

    def _prepare_source_images(self, source_images):
        if not source_images:
            return None, []

        condition_images = []
        vae_images = []
        for image in source_images:
            pil_image = self._tensor_to_pil(image[0])
            ratio = pil_image.width / pil_image.height
            condition_width, condition_height = _calculate_dimensions(CONDITION_IMAGE_AREA, ratio)
            vae_width, vae_height = _calculate_dimensions(VAE_IMAGE_AREA, ratio)
            condition_images.append(self.image_processor.resize(pil_image, condition_height, condition_width))
            vae_images.append(self.image_processor.preprocess(pil_image, vae_height, vae_width).unsqueeze(2))
        return condition_images, vae_images

    def _tensor_to_pil(self, image):
        image = ((image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte()
        array = image.permute(1, 2, 0).numpy()
        return Image.fromarray(array, mode="RGB")

    def _encode_source_image_latents(self, vae_images):
        packed_latents = []
        img_shapes = []
        for image in vae_images:
            image = image.to(device=self.device, dtype=self.running_dtype)
            latent = self.vae.encode(image).latent_dist.mode()
            latent = self._normalize_latents(latent)

            n, c, _, h, w = latent.shape
            packed_latents.append(self.pipeline_cls._pack_latents(latent, n, c, h, w))
            img_shapes.append((1, h // 2, w // 2))

        return torch.cat(packed_latents, dim=1), img_shapes

    def _get_additional_image_tokens(self, condition):
        if condition is None:
            raise ValueError("QwenImageEditModel.prepare_denoiser_input requires condition.")
        return condition.get("source_latents"), condition.get("source_img_shapes", [])

    def get_pipeline_extra_components(self):
        return {"processor": self.text_pipeline.processor}

    def get_pipeline_infer_kwargs(self, infer_config):
        kwargs = super().get_pipeline_infer_kwargs(infer_config)
        if infer_config.get("height") is None:
            kwargs.pop("height")
        if infer_config.get("width") is None:
            kwargs.pop("width")
        return kwargs

    def get_pipeline_sample_kwargs(self, sample):
        source_image_paths = sample["meta"].get("source_image_paths", [])
        if not source_image_paths:
            return {}

        images = []
        for path in source_image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        return {"image": images[0] if len(images) == 1 else images}
