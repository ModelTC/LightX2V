from functools import cached_property

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

from lightx2v_train.model_zoo.qwen_image.capability_adapters import QwenImageEditDistributionMatchingCapability
from lightx2v_train.utils.registry import MODEL_REGISTER

from .data_process import QwenImageEditDataProcessor
from .qwen_image import QwenImageModel


@MODEL_REGISTER("qwen_image_edit")
class QwenImageEditModel(QwenImageModel):
    """Supports weights from these Hugging Face repos:
    - https://huggingface.co/Qwen/Qwen-Image-Edit-2511
    """

    pipeline_cls = QwenImageEditPlusPipeline
    distribution_matching_capability_cls = QwenImageEditDistributionMatchingCapability

    @cached_property
    def data_processor(self):
        return QwenImageEditDataProcessor(self.image_processor, self.config)

    def encode_condition(self, sample):
        prompt = sample["conditioning"]["prompt"]
        return self.encode_conditions_with_source(sample, [prompt])[0]

    def encode_conditions_with_source(self, sample, prompts):
        condition_images, vae_images = self.data_processor.process_source_images(sample)
        conditions = [self.encode_prompt_condition(prompt, image=condition_images) for prompt in prompts]
        if vae_images:
            source_latents, source_img_shapes = self._encode_source_image_latents(vae_images)
            for condition in conditions:
                condition["source_latents"] = source_latents
                condition["source_img_shapes"] = source_img_shapes
        return conditions

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
