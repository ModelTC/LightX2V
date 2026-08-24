import torch
from diffusers import QwenImageEditPlusPipeline

from lightx2v_train.model_zoo.qwen_image.capability_adapters import QwenImageEditDistributionMatchingCapability
from lightx2v_train.utils.image_ops import image_tensor_to_pil
from lightx2v_train.utils.registry import MODEL_REGISTER

from .qwen_image import QwenImageModel


@MODEL_REGISTER("qwen_image_edit")
class QwenImageEditModel(QwenImageModel):
    """Supports weights from these Hugging Face repos:
    - https://huggingface.co/Qwen/Qwen-Image-Edit-2511
    """

    pipeline_cls = QwenImageEditPlusPipeline
    distribution_matching_capability_cls = QwenImageEditDistributionMatchingCapability
    shared_condition_keys = ("source_latents", "source_img_shapes")

    def encode_condition(self, sample):
        prompt = sample["conditioning"]["prompt"]
        return self.encode_conditions_with_source(sample, [prompt])[0]

    def encode_conditions_with_context(self, sample, prompts):
        return self.encode_conditions_with_source(sample, prompts)

    def encode_conditions_with_source(self, sample, prompts):
        inputs = sample["inputs"]
        condition_tensors = inputs.get("source_condition_images", [])
        condition_images = [image_tensor_to_pil(image) for image in condition_tensors] or None
        vae_images = inputs.get("source_vae_pixel_values", [])
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
            if image.ndim == 4:
                image = image.unsqueeze(0)
            if image.ndim != 5 or image.shape[0] != 1:
                raise ValueError(f"Expected one source VAE image with shape [1, C, T, H, W], got {tuple(image.shape)}")
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
