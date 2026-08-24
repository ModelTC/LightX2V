from diffusers import LongCatImageEditPipeline
from diffusers.pipelines.longcat_image.pipeline_longcat_image import prepare_pos_ids

from lightx2v_train.model_zoo.longcat_image.capability_adapters import LongCatImageEditDistributionMatchingCapability
from lightx2v_train.utils.image_ops import image_tensor_to_pil
from lightx2v_train.utils.registry import MODEL_REGISTER

from .longcat_image import LongCatImageModel


@MODEL_REGISTER("longcat_image_edit")
class LongCatImageEditModel(LongCatImageModel):
    """Supports weights from these Hugging Face repos:
    - https://huggingface.co/meituan-longcat/LongCat-Image-Edit
    """

    pipeline_cls = LongCatImageEditPipeline
    distribution_matching_capability_cls = LongCatImageEditDistributionMatchingCapability
    supports_cfg_renorm = False
    supports_prompt_rewrite = False
    default_unconditional_prompt = ""
    shared_condition_keys = ("source_tokens", "source_height", "source_width")

    def encode_condition(self, sample):
        prompt = sample["conditioning"]["prompt"]
        return self.encode_conditions_with_source(sample, [prompt])[0]

    def encode_conditions_with_context(self, sample, prompts):
        return self.encode_conditions_with_source(sample, prompts)

    def encode_inference_condition(self, sample, *, is_negative=False):
        del is_negative
        return self.encode_condition(sample)

    def encode_conditions_with_source(self, sample, prompts):
        inputs = sample["inputs"]
        source_image = inputs.get("source_condition_image")
        source_pixels = inputs.get("source_vae_pixel_values")
        if source_image is None or source_pixels is None:
            raise ValueError("LongCat Image Edit requires source_condition_image and source_vae_pixel_values")

        source_image = image_tensor_to_pil(source_image)
        source_tokens, source_height, source_width = self._encode_source_latents(source_pixels)
        conditions = []
        for prompt in prompts:
            prompt_embed, text_ids = self.text_pipeline.encode_prompt(
                prompt=prompt,
                image=source_image,
                num_images_per_prompt=1,
            )
            conditions.append(
                {
                    "prompt_embed": prompt_embed,
                    "text_ids": text_ids,
                    "source_tokens": source_tokens,
                    "source_height": source_height,
                    "source_width": source_width,
                }
            )
        return conditions

    def _encode_source_latents(self, source_pixels):
        if source_pixels.ndim == 3:
            source_pixels = source_pixels.unsqueeze(0)
        if source_pixels.ndim != 4 or source_pixels.shape[0] != 1:
            raise ValueError(f"Expected one source image with shape [1, C, H, W], got {tuple(source_pixels.shape)}")

        source_pixels = source_pixels.to(device=self.device, dtype=self.running_dtype)
        latents = self._normalize_latents(self.vae.encode(source_pixels).latent_dist.mode())
        batch_size, channels, height, width = latents.shape
        tokens = self.pipeline_cls._pack_latents(latents, batch_size, channels, height, width)
        return tokens, height, width

    def _get_additional_image_condition(self, condition, position_start):
        source_tokens = condition.get("source_tokens")
        if source_tokens is None:
            raise ValueError("LongCat Image Edit condition is missing source_tokens")
        source_ids = prepare_pos_ids(
            modality_id=2,
            type="image",
            start=(position_start, position_start),
            height=int(condition["source_height"]) // 2,
            width=int(condition["source_width"]) // 2,
        ).to(self.device)
        return source_tokens, source_ids
