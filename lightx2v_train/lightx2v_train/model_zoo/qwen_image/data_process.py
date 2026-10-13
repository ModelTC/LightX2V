import torch
from PIL import Image
from diffusers import AutoencoderKLQwenImage
from diffusers.image_processor import VaeImageProcessor

from lightx2v_train.utils.image_ops import (
    align_dimension,
    calculate_area_dimensions,
    pil_to_image_tensor,
    resize_and_center_crop,
)
from lightx2v_train.utils.registry import SAMPLE_PROCESSOR_REGISTER

CONDITION_IMAGE_AREA = 384 * 384
VAE_IMAGE_AREA = 1024 * 1024


def _size_multiple_from_config(config):
    processor_config = config.get("data", {}).get("processor", {})
    value = processor_config.get("size_multiple")
    if value is None:
        model_path = config["model"]["pretrained_model_name_or_path"]
        vae_config = AutoencoderKLQwenImage.load_config(model_path, subfolder="vae")
        value = 2 ** len(vae_config["temperal_downsample"]) * 2
    value = int(value)
    if value <= 0:
        raise ValueError(f"size_multiple must be positive, got {value}")
    return value


def _target_area_from_config(config):
    data_config = config.get("data", {})
    for split in ("train", "val"):
        if "target_area" in data_config.get(split, {}):
            return int(data_config[split]["target_area"])
    return 1024 * 1024


def _optional_scalar(mapping, key):
    value = mapping.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{key} must contain one value, got {value.numel()}")
        value = value.item()
    elif isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{key} must contain one value, got {len(value)}")
        value = value[0]
        if torch.is_tensor(value):
            value = value.item()
    return int(value)


def _explicit_target_size(sample):
    meta = sample.get("meta", {})
    height = _optional_scalar(meta, "target_height")
    width = _optional_scalar(meta, "target_width")
    if (height is None) != (width is None):
        raise ValueError("meta.target_height and meta.target_width must be provided together")
    return height, width


@SAMPLE_PROCESSOR_REGISTER("qwen_image")
class QwenImageDataProcessor:
    def __init__(self, config):
        self.image_processor = VaeImageProcessor(vae_scale_factor=_size_multiple_from_config(config))
        self.target_area = _target_area_from_config(config)
        self.unconditional_prompt = config["model"].get("unconditional_prompt", " ")

    def __call__(self, sample):
        inputs = sample["inputs"]
        image = inputs.pop("target_image", None)
        if image is not None:
            inputs["target_pixel_values"] = self._process_target(image, sample)
        return sample

    def infer_target_size(self, sample, default_height, default_width):
        height, width = _explicit_target_size(sample)
        height = default_height if height is None else height
        width = default_width if width is None else width
        multiple = int(self.image_processor.config.vae_scale_factor)
        return align_dimension(int(height), multiple), align_dimension(int(width), multiple)

    def _process_target(self, image, sample, reference_image=None, area_multiple=None):
        reference = image if reference_image is None else reference_image
        width, height = self._resolve_target_size(
            sample,
            reference.width / reference.height,
            self.target_area,
            area_multiple,
        )
        image = resize_and_center_crop(image, width, height)
        return self.image_processor.preprocess(image)[0]

    def _resolve_target_size(self, sample, ratio, target_area, area_multiple=None):
        height, width = _explicit_target_size(sample)
        multiple = int(self.image_processor.config.vae_scale_factor)
        if height is not None:
            return align_dimension(width, multiple), align_dimension(height, multiple)
        return calculate_area_dimensions(target_area, ratio, area_multiple or multiple)


@SAMPLE_PROCESSOR_REGISTER("qwen_image_edit")
class QwenImageEditDataProcessor(QwenImageDataProcessor):
    def __call__(self, sample):
        inputs = sample["inputs"]
        meta = sample["meta"]
        target_image = inputs.pop("target_image", None)
        source_images = inputs.pop("source_images", [])
        reference_image = source_images[-1] if source_images else None

        if reference_image is not None:
            meta["reference_image_height"] = reference_image.height
            meta["reference_image_width"] = reference_image.width
        if target_image is not None:
            target_pixels = self._process_target(
                target_image,
                sample,
                reference_image=reference_image,
                area_multiple=32,
            )
            inputs["target_pixel_values"] = target_pixels
            meta["target_height"] = int(target_pixels.shape[-2])
            meta["target_width"] = int(target_pixels.shape[-1])
        elif reference_image is not None:
            target_width, target_height = self._resolve_target_size(
                sample,
                reference_image.width / reference_image.height,
                self.target_area,
                area_multiple=32,
            )
            meta["target_height"] = target_height
            meta["target_width"] = target_width
        if source_images:
            condition_images, vae_images = self._process_source_images(source_images)
            inputs["source_condition_images"] = condition_images
            inputs["source_vae_pixel_values"] = vae_images
        return sample

    def infer_target_size(self, sample, default_height, default_width):
        meta = sample.get("meta", {})
        reference_height = _optional_scalar(meta, "reference_image_height")
        reference_width = _optional_scalar(meta, "reference_image_width")
        if (reference_height is None) != (reference_width is None):
            raise ValueError("meta.reference_image_height and meta.reference_image_width must be provided together")
        if reference_height is None:
            return super().infer_target_size(sample, default_height, default_width)
        width, height = self._resolve_target_size(
            sample,
            reference_width / reference_height,
            int(default_height) * int(default_width),
            area_multiple=32,
        )
        return height, width

    def _process_source_images(self, source_images):
        condition_images = []
        vae_images = []
        for image in source_images:
            ratio = image.width / image.height
            condition_width, condition_height = calculate_area_dimensions(CONDITION_IMAGE_AREA, ratio, multiple=32)
            vae_width, vae_height = calculate_area_dimensions(VAE_IMAGE_AREA, ratio, multiple=32)
            condition_image = self.image_processor.resize(image, condition_height, condition_width)
            condition_images.append(pil_to_image_tensor(condition_image))
            vae_images.append(self.image_processor.preprocess(image, vae_height, vae_width)[0].unsqueeze(1))
        return condition_images, vae_images


@SAMPLE_PROCESSOR_REGISTER("qwen_image_2_1")
class QwenImage21DataProcessor(QwenImageDataProcessor):
    """Prepare RGB/RGBA targets and optional reference images for Qwen-Image-2.1."""

    image_mode = "RGBA"

    def __init__(self, config):
        processor_config = config.get("data", {}).get("processor", {})
        size_multiple = int(processor_config.get("size_multiple", 32))
        # 16x VAE downsampling and 2x2 VLM image slots require multiples of 32.
        if size_multiple <= 0 or size_multiple % 32:
            raise ValueError("Qwen-Image-2.1 size_multiple must be a positive multiple of 32")
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=size_multiple,
            vae_latent_channels=64,
            do_convert_rgb=False,
            do_resize=False,
        )
        self.target_area = _target_area_from_config(config)
        self.reference_image_area = int(processor_config.get("reference_image_area", self.target_area))
        self.unconditional_prompt = config["model"].get("unconditional_prompt", " ")

    def __call__(self, sample):
        inputs = sample["inputs"]
        meta = sample["meta"]
        target_image = inputs.pop("target_image", None)
        source_images = inputs.pop("source_images", [])
        if target_image is not None:
            # Training follows the target image's aspect ratio.
            target_pixels = self._process_target(target_image, sample)
            inputs["target_pixel_values"] = target_pixels
            meta["target_height"] = int(target_pixels.shape[-2])
            meta["target_width"] = int(target_pixels.shape[-1])
        if source_images:
            reference_image = source_images[-1]
            meta["reference_image_height"] = reference_image.height
            meta["reference_image_width"] = reference_image.width
            condition_images, vae_images = self._process_source_images(source_images)
            inputs["source_condition_images"] = condition_images
            inputs["source_vae_pixel_values"] = vae_images
        return sample

    def infer_target_size(self, sample, default_height, default_width):
        meta = sample["meta"]
        height, _ = _explicit_target_size(sample)
        if height is not None or "reference_image_height" not in meta:
            return super().infer_target_size(sample, default_height, default_width)
        reference_height = _optional_scalar(meta, "reference_image_height")
        reference_width = _optional_scalar(meta, "reference_image_width")
        width, height = self._resolve_target_size(sample, reference_width / reference_height, int(default_height) * int(default_width))
        return height, width

    def _process_source_images(self, source_images):
        condition_images = []
        vae_images = []
        multiple = int(self.image_processor.config.vae_scale_factor)
        for image in source_images:
            width, height = calculate_area_dimensions(self.reference_image_area, image.width / image.height, multiple)
            # Both encoders share geometry; only VLM composites alpha over white.
            rgba_image = self.image_processor.resize(image.convert(self.image_mode), height, width)
            white = Image.new("RGB", rgba_image.size, (255, 255, 255))
            white.paste(rgba_image, mask=rgba_image.getchannel("A"))
            condition_images.append(pil_to_image_tensor(white))
            vae_images.append(self.image_processor.preprocess(rgba_image)[0].unsqueeze(1))
        return condition_images, vae_images

    def _process_target(self, image, sample):
        return super()._process_target(image.convert(self.image_mode), sample)
