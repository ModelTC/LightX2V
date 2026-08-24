import math

import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

from lightx2v_train.utils.image_ops import (
    align_dimension,
    calculate_area_dimensions,
    pil_to_image_tensor,
    resize_and_center_crop,
)
from lightx2v_train.utils.registry import SAMPLE_PROCESSOR_REGISTER


def _size_multiple_from_config(config):
    processor_config = config.get("data", {}).get("processor", {})
    preprocessing = config.get("model", {}).get("input_preprocessing", {})
    value = processor_config.get("size_multiple", preprocessing.get("size_multiple"))
    if value is None:
        model_path = config["model"]["pretrained_model_name_or_path"]
        vae_config = AutoencoderKL.load_config(model_path, subfolder="vae")
        value = 2 ** (len(vae_config["block_out_channels"]) - 1) * 2
    value = int(value)
    if value <= 0:
        raise ValueError(f"size_multiple must be positive, got {value}")
    return value


def _target_area_from_config(config):
    processor_config = config.get("data", {}).get("processor", {})
    if "target_area" in processor_config:
        return int(processor_config["target_area"])
    preprocessing = config.get("model", {}).get("input_preprocessing", {})
    if "target_area" in preprocessing:
        return int(preprocessing["target_area"])
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


def _align_up(value, multiple):
    if value <= 0:
        raise ValueError(f"Image dimensions must be positive, got {value}")
    return math.ceil(value / multiple) * multiple


def _calculate_edit_dimensions(target_area, ratio, multiple):
    if target_area <= 0:
        raise ValueError(f"target_area must be positive, got {target_area}")
    if ratio <= 0:
        raise ValueError(f"Image aspect ratio must be positive, got {ratio}")
    width = _align_up(math.sqrt(target_area * ratio), multiple)
    height = _align_up(math.sqrt(target_area / ratio), multiple)
    return width, height


@SAMPLE_PROCESSOR_REGISTER("longcat_image")
class LongCatImageDataProcessor:
    def __init__(self, config):
        self.image_processor = VaeImageProcessor(vae_scale_factor=_size_multiple_from_config(config))
        self.target_area = _target_area_from_config(config)
        self.unconditional_prompt = config["model"].get("unconditional_prompt", " ")

    def __call__(self, sample):
        inputs = sample["inputs"]
        image = inputs.pop("target_image", None)
        if image is None:
            return sample

        height, width = _explicit_target_size(sample)
        multiple = int(self.image_processor.config.vae_scale_factor)
        if height is None:
            ratio = image.width / image.height
            width, height = calculate_area_dimensions(self.target_area, ratio, multiple)
        else:
            width = align_dimension(width, multiple)
            height = align_dimension(height, multiple)
        image = resize_and_center_crop(image, width, height)
        inputs["target_pixel_values"] = self.image_processor.preprocess(image)[0]
        return sample

    def infer_target_size(self, sample, default_height, default_width):
        height, width = _explicit_target_size(sample)
        height = default_height if height is None else height
        width = default_width if width is None else width
        multiple = int(self.image_processor.config.vae_scale_factor)
        return align_dimension(int(height), multiple), align_dimension(int(width), multiple)


@SAMPLE_PROCESSOR_REGISTER("longcat_image_edit")
class LongCatImageEditDataProcessor(LongCatImageDataProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.unconditional_prompt = config["model"].get("unconditional_prompt", "")

    def __call__(self, sample):
        inputs = sample["inputs"]
        target_image = inputs.pop("target_image", None)
        source_images = inputs.pop("source_images", [])
        if len(source_images) != 1:
            raise ValueError(f"LongCat Image Edit requires exactly one source image, got {len(source_images)}")

        source_image = source_images[0]
        height, width = self._resolve_edit_size(sample, source_image)
        sample["meta"]["target_height"] = height
        sample["meta"]["target_width"] = width
        sample["meta"]["reference_image_height"] = source_image.height
        sample["meta"]["reference_image_width"] = source_image.width

        source_image = self.image_processor.resize(source_image, height, width)
        prompt_image = self.image_processor.resize(source_image, height // 2, width // 2)
        inputs["source_condition_image"] = pil_to_image_tensor(prompt_image)
        inputs["source_vae_pixel_values"] = self.image_processor.preprocess(source_image)[0]

        if target_image is not None:
            target_image = resize_and_center_crop(target_image, width, height)
            inputs["target_pixel_values"] = self.image_processor.preprocess(target_image)[0]
        return sample

    def infer_target_size(self, sample, default_height, default_width):
        del default_height, default_width
        height, width = _explicit_target_size(sample)
        if height is None:
            raise ValueError("LongCat Image Edit requires source-derived target dimensions")
        return height, width

    def _resolve_edit_size(self, sample, source_image):
        height, width = _explicit_target_size(sample)
        multiple = int(self.image_processor.config.vae_scale_factor)
        if height is not None:
            return _align_up(height, multiple), _align_up(width, multiple)
        ratio = source_image.width / source_image.height
        width, height = _calculate_edit_dimensions(self.target_area, ratio, multiple)
        return height, width
