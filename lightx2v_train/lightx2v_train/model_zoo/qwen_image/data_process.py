import torch

from lightx2v_train.utils.image_ops import (
    align_dimension,
    calculate_area_dimensions,
    image_tensor_to_pil,
    resize_and_center_crop,
)

CONDITION_IMAGE_AREA = 384 * 384
VAE_IMAGE_AREA = 1024 * 1024


def _target_area_from_config(config):
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


class QwenImageDataProcessor:
    def __init__(self, image_processor, config):
        self.image_processor = image_processor
        self.target_area = _target_area_from_config(config)

    def process_target(self, sample):
        return self._process_target(sample)

    def infer_target_size(self, sample, default_height, default_width):
        height, width = _explicit_target_size(sample)
        height = default_height if height is None else height
        width = default_width if width is None else width
        multiple = int(self.image_processor.config.vae_scale_factor)
        return align_dimension(int(height), multiple), align_dimension(int(width), multiple)

    def _process_target(self, sample, reference_tensor=None, area_multiple=None):
        image = image_tensor_to_pil(sample["inputs"]["target_image"])
        reference = image if reference_tensor is None else image_tensor_to_pil(reference_tensor)
        width, height = self._resolve_target_size(
            sample,
            reference,
            self.target_area,
            area_multiple,
        )
        image = resize_and_center_crop(image, width, height)
        return self.image_processor.preprocess(image)

    def _resolve_target_size(self, sample, reference_image, target_area, area_multiple=None):
        height, width = _explicit_target_size(sample)
        multiple = int(self.image_processor.config.vae_scale_factor)
        if height is not None:
            return align_dimension(width, multiple), align_dimension(height, multiple)
        ratio = reference_image.width / reference_image.height
        return calculate_area_dimensions(target_area, ratio, area_multiple or multiple)


class QwenImageEditDataProcessor(QwenImageDataProcessor):
    def process_target(self, sample):
        source_images = sample["inputs"].get("source_images")
        reference_tensor = source_images[-1] if source_images else None
        return self._process_target(
            sample,
            reference_tensor=reference_tensor,
            area_multiple=32,
        )

    def infer_target_size(self, sample, default_height, default_width):
        source_images = sample["inputs"].get("source_images")
        if not source_images:
            return super().infer_target_size(sample, default_height, default_width)
        reference_image = image_tensor_to_pil(source_images[-1])
        width, height = self._resolve_target_size(
            sample,
            reference_image,
            int(default_height) * int(default_width),
            area_multiple=32,
        )
        return height, width

    def process_source_images(self, sample):
        source_images = self._source_images(sample)
        if not source_images:
            return None, []

        condition_images = []
        vae_images = []
        for image in source_images:
            pil_image = image_tensor_to_pil(image)
            ratio = pil_image.width / pil_image.height
            condition_width, condition_height = calculate_area_dimensions(CONDITION_IMAGE_AREA, ratio, multiple=32)
            vae_width, vae_height = calculate_area_dimensions(VAE_IMAGE_AREA, ratio, multiple=32)
            condition_images.append(self.image_processor.resize(pil_image, condition_height, condition_width))
            vae_images.append(self.image_processor.preprocess(pil_image, vae_height, vae_width).unsqueeze(2))
        return condition_images, vae_images

    @staticmethod
    def _source_images(sample):
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
