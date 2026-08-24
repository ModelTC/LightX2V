import torch
from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from lightx2v_train.utils.image_ops import align_dimension, calculate_area_dimensions, resize_and_center_crop
from lightx2v_train.utils.registry import SAMPLE_PROCESSOR_REGISTER


def _size_multiple_from_config(config):
    processor_config = config.get("data", {}).get("processor", {})
    value = processor_config.get("size_multiple")
    if value is None:
        model_path = config["model"]["pretrained_model_name_or_path"]
        vae_config = AutoencoderKLFlux2.load_config(model_path, subfolder="vae")
        value = 2 ** (len(vae_config["block_out_channels"]) - 1) * 2
    value = int(value)
    if value <= 0:
        raise ValueError(f"size_multiple must be positive, got {value}")
    return value


def _target_area_from_config(config):
    processor_config = config.get("data", {}).get("processor", {})
    return int(processor_config.get("target_area", 1024 * 1024))


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


@SAMPLE_PROCESSOR_REGISTER("flux2_dev")
@SAMPLE_PROCESSOR_REGISTER("flux2_klein")
class Flux2DataProcessor:
    def __init__(self, config):
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=_size_multiple_from_config(config))
        self.target_area = _target_area_from_config(config)
        self.unconditional_prompt = config["model"].get("unconditional_prompt", "")

    def __call__(self, sample):
        inputs = sample["inputs"]
        image = inputs.pop("target_image", None)
        if image is not None:
            inputs["target_pixel_values"] = self._process_target(image, sample)
        return sample

    def _process_target(self, image, sample, fallback_size=None):
        height, width = _explicit_target_size(sample)
        multiple = int(self.image_processor.config.vae_scale_factor)
        if height is not None:
            height = align_dimension(height, multiple)
            width = align_dimension(width, multiple)
        elif fallback_size is not None:
            height, width = fallback_size
        else:
            width, height = calculate_area_dimensions(
                self.target_area,
                image.width / image.height,
                multiple,
            )
        image = resize_and_center_crop(image, width, height)
        return self.image_processor.preprocess(image)[0]

    def infer_target_size(self, sample, default_height, default_width):
        height, width = _explicit_target_size(sample)
        height = default_height if height is None else height
        width = default_width if width is None else width
        multiple = int(self.image_processor.config.vae_scale_factor)
        return align_dimension(int(height), multiple), align_dimension(int(width), multiple)


@SAMPLE_PROCESSOR_REGISTER("flux2_dev_edit")
@SAMPLE_PROCESSOR_REGISTER("flux2_klein_edit")
class Flux2EditDataProcessor(Flux2DataProcessor):
    def __call__(self, sample):
        inputs = sample["inputs"]
        meta = sample["meta"]
        target_image = inputs.pop("target_image", None)
        source_images = inputs.pop("source_images", [])
        if not source_images:
            raise ValueError("Flux2 image editing requires at least one source image")

        source_pixels = []
        source_sizes = []
        for image in source_images:
            pixels, size = self._process_source(image)
            source_pixels.append(pixels)
            source_sizes.append(size)

        reference_height, reference_width = source_sizes[0]
        meta["reference_image_height"] = reference_height
        meta["reference_image_width"] = reference_width
        inputs["source_vae_pixel_values"] = source_pixels
        if target_image is not None:
            target_pixels = self._process_target(
                target_image,
                sample,
                fallback_size=(reference_height, reference_width),
            )
            inputs["target_pixel_values"] = target_pixels
            meta["target_height"] = int(target_pixels.shape[-2])
            meta["target_width"] = int(target_pixels.shape[-1])
        else:
            target_height, target_width = _explicit_target_size(sample)
            if target_height is None:
                target_height, target_width = reference_height, reference_width
            else:
                multiple = int(self.image_processor.config.vae_scale_factor)
                target_height = align_dimension(target_height, multiple)
                target_width = align_dimension(target_width, multiple)
            meta["target_height"] = target_height
            meta["target_width"] = target_width
        return sample

    def _process_source(self, image):
        self.image_processor.check_image_input(image)
        if image.width * image.height > self.target_area:
            image = self.image_processor._resize_to_target_area(image, self.target_area)

        multiple = int(self.image_processor.config.vae_scale_factor)
        width = align_dimension(image.width, multiple)
        height = align_dimension(image.height, multiple)
        pixels = self.image_processor.preprocess(
            image,
            height=height,
            width=width,
            resize_mode="crop",
        )[0]
        return pixels, (height, width)

    def infer_target_size(self, sample, default_height, default_width):
        del default_height, default_width
        height, width = _explicit_target_size(sample)
        if height is not None:
            multiple = int(self.image_processor.config.vae_scale_factor)
            return align_dimension(height, multiple), align_dimension(width, multiple)

        meta = sample.get("meta", {})
        height = _optional_scalar(meta, "reference_image_height")
        width = _optional_scalar(meta, "reference_image_width")
        if height is None or width is None:
            raise ValueError("Flux2 image editing requires source-derived target dimensions")
        return height, width
