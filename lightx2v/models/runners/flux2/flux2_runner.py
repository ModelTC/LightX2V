import gc
import math
import os

import numpy as np
import torch
from loguru import logger

from lightx2v.models.networks.flux2.model import Flux2DevTransformerModel, Flux2KleinTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.flux2.feature_caching.scheduler import Flux2DevSchedulerCaching, Flux2SchedulerCaching
from lightx2v.models.schedulers.flux2.scheduler import Flux2DevScheduler, Flux2Scheduler
from lightx2v.models.video_encoders.hf.flux2.vae import Flux2VAE
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import is_main_process
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


class Flux2BaseRunner(DefaultRunner):
    """Shared base runner for Flux2 Klein and Dev models."""

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, config):
        config["vae_scale_factor"] = config.get("vae_scale_factor", 16)
        super().__init__(config)

    def _get_scheduler_class(self):
        if self.config.get("feature_caching", "NoCaching") in ("NoCaching", "None"):
            return None
        if self.config.get("feature_caching") == "Ada":
            return Flux2SchedulerCaching
        raise NotImplementedError(f"Unsupported feature_caching type: {self.config.get('feature_caching')}")

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()
        self.model = self.load_transformer()

    def load_vae(self):
        return Flux2VAE(self.config)

    def init_modules(self):
        logger.info(f"Initializing {self.config['model_cls']} modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
            self.model.set_scheduler(self.scheduler)
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False)

        task = self.config.get("task", "t2i")
        if task == "i2i":
            self.run_input_encoder = self._run_input_encoder_local_i2i
            self.run_dit = self._run_dit_local_i2i
        else:
            self.run_input_encoder = self._run_input_encoder_local_t2i
            self.run_dit = self._run_dit_local

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2i(self):
        prompt = self.input_info.prompt
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()
        text_encoder_output = self.run_text_encoder(prompt, neg_prompt=self.input_info.negative_prompt)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    @ProfilingContext4DebugL2("Run Encoders I2I")
    def _run_input_encoder_local_i2i(self):
        prompt = self.input_info.prompt
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()
        text_encoder_output = self.run_text_encoder(prompt, neg_prompt=self.input_info.negative_prompt)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]

        image_path = self.input_info.image_path
        from PIL import Image

        if isinstance(image_path, str):
            if os.path.isdir(image_path):
                image_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"))])
                input_image = [Image.open(img_file).convert("RGB") for img_file in image_files]
            else:
                input_image = Image.open(image_path).convert("RGB")
        else:
            input_image = image_path

        vae_scale_factor = self.config.get("vae_scale_factor", 8)
        from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

        image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor)

        if not isinstance(input_image, list):
            input_image = [input_image]

        condition_images = []
        max_image_area = self.config.get("max_image_area", 1024 * 1024)
        inpaint_mask = None
        inpaint_mask_enabled = self.config.get("inpaint_mask_enabled", False)
        if inpaint_mask_enabled:
            main_img = input_image[0]
            image_processor.check_image_input(main_img)
            processed_img, target_shape = self._preprocess_condition_image(image_processor, main_img, max_image_area, vae_scale_factor)
            self.input_info.target_shape = target_shape
            processed_tensor = processed_img.to(AI_DEVICE)
            condition_images.extend([processed_tensor, processed_tensor])

            if len(input_image) > 1:
                image_processor.check_image_input(input_image[1])
                inpaint_mask = self._preprocess_inpaint_mask_image(image_processor, input_image[1], main_img, max_image_area, target_shape)
        else:
            for index, img in enumerate(input_image):
                image_processor.check_image_input(img)
                processed_img, target_shape = self._preprocess_condition_image(image_processor, img, max_image_area, vae_scale_factor)
                condition_images.append(processed_img.to(AI_DEVICE))
                if index == 0:
                    self.input_info.target_shape = target_shape

        torch_device_module.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {"image_tensor": condition_images, "inpaint_mask": inpaint_mask},
        }

    @staticmethod
    def _maybe_resize_to_max_area(image_processor, img, max_image_area):
        width, height = img.size
        if max_image_area is not None and max_image_area > 0 and width * height > max_image_area:
            img = image_processor._resize_to_target_area(img, max_image_area)
        return img

    @staticmethod
    def _snap_image_dimensions(width, height, vae_scale_factor):
        multiple_of = vae_scale_factor * 2
        return (width // multiple_of) * multiple_of, (height // multiple_of) * multiple_of

    def _preprocess_condition_image(self, image_processor, img, max_image_area, vae_scale_factor):
        img = self._maybe_resize_to_max_area(image_processor, img, max_image_area)
        image_width, image_height = self._snap_image_dimensions(*img.size, vae_scale_factor)
        img = image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
        return img, (image_height, image_width)

    def _preprocess_inpaint_mask_image(self, image_processor, mask_img, reference_img, max_image_area, target_shape):
        mask_img = mask_img.convert("RGB")
        if mask_img.size != reference_img.size:
            mask_img = mask_img.resize(reference_img.size)
        mask_img = self._maybe_resize_to_max_area(image_processor, mask_img, max_image_area)
        image_height, image_width = target_shape
        cropped_mask = image_processor.resize(mask_img, image_height, image_width, resize_mode="crop")
        return self._prepare_inpaint_mask(cropped_mask)

    def _prepare_inpaint_mask(self, mask):
        if mask is None:
            return None

        from PIL import Image

        height, width = self.input_info.target_shape
        multiple_of = self.config.get("vae_scale_factor", 8) * 2
        packed_h = height // multiple_of
        packed_w = width // multiple_of

        resample = getattr(Image, "Resampling", Image).BILINEAR
        mask = mask.convert("RGB").resize((packed_w, packed_h), resample)
        mask = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)
        mask = mask.permute(2, 0, 1).unsqueeze(0)
        mask = mask.mean(dim=1, keepdim=True)

        blur_size = getattr(self.input_info, "inpaint_blur_size", None)
        blur_sigma = getattr(self.input_info, "inpaint_blur_sigma", None)
        if blur_size is not None and blur_sigma is not None:
            from torchvision.transforms import GaussianBlur

            blur = GaussianBlur(kernel_size=blur_size * 2 + 1, sigma=blur_sigma)
            mask = blur(mask)

        return mask.clamp(0, 1).view(1, packed_h * packed_w, 1).to(AI_DEVICE)

    def _prepare_text_ids(self, x):
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t, h, w, c = torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, c)
            out_ids.append(coords)
        return torch.stack(out_ids)

    def _prepare_latent_ids(self, batch_size, height, width):
        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        c = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, c)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    @ProfilingContext4DebugL2("Run DiT")
    def _run_dit_local(self, total_steps=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        self.model.scheduler.prepare(self.input_info)
        latents, generator = self.run(total_steps)
        return latents, generator

    @ProfilingContext4DebugL2("Run DiT I2I")
    def _run_dit_local_i2i(self, total_steps=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

        image_encoder_output = self.inputs["image_encoder_output"]
        input_image_tensor = image_encoder_output["image_tensor"]
        inpaint_mask = image_encoder_output.get("inpaint_mask")

        self.model.scheduler.prepare_i2i(self.input_info, input_image_tensor, self.vae, inpaint_mask=inpaint_mask)

        latents, generator = self.run(total_steps)
        return latents, generator

    def run(self, total_steps=None):
        if total_steps is None:
            total_steps = self.model.scheduler.infer_steps
        for step_index in range(total_steps):
            logger.info(f"==> step_index: {step_index + 1} / {total_steps}")

            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(self.inputs)

            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()

            if self.progress_callback:
                self.progress_callback(((step_index + 1) / total_steps) * 100, 100)

        return self.model.scheduler.latents, self.model.scheduler.generator

    def get_custom_shape(self):
        default_aspect_ratios = {
            "16:9": [1344, 768],
            "9:16": [768, 1344],
            "1:1": [1024, 1024],
            "4:3": [1152, 864],
            "3:4": [864, 1152],
            "3:2": [1216, 832],
            "2:3": [832, 1216],
        }
        as_maps = self.config.get("aspect_ratios", {})
        as_maps.update(default_aspect_ratios)
        max_size = self.config.get("max_custom_size", 1664)
        min_size = self.config.get("min_custom_size", 256)

        if len(self.input_info.target_shape) == 2:
            height, width = self.input_info.target_shape
            height = int(height)
            width = int(width)
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                width, height = int(width * scale), int(height * scale)
                logger.warning(f"Custom shape is too large, scaled to {width}x{height}")
            width, height = max(width, min_size), max(height, min_size)
            logger.info(f"Flux2 Image Runner got custom shape: {width}x{height}")
            return (width, height)

        if self.input_info.aspect_ratio and not self.config.get("_auto_resize", False):
            if self.input_info.aspect_ratio in as_maps:
                logger.info(f"Flux2 Image Runner got aspect ratio: {self.input_info.aspect_ratio}")
                width, height = as_maps[self.input_info.aspect_ratio]
                return (width, height)
            logger.warning(f"Invalid aspect ratio: {self.input_info.aspect_ratio}, not in {as_maps.keys()}")

        width, height = as_maps[self.config.get("aspect_ratio", "16:9")]
        return (width, height)

    def set_target_shape(self):
        task = self.config.get("task", "t2i")
        if task == "i2i":
            height, width = self.input_info.target_shape
        else:
            custom_shape = self.get_custom_shape()
            if custom_shape is not None:
                width, height = custom_shape
            else:
                calculated_width, calculated_height, _ = calculate_dimensions(self.resolution * self.resolution, 16 / 9)
                multiple_of = self.config.get("vae_scale_factor", 8) * 2
                width = calculated_width // multiple_of * multiple_of
                height = calculated_height // multiple_of * multiple_of
                self.input_info.target_shape = (height, width)

        multiple_of = self.config.get("vae_scale_factor", 8) * 2

        packed_batch = 1
        packed_h = height // multiple_of
        packed_w = width // multiple_of
        packed_channels = 128

        self.num_channels_latents = packed_channels
        self.input_info.latent_shape = (packed_batch, packed_h * packed_w, packed_channels)
        self.input_info.latent_image_ids = self._prepare_latent_ids(packed_batch, packed_h, packed_w).to(AI_DEVICE)

    def set_img_shapes(self):
        pass

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae = self.load_vae()

        B, _, C = latents.shape

        H = int((self.input_info.latent_image_ids[0, :, 1].max() + 1).item())
        W = int((self.input_info.latent_image_ids[0, :, 2].max() + 1).item())

        latents = latents.view(B, H, W, C).permute(0, 3, 1, 2)

        bn_mean = self.vae.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.vae.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        latents = latents * bn_std + bn_mean

        latents = latents.reshape(B, C // 4, 2, 2, H, W)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(B, C // 4, H * 2, W * 2)

        images = self.vae.decode(latents, self.input_info)

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()

        return images

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info
        self.inputs = self.run_input_encoder()
        logger.info(f"input_info: {self.input_info}")

        self.set_target_shape()
        self.set_img_shapes()

        latents, generator = self.run_dit()
        images = self.run_vae_decoder(latents)

        if not input_info.return_result_tensor and is_main_process():
            image = images[0]
            image.save(input_info.save_result_path)
            logger.info(f"Image saved: {input_info.save_result_path}")

        torch_device_module.empty_cache()
        gc.collect()

        if input_info.return_result_tensor:
            return {"images": images}
        return {"images": None}


@RUNNER_REGISTER("flux2_klein")
class Flux2KleinRunner(Flux2BaseRunner):
    def load_transformer(self):
        model_kwargs = {
            "model_path": os.path.join(self.config["model_path"], "transformer"),
            "config": self.config,
            "device": self.init_device,
        }
        return Flux2KleinTransformerModel(**model_kwargs)

    def load_text_encoder(self):
        from lightx2v.models.input_encoders.hf.flux2.qwen3_model import Flux2Klein_TextEncoder

        text_encoder = Flux2Klein_TextEncoder(self.config)
        return [text_encoder]

    def init_scheduler(self):
        caching_scheduler_class = self._get_scheduler_class()
        if caching_scheduler_class is not None:
            self.scheduler = caching_scheduler_class(self.config)
        else:
            self.scheduler = Flux2Scheduler(self.config)

    @ProfilingContext4DebugL1("Run Text Encoder")
    def run_text_encoder(self, text, image_list=None, neg_prompt=None):
        prompt_embeds_list, _ = self.text_encoders[0].infer([text])
        prompt_embeds = prompt_embeds_list[0].unsqueeze(0)
        text_ids = self._prepare_text_ids(prompt_embeds).to(AI_DEVICE)

        text_encoder_output = {"prompt_embeds": prompt_embeds, "text_ids": text_ids}

        if self.config.get("sample_guide_scale", 1.0) > 1.0 or self.config.get("enable_cfg", True):
            neg_prompt_embeds_list, _ = self.text_encoders[0].infer([""])
            neg_prompt_embeds = neg_prompt_embeds_list[0].unsqueeze(0)
            neg_text_ids = self._prepare_text_ids(neg_prompt_embeds).to(AI_DEVICE)

            text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
            text_encoder_output["negative_text_ids"] = neg_text_ids

        return text_encoder_output


@RUNNER_REGISTER("flux2_dev")
class Flux2DevRunner(Flux2BaseRunner):
    def load_transformer(self):
        model_kwargs = {
            "model_path": os.path.join(self.config["model_path"], "transformer"),
            "config": self.config,
            "device": self.init_device,
        }
        return Flux2DevTransformerModel(**model_kwargs)

    def load_text_encoder(self):
        from lightx2v.models.input_encoders.hf.flux2.mistral3_model import Flux2Dev_TextEncoder

        text_encoder = Flux2Dev_TextEncoder(self.config)
        return [text_encoder]

    def init_scheduler(self):
        caching_scheduler_class = self._get_scheduler_class()
        if caching_scheduler_class is not None:
            self.scheduler = Flux2DevSchedulerCaching(self.config)
        else:
            self.scheduler = Flux2DevScheduler(self.config)

    @ProfilingContext4DebugL1("Run Text Encoder")
    def run_text_encoder(self, text, image_list=None, neg_prompt=None):
        prompt_embeds_list, _ = self.text_encoders[0].infer([text])
        prompt_embeds = prompt_embeds_list[0].unsqueeze(0)
        text_ids = self._prepare_text_ids(prompt_embeds).to(AI_DEVICE)

        text_encoder_output = {"prompt_embeds": prompt_embeds, "text_ids": text_ids}

        return text_encoder_output
