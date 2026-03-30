import gc
import math
import os

import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.flux2_klein.qwen3_model import Flux2Klein_TextEncoder
from lightx2v.models.networks.flux2_klein.model import Flux2KleinTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.flux2_klein.scheduler import Flux2KleinScheduler
from lightx2v.models.video_encoders.hf.flux2_klein.vae import Flux2KleinVAE
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


@RUNNER_REGISTER("flux2_klein")
class Flux2KleinRunner(DefaultRunner):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, config):
        config["vae_scale_factor"] = config.get("vae_scale_factor", 16)  # 8 * 2 for patch config
        super().__init__(config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()
        self.model = self.load_transformer()

    def load_transformer(self):
        model_kwargs = {
            "model_path": os.path.join(self.config["model_path"], "transformer"),
            "config": self.config,
            "device": self.init_device,
        }
        return Flux2KleinTransformerModel(**model_kwargs)

    def load_text_encoder(self):
        text_encoder = Flux2Klein_TextEncoder(self.config)
        return [text_encoder]

    def load_vae(self):
        vae = Flux2KleinVAE(self.config)
        return vae

    def init_scheduler(self):
        self.scheduler = Flux2KleinScheduler(self.config)

    def init_modules(self):
        logger.info("Initializing Flux2Klein modules...")
        self.load_model()
        self.model.set_scheduler(self.scheduler)

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
        text_encoder_output = self.run_text_encoder(prompt, neg_prompt=self.input_info.negative_prompt)
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    @ProfilingContext4DebugL2("Run Encoders I2I")
    def _run_input_encoder_local_i2i(self):
        prompt = self.input_info.prompt
        text_encoder_output = self.run_text_encoder(prompt, neg_prompt=self.input_info.negative_prompt)

        image_path = self.input_info.image_path
        from PIL import Image

        if isinstance(image_path, str):
            # Handle both file and directory paths
            if os.path.isdir(image_path):
                # Load all image files from directory, sorted by name
                image_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"))])
                input_image = [Image.open(img_file).convert("RGB") for img_file in image_files]
            else:
                # Single image file
                input_image = Image.open(image_path).convert("RGB")
        else:
            input_image = image_path

        vae_scale_factor = self.config.get("vae_scale_factor", 8)
        from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

        image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor)

        if not isinstance(input_image, list):
            input_image = [input_image]

        condition_images = []
        for img in input_image:
            image_processor.check_image_input(img)
            image_width, image_height = img.size
            if image_width * image_height > 1024 * 1024:
                img = image_processor._resize_to_target_area(img, 1024 * 1024)
                image_width, image_height = img.size

            import debugpy

            debugpy.breakpoint()
            multiple_of = vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            img = image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
            condition_images.append(img.to(AI_DEVICE))
            if not hasattr(self.input_info, "auto_width"):
                self.input_info.auto_width = image_width
                self.input_info.auto_height = image_height

        torch.cuda.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {"image_tensor": condition_images},
        }

    # Copied from diffusers/pipelines/flux2/pipeline_flux2.py _prepare_text_ids
    def _prepare_text_ids(self, x):
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t, h, w, c = torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, c)
            out_ids.append(coords)
        return torch.stack(out_ids)

    # Copied from diffusers/pipelines/flux2/pipeline_flux2.py _prepare_latent_ids
    def _prepare_latent_ids(self, batch_size, height, width):
        t = torch.arange(1)
        # Patched height/width are used here
        h = torch.arange(height)
        w = torch.arange(width)
        c = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, c)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    @ProfilingContext4DebugL1("Run Text Encoder")
    def run_text_encoder(self, text, image_list=None, neg_prompt=None):
        prompt_embeds_list, _ = self.text_encoders[0].infer([text])
        prompt_embeds = prompt_embeds_list[0].unsqueeze(0)  # (1, seq_len, dim)
        text_ids = self._prepare_text_ids(prompt_embeds).to(AI_DEVICE)

        text_encoder_output = {"prompt_embeds": prompt_embeds, "text_ids": text_ids}

        # Flux2Klein uses empty string as negative prompt for CFG
        if self.config.get("sample_guide_scale", 1.0) > 1.0 or self.config.get("enable_cfg", True):
            neg_prompt_embeds_list, _ = self.text_encoders[0].infer([""])
            neg_prompt_embeds = neg_prompt_embeds_list[0].unsqueeze(0)
            neg_text_ids = self._prepare_text_ids(neg_prompt_embeds).to(AI_DEVICE)

            text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
            text_encoder_output["negative_text_ids"] = neg_text_ids

        return text_encoder_output

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

        input_image_tensor = self.inputs["image_encoder_output"]["image_tensor"]

        self.model.scheduler.prepare_i2i(self.input_info, input_image_tensor, self.vae)

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
            logger.info(f"Flux2Klein Image Runner got custom shape: {width}x{height}")
            return (width, height)

        if self.input_info.aspect_ratio and not self.config.get("_auto_resize", False):
            if self.input_info.aspect_ratio in as_maps:
                logger.info(f"Flux2Klein Image Runner got aspect ratio: {self.input_info.aspect_ratio}")
                width, height = as_maps[self.input_info.aspect_ratio]
                return (width, height)
            logger.warning(f"Invalid aspect ratio: {self.input_info.aspect_ratio}, not in {as_maps.keys()}")

        width, height = as_maps[self.config.get("aspect_ratio", "16:9")]
        return (width, height)

    def set_target_shape(self):
        multiple_of = self.config.get("vae_scale_factor", 8) * 2

        task = self.config.get("task", "t2i")
        if task == "i2i" and hasattr(self.input_info, "auto_width"):
            width = self.input_info.auto_width
            height = self.input_info.auto_height
        else:
            custom_shape = self.get_custom_shape()
            if custom_shape is not None:
                width, height = custom_shape
            else:
                calculated_width, calculated_height, _ = calculate_dimensions(self.resolution * self.resolution, 16 / 9)
                width = calculated_width // multiple_of * multiple_of
                height = calculated_height // multiple_of * multiple_of

            self.input_info.auto_width = width
            self.input_info.auto_height = height

        self.input_info.target_shape = (height, width)
        logger.info(f"Flux2Klein Image Runner set target shape: {width}x{height}")

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
        B, _, C = latents.shape

        # Need original packed height and width. We can recover from latent_image_ids max
        H = int((self.input_info.latent_image_ids[0, :, 1].max() + 1).item())
        W = int((self.input_info.latent_image_ids[0, :, 2].max() + 1).item())

        latents = latents.view(B, H, W, C).permute(0, 3, 1, 2)

        bn_mean = self.vae.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.vae.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.vae.config.batch_norm_eps)
        latents = latents * bn_std + bn_mean

        latents = latents.reshape(B, C // 4, 2, 2, H, W)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(B, C // 4, H * 2, W * 2)

        images = self.vae.decode(latents, self.input_info)
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

        if not input_info.return_result_tensor:
            image = images[0]
            image.save(input_info.save_result_path)
            logger.info(f"Image saved: {input_info.save_result_path}")

        torch.cuda.empty_cache()
        gc.collect()

        if input_info.return_result_tensor:
            return {"images": images}
        return {"images": None}
