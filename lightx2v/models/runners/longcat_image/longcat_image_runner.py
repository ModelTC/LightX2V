import gc
import math

import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.longcat.longcat_text_encoder import LongCatImageTextEncoder
from lightx2v.models.networks.longcat_image.model import LongCatImageTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.longcat_image.scheduler import LongCatImageScheduler
from lightx2v.models.video_encoders.hf.longcat_image.vae import LongCatImageVAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


@RUNNER_REGISTER("longcat_image")
class LongCatImageRunner(DefaultRunner):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, config):
        super().__init__(config)
        self.resolution = self.config.get("resolution", 1024)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()

    def load_transformer(self):
        model = LongCatImageTransformerModel(self.config)
        return model

    def load_text_encoder(self):
        text_encoder = LongCatImageTextEncoder(self.config)
        text_encoders = [text_encoder]
        return text_encoders

    def load_image_encoder(self):
        pass

    def load_vae(self):
        vae = LongCatImageVAE(self.config)
        return vae

    def init_modules(self):
        logger.info("Initializing LongCat runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
            self.model.set_scheduler(self.scheduler)
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False)
        self.run_dit = self._run_dit_local
        # LongCat only supports T2I
        self.run_input_encoder = self._run_input_encoder_local_t2i

    @ProfilingContext4DebugL2("Run DiT")
    def _run_dit_local(self, total_steps=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        self.model.scheduler.prepare(self.input_info)
        latents, generator = self.run(total_steps)
        return latents, generator

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

    @ProfilingContext4DebugL1(
        "Run Text Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_text_encode_duration,
        metrics_labels=["LongCatImageRunner"],
    )
    def run_text_encoder(self, text, neg_prompt=None):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_prompt_len.observe(len(text))
        text_encoder_output = {}

        # Check if prompt rewrite is enabled
        enable_prompt_rewrite = self.config.get("enable_prompt_rewrite", False)

        # Optionally rewrite the prompt
        if enable_prompt_rewrite:
            text = self.text_encoders[0].rewire_prompt(text)[0]
            logger.info(f"Rewritten prompt: {text}")

        prompt_embeds, prompt_embeds_mask, _ = self.text_encoders[0].infer([text])
        # LongCat uses full sequence length (512) for position embeddings, not just valid tokens
        self.input_info.txt_seq_lens = [prompt_embeds.shape[1]]
        text_encoder_output["prompt_embeds"] = prompt_embeds

        if self.config["enable_cfg"] and neg_prompt is not None:
            neg_prompt_embeds, neg_prompt_embeds_mask, _ = self.text_encoders[0].infer([neg_prompt])
            self.input_info.txt_seq_lens.append(neg_prompt_embeds.shape[1])
            text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds

        return text_encoder_output

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_decode_duration,
        metrics_labels=["LongCatImageRunner"],
    )
    def run_vae_decoder(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae = self.load_vae()
        images = self.vae.decode(latents, self.input_info)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()
        return images

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

        if len(self.input_info.custom_shape) == 2:
            height, width = self.input_info.custom_shape
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                width, height = int(width * scale), int(height * scale)
                logger.warning(f"Custom shape is too large, scaled to {width}x{height}")
            width, height = max(width, min_size), max(height, min_size)
            logger.info(f"LongCat Image Runner got custom shape: {width}x{height}")
            return (width, height)

        if self.input_info.aspect_ratio and not self.config.get("_auto_resize", False):
            if self.input_info.aspect_ratio in as_maps:
                logger.info(f"LongCat Image Runner got aspect ratio: {self.input_info.aspect_ratio}")
                width, height = as_maps[self.input_info.aspect_ratio]
                return (width, height)
            logger.warning(f"Invalid aspect ratio: {self.input_info.aspect_ratio}, not in {as_maps.keys()}")

        width, height = as_maps[self.config.get("aspect_ratio", "16:9")]
        return (width, height)

    def set_target_shape(self):
        custom_shape = self.get_custom_shape()
        if custom_shape is not None:
            width, height = custom_shape
        else:
            calculated_width, calculated_height, _ = calculate_dimensions(
                self.resolution * self.resolution, 16 / 9
            )
            multiple_of = self.config.get("vae_scale_factor", 8) * 2
            width = calculated_width // multiple_of * multiple_of
            height = calculated_height // multiple_of * multiple_of

        logger.info(f"LongCat Image Runner set target shape: {width}x{height}")
        self.input_info.auto_width = width
        self.input_info.auto_height = height

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        vae_scale_factor = self.config.get("vae_scale_factor", 8)
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        num_channels_latents = 16  # LongCat uses 16 latent channels
        self.input_info.target_shape = (1, num_channels_latents, height, width)

    def set_img_shapes(self):
        width, height = self.input_info.auto_width, self.input_info.auto_height
        vae_scale_factor = self.config.get("vae_scale_factor", 8)
        # For T2I task
        image_shapes = [(1, height // vae_scale_factor // 2, width // vae_scale_factor // 2)] * 1
        self.input_info.image_shapes = image_shapes

    def init_scheduler(self):
        self.scheduler = LongCatImageScheduler(self.config)

    def get_encoder_output_i2v(self):
        pass

    def run_image_encoder(self):
        pass

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.image_encoder = self.load_image_encoder()
        self.vae = self.load_vae()
        self.vfi_model = self.load_vfi_model() if "video_frame_interpolation" in self.config else None

    def run_pipeline(self, input_info):
        self.input_info = input_info

        self.inputs = self.run_input_encoder()
        self.set_target_shape()
        self.set_img_shapes()
        logger.info(f"input_info: {self.input_info}")
        latents, generator = self.run_dit()
        images = self.run_vae_decoder(latents)
        self.end_run()

        image = images[0]
        image.save(f"{input_info.save_result_path}")
        logger.info(f"Image saved: {input_info.save_result_path}")

        del latents, generator
        torch_device_module.empty_cache()
        gc.collect()

        # Return (images, audio) - audio is None for default runner
        return images, None
