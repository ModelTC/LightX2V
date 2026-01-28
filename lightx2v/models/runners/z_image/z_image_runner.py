import gc
import math

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from loguru import logger

from lightx2v.models.input_encoders.hf.z_image.qwen3_model import Qwen3Model_TextEncoder
from lightx2v.models.networks.z_image.model import ZImageTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.z_image.scheduler import ZImageScheduler
from lightx2v.models.video_encoders.hf.z_image.vae import AutoencoderKLZImageVAE
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


@RUNNER_REGISTER("z_image")
class ZImageRunner(DefaultRunner):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, config):
        super().__init__(config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()

    def load_transformer(self):
        model = ZImageTransformerModel(self.config)
        return model

    def load_text_encoder(self):
        text_encoder = Qwen3Model_TextEncoder(self.config)
        text_encoders = [text_encoder]
        return text_encoders

    def load_image_encoder(self):
        pass

    def load_vae(self):
        vae = AutoencoderKLZImageVAE(self.config)
        return vae

    def init_modules(self):
        logger.info("Initializing runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
            self.model.set_scheduler(self.scheduler)
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False)
        self.run_dit = self._run_dit_local
        if self.config["task"] == "t2i":
            self.run_input_encoder = self._run_input_encoder_local_t2i
        elif self.config["task"] == "i2i":
            self.run_input_encoder = self._run_input_encoder_local_i2i
        else:
            assert NotImplementedError

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

    def read_image_input(self, img_path):
        if isinstance(img_path, Image.Image):
            img_ori = img_path
        else:
            img_ori = Image.open(img_path).convert("RGB")

        # Get image dimensions
        width, height = img_ori.size

        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_image_len.observe(width * height)

        vae_scale_factor = self.config["vae_scale_factor"]
        vae_scale = vae_scale_factor * 2
        if height % vae_scale != 0 or width % vae_scale != 0:
            logger.warning(f"Image dimensions ({height}, {width}) are not divisible by {vae_scale}. Resizing to nearest valid dimensions.")
            # Resize to nearest valid dimensions
            new_height = (height // vae_scale) * vae_scale
            new_width = (width // vae_scale) * vae_scale
            if new_height == 0:
                new_height = vae_scale
            if new_width == 0:
                new_width = vae_scale
            img_ori = img_ori.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image to ({new_height}, {new_width})")

        img = TF.to_tensor(img_ori).sub_(0.5).div_(0.5).unsqueeze(0).to(AI_DEVICE)
        self.input_info.original_size.append(img_ori.size)
        return img, img_ori

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2i(self):
        image_paths_list = self.input_info.image_path.split(",")
        images_list = []
        for image_path in image_paths_list:
            _, image = self.read_image_input(image_path)
            images_list.append(image)

        prompt = self.input_info.prompt
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()
        text_encoder_output = self.run_text_encoder(prompt, images_list, neg_prompt=self.input_info.negative_prompt)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]

        image_encoder_output_list = []
        for vae_image in text_encoder_output["image_info"]["vae_image_list"]:
            image_encoder_output = self.run_vae_encoder(image=vae_image)
            image_encoder_output_list.append(image_encoder_output)
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output_list,
        }

    @ProfilingContext4DebugL1("Run Text Encoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_text_encode_duration, metrics_labels=["ZImageRunner"])
    def run_text_encoder(self, text, image_list=None, neg_prompt=None):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_prompt_len.observe(len(text))
        text_encoder_output = {}

        if self.config["task"] == "t2i":
            # T2I task: only text encoding
            # qwen3_model.infer always returns (embedding_list, image_info)
            # For t2i, image_info is empty dict {}
            prompt_embeds_list, _ = self.text_encoders[0].infer([text])
            prompt_embeds = prompt_embeds_list[0]  # Get first (and only) embedding
            # embedding_list[0] shape is (seq_len, hidden_dim), use shape[0] for sequence length
            self.input_info.txt_seq_lens = [prompt_embeds.shape[0]]
            text_encoder_output["prompt_embeds"] = prompt_embeds
            if self.config["enable_cfg"] and neg_prompt is not None:
                neg_prompt_embeds_list, _ = self.text_encoders[0].infer([neg_prompt])
                neg_prompt_embeds = neg_prompt_embeds_list[0]
                self.input_info.txt_seq_lens.append(neg_prompt_embeds.shape[0])
                text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
        elif self.config["task"] == "i2i":
            # I2I task: text encoding + image preprocessing
            if image_list is not None:
                prompt_embeds_list, image_info = self.text_encoders[0].infer([text], image_list)
                prompt_embeds = prompt_embeds_list[0]  # Get first (and only) embedding
                # embedding_list[0] shape is (seq_len, hidden_dim), use shape[0] for sequence length
                self.input_info.txt_seq_lens = [prompt_embeds.shape[0]]
                text_encoder_output["prompt_embeds"] = prompt_embeds
                text_encoder_output["image_info"] = image_info
                if self.config["enable_cfg"] and neg_prompt is not None:
                    neg_prompt_embeds_list, _ = self.text_encoders[0].infer([neg_prompt], image_list)
                    neg_prompt_embeds = neg_prompt_embeds_list[0]
                    self.input_info.txt_seq_lens.append(neg_prompt_embeds.shape[0])
                    text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
            else:
                # No images provided, treat as t2i
                prompt_embeds_list, _ = self.text_encoders[0].infer([text])
                prompt_embeds = prompt_embeds_list[0]
                self.input_info.txt_seq_lens = [prompt_embeds.shape[0]]
                text_encoder_output["prompt_embeds"] = prompt_embeds
                if self.config["enable_cfg"] and neg_prompt is not None:
                    neg_prompt_embeds_list, _ = self.text_encoders[0].infer([neg_prompt])
                    neg_prompt_embeds = neg_prompt_embeds_list[0]
                    self.input_info.txt_seq_lens.append(neg_prompt_embeds.shape[0])
                    text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
        else:
            # Default: t2i behavior
            prompt_embeds_list, _ = self.text_encoders[0].infer([text])
            prompt_embeds = prompt_embeds_list[0]
            self.input_info.txt_seq_lens = [prompt_embeds.shape[0]]
            text_encoder_output["prompt_embeds"] = prompt_embeds
            if self.config["enable_cfg"] and neg_prompt is not None:
                neg_prompt_embeds_list, _ = self.text_encoders[0].infer([neg_prompt])
                neg_prompt_embeds = neg_prompt_embeds_list[0]
                self.input_info.txt_seq_lens.append(neg_prompt_embeds.shape[0])
                text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds

        return text_encoder_output

    @ProfilingContext4DebugL1("Run VAE Encoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration, metrics_labels=["ZImageRunner"])
    def run_vae_encoder(self, image):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae = self.load_vae()
        image_latents = self.vae.encode_vae_image(image.to(GET_DTYPE()))
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()
        return {"image_latents": image_latents}

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

    def get_input_target_shape(self):
        default_aspect_ratios = {
            "16:9": [1664, 928],
            "9:16": [928, 1664],
            "1:1": [1328, 1328],
            "4:3": [1472, 1140],
            "3:4": [768, 1024],
        }
        as_maps = self.config.get("aspect_ratios", {})
        as_maps.update(default_aspect_ratios)
        max_size = self.config.get("max_custom_size", 1664)
        min_size = self.config.get("min_custom_size", 256)

        if len(self.input_info.target_shape) == 2:
            height, width = self.input_info.target_shape
            height, width = int(height), int(width)
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                width, height = int(width * scale), int(height * scale)
                logger.warning(f"Custom shape is too large, scaled to {width}x{height}")
            width, height = max(width, min_size), max(height, min_size)
            logger.info(f"Qwen Image Runner got custom shape: {width}x{height}")
            return (width, height)

        aspect_ratio = self.input_info.aspect_ratio if self.input_info.aspect_ratio else self.config.get("aspect_ratio", None)
        if aspect_ratio in as_maps:
            logger.info(f"Qwen Image Runner got aspect ratio: {aspect_ratio}")
            width, height = as_maps[aspect_ratio]
            return (width, height)
        logger.warning(f"Invalid aspect ratio: {aspect_ratio}, not in {as_maps.keys()}")

        raise NotImplementedError

    def set_target_shape(self):
        height, width = self.get_input_target_shape()

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        # Use config vae_scale_factor to match official pipeline calculation
        vae_scale_factor = self.config["vae_scale_factor"]
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        num_channels_latents = self.config.get("num_channels_latents", 16)
        self.input_info.target_shape = (1, num_channels_latents, height, width)

    def set_img_shapes(self):
        if hasattr(self.input_info, "target_shape") and self.input_info.target_shape is not None:
            if len(self.input_info.target_shape) != 4:
                raise ValueError(f"target_shape must be 4D [B, C, H, W], got {len(self.input_info.target_shape)}D: {self.input_info.target_shape}")
            _, _, latent_height, latent_width = self.input_info.target_shape
        else:
            height, width = self.get_input_target_shape()

            vae_scale_factor = self.config["vae_scale_factor"]
            latent_height = 2 * (int(height) // (vae_scale_factor * 2))
            latent_width = 2 * (int(width) // (vae_scale_factor * 2))

        patch_size = self.config.get("patch_size", 2)
        patch_height = latent_height // patch_size
        patch_width = latent_width // patch_size

        image_shapes = [(1, patch_height, patch_width)]
        self.input_info.image_shapes = image_shapes

    def init_scheduler(self):
        self.scheduler = ZImageScheduler(self.config)

    def get_encoder_output_i2v(self):
        pass

    def run_image_encoder(self):
        pass

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_decode_duration,
        metrics_labels=["ZImageRunner"],
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

    def run_pipeline(self, input_info):
        self.input_info = input_info

        self.inputs = self.run_input_encoder()
        # Store image_encoder_output in input_info for scheduler to access
        if self.config["task"] == "i2i" and "image_encoder_output" in self.inputs:
            self.input_info.image_encoder_output = self.inputs["image_encoder_output"]

        self.set_target_shape()
        self.set_img_shapes()
        logger.info(f"input_info: {self.input_info}")

        latents, generator = self.run_dit()
        images = self.run_vae_decoder(latents)
        self.end_run()

        if not input_info.return_result_tensor:
            image = images[0]
            image.save(input_info.save_result_path)
            logger.info(f"Image saved: {input_info.save_result_path}")

        del latents, generator
        torch_device_module.empty_cache()
        gc.collect()

        if input_info.return_result_tensor:
            return {"images": images}
        elif input_info.save_result_path is not None:
            return {"images": None}
