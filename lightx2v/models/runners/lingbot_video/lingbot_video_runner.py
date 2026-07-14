import gc
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger

from lightx2v.models.input_encoders.hf.lingbot_video.qwen3vl import LingBotVideoQwen3VLTextEncoder
from lightx2v.models.networks.lingbot_video.model import LingBotVideoTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.lingbot_video.scheduler import LingBotVideoScheduler
from lightx2v.models.video_encoders.hf.lingbot_video.vae import LingBotVideoWanVAE
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import save_to_video
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2


def _round_by_factor(number, factor):
    return round(number / factor) * factor


def _ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def _floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor, min_pixels=None, max_pixels=None):
    max_pixels = max_pixels if max_pixels is not None else IMAGE_MAX_TOKEN_NUM * factor**2
    min_pixels = min_pixels if min_pixels is not None else IMAGE_MIN_TOKEN_NUM * factor**2
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}.")
    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


@RUNNER_REGISTER("lingbot_video")
class LingBotVideoRunner(DefaultRunner):
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(self, config):
        if config.get("task") not in {"t2i", "t2v", "i2v"}:
            raise NotImplementedError("LingBot-Video LightX2V backend currently supports t2i, t2v, and i2v.")
        if config.get("lazy_load", False) or config.get("unload_modules", False):
            raise NotImplementedError("LingBot-Video lazy_load/unload_modules are not implemented yet.")
        super().__init__(config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.image_encoder = None
        self.vae = self.load_vae()

    def load_transformer(self):
        return LingBotVideoTransformerModel(
            model_path=os.path.join(self.config["model_path"], self.config.get("transformer_subfolder", "transformer")),
            config=self.config,
            device=self.init_device,
        )

    def load_text_encoder(self):
        return [LingBotVideoQwen3VLTextEncoder(self.config)]

    def load_image_encoder(self):
        return None

    def load_vae(self):
        return LingBotVideoWanVAE(self.config)

    def init_modules(self):
        super().init_modules()
        if self.config["task"] in {"t2i", "t2v"}:
            self.run_input_encoder = self._run_input_encoder_local_t2v
        elif self.config["task"] == "i2v":
            self.run_input_encoder = self._run_input_encoder_local_i2v
        self.run_dit = self._run_dit_local
        self.config.lock()

    def init_scheduler(self):
        super().init_scheduler()
        self.scheduler = LingBotVideoScheduler(self.config)

    @ProfilingContext4DebugL1("Run Text Encoder")
    def run_text_encoder(self, prompt, neg_prompt=None, images=None):
        text_encoder_output = {}
        prompt_output = self.text_encoders[0].infer(prompt, images=images)
        text_encoder_output["prompt_embeds"] = prompt_output["prompt_embeds"]
        text_encoder_output["prompt_mask"] = prompt_output["prompt_mask"]
        if hasattr(self.input_info, "txt_seq_lens"):
            self.input_info.txt_seq_lens = [prompt_output["prompt_embeds"].shape[1]]

        if self.config.get("enable_cfg", True):
            neg_prompt = "" if neg_prompt is None else neg_prompt
            negative_output = self.text_encoders[0].infer(neg_prompt, images=images)
            text_encoder_output["negative_prompt_embeds"] = negative_output["prompt_embeds"]
            text_encoder_output["negative_prompt_mask"] = negative_output["prompt_mask"]
            if hasattr(self.input_info, "txt_seq_lens"):
                self.input_info.txt_seq_lens.append(negative_output["prompt_embeds"].shape[1])
        return text_encoder_output

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2v(self):
        negative_prompt = self.input_info.negative_prompt or ""
        text_encoder_output = self.run_text_encoder(self.input_info.prompt, neg_prompt=negative_prompt)
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    def preprocess_image(self, image, height, width):
        raw = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).contiguous()
        old_h, old_w = raw.shape[-2:]
        scale = max(height / old_h, width / old_w)
        new_h = max(math.ceil(old_h * scale), height)
        new_w = max(math.ceil(old_w * scale), width)
        resized = F.interpolate(raw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        top = int(round((new_h - height) / 2.0))
        left = int(round((new_w - width) / 2.0))
        return resized[:, :, top : top + height, left : left + width].float().div_(255.0).unsqueeze(2)

    def _vision_patch_size(self):
        text_encoder = self.text_encoders[0]
        for obj in (
            getattr(getattr(text_encoder.text_encoder, "config", None), "vision_config", None),
            getattr(getattr(text_encoder.processor, "image_processor", None), "config", None),
            getattr(text_encoder.processor, "image_processor", None),
        ):
            patch = getattr(obj, "patch_size", None)
            if patch is not None:
                return int(patch)
        return 16

    def _vlm_image(self, pixel):
        frame = pixel[0, :, 0].detach().cpu().clamp(0, 1)
        image = Image.fromarray(frame.permute(1, 2, 0).mul(255).byte().numpy(), mode="RGB")
        patch_factor = self._vision_patch_size() * SPATIAL_MERGE_SIZE
        width, height = image.size
        resized_height, resized_width = smart_resize(height, width, factor=patch_factor)
        return image.resize((resized_width, resized_height))

    def _ensure_scheduler_generator(self):
        if self.scheduler.generator is None:
            self.scheduler.generator = torch.Generator(device=AI_DEVICE).manual_seed(int(self.input_info.seed))
        return self.scheduler.generator

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        height, width = self._resolve_output_size()
        image_path = self.input_info.image_path.split(",")[0]
        if not image_path:
            raise ValueError("LingBot-Video i2v requires --image_path.")
        image = Image.open(image_path).convert("RGB")
        pixel = self.preprocess_image(image, height, width)
        vlm_image = self._vlm_image(pixel)
        generator = self._ensure_scheduler_generator()
        cond_latent = self.vae.encode_image_latent(pixel, generator=generator)
        negative_prompt = self.input_info.negative_prompt or ""
        text_encoder_output = self.run_text_encoder(self.input_info.prompt, neg_prompt=negative_prompt, images=[vlm_image])
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {"cond_latent": cond_latent},
        }

    def _resolve_output_size(self):
        if len(self.input_info.target_shape) == 2:
            height, width = int(self.input_info.target_shape[0]), int(self.input_info.target_shape[1])
        else:
            height, width = int(self.config["target_height"]), int(self.config["target_width"])
        return height, width

    def set_target_shape(self):
        height, width = self._resolve_output_size()
        if height <= 0 or width <= 0:
            raise ValueError(f"LingBot-Video target shape must be positive, got {height}x{width}.")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"LingBot-Video target shape must be divisible by 16, got {height}x{width}.")

        if self.config["task"] == "t2i":
            frames = 1
        else:
            frames = int(self.config["target_video_length"])
        if frames != 1 and (frames - 1) % int(self.config.get("vae_scale_factor_temporal", 4)) != 0:
            raise ValueError(f"LingBot-Video target_video_length must be 1 or 4n+1, got {frames}.")
        latent_t = (frames - 1) // int(self.config.get("vae_scale_factor_temporal", 4)) + 1
        latent_h = height // int(self.config.get("vae_scale_factor_spatial", 8))
        latent_w = width // int(self.config.get("vae_scale_factor_spatial", 8))
        latent_shape = (1, int(self.config.get("in_channels", 16)), latent_t, latent_h, latent_w)

        self.input_info.auto_height = height
        self.input_info.auto_width = width
        self.input_info.target_shape = latent_shape
        self.input_info.latent_shape = latent_shape
        patch_h, patch_w = self.config.get("patch_size", [1, 2, 2])[1:]
        if hasattr(self.input_info, "image_shapes"):
            self.input_info.image_shapes = [[(latent_t, latent_h // patch_h, latent_w // patch_w)]]
        logger.info(f"LingBot-Video target shape: frames={frames}, image={height}x{width}, latent={latent_shape}")

    def _apply_condition_latent(self):
        image_encoder_output = self.inputs.get("image_encoder_output") or {}
        cond_latent = image_encoder_output.get("cond_latent")
        if cond_latent is None:
            return
        latents = self.model.scheduler.latents
        cond_latent = cond_latent.to(device=latents.device, dtype=torch.float32)
        latents[:, :, : cond_latent.shape[2]] = cond_latent
        self.model.scheduler.latents = latents

    @ProfilingContext4DebugL2("Run DiT")
    def _run_dit_local(self, total_steps=None):
        self.model.scheduler.prepare(self.input_info)
        self._apply_condition_latent()
        latents, generator = self.run(total_steps)
        return latents, generator

    def run(self, total_steps=None):
        if total_steps is None:
            total_steps = self.model.scheduler.infer_steps
        for step_index in range(total_steps):
            logger.info(f"==> step_index: {step_index + 1} / {total_steps}")
            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)
            with ProfilingContext4DebugL1("infer_main"):
                self.model.infer(self.inputs)
            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()
                self._apply_condition_latent()
            if self.progress_callback:
                self.progress_callback(((step_index + 1) / total_steps) * 100, 100)
        return self.model.scheduler.latents, self.model.scheduler.generator

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        return self.vae.decode(latents, self.input_info)

    def _save_outputs(self, outputs, input_info):
        if input_info.return_result_tensor or not input_info.save_result_path:
            return
        save_result_path = input_info.save_result_path
        os.makedirs(os.path.dirname(os.path.abspath(save_result_path)), exist_ok=True)
        if self.config["task"] == "t2i":
            image = outputs[0] if isinstance(outputs, list) else outputs
            image.save(save_result_path)
            logger.info(f"Image saved: {save_result_path}")
        else:
            save_to_video(outputs, save_result_path, fps=float(self.config.get("target_fps", 24)), method="ffmpeg")
            logger.info(f"Video saved: {save_result_path}")

    def _finalize_pipeline_outputs(self, outputs, latents=None, generator=None):
        if latents is not None:
            del latents
        if generator is not None:
            del generator
        torch_device_module.empty_cache()
        gc.collect()
        if self.input_info.return_result_tensor:
            return {"images": outputs} if self.config["task"] == "t2i" else {"video": outputs}
        return {"images": None} if self.config["task"] == "t2i" else {"video": None}

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info
        self.inputs = self.run_input_encoder()
        self.set_target_shape()
        logger.info(f"input_info: {self.input_info}")
        latents, generator = self.run_dit()
        outputs = self.run_vae_decoder(latents)
        self._save_outputs(outputs, input_info)
        outputs = self._finalize_pipeline_outputs(outputs, latents=latents, generator=generator)
        self.end_run()
        return outputs
