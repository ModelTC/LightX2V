import gc
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from loguru import logger
from transformers import AutoTokenizer

from lightx2v.models.networks.cosmos3.model import Cosmos3TransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.cosmos3.scheduler import Cosmos3Scheduler
from lightx2v.models.video_encoders.hf.cosmos3.vae import Cosmos3WanVAE
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import save_to_video
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."


@RUNNER_REGISTER("cosmos3")
class Cosmos3Runner(DefaultRunner):
    model_cpu_offload_seq = "transformer->vae"
    _callback_tensor_inputs = ["latents"]

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_tokenizer = self.load_text_encoder()
        self.vae = self.load_vae()

    def load_transformer(self):
        return Cosmos3TransformerModel(
            model_path=os.path.join(self.config["model_path"], "transformer"),
            config=self.config,
            device=self.init_device,
        )

    def load_text_encoder(self):
        tokenizer_path = self.config.get("text_tokenizer_path", os.path.join(self.config["model_path"], "text_tokenizer"))
        if not os.path.exists(tokenizer_path):
            tokenizer_path = self.config["model_path"]
        logger.info(f"Loading Cosmos3 tokenizer from {tokenizer_path}")
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def load_image_encoder(self):
        return None

    def load_vae(self):
        return Cosmos3WanVAE(self.config)

    def init_modules(self):
        logger.info("Initializing Cosmos3 runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False)
        if hasattr(self, "model") and self.model is not None:
            self.model.set_scheduler(self.scheduler)
        if self.config["task"] not in ("t2i", "i2v"):
            raise NotImplementedError(f"Cosmos3Runner currently supports tasks t2i/i2v, got {self.config['task']}")
        self.run_input_encoder = self._run_input_encoder_local
        self.run_dit = self._run_dit_local
        self.config.lock()

    def init_scheduler(self):
        super().init_scheduler()
        if self.config.get("disagg_mode") == "decode":
            return
        self.scheduler = Cosmos3Scheduler(self.config)

    def _append_prompt_template(self, base: str, addition: str) -> str:
        base = (base or "").rstrip(".")
        return f"{base}. {addition}" if base else addition

    def _tokenize_chat(self, text: str, is_image: bool):
        conversations = [
            {"role": "system", "content": _SYSTEM_PROMPT_IMAGE if is_image else _SYSTEM_PROMPT_VIDEO},
            {"role": "user", "content": text},
        ]
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "add_vision_id": False,
            "return_dict": True,
        }
        try:
            encodings = self.text_tokenizer.apply_chat_template(conversations, **kwargs)
        except TypeError:
            kwargs.pop("add_vision_id", None)
            try:
                encodings = self.text_tokenizer.apply_chat_template(conversations, **kwargs)
            except TypeError:
                kwargs.pop("return_dict", None)
                encodings = self.text_tokenizer.apply_chat_template(conversations, **kwargs)

        if isinstance(encodings, dict):
            input_ids = encodings["input_ids"]
        elif hasattr(encodings, "input_ids"):
            input_ids = encodings.input_ids
        else:
            input_ids = encodings
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        return list(input_ids)

    def tokenize_prompt(self, prompt, negative_prompt=None):
        height = int(self.input_info.auto_height)
        width = int(self.input_info.auto_width)
        num_frames = int(self.config.get("target_video_length", 1))
        fps = float(self.config.get("target_fps", 24.0))
        is_image = num_frames == 1
        negative_prompt = "" if negative_prompt is None else negative_prompt

        cond_text = prompt
        uncond_text = negative_prompt
        if not is_image and self.config.get("add_duration_template", True):
            cond_text = self._append_prompt_template(cond_text, f"The video is {num_frames / fps:.1f} seconds long and is of {fps:.0f} FPS.")
            uncond_text = self._append_prompt_template(uncond_text, f"The video is not {num_frames / fps:.1f} seconds long and is not of {fps:.0f} FPS.")
        if self.config.get("add_resolution_template", True):
            if is_image:
                cond_text = self._append_prompt_template(cond_text, f"This image is of {height}x{width} resolution.")
                uncond_text = self._append_prompt_template(uncond_text, f"This image is not of {height}x{width} resolution.")
            else:
                cond_text = self._append_prompt_template(cond_text, f"This video is of {height}x{width} resolution.")
                uncond_text = self._append_prompt_template(uncond_text, f"This video is not of {height}x{width} resolution.")

        eos_token_id = self.text_tokenizer.eos_token_id
        vision_start_id = self.text_tokenizer.convert_tokens_to_ids("<|vision_start|>")
        if eos_token_id is None or vision_start_id is None:
            raise ValueError("Cosmos3 tokenizer must provide eos_token_id and <|vision_start|>.")

        cond_input_ids = self._tokenize_chat(cond_text, is_image=is_image) + [eos_token_id, vision_start_id]
        uncond_input_ids = self._tokenize_chat(uncond_text, is_image=is_image) + [eos_token_id, vision_start_id]
        return cond_input_ids, uncond_input_ids

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local(self):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_tokenizer = self.load_text_encoder()
        cond_input_ids, uncond_input_ids = self.tokenize_prompt(
            self.input_info.prompt,
            negative_prompt=self.input_info.negative_prompt,
        )
        self.input_info.txt_seq_lens = [len(cond_input_ids), len(uncond_input_ids)]
        return {
            "text_encoder_output": {
                "cond_input_ids": cond_input_ids,
                "uncond_input_ids": uncond_input_ids,
            },
            "image_encoder_output": None,
        }

    def set_target_shape(self):
        if len(self.input_info.target_shape) == 2:
            height, width = self.input_info.target_shape
            height, width = int(height), int(width)
        else:
            height = int(self.config.get("target_height", 1024))
            width = int(self.config.get("target_width", 1024))

        spatial_scale = int(self.config.get("vae_scale_factor_spatial", self.config.get("vae_scale_factor", 16)))
        temporal_scale = int(self.config.get("vae_scale_factor_temporal", 4))
        if height < spatial_scale or width < spatial_scale:
            raise ValueError(f"Cosmos3 target size must be at least {spatial_scale}x{spatial_scale}, got {height}x{width}.")
        rounded_height = height // spatial_scale * spatial_scale
        rounded_width = width // spatial_scale * spatial_scale
        if rounded_height != height or rounded_width != width:
            logger.warning(f"Cosmos3 target shape rounded from {height}x{width} to {rounded_height}x{rounded_width}")
            height, width = rounded_height, rounded_width

        latent_channels = int(self.config.get("latent_channel", 48))
        pixel_frames = int(self.config.get("target_video_length", 1))
        latent_frames = (pixel_frames - 1) // temporal_scale + 1
        self.input_info.auto_height = height
        self.input_info.auto_width = width
        self.input_info.target_shape = (1, latent_channels, latent_frames, height // spatial_scale, width // spatial_scale)
        self.input_info.image_shapes = [[(latent_frames, height // spatial_scale, width // spatial_scale)]]
        logger.info(f"Cosmos3 Runner set target shape: {width}x{height}, latent: {self.input_info.target_shape}")

    def _load_i2v_condition_frame(self):
        image_path = getattr(self.input_info, "image_path", "")
        if not image_path:
            raise ValueError("Cosmos3 i2v requires --image_path.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Cosmos3 i2v image_path does not exist: {image_path}")
        height = int(self.input_info.auto_height)
        width = int(self.input_info.auto_width)
        resample = getattr(Image, "Resampling", Image).BILINEAR
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((width, height), resample=resample)
            frame = np.asarray(image).astype(np.float32) / 127.5 - 1.0
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return frame.to(device=AI_DEVICE, dtype=GET_DTYPE())

    @ProfilingContext4DebugL2("Prepare i2v condition")
    def _prepare_i2v_condition_latents(self):
        if self.config["task"] != "i2v":
            return
        if hasattr(self.input_info, "vision_condition_latents") and self.input_info.vision_condition_latents is not None:
            return
        loaded_vae_here = not hasattr(self, "vae") or self.vae is None
        if loaded_vae_here:
            self.vae = self.load_vae()
        frame = self._load_i2v_condition_frame()
        num_frames = int(self.config.get("target_video_length", 189))
        video = frame.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()
        condition_latents = self.vae.encode(video)
        self.input_info.vision_condition_latents = condition_latents
        self.input_info.vision_condition_frame_indexes = [0]
        del video, frame
        if loaded_vae_here and (self.config.get("lazy_load", False) or self.config.get("unload_modules", False)):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()

    @ProfilingContext4DebugL2("Run DiT")
    def _run_dit_local(self, total_steps=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        self._prepare_i2v_condition_latents()
        self.model.scheduler.prepare(self.input_info)
        if hasattr(self.input_info, "vision_condition_latents"):
            self.input_info.vision_condition_latents = None
        return self.run(total_steps)

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_labels=["Cosmos3Runner"],
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
            with ProfilingContext4DebugL1("infer_main"):
                self.model.infer(self.inputs)
            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()
            if self.progress_callback:
                self.progress_callback(((step_index + 1) / total_steps) * 100, 100)
        return self.model.scheduler.latents, self.model.scheduler.generator

    def _save_images(self, images, input_info, log_prefix="Image saved"):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if input_info.return_result_tensor or not input_info.save_result_path:
            return
        save_path = input_info.save_result_path
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if self.config["task"] == "i2v":
            if isinstance(images, torch.Tensor):
                if images.dim() == 5:
                    video = images[0].permute(1, 2, 3, 0).contiguous()
                elif images.dim() == 4:
                    video = images
                else:
                    raise ValueError(f"Cosmos3 i2v tensor output must be 4D or 5D, got {tuple(images.shape)}")
            else:
                frames = [torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0) for image in images]
                video = torch.stack(frames, dim=0)
            save_to_video(
                video.clamp(0, 1),
                save_path,
                fps=float(self.config.get("target_fps", 24.0)),
                method=self.config.get("save_video_method", "ffmpeg"),
            )
            logger.info(f"Video saved: {save_path}")
            return
        image_prefix, image_suffix = os.path.splitext(save_path)
        image_suffix = image_suffix.lstrip(".") or "png"
        if isinstance(images, list) and len(images) > 1:
            for idx, image in enumerate(images):
                path = f"{image_prefix}_{idx:05d}.{image_suffix}"
                image.save(path)
                logger.info(f"{log_prefix}: {path}")
            return
        image = images[0] if isinstance(images, list) else images
        image.save(f"{image_prefix}.{image_suffix}")
        logger.info(f"{log_prefix}: {image_prefix}.{image_suffix}")

    def _finalize_pipeline_outputs(self, input_info, images, latents=None, generator=None):
        if latents is not None:
            del latents
        if generator is not None:
            del generator
        torch_device_module.empty_cache()
        gc.collect()
        if input_info.return_result_tensor:
            return {"video" if self.config["task"] == "i2v" else "images": images}
        if input_info.save_result_path is not None:
            return {"video" if self.config["task"] == "i2v" else "images": None}
        return {"video" if self.config["task"] == "i2v" else "images": images}

    def end_run(self):
        if hasattr(self, "model") and self.model is not None:
            self.model.scheduler.clear()
        elif hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.clear()
        if hasattr(self, "inputs"):
            del self.inputs
        self.input_info = None
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "text_tokenizer"):
                del self.text_tokenizer
            torch_device_module.empty_cache()
            gc.collect()

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info
        self.set_target_shape()
        self.inputs = self.run_input_encoder()
        logger.info(f"input_info: {self.input_info}")
        latents, generator = self.run_dit()
        images = self.run_vae_decoder(latents)
        self.end_run()
        self._save_images(images, input_info, log_prefix="Image saved")
        return self._finalize_pipeline_outputs(input_info, images, latents=latents, generator=generator)
