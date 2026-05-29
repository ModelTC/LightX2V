import gc
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.input_encoders.hf.ernie_image.text_encoder import ErnieImageTextEncoder
from lightx2v.models.networks.ernie_image.model import ErnieImageTransformerModel
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.ernie_image.components import ErnieImagePipelineComponents
from lightx2v.models.schedulers.ernie_image.scheduler import ErnieImageScheduler
from lightx2v.models.video_encoders.hf.ernie_image.vae import AutoencoderKLErnieImageVAE
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

ERNIE_IMAGE_ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (720, 1280),
    "9:16": (1280, 720),
    "4:3": (896, 1152),
    "3:4": (1152, 896),
    "3:2": (832, 1248),
    "2:3": (1248, 832),
}


def _truthy(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _validate_image_shape(height, width):
    if height <= 0 or width <= 0:
        raise ValueError(f"ERNIE-Image height and width must be positive, got {height}x{width}.")
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"ERNIE-Image height and width must be divisible by 16, got {height}x{width}.")


def resolve_ernie_image_shape(input_info, config):
    target_shape = getattr(input_info, "target_shape", None)
    if target_shape:
        if len(target_shape) != 2:
            raise ValueError(f"ERNIE-Image target_shape must be [height, width], got {target_shape}.")
        height, width = int(target_shape[0]), int(target_shape[1])
        _validate_image_shape(height, width)
        return height, width

    target_height = config.get("target_height")
    target_width = config.get("target_width")
    if target_height and target_width:
        height, width = int(target_height), int(target_width)
        _validate_image_shape(height, width)
        return height, width

    aspect_ratio = getattr(input_info, "aspect_ratio", None) or config.get("aspect_ratio", "1:1")
    if not aspect_ratio:
        aspect_ratio = "1:1"
    if aspect_ratio not in ERNIE_IMAGE_ASPECT_RATIOS:
        supported = ", ".join(sorted(ERNIE_IMAGE_ASPECT_RATIOS))
        raise ValueError(f"Unsupported ERNIE-Image aspect_ratio '{aspect_ratio}'. Supported ratios: {supported}.")

    height, width = ERNIE_IMAGE_ASPECT_RATIOS[aspect_ratio]
    _validate_image_shape(height, width)
    return height, width


@RUNNER_REGISTER("ernie_image")
class ErnieImageRunner(BaseRunner):
    model_cpu_offload_seq = "pe->text_encoder->transformer->vae"

    def __init__(self, config):
        super().__init__(config)
        self.components = None
        self.pipe = None
        self.model = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self.progress_callback = None

    def set_config(self, config_modify):
        logger.info(f"modify config: {config_modify}")
        if hasattr(self.config, "temporarily_unlocked"):
            with self.config.temporarily_unlocked():
                self.config.update(config_modify)
        else:
            self.config.update(config_modify)

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def init_modules(self):
        logger.info("Initializing ERNIE-Image Diffusers runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False), "ERNIE-Image lazy_load requires cpu_offload."

        if hasattr(self.config, "lock"):
            self.config.lock()

    def _get_target_device(self):
        if AI_DEVICE is not None:
            return AI_DEVICE
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _from_pretrained_kwargs(self):
        kwargs = {"torch_dtype": GET_DTYPE()}
        for key in ("revision", "variant", "cache_dir", "local_files_only"):
            value = self.config.get(key)
            if value is not None:
                kwargs[key] = value
        return kwargs

    def load_model(self):
        logger.info(f"Loading ERNIE-Image pipeline from {self.config['model_path']}")
        self.components = ErnieImagePipelineComponents.from_pretrained(
            self.config["model_path"],
            self._from_pretrained_kwargs(),
            target_device=self._get_target_device(),
            cpu_offload=self.config.get("cpu_offload", False),
        )
        self.pipe = self.components.pipeline
        self.text_encoder = self.load_text_encoder()
        self.scheduler = self.load_scheduler()
        self.model = self.load_transformer()
        self.model.set_scheduler(self.scheduler)
        self.vae = self.load_vae()
        return self.pipe

    def load_transformer(self):
        model = ErnieImageTransformerModel(
            self.config,
            self.components.transformer,
            diffusers_cpu_offload=self.config.get("cpu_offload", False),
        )
        if self.scheduler is not None:
            model.set_scheduler(self.scheduler)
        return model

    def load_text_encoder(self):
        return ErnieImageTextEncoder(
            self.config,
            self.components,
            diffusers_cpu_offload=self.config.get("cpu_offload", False),
        )

    def load_scheduler(self):
        return ErnieImageScheduler(self.config, self.components.scheduler)

    def load_vae(self):
        return AutoencoderKLErnieImageVAE(
            self.config,
            self.components.vae,
            unpatchify_latents=self.components.unpatchify_latents,
            diffusers_cpu_offload=self.config.get("cpu_offload", False),
        )

    def _ensure_pipe(self):
        if self.pipe is None:
            self.load_model()
        else:
            if self.text_encoder is None:
                self.text_encoder = self.load_text_encoder()
            if self.scheduler is None:
                self.scheduler = self.load_scheduler()
            if self.model is None:
                self.model = self.load_transformer()
            if self.model.scheduler is not self.scheduler:
                self.model.set_scheduler(self.scheduler)
            if self.vae is None:
                self.vae = self.load_vae()

    def _make_generator(self, seed):
        if seed is None:
            return None
        return torch.Generator(device="cpu").manual_seed(int(seed))

    def _wrap_progress_callback(self):
        total_steps = max(int(self.config.get("infer_steps", self.config.get("num_inference_steps", 50))), 1)

        def _callback_on_step_end(pipe, step, timestep, callback_kwargs):
            del pipe, timestep
            self.progress_callback(((step + 1) / total_steps) * 100, 100)
            return callback_kwargs

        return _callback_on_step_end

    def _build_generation_kwargs(self, input_info):
        height, width = resolve_ernie_image_shape(input_info, self.config)
        kwargs = {
            "prompt": input_info.prompt,
            "negative_prompt": getattr(input_info, "negative_prompt", ""),
            "height": height,
            "width": width,
            "num_inference_steps": int(self.config.get("infer_steps", self.config.get("num_inference_steps", 50))),
            "guidance_scale": float(self.config.get("sample_guide_scale", self.config.get("guidance_scale", 4.0))),
            "seed": getattr(input_info, "seed", None),
            "use_pe": _truthy(self.config.get("use_pe", True)),
            "output_type": self.config.get("output_type", "pil"),
        }

        for key in ("num_images_per_prompt", "callback_on_step_end_tensor_inputs"):
            value = self.config.get(key)
            if value is not None:
                kwargs[key] = value

        return kwargs

    def _build_pipeline_kwargs(self, input_info):
        kwargs = self._build_generation_kwargs(input_info)
        seed = kwargs.pop("seed", None)
        kwargs["generator"] = self._make_generator(seed)
        if self.progress_callback is not None:
            kwargs["callback_on_step_end"] = self._wrap_progress_callback()

        return kwargs

    def _execution_device(self):
        device = self.components.execution_device
        if device is not None:
            return device
        return torch.device(self._get_target_device())

    def _encode_prompts(self, generation_kwargs, device, dtype):
        if self.text_encoder is None:
            self.text_encoder = self.load_text_encoder()
        return self.text_encoder.encode(
            generation_kwargs,
            device,
            dtype,
            do_classifier_free_guidance=self._do_classifier_free_guidance(generation_kwargs),
        )

    def _do_classifier_free_guidance(self, generation_kwargs):
        return float(generation_kwargs["guidance_scale"]) > 1.0

    def run_vae_decoder(self, latents, output_type):
        if self.vae is None:
            self.vae = self.load_vae()
        return self.vae.decode(latents, output_type=output_type)

    def run_scheduler_step(self, pred, timestep, latents=None):
        if self.scheduler is None:
            self.scheduler = self.load_scheduler()
        return self.scheduler.step(pred, timestep, latents)

    def _run_decomposed_pipeline(self, input_info):
        generation_kwargs = self._build_generation_kwargs(input_info)
        device = self._execution_device()
        if self.model is None:
            self.model = self.load_transformer()
        dtype = self.model.dtype or GET_DTYPE()
        num_inference_steps = int(generation_kwargs["num_inference_steps"])
        guidance_scale = float(generation_kwargs["guidance_scale"])
        num_images_per_prompt = int(generation_kwargs.get("num_images_per_prompt", 1))

        batch_size, text_bth, text_lens = self._encode_prompts(generation_kwargs, device, dtype)
        total_batch_size = batch_size * num_images_per_prompt

        latent_height = generation_kwargs["height"] // self.components.vae_scale_factor
        latent_width = generation_kwargs["width"] // self.components.vae_scale_factor
        latent_channels = self.components.transformer.config.in_channels
        latent_shape = (total_batch_size, latent_channels, latent_height, latent_width)

        if self.scheduler is None:
            self.scheduler = self.load_scheduler()
        if self.model.scheduler is not self.scheduler:
            self.model.set_scheduler(self.scheduler)
        timesteps = self.scheduler.prepare(
            num_inference_steps,
            device,
            latent_shape=latent_shape,
            dtype=dtype,
            seed=generation_kwargs["seed"],
        )
        latents = self.scheduler.latents

        do_cfg = self._do_classifier_free_guidance(generation_kwargs)
        with torch.no_grad():
            for step_index, timestep in enumerate(timesteps):
                self.scheduler.step_pre(step_index)
                self.model.infer(
                    {
                        "timestep": timestep,
                        "text_bth": text_bth,
                        "text_lens": text_lens,
                        "device": device,
                        "dtype": dtype,
                        "total_batch_size": total_batch_size,
                        "guidance_scale": guidance_scale,
                        "do_classifier_free_guidance": do_cfg,
                    }
                )

                latents = self.run_scheduler_step(self.scheduler.noise_pred, timestep)

                if self.progress_callback:
                    self.progress_callback(((step_index + 1) / num_inference_steps) * 100, 100)

        images = self.run_vae_decoder(latents, generation_kwargs["output_type"])
        self.components.maybe_free_model_hooks()
        return images

    def _run_diffusers_pipeline(self, input_info):
        output = self.components.run_pipeline_call(self._build_pipeline_kwargs(input_info))
        if isinstance(output, torch.Tensor):
            return output
        return output.images

    def _unload_pipe(self):
        if self.components is None:
            self.pipe = None
            self.model = None
            self.text_encoder = None
            self.vae = None
            self.scheduler = None
            return
        self.components = None
        self.model = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        del self.pipe
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_images(self, images, save_result_path):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if not save_result_path:
            return

        output_path = Path(save_result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_prefix = str(output_path.with_suffix(""))
        image_suffix = output_path.suffix[1:] if output_path.suffix else "png"

        if len(images) == 1:
            images[0].save(str(output_path))
            logger.info(f"Image saved: {output_path}")
            return

        for idx, image in enumerate(images):
            image_path = f"{image_prefix}_{idx:05d}.{image_suffix}"
            image.save(image_path)
            logger.info(f"Image saved: {image_path}")

    @torch.no_grad()
    def run_pipeline(self, input_info):
        self.input_info = input_info
        self._ensure_pipe()

        logger.info(f"input_info: {self.input_info}")
        try:
            if self.config.get("use_diffusers_pipeline_call", False):
                images = self._run_diffusers_pipeline(input_info)
            else:
                images = self._run_decomposed_pipeline(input_info)

            if getattr(input_info, "return_result_tensor", False):
                return {"images": images}

            self._save_images(images, getattr(input_info, "save_result_path", None))
            return {"images": None}
        finally:
            if self.config.get("unload_modules", False):
                self._unload_pipe()
