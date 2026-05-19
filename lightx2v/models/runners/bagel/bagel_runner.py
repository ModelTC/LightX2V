import gc
import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.bagel.model import BagelModel
from lightx2v.models.runners.bagel.t2i_utils import get_bagel_latent_downsample, resolve_bagel_t2i_image_shape
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.bagel.scheduler import BagelScheduler
from lightx2v.models.video_encoders.hf.bagel.vae import BagelVae
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def _has_save_path(input_info):
    return bool(getattr(input_info, "save_result_path", None))


@RUNNER_REGISTER("bagel")
class BagelRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)

    def init_scheduler(self):
        self.scheduler = BagelScheduler(self.config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_bagel_model()
        self.vae_decoder = self.load_vae_decoder()

    def load_bagel_model(self):
        model = BagelModel(self.config)
        return model

    def load_vae_decoder(self):
        vae_model = BagelVae(self.config)
        return vae_model

    def init_modules(self):
        logger.info("Initializing runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False)
        self.run_dit = self._run_dit_local

    def set_image_shapes(self):
        image_shape = resolve_bagel_t2i_image_shape(self.input_info, self.config)
        self.input_info.image_shapes = image_shape
        self.input_info.target_shape = list(image_shape)
        logger.info(f"BAGEL T2I image shape: {image_shape[0]}x{image_shape[1]}")
        return image_shape

    def run(self, total_steps=None):
        if total_steps is None:
            total_steps = self.model.scheduler.infer_steps - 1

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

    @ProfilingContext4DebugL2("Run DiT")
    def _run_dit_local(self, total_steps=None):
        latents, generator = self.run(total_steps)
        return latents, generator

    def _refresh_scheduler_from_config(self):
        self.scheduler.infer_steps = int(self.config.get("infer_steps", self.scheduler.infer_steps))
        self.scheduler.timestep_shift = self.config["inference_hyper"]["timestep_shift"]
        self.scheduler.set_timesteps()

    @ProfilingContext4DebugL1("Run VAE Decoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_vae_decode_duration, metrics_labels=["DefaultRunner"])
    def run_vae_decoder(self, latents, decode_info):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae_decoder()
        images = self.vae_decoder.decode(latents, decode_info)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch_device_module.empty_cache()
            gc.collect()
        return images

    def _build_decode_info(self, image_shape):
        return {
            "packed_seqlens": self.inputs.generation_input["packed_seqlens"],
            "image_shape": image_shape,
            "latent_downsample": get_bagel_latent_downsample(self.config),
            "latent_channel": self.config["vae_config"]["z_channels"],
            "latent_patch_size": self.config["latent_patch_size"],
        }

    def _save_images(self, images, input_info, log_prefix="Image saved"):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if input_info.return_result_tensor:
            return
        if not _has_save_path(input_info):
            return

        image_prefix, image_suffix = os.path.splitext(input_info.save_result_path)
        image_suffix = image_suffix.lstrip(".") or "png"
        if isinstance(images[0], list) and len(images[0]) > 1:
            for idx, image in enumerate(images[0]):
                out_path = f"{image_prefix}_{idx:05d}.{image_suffix}"
                image.save(out_path)
                logger.info(f"{log_prefix}: {out_path}")
        else:
            out_path = f"{image_prefix}.{image_suffix}"
            images[0].save(out_path)
            logger.info(f"{log_prefix}: {out_path}")

    def _finalize_pipeline_outputs(self, input_info, images, latents=None, generator=None):
        if latents is not None:
            del latents
        if generator is not None:
            del generator
        torch_device_module.empty_cache()
        gc.collect()

        if input_info.return_result_tensor:
            return {"images": images}
        if _has_save_path(input_info):
            return {"images": None}
        return {"images": images}

    def run_pipeline(self, input_info):
        if self.config["task"] != "t2i":
            raise NotImplementedError("BAGEL v1 support in LightX2V currently only implements task='t2i'")

        self.input_info = input_info
        logger.info(f"input_info: {self.input_info}")
        if getattr(self.input_info, "negative_prompt", ""):
            logger.warning("BAGEL T2I v1 does not use negative_prompt; the value will be ignored.")

        self._refresh_scheduler_from_config()
        image_shape = self.set_image_shapes()

        self.inputs, self.scheduler = self.model.prepare_inputs(self.input_info, self.scheduler)
        self.model.set_scheduler(self.scheduler)

        latents, generator = self.run_dit()
        decode_info = self._build_decode_info(image_shape)
        images = self.run_vae_decoder(latents, decode_info)
        self.end_run()

        self._save_images(images, input_info)
        return self._finalize_pipeline_outputs(input_info, images, latents=latents, generator=generator)
