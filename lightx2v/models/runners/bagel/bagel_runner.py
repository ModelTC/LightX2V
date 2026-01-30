import gc

import torch
from loguru import logger

from lightx2v.models.networks.bagel.model import BagelModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.bagel.scheduler import BagelScheduler
from lightx2v.models.video_encoders.hf.bagel.vae import BagelVae
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


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
        self.input_info.image_shapes = (1024, 1024)

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

    def run_pipeline(self, input_info):
        self.input_info = input_info
        logger.info(f"input_info: {self.input_info}")

        self.inputs, self.scheduler = self.model.prepare_inputs(self.input_info, self.scheduler)
        self.model.set_scheduler(self.scheduler)

        self.set_image_shapes()
        latents, generator = self.run_dit()
        decode_info = {
            "packed_seqlens": self.inputs.generation_input["packed_seqlens"],
            "image_shape": [1024, 1024],
            "latent_downsample": 16,
            "latent_channel": 16,
            "latent_patch_size": 2,
        }
        images = self.run_vae_decoder(latents, decode_info)
        self.end_run()

        if isinstance(images[0], list) and len(images[0]) > 1:
            image_prefix = f"{input_info.save_result_path}".split(".")[0]
            for idx, image in enumerate(images[0]):
                image.save(f"{image_prefix}_{idx}.png")
                logger.info(f"Image saved: {image_prefix}_{idx}.png")
        else:
            image = images[0]
            image.save(f"{input_info.save_result_path}")
            logger.info(f"Image saved: {input_info.save_result_path}")

        del latents, generator
        torch_device_module.empty_cache()
        gc.collect()

        # Return (images, audio) - audio is None for default runner
        return images, None
