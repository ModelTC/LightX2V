import gc
import math
import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.input_encoders.hf.ernie_image.mistral3_model import ErnieImageTextEncoder
from lightx2v.models.networks.ernie_image.model import ErnieImageTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.ernie_image.scheduler import ErnieImageScheduler
from lightx2v.models.video_encoders.hf.ernie_image.vae import ErnieImageVAE
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def calculate_dimensions(target_area, ratio, multiple_of):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / multiple_of) * multiple_of
    height = round(height / multiple_of) * multiple_of
    return int(width), int(height)


@RUNNER_REGISTER("ernie_image_turbo")
@RUNNER_REGISTER("ernie_image")
class ErnieImageRunner(DefaultRunner):
    model_cpu_offload_seq = "pe->text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, config):
        super().__init__(config)
        self.resolution = config.get("resolution", 1024)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()

    def load_transformer(self):
        return ErnieImageTransformerModel(
            model_path=os.path.join(self.config["model_path"], "transformer"),
            config=self.config,
            device=self.init_device,
        )

    def load_text_encoder(self):
        return [ErnieImageTextEncoder(self.config)]

    def load_image_encoder(self):
        return None

    def load_vae(self):
        return ErnieImageVAE(self.config)

    def init_scheduler(self):
        self.scheduler = ErnieImageScheduler(self.config)

    def init_modules(self):
        logger.info("Initializing ERNIE-Image runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
            self.model.set_scheduler(self.scheduler)
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False)
        self.run_dit = self._run_dit_local
        if self.config["task"] != "t2i":
            raise NotImplementedError(f"ErnieImageRunner only supports t2i, got: {self.config['task']}")
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
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()
        text_encoder_output = self.run_text_encoder(
            self.input_info.prompt,
            neg_prompt=self.input_info.negative_prompt,
        )
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    @ProfilingContext4DebugL1("Run Text Encoder")
    def run_text_encoder(self, text, neg_prompt=None):
        width = getattr(self.input_info, "auto_width", self.config.get("target_width", 1024))
        height = getattr(self.input_info, "auto_height", self.config.get("target_height", 1024))
        prompt_embeds_list, revised_prompts = self.text_encoders[0].infer(
            [text],
            use_pe=self.config.get("use_pe", True),
            width=width,
            height=height,
        )
        prompt_embeds = prompt_embeds_list[0]
        self.input_info.txt_seq_lens = [prompt_embeds.shape[0]]
        self.input_info.revised_prompts = revised_prompts

        text_encoder_output = {"prompt_embeds": prompt_embeds}
        if self.config.get("enable_cfg", False):
            if neg_prompt is None:
                neg_prompt = ""
            negative_prompt_embeds_list, _ = self.text_encoders[0].infer(
                [neg_prompt],
                use_pe=False,
                width=width,
                height=height,
            )
            negative_prompt_embeds = negative_prompt_embeds_list[0]
            self.input_info.txt_seq_lens.append(negative_prompt_embeds.shape[0])
            text_encoder_output["negative_prompt_embeds"] = negative_prompt_embeds
        return text_encoder_output

    def set_target_shape(self):
        vae_scale_factor = self.config.get("vae_scale_factor", 16)
        if len(self.input_info.target_shape) == 2:
            height, width = [int(v) for v in self.input_info.target_shape]
        else:
            target_height = self.config.get("target_height", None)
            target_width = self.config.get("target_width", None)
            if target_height and target_width:
                height, width = int(target_height), int(target_width)
            else:
                aspect_ratio = self.input_info.aspect_ratio or self.config.get("aspect_ratio", "1:1")
                if ":" in aspect_ratio:
                    w_ratio, h_ratio = [float(item) for item in aspect_ratio.split(":", 1)]
                    ratio = w_ratio / h_ratio
                else:
                    ratio = float(aspect_ratio)
                width, height = calculate_dimensions(
                    self.resolution * self.resolution,
                    ratio,
                    vae_scale_factor,
                )

        if height % vae_scale_factor != 0 or width % vae_scale_factor != 0:
            raise ValueError(f"Height and width must be divisible by {vae_scale_factor}, got {height}x{width}.")

        self.input_info.auto_width = width
        self.input_info.auto_height = height
        self.input_info.target_shape = (
            1,
            self.config.get("in_channels", 128),
            height // vae_scale_factor,
            width // vae_scale_factor,
        )
        logger.info(f"ERNIE-Image target shape: {width}x{height}, latent shape: {self.input_info.target_shape}")

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

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae = self.load_vae()
        images = self.vae.decode(latents.to(GET_DTYPE()), self.input_info)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()
        return images

    def _save_images(self, images, input_info):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if input_info.return_result_tensor or not input_info.save_result_path:
            return
        image_prefix = input_info.save_result_path.rsplit(".", 1)[0]
        image_suffix = input_info.save_result_path.rsplit(".", 1)[1] if len(input_info.save_result_path.rsplit(".", 1)) > 1 else "png"
        for idx, image in enumerate(images):
            if idx == 0:
                image_path = f"{image_prefix}.{image_suffix}"
            else:
                image_path = f"{image_prefix}_{idx:05d}.{image_suffix}"
            image.save(image_path)
            logger.info(f"Image saved: {image_path}")

    def _finalize_pipeline_outputs(self, input_info, images, revised_prompts, latents=None, generator=None):
        if latents is not None:
            del latents
        if generator is not None:
            del generator
        torch_device_module.empty_cache()
        gc.collect()
        if input_info.return_result_tensor:
            return {"images": images, "revised_prompts": revised_prompts}
        return {"images": None, "revised_prompts": revised_prompts}

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info
        self.set_target_shape()
        self.inputs = self.run_input_encoder()
        logger.info(f"input_info: {self.input_info}")
        latents, generator = self.run_dit()
        images = self.run_vae_decoder(latents)
        revised_prompts = getattr(self.input_info, "revised_prompts", None)
        self._save_images(images, input_info)
        self.end_run()
        return self._finalize_pipeline_outputs(input_info, images, revised_prompts, latents=latents, generator=generator)
