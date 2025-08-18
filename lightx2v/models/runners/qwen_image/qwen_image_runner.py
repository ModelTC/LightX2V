import gc

import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.qwen25.qwen25_vlforconditionalgeneration import Qwen25_VLForConditionalGeneration_TextEncoder
from lightx2v.models.networks.qwen_image.model import QwenImageTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.qwen_image.scheduler import QwenImageScheduler
from lightx2v.models.video_encoders.hf.qwen_image.vae import AutoencoderKLQwenImageVAE
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("qwen_image")
class QwenImageRunner(DefaultRunner):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, config):
        super().__init__(config)

    @ProfilingContext("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()

    def load_transformer(self):
        model = QwenImageTransformerModel(self.config)
        return model

    def load_text_encoder(self):
        text_encoder = Qwen25_VLForConditionalGeneration_TextEncoder(self.config)
        text_encoders = [text_encoder]
        return text_encoders

    def load_image_encoder(self):
        pass

    def load_vae(self):
        vae = AutoencoderKLQwenImageVAE(self.config)
        return vae

    def init_modules(self):
        logger.info("Initializing runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False)
        self.run_dit = self._run_dit_local
        self.run_vae_decoder = self._run_vae_decoder_local
        if self.config["task"] == "t2i":
            self.run_input_encoder = self._run_input_encoder_local_i2v
        else:
            assert NotImplementedError

    @ProfilingContext("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
        text_encoder_output = self.run_text_encoder(prompt)
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    def run_text_encoder(self, text):
        text_encoder_output = {}
        prompt_embeds, prompt_embeds_mask = self.text_encoders[0].infer([text])
        text_encoder_output["prompt_embeds"] = prompt_embeds
        text_encoder_output["prompt_embeds_mask"] = prompt_embeds_mask
        return text_encoder_output

    def set_target_shape(self):
        self.vae_scale_factor = self.vae.vae_scale_factor if getattr(self, "vae", None) else 8
        width, height = self.config.aspect_ratios[self.config.aspect_ratio]
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        num_channels_latents = self.model.in_channels // 4
        self.config.target_shape = (self.config.batchsize, 1, num_channels_latents, height, width)

    def init_scheduler(self):
        scheduler = QwenImageScheduler(self.config)
        self.model.set_scheduler(scheduler)
        self.model.pre_infer.set_scheduler(scheduler)
        self.model.transformer_infer.set_scheduler(scheduler)
        self.model.post_infer.set_scheduler(scheduler)

    def get_encoder_output_i2v(self):
        pass

    def run_image_encoder(self):
        pass

    def run_vae_encoder(self):
        pass

    @ProfilingContext("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.image_encoder = self.load_image_encoder()
        self.vae = self.load_vae()
        self.vfi_model = self.load_vfi_model() if "video_frame_interpolation" in self.config else None

    @ProfilingContext("Run VAE Decoder")
    def _run_vae_decoder_local(self, latents, generator):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae()
        images = self.vae.decode(latents)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()
        return images

    def run_pipeline(self, save_image=True):
        if self.config["use_prompt_enhancer"]:
            self.config["prompt_enhanced"] = self.post_prompt_enhancer()

        self.inputs = self.run_input_encoder()
        self.set_target_shape()
        latents, generator = self.run_dit()

        images = self.run_vae_decoder(latents, generator)
        image = images[0]
        image.save(f"{self.config.save_video_path}")

        del latents, generator
        torch.cuda.empty_cache()
        gc.collect()

        # Return (images, audio) - audio is None for default runner
        return images, None
