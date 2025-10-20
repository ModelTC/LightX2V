import gc
import math

import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.qwen25.qwen25_vlforconditionalgeneration import Qwen25_VLForConditionalGeneration_TextEncoder
from lightx2v.models.networks.qwen_image.model import QwenImageTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.qwen_image.scheduler import QwenImageScheduler
from lightx2v.models.video_encoders.hf.qwen_image.vae import AutoencoderKLQwenImageVAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


@RUNNER_REGISTER("qwen_image")
class QwenImageRunner(DefaultRunner):
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
        if self.config["task"] == "t2i":
            self.run_input_encoder = self._run_input_encoder_local_t2i
        elif self.config["task"] == "i2i":
            self.run_input_encoder = self._run_input_encoder_local_i2i
        else:
            assert NotImplementedError

        self.model.set_scheduler(self.scheduler)

    @ProfilingContext4DebugL2("Run DiT")
    def _run_dit_local(self, total_steps=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
        self.model.scheduler.prepare(self.input_info)
        latents, generator = self.run(total_steps)
        return latents, generator

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

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2i(self):
        _, image = self.read_image_input(self.input_info.image_path)
        prompt = self.input_info.prompt
        text_encoder_output = self.run_text_encoder(prompt, image, neg_prompt=self.input_info.negative_prompt)
        image_encoder_output = self.run_vae_encoder(image=text_encoder_output["preprocessed_image"])
        image_encoder_output["image_info"] = text_encoder_output["image_info"]
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

    @ProfilingContext4DebugL1("Run Text Encoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_text_encode_duration, metrics_labels=["QwenImageRunner"])
    def run_text_encoder(self, text, image=None, neg_prompt=None):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_prompt_len.observe(len(text))
        text_encoder_output = {}
        if self.config["task"] == "t2i":
            prompt_embeds, prompt_embeds_mask, _, _ = self.text_encoders[0].infer([text])
            text_encoder_output["prompt_embeds"] = prompt_embeds
            text_encoder_output["prompt_embeds_mask"] = prompt_embeds_mask
            if self.config["do_true_cfg"] and neg_prompt is not None:
                neg_prompt_embeds, neg_prompt_embeds_mask, _, _ = self.text_encoders[0].infer([neg_prompt])
                text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
                text_encoder_output["negative_prompt_embeds_mask"] = neg_prompt_embeds_mask
        elif self.config["task"] == "i2i":
            prompt_embeds, prompt_embeds_mask, preprocessed_image, image_info = self.text_encoders[0].infer([text], image)
            text_encoder_output["prompt_embeds"] = prompt_embeds
            text_encoder_output["prompt_embeds_mask"] = prompt_embeds_mask
            text_encoder_output["preprocessed_image"] = preprocessed_image
            text_encoder_output["image_info"] = image_info
            if self.config["do_true_cfg"] and neg_prompt is not None:
                neg_prompt_embeds, neg_prompt_embeds_mask, _, _ = self.text_encoders[0].infer([neg_prompt], image)
                text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
                text_encoder_output["negative_prompt_embeds_mask"] = neg_prompt_embeds_mask
        return text_encoder_output

    @ProfilingContext4DebugL1("Run VAE Encoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_vae_encode_duration, metrics_labels=["QwenImageRunner"])
    def run_vae_encoder(self, image):
        image_latents = self.vae.encode_vae_image(image, self.input_info)
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

    def set_target_shape(self):
        if not self.config["_auto_resize"]:
            width, height = self.config["aspect_ratios"][self.config["aspect_ratio"]]
        else:
            width, height = self.input_info.original_size
            calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, width / height)
            multiple_of = self.vae.vae_scale_factor * 2
            width = calculated_width // multiple_of * multiple_of
            height = calculated_height // multiple_of * multiple_of
            self.input_info.auto_width = width
            self.input_info.auto_hight = height

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae.vae_scale_factor * 2))
        num_channels_latents = self.model.in_channels // 4
        self.input_info.target_shape = (self.config["batchsize"], 1, num_channels_latents, height, width)

    def set_img_shapes(self):
        if self.config["task"] == "t2i":
            width, height = self.config["aspect_ratios"][self.config["aspect_ratio"]]
            img_shapes = [(1, height // self.config["vae_scale_factor"] // 2, width // self.config["vae_scale_factor"] // 2)] * self.config["batchsize"]
        elif self.config["task"] == "i2i":
            image_height, image_width = self.inputs["image_encoder_output"]["image_info"]
            img_shapes = [
                [
                    (1, self.input_info.auto_hight // self.config["vae_scale_factor"] // 2, self.input_info.auto_width // self.config["vae_scale_factor"] // 2),
                    (1, image_height // self.config["vae_scale_factor"] // 2, image_width // self.config["vae_scale_factor"] // 2),
                ]
            ]
        self.inputs["img_shapes"] = img_shapes

    def init_scheduler(self):
        self.scheduler = QwenImageScheduler(self.config)

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

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_decode_duration,
        metrics_labels=["QwenImageRunner"],
    )
    def run_vae_decoder(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae()
        images = self.vae.decode(latents, self.input_info)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()
        return images

    def run_pipeline(self, input_info):
        self.input_info = input_info

        self.inputs = self.run_input_encoder()
        self.set_target_shape()
        self.set_img_shapes()

        latents, generator = self.run_dit()
        images = self.run_vae_decoder(latents)
        self.end_run()

        image = images[0]
        image.save(f"{input_info.save_result_path}")

        del latents, generator
        torch.cuda.empty_cache()
        gc.collect()

        # Return (images, audio) - audio is None for default runner
        return images, None
