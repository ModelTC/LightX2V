import gc
import math

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from loguru import logger

from lightx2v.disagg.disagg_mixin import DisaggMixin
from lightx2v.models.input_encoders.hf.qwen25.qwen25_vlforconditionalgeneration import Qwen25_VLForConditionalGeneration_TextEncoder
from lightx2v.models.networks.lora_adapter import LoraAdapter
from lightx2v.models.networks.qwen_image.model import QwenImageTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.qwen_image.scheduler import QwenImageScheduler
from lightx2v.models.video_encoders.hf.qwen_image.vae import AutoencoderKLQwenImageVAE
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


def build_qwen_image_model_with_lora(qwen_module, config, model_kwargs, lora_configs):
    lora_dynamic_apply = config.get("lora_dynamic_apply", False)

    if lora_dynamic_apply:
        lora_path = lora_configs[0]["path"]
        lora_strength = lora_configs[0]["strength"]
        model_kwargs["lora_path"] = lora_path
        model_kwargs["lora_strength"] = lora_strength
        model = qwen_module(**model_kwargs)
    else:
        assert not config.get("dit_quantized", False), "Online LoRA only for quantized models; merging LoRA is unsupported."
        assert not config.get("lazy_load", False), "Lazy load mode does not support LoRA merging."
        model = qwen_module(**model_kwargs)
        lora_adapter = LoraAdapter(model)
        lora_adapter.apply_lora(lora_configs)
    return model


@RUNNER_REGISTER("qwen_image")
class QwenImageRunner(DisaggMixin, DefaultRunner):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, config):
        super().__init__(config)
        self.is_layered = self.config.get("layered", False)
        if self.is_layered:
            self.layers = self.config.get("layers", 4)
        self.resolution = self.config.get("resolution", 1024)

        # Text encoder type: "lightllm_service", "lightllm_kernel", or default (baseline)
        self.text_encoder_type = config.get("text_encoder_type", "baseline")

        if self.text_encoder_type in ["lightllm_service", "lightllm_kernel"]:
            logger.info(f"Using LightLLM text encoder: {self.text_encoder_type}")

    def set_config(self, config_modify):
        """Apply per-request overrides and optionally sync disagg fields."""
        super().set_config(config_modify)
        self.apply_disagg_request_overrides(config_modify)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        disagg_mode = self.config.get("disagg_mode")

        if disagg_mode == "encoder":
            logger.info("[Disagg] Loading models for ENCODER role (QwenImage)...")
            self.model = None
            self.text_encoders = self.load_text_encoder()
            self.vae = self.load_vae()
        elif disagg_mode == "transformer":
            logger.info("[Disagg] Loading models for TRANSFORMER role (QwenImage)...")
            self.model = self.load_transformer()
            self.text_encoders = None
            # Skip VAE when a dedicated Decoder service handles Phase 2 (3-way disagg)
            if self.config.get("disagg_config", {}).get("decoder_engine_rank") is not None:
                self.vae = None
            else:
                self.vae = self.load_vae()
        elif disagg_mode == "decode":
            logger.info("[Disagg] Loading models for DECODE role (QwenImage)...")
            self.model = None
            self.text_encoders = None
            self.vae = self.load_vae()
        else:
            self.model = self.load_transformer()
            self.text_encoders = self.load_text_encoder()
            self.image_encoder = self.load_image_encoder()
            self.vae = self.load_vae()
            self.vfi_model = self.load_vfi_model() if "video_frame_interpolation" in self.config else None

    def load_transformer(self):
        qwen_image_model_kwargs = {
            "model_path": os.path.join(self.config["model_path"], "transformer"),
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = QwenImageTransformerModel(**qwen_image_model_kwargs)
        else:
            model = build_qwen_image_model_with_lora(QwenImageTransformerModel, self.config, qwen_image_model_kwargs, lora_configs)
        return model

    def load_text_encoder(self):
        """Load text encoder based on text_encoder_type configuration.

        Supported types:
        - "lightllm_service": LightLLM HTTP service mode
        - "lightllm_kernel": HuggingFace model with Triton kernel optimizations
        - "baseline" (default): HuggingFace baseline implementation
        """
        encoder_config = dict(self.config)
        encoder_config.update(self.config.get("lightllm_config", {}))

        if self.text_encoder_type == "lightllm_service":
            from lightx2v.models.input_encoders.lightllm import LightLLMServiceTextEncoder

            logger.info("Loading LightLLM service-based text encoder")
            text_encoder = LightLLMServiceTextEncoder(encoder_config)
        elif self.text_encoder_type == "lightllm_kernel":
            from lightx2v.models.input_encoders.lightllm import LightLLMKernelTextEncoder

            logger.info("Loading LightLLM Kernel-optimized text encoder")
            text_encoder = LightLLMKernelTextEncoder(encoder_config)
        else:  # baseline or default
            logger.info("Loading HuggingFace baseline text encoder")
            text_encoder = Qwen25_VLForConditionalGeneration_TextEncoder(self.config)

        text_encoders = [text_encoder]
        return text_encoders

    def load_image_encoder(self):
        pass

    def load_vae(self):
        """Load VAE based on vae_type configuration.

        Supported types:
        - "tensorrt": TensorRT-accelerated VAE (requires pre-built engines)
        - "baseline" (default): HuggingFace baseline implementation
        """
        vae_type = self.config.get("vae_type", "baseline")
        trt_vae_config = self.config.get("trt_vae_config", {})

        if vae_type == "tensorrt":
            try:
                from lightx2v.models.video_encoders.trt.qwen_image.vae_trt import TensorRTVAE

                logger.info("Loading TensorRT-accelerated VAE")
                vae = TensorRTVAE(self.config)
                return vae
            except ImportError as e:
                logger.warning(f"TensorRT not available, falling back to PyTorch VAE: {e}")
            except FileNotFoundError as e:
                logger.warning(f"TensorRT engine files not found, falling back to PyTorch VAE: {e}")
            except Exception as e:
                logger.warning(f"Failed to load TensorRT VAE, falling back to PyTorch VAE: {e}")

        # Baseline or fallback
        logger.info("Loading PyTorch baseline VAE")
        vae = AutoencoderKLQwenImageVAE(self.config)
        return vae

    def init_modules(self):
        if self.config.get("disagg_mode"):
            self.init_disagg(self.config)
        super().init_modules()
        self.run_dit = self._run_dit_local

        disagg_mode = self.config.get("disagg_mode")
        if disagg_mode == "decode":
            # Decoder role does not need a task-specific input encoder
            return
        if self.config["task"] == "t2i":
            self.run_input_encoder = self._run_input_encoder_local_t2i
        elif self.config["task"] == "i2i":
            self.run_input_encoder = self._run_input_encoder_local_i2i
        else:
            raise NotImplementedError(f"QwenImageRunner does not support task: {self.config['task']}")

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
            if self.config.get("layered", False):
                img_ori = Image.open(img_path).convert("RGBA")
            else:
                img_ori = Image.open(img_path).convert("RGB")
        if GET_RECORDER_MODE():
            width, height = img_ori.size
            monitor_cli.lightx2v_input_image_len.observe(width * height)
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
            # Offload text encoder (service mode doesn't need offload)
            if self.text_encoder_type == "lightllm_service":
                pass  # Service mode: no local model to offload
            else:
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

    @ProfilingContext4DebugL1("Run Text Encoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_text_encode_duration, metrics_labels=["QwenImageRunner"])
    def run_text_encoder(self, text, image_list=None, neg_prompt=None):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_prompt_len.observe(len(text))
        text_encoder_output = {}
        if self.config["task"] == "t2i":
            prompt_embeds, _, _ = self.text_encoders[0].infer([text])
            self.input_info.txt_seq_lens = [prompt_embeds.shape[1]]
            text_encoder_output["prompt_embeds"] = prompt_embeds
            if self.config["enable_cfg"] and neg_prompt is not None:
                neg_prompt_embeds, _, _ = self.text_encoders[0].infer([neg_prompt])
                self.input_info.txt_seq_lens.append(neg_prompt_embeds.shape[1])
                text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
        elif self.config["task"] == "i2i":
            prompt_embeds, _, image_info = self.text_encoders[0].infer([text], image_list)
            self.input_info.txt_seq_lens = [prompt_embeds.shape[1]]
            text_encoder_output["prompt_embeds"] = prompt_embeds
            text_encoder_output["image_info"] = image_info
            if self.config["enable_cfg"] and neg_prompt is not None:
                neg_prompt_embeds, _, _ = self.text_encoders[0].infer([neg_prompt], image_list)
                self.input_info.txt_seq_lens.append(neg_prompt_embeds.shape[1])
                text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
        return text_encoder_output

    @ProfilingContext4DebugL1("Run VAE Encoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration, metrics_labels=["QwenImageRunner"])
    def run_vae_encoder(self, image):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae = self.load_vae()
        image_latents = self.vae.encode_vae_image(image.to(GET_DTYPE()))
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()
        return {"image_latents": image_latents}

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_decode_duration,
        metrics_labels=["QwenImageRunner"],
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

        return None

    def set_target_shape(self):
        # In disagg transformer mode, use the shape transmitted from encoder
        if self.config.get("disagg_mode") == "transformer" and getattr(self, "inputs", {}).get("latent_shape"):
            latent_shape = self.inputs["latent_shape"]
            self.input_info.target_shape = tuple(latent_shape)
            # Reconstruct auto_height and auto_width
            scale_factor = self.config["vae_scale_factor"]
            self.input_info.auto_height = latent_shape[-2] * scale_factor
            self.input_info.auto_width = latent_shape[-1] * scale_factor
            logger.info(f"Qwen Image Runner restored target shape from disagg: {latent_shape}")
            return

        custom_shape = self.get_custom_shape()
        if custom_shape is not None:
            width, height = custom_shape
        else:
            width, height = self.input_info.original_size[-1]
            calculated_width, calculated_height, _ = calculate_dimensions(self.resolution * self.resolution, width / height)
            multiple_of = self.config["vae_scale_factor"] * 2
            width = calculated_width // multiple_of * multiple_of
            height = calculated_height // multiple_of * multiple_of
        logger.info(f"Qwen Image Runner set target shape: {width}x{height}")
        self.input_info.auto_width = width
        self.input_info.auto_height = height

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.config["vae_scale_factor"] * 2))
        width = 2 * (int(width) // (self.config["vae_scale_factor"] * 2))
        num_channels_latents = self.config["in_channels"] // 4
        if not self.is_layered:
            self.input_info.target_shape = (1, 1, num_channels_latents, height, width)
        else:
            self.input_info.target_shape = (1, self.layers + 1, num_channels_latents, height, width)

    def set_img_shapes(self):
        width, height = self.input_info.auto_width, self.input_info.auto_height
        if self.config["task"] == "t2i":
            image_shapes = [[(1, height // self.config["vae_scale_factor"] // 2, width // self.config["vae_scale_factor"] // 2)]]
        elif self.config["task"] == "i2i":
            if self.is_layered:
                image_shapes = [
                    [
                        *[(1, height // self.config["vae_scale_factor"] // 2, width // self.config["vae_scale_factor"] // 2) for _ in range(self.layers + 1)],
                        (1, height // self.config["vae_scale_factor"] // 2, width // self.config["vae_scale_factor"] // 2),
                    ]
                ]
            else:
                image_shapes = [[(1, height // self.config["vae_scale_factor"] // 2, width // self.config["vae_scale_factor"] // 2)]]
                for image_height, image_width in self.inputs["text_encoder_output"]["image_info"]["vae_image_info_list"]:
                    image_shapes[0].append((1, image_height // self.config["vae_scale_factor"] // 2, image_width // self.config["vae_scale_factor"] // 2))
        self.input_info.image_shapes = image_shapes

    def init_scheduler(self):
        super().init_scheduler()
        if self.config.get("disagg_mode") == "decode":
            return
        self.scheduler = QwenImageScheduler(self.config)

    def get_encoder_output_i2v(self):
        pass

    def run_image_encoder(self):
        pass

    def _save_images(self, images, input_info, log_prefix="Image saved"):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if input_info.return_result_tensor:
            return

        image_prefix = input_info.save_result_path.rsplit(".", 1)[0]
        image_suffix = input_info.save_result_path.rsplit(".", 1)[1] if len(input_info.save_result_path.rsplit(".", 1)) > 1 else "png"
        if isinstance(images[0], list) and len(images[0]) > 1:
            for idx, image in enumerate(images[0]):
                image.save(f"{image_prefix}_{idx:05d}.{image_suffix}")
                logger.info(f"{log_prefix}: {image_prefix}_{idx:05d}.{image_suffix}")
        else:
            image = images[0]
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
            return {"images": images}
        elif input_info.save_result_path is not None:
            return {"images": None}

    def _run_pipeline_local(self, input_info):
        self.inputs = self.run_input_encoder()
        self.set_target_shape()
        self.set_img_shapes()
        logger.info(f"input_info: {self.input_info}")
        latents, generator = self.run_dit()
        images = self.run_vae_decoder(latents)
        self.end_run()
        self._save_images(images, input_info, log_prefix="Image saved")
        return self._finalize_pipeline_outputs(input_info, images, latents=latents, generator=generator)

    def _run_pipeline_disagg_encoder(self):
        self.inputs = self.run_input_encoder()
        self.set_target_shape()
        self.set_img_shapes()
        logger.info(f"input_info: {self.input_info}")
        latent_shape = list(self.input_info.target_shape)
        self.send_encoder_outputs(self.inputs, latent_shape)
        logger.info("[Disagg] Encoder role completed. Skipping DiT run_main.")
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        return None

    def _run_pipeline_disagg_transformer(self, input_info):
        self.inputs = self.receive_encoder_outputs()
        prompt_embeds = self.inputs.get("text_encoder_output", {}).get("prompt_embeds")
        if prompt_embeds is not None:
            self.input_info.txt_seq_lens = [prompt_embeds.shape[1]]
            neg_embeds = self.inputs.get("text_encoder_output", {}).get("negative_prompt_embeds")
            if neg_embeds is not None:
                self.input_info.txt_seq_lens.append(neg_embeds.shape[1])

        self.set_target_shape()
        self.set_img_shapes()
        logger.info(f"input_info: {self.input_info}")

        latents, generator = self.run_dit()
        if getattr(self, "_disagg_p2_sender", None) is not None:
            self.send_transformer_outputs(latents)
            self.end_run()
            if GET_RECORDER_MODE():
                monitor_cli.lightx2v_worker_request_success.inc()
            return None

        images = self.run_vae_decoder(latents)
        self.end_run()
        self._save_images(images, input_info, log_prefix="Image saved")
        return self._finalize_pipeline_outputs(input_info, images, latents=latents, generator=generator)

    def _run_pipeline_disagg_decode(self, input_info):
        # Decoder role: receive DiT latents from Transformer, decode with VAE, save image
        latents = self.receive_transformer_outputs()

        scale_factor = self.config["vae_scale_factor"]
        p2_meta = getattr(self, "_p2_receive_meta", {})
        auto_height = p2_meta.get("auto_height")
        auto_width = p2_meta.get("auto_width")
        if auto_height is None or auto_width is None:
            # Fallback for spatial-format latents (non-packed models)
            latent_h = latents.shape[-2]
            latent_w = latents.shape[-1]
            auto_height = latent_h * scale_factor * 2
            auto_width = latent_w * scale_factor * 2
        self.input_info.auto_height = int(auto_height)
        self.input_info.auto_width = int(auto_width)
        # Compute image_shapes: number of spatial patches per image
        h_patches = int(auto_height) // (scale_factor * 2)
        w_patches = int(auto_width) // (scale_factor * 2)
        self.input_info.image_shapes = [[(1, h_patches, w_patches)]]
        images = self.run_vae_decoder(latents)
        self.end_run()

        self._save_images(images, input_info, log_prefix="[Disagg] Decode: image saved")

        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        if input_info.return_result_tensor:
            return {"images": images}
        return {"images": None}

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info
        disagg_mode = self.config.get("disagg_mode")

        if disagg_mode == "decode":
            return self._run_pipeline_disagg_decode(input_info)
        if disagg_mode == "encoder":
            return self._run_pipeline_disagg_encoder()
        if disagg_mode == "transformer":
            return self._run_pipeline_disagg_transformer(input_info)
        return self._run_pipeline_local(input_info)
