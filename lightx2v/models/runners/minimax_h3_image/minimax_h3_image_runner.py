import math
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from loguru import logger
from torch.nn.attention import SDPBackend, sdpa_kernel

from lightx2v.models.input_encoders.hf.minimax_h3_image import MiniMaxH3ImageTextEncoder
from lightx2v.models.input_encoders.hf.minimax_h3_image.preprocess import get_image_canvas_size, image_to_tensor, prepare_reference_images
from lightx2v.models.networks.minimax_h3_image.model import MiniMaxH3ImageModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.request_fields import IMAGE_REQUEST_FIELDS
from lightx2v.models.schedulers.minimax_h3_image import MiniMaxH3ImageScheduler
from lightx2v.models.video_encoders.hf.minimax_h3_image import MiniMaxH3ImageVAE
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import is_main_process
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("minimax_h3_image")
class MiniMaxH3ImageRunner(DefaultRunner):
    supported_request_fields_by_task = {"t2i": IMAGE_REQUEST_FIELDS, "i2i": IMAGE_REQUEST_FIELDS | {"image_path"}}

    def __init__(self, config):
        if config.get("enable_cfg", False) or config.get("cfg_parallel", False):
            raise ValueError("MiniMax-H3-Image is guidance-distilled and does not have a CFG/unconditional branch")
        for option in ("lazy_load", "unload_modules", "pipefusion_parallel", "disagg_mode"):
            if config.get(option):
                raise ValueError(f"MiniMax-H3-Image does not support {option}")
        if config.get("feature_caching", "NoCaching") != "NoCaching":
            raise ValueError("MiniMax-H3-Image requires feature_caching='NoCaching'")
        if config.get("negative_prompt") or config.get("sample_guide_scale", 1.0) != 1.0:
            raise ValueError("MiniMax-H3-Image uses one positive branch without negative prompts or guidance")
        super().__init__(config)

    def get_supported_tasks(self):
        return super().get_supported_tasks() if self.config.get("task") is not None else ("t2i", "i2i")

    def init_scheduler(self):
        self.scheduler = MiniMaxH3ImageScheduler(self.config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = MiniMaxH3ImageModel(self.config["model_path"], self.config, self.init_device)
        self.text_encoders = []
        self.vae = None
        self.text_encoder_tensor_parallel = self.config.get("text_encoder_tensor_parallel", False)
        if self.text_encoder_tensor_parallel:
            self.text_encoders = [MiniMaxH3ImageTextEncoder(self.config)]
        self._run_on_rank0(self._load_input_encoders)

    def _load_input_encoders(self):
        if not self.text_encoder_tensor_parallel:
            encoder_config = dict(self.config)
            encoder_config["text_encoder_tensor_parallel"] = False
            self.text_encoders = [MiniMaxH3ImageTextEncoder(encoder_config)]
        self.vae = MiniMaxH3ImageVAE.from_pretrained(
            self.config["model_path"],
            device=AI_DEVICE,
            cpu_offload=self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False)),
            attn_type=self.config.get("vae_attn_type", "torch_sdpa"),
        )

    def _run_on_rank0(self, function):
        if not dist.is_initialized():
            return function()
        result, error = None, None
        if is_main_process():
            try:
                result = function()
            except Exception as exc:
                error = exc
        status = [f"{type(error).__name__}: {error}" if error is not None else None]
        dist.broadcast_object_list(status, src=0)
        if error is not None:
            raise error
        if status[0] is not None:
            raise RuntimeError(f"MiniMax-H3-Image rank 0 failed: {status[0]}")
        return result

    def _resolve_request_geometry(self):
        size = self.input_info.size
        if not size and self.input_info.aspect_ratio:
            dimensions = self.input_info.aspect_ratio.split(":")
            if len(dimensions) != 2:
                raise ValueError("MiniMax-H3-Image aspect_ratio must be WIDTH:HEIGHT")
            ratio_width, ratio_height = (float(value) for value in dimensions)
            if not all(math.isfinite(value) and value > 0 for value in (ratio_width, ratio_height)):
                raise ValueError("MiniMax-H3-Image aspect_ratio dimensions must be positive and finite")
            base_height, base_width = self.config["size"]
            area = base_height * base_width
            size = [max(32, round(math.sqrt(area * ratio_height / ratio_width))), max(32, round(math.sqrt(area * ratio_width / ratio_height)))]
        size = size or self.config["size"]
        if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
            raise ValueError("MiniMax-H3-Image size must be [positive height, positive width]")
        self.output_size = tuple(size)
        self.request_height, self.request_width = get_image_canvas_size(*size, min_generation_area=self.config.get("min_generation_area", 2048 * 2048))
        self.input_info.size = list(size)

    @ProfilingContext4DebugL1("Run Text Encoder")
    def run_text_encoder(self, images):
        return self.text_encoders[0].infer(self.input_info.prompt, image_list=images)

    @ProfilingContext4DebugL2("Run Encoders")
    def run_input_encoder(self):
        if self.text_encoder_tensor_parallel:
            references = self._prepare_input_images()
            text_output = self.run_text_encoder([reference.qwen_image for reference in references])
            inputs = self._run_on_rank0(lambda: self._sample_inputs(text_output, references))
        else:
            inputs = self._run_on_rank0(self._encode_inputs)
        return self._broadcast_encoded_inputs(inputs)

    def _prepare_input_images(self):
        self._resolve_request_geometry()
        if not isinstance(self.input_info.prompt, str) or not self.input_info.prompt:
            raise ValueError("MiniMax-H3-Image requires a nonempty prompt string")
        references = prepare_reference_images(self.input_info.image_path) if self.input_info.task == "i2i" else []
        if self.input_info.task == "i2i" and not references:
            raise ValueError("MiniMax-H3-Image i2i requires at least one reference in image_path")
        return references

    def _encode_inputs(self):
        references = self._prepare_input_images()
        text_output = self.run_text_encoder([reference.qwen_image for reference in references])
        return self._sample_inputs(text_output, references)

    def _sample_inputs(self, text_output, references):
        # Modular Diffusers samples reference posteriors before target noise
        # and reference-anchor noise, using one CPU random stream.
        generator = torch.Generator(device="cpu").manual_seed(int(self.input_info.seed))
        reference_latents = []
        with ProfilingContext4DebugL1("Run VAE Encoder"):
            for reference in references:
                pixels = image_to_tensor(reference.vae_image).unsqueeze(2)
                reference_latents.append(self.vae.encode_condition(pixels, generator=generator))
        target_noise = torch.randn((1, 24, 1, self.request_height // 16, self.request_width // 16), generator=generator, dtype=torch.float32)
        reference_noises = [torch.randn(latent.shape, generator=generator, dtype=torch.float32) for latent in reference_latents]
        return {"text_encoder_output": text_output, "reference_latents": reference_latents, "target_noise": target_noise, "reference_noises": reference_noises}

    def _broadcast_encoded_inputs(self, inputs):
        if not dist.is_initialized():
            return inputs
        tensors = []
        metadata = [None]
        if is_main_process():
            text_output = inputs["text_encoder_output"]
            tensors = [text_output["prompt_embeds"], text_output["text_token_tags"], inputs["target_noise"], *inputs["reference_latents"], *inputs["reference_noises"]]
            metadata[0] = (self.output_size, self.request_height, self.request_width, len(inputs["reference_latents"]), [(tensor.shape, tensor.dtype) for tensor in tensors])
        dist.broadcast_object_list(metadata, src=0)
        self.output_size, self.request_height, self.request_width, num_references, tensor_metadata = metadata[0]
        self.input_info.size = list(self.output_size)
        received = []
        for index, (shape, dtype) in enumerate(tensor_metadata):
            tensor = tensors[index].to(AI_DEVICE).contiguous() if is_main_process() else torch.empty(shape, dtype=dtype, device=AI_DEVICE)
            dist.broadcast(tensor, src=0)
            received.append(tensor)
        return {
            "text_encoder_output": {"prompt_embeds": received[0], "text_token_tags": received[1]},
            "target_noise": received[2],
            "reference_latents": received[3 : 3 + num_references],
            "reference_noises": received[3 + num_references :],
        }

    @ProfilingContext4DebugL2("Prepare DiT")
    def init_run(self):
        self.scheduler.prepare(
            height=self.request_height,
            width=self.request_width,
            text_token_tags=self.inputs["text_encoder_output"]["text_token_tags"],
            reference_latents=self.inputs["reference_latents"],
            target_noise=self.inputs["target_noise"],
            reference_noises=self.inputs["reference_noises"],
        )
        if self.config.get("cpu_offload", False) and not self.model.block_offload:
            self.model.to_cuda()
        logger.info(f"MiniMax-H3-Image canvas: {self.request_height}x{self.request_width}; packed rows: {self.scheduler.video_latents.shape[0]}")

    def run_segment(self, segment_idx=0):
        for step in range(self.scheduler.infer_steps):
            self.check_stop()
            logger.info(f"==> MiniMax-H3-Image step: {step + 1} / {self.scheduler.infer_steps}")
            with ProfilingContext4DebugL1("Run Dit every step"):
                self.scheduler.step_pre(step)
                self.model.infer(self.inputs)
                self.scheduler.step_post()
            if self.progress_callback:
                self.progress_callback((step + 1) * 100 / self.scheduler.infer_steps, 100)
        return self.scheduler.unpack_target_latents()

    def _offload_transformer(self):
        if not self.config.get("cpu_offload", False):
            return
        self.model.to_cpu()
        getattr(torch, AI_DEVICE).synchronize()
        self.maybe_empty_cache(force=True, collect_garbage=True)

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        try:
            self.init_run()
            latents = self.run_segment()
        except BaseException:
            with suppress(Exception):
                self._offload_transformer()
            raise
        self._offload_transformer()
        return latents

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        return self.vae.decode(latents)

    def process_images_after_vae_decoder(self, value):
        # The reference pipeline quantizes before resizing the canvas to the requested size.
        pixels = value[0, :, 0].float().clamp(0, 1).permute(1, 2, 0).mul(255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(pixels)
        height, width = self.output_size
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        if self.input_info.save_result_path and not self.input_info.return_result_tensor:
            output_path = Path(self.input_info.save_result_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            logger.info(f"Image saved to {output_path}")
        # Match the BGR numpy image convention used by the other image runners.
        images = [np.asarray(image)[..., ::-1].copy()] if self.input_info.return_result_tensor else None
        return {"images": images}

    def _decode_output(self, latents):
        return self.process_images_after_vae_decoder(self.run_vae_decoder(latents))

    def _broadcast_output(self, result):
        if not dist.is_initialized():
            return result
        if not self.input_info.return_result_tensor:
            return {"images": None}
        shape = [result["images"][0].shape if is_main_process() else None]
        dist.broadcast_object_list(shape, src=0)
        if is_main_process():
            pixels = torch.from_numpy(result["images"][0]).to(AI_DEVICE)
        else:
            pixels = torch.empty(shape[0], dtype=torch.uint8, device=AI_DEVICE)
        dist.broadcast(pixels, src=0)
        return {"images": [pixels.cpu().numpy()]}

    @torch.inference_mode()
    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info
        try:
            # Optional attention imports can disable cuDNN SDPA globally.
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                self.inputs = self.run_input_encoder()
                latents = self.run_main()
                result = self._run_on_rank0(lambda: self._decode_output(latents))
                return self._broadcast_output(result)
        finally:
            self.scheduler.clear()
            self.inputs = None
            self.input_info = None
