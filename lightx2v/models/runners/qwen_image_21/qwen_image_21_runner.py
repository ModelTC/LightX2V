import math
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from loguru import logger

from lightx2v.common.ops.mm.fp8_f16_accum import fp8_f16_accum_mm_unavailable_reason, validate_fp8_f16_accum_qmax
from lightx2v.models.input_encoders.hf.qwen_image_21.qwen3vl import QwenImage21TextEncoder
from lightx2v.models.networks.qwen_image_21.fp8_f16_accum_policy import ACTIVATION_QMAX, validate_checkpoint
from lightx2v.models.networks.qwen_image_21.infer.pre_infer import build_token_layout
from lightx2v.models.networks.qwen_image_21.model import QwenImage21TransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.request_fields import IMAGE_REQUEST_FIELDS
from lightx2v.models.schedulers.qwen_image_21.scheduler import QwenImage21Scheduler
from lightx2v.models.video_encoders.hf.qwen_image_21.vae import QwenImage21VAE
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def image_dimensions(resolution, ratio):
    width = math.sqrt(resolution * resolution * ratio)
    height = width / ratio
    return max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)


@RUNNER_REGISTER("qwen_image_21")
class QwenImage21Runner(DefaultRunner):
    _WARMUP_RESOLUTIONS = ((1024, 1024), (768, 960))
    supported_request_fields_by_task = {"t2i": IMAGE_REQUEST_FIELDS, "i2i": IMAGE_REQUEST_FIELDS | {"image_path"}}

    def __init__(self, config):
        unsupported = (
            "vae_cpu_offload",
            "lazy_load",
            "unload_modules",
            "cfg_parallel",
            "pipefusion_parallel",
            "disagg_mode",
            "lora_configs",
            "vae_tiling",
        )
        for key in unsupported:
            if config.get(key):
                raise ValueError(f"qwen_image_21 does not yet support {key}")
        if config.get("cpu_offload", False) and config.get("offload_granularity", "model") not in {"model", "block"}:
            raise NotImplementedError("qwen_image_21 supports model and block CPU offload")
        if config.get("seq_parallel"):
            parallel = config["parallel"]
            seq_p_size = int(parallel.get("seq_p_size", 1))
            seq_p_attn_type = parallel.get("seq_p_attn_type", "ulysses")
            if seq_p_attn_type != "ulysses":
                raise ValueError("qwen_image_21 sequence parallel currently supports only seq_p_attn_type='ulysses'")
            tensor_p_size = int(parallel.get("tensor_p_size", 1))
            global_heads = int(config["num_attention_heads"])
            if global_heads % tensor_p_size:
                raise ValueError(f"qwen_image_21 TP requires num_attention_heads ({global_heads}) to be divisible by tensor_p_size ({tensor_p_size})")
            local_heads = global_heads // tensor_p_size
            if local_heads % seq_p_size:
                raise ValueError(
                    "qwen_image_21 Ulysses requires TP-local attention heads to be divisible by "
                    f"seq_p_size: global_heads={config['num_attention_heads']}, "
                    f"tensor_p_size={tensor_p_size}, local_heads={local_heads}, seq_p_size={seq_p_size}"
                )
            head_parallel_group_size = parallel.get("seq_p_head_parallel_group_size", 1)
            if not isinstance(head_parallel_group_size, int) or isinstance(head_parallel_group_size, bool):
                raise TypeError("qwen_image_21 seq_p_head_parallel_group_size must be an integer")
            heads_per_rank = local_heads // seq_p_size
            if not 1 <= head_parallel_group_size <= heads_per_rank:
                raise ValueError(f"qwen_image_21 seq_p_head_parallel_group_size must be in [1, {heads_per_rank}], got {head_parallel_group_size}")
            if head_parallel_group_size != 1 and not parallel.get("seq_p_head_parallel", False):
                raise ValueError("qwen_image_21 seq_p_head_parallel_group_size requires seq_p_head_parallel=true")
            if parallel.get("seq_p_quant_scheme") not in (None, "fp8", "fp4"):
                raise ValueError("qwen_image_21 seq_p_quant_scheme supports only fp8 and fp4")
        if config.get("dit_quantized"):
            if config.get("dit_quant_scheme") not in ("fp8-sgl", "fp8-f16-accum"):
                raise ValueError("qwen_image_21 DiT quantization supports only fp8-sgl and fp8-f16-accum")
            if not config.get("dit_quantized_ckpt"):
                raise ValueError("qwen_image_21 FP8 requires dit_quantized_ckpt")
            if config["dit_quant_scheme"] == "fp8-f16-accum":
                validate_fp8_f16_accum_qmax(config.get("dit_fp8_activation_qmax", ACTIVATION_QMAX))
        elif config.get("dit_quant_scheme", "Default") != "Default":
            raise ValueError("dit_quant_scheme requires dit_quantized=true")
        if config.get("feature_caching", "NoCaching") != "NoCaching":
            raise ValueError("qwen_image_21 supports exact condition KV caching, not feature caching")
        if not config["causal_condition"]:
            raise ValueError("Condition KV caching requires causal_condition=true")
        if config["enable_cfg"]:
            assert config["sample_guide_scale"] > 1, "enable_cfg=true requires sample_guide_scale > 1"
        if config["infer_steps"] < 1:
            raise ValueError("infer_steps must be positive")
        super().__init__(config)

    def get_supported_tasks(self):
        if self.config.get("task") is not None:
            return super().get_supported_tasks()
        return ("t2i", "i2i")

    @torch.inference_mode()
    @ProfilingContext4DebugL1("Warmup")
    def run_warmup(self):
        if type(self) is not QwenImage21Runner:
            raise NotImplementedError(f"Qwen-Image-2.1 warmup is not implemented for {type(self).__name__}")
        self._run_warmup()
        self._maybe_freeze_gc()

    def _run_warmup(self):
        # A temporary reference image reuses the complete I2I preprocessing path.
        with TemporaryDirectory(prefix="qwen_image_21_warmup_") as directory:
            image_path = Path(directory) / "reference.png"
            for task in ("t2i", "i2i"):
                for height, width in self._WARMUP_RESOLUTIONS:
                    logger.info(f"Warmup: {task}, {height}x{width}")
                    transformer_offloaded = not self.config.get("cpu_offload", False)
                    try:
                        self.scheduler.generator = None
                        request = {"task": task, "prompt": "warmup", "seed": 0, "size": [height, width]}
                        if task == "i2i":
                            Image.new("RGBA", (width, height), color=(0, 0, 0, 255)).save(image_path)
                            request["image_path"] = str(image_path)
                        # Internal warmup covers both paths even for a single-task runner.
                        self.input_info = self.create_input_info(request)
                        self.inputs = self.run_input_encoder()
                        self.init_run()
                        transformer_offloaded = False
                        self.model.prefill_condition_kv(self.inputs)
                        self.scheduler.step_pre(step_index=0)
                        self.model.infer(self.inputs)
                        self.scheduler.step_post()
                        if self.config.get("cpu_offload", False):
                            self._offload_transformer()
                            transformer_offloaded = True
                        self.run_vae_decoder(self.scheduler.latents)
                        torch_device_module.synchronize()
                    finally:
                        if self.config.get("cpu_offload", False) and not transformer_offloaded:
                            with suppress(Exception):
                                self._offload_transformer()
                        self.end_run()
                        self.__dict__.pop("inputs", None)
        logger.info("[Warmup] Warmup completed")

    def init_scheduler(self):
        self.scheduler = QwenImage21Scheduler(self.config)

    def load_transformer(self):
        if self.config.get("dit_quant_scheme") == "fp8-f16-accum":
            reason = fp8_f16_accum_mm_unavailable_reason()
            if reason is not None:
                raise RuntimeError(f"qwen_image_21 fp8-f16-accum is unavailable: {reason}")
            validate_checkpoint(self.config["dit_quantized_ckpt"])
        return QwenImage21TransformerModel(str(Path(self.config["model_path"]) / "transformer"), self.config, self.init_device)

    def load_text_encoder(self):
        return [QwenImage21TextEncoder(self.config)]

    def load_vae(self):
        return QwenImage21VAE(self.config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()

    @ProfilingContext4DebugL2("Run Encoders")
    def run_input_encoder(self):
        info = self.input_info
        resolution = self.config["resolution"]
        images = []
        if info.task == "i2i":
            if not info.image_path:
                raise ValueError("Image editing requires --image_path (comma separated for multiple images)")
            for path in info.image_path.split(","):
                with Image.open(path.strip()) as image:
                    image = image.convert("RGBA")
                    dimensions = image_dimensions(resolution, image.width / image.height)
                    images.append(image.resize(dimensions, Image.Resampling.LANCZOS))
        if info.size:
            height, width = info.size
        elif info.aspect_ratio:
            a, b = (float(s) for s in info.aspect_ratio.split(":"))
            width, height = image_dimensions(resolution, a / b)
        elif images:
            width, height = images[-1].size
        else:
            height = width = resolution
        if height < 32 or width < 32:
            raise ValueError("Output height and width must be at least 32")
        if height % 32 or width % 32:
            logger.warning(f"Output height and width ({height}, {width}) are not divisible by 32 and will be rounded down to multiples of 32")
        height = height // 32 * 32
        width = width // 32 * 32
        info.size = [height, width]
        scale = self.config["vae_scale_factor"]
        info.latent_shape = (1, 1, self.config["in_channels"], height // scale, width // scale)
        shapes = [(1, image.height // scale, image.width // scale) for image in images] + [(1, height // scale, width // scale)]
        outputs = self.run_text_encoder(images, shapes)
        outputs["image_latents"] = self.run_vae_encoder(images) if images else None
        return outputs

    @ProfilingContext4DebugL1("Run Text Encoder")
    def run_text_encoder(self, images, shapes):
        info = self.input_info
        outputs = {}
        branches = [("cond", info.prompt)]
        if self.config["enable_cfg"]:
            branches.append(("uncond", info.negative_prompt or ""))
        encoder = self.text_encoders[0]
        offload = self.config.get("text_encoder_cpu_offload", False)
        try:
            if offload:
                with ProfilingContext4DebugL1("Onload Text Encoder"):
                    encoder.to_cuda()
            for name, prompt in branches:
                branch = encoder.infer(prompt, images)
                branch["layout"] = build_token_layout(branch["image_mask"], shapes, self.config["axes_dims_rope"], rope=self.model.transformer_infer.rope)
                outputs[name] = branch
        finally:
            if offload:
                with ProfilingContext4DebugL1("Offload Text Encoder"):
                    encoder.to_cpu()
        return outputs

    @ProfilingContext4DebugL1("Run VAE Encoder")
    def run_vae_encoder(self, images):
        image_latents = []
        for image in images:
            # Preserve the reference processor's NHWC batch strides. Adding a
            # batch axis after permutation can select a different BF16 conv
            # kernel even though the pixel values are identical.
            value = torch.from_numpy(np.array(image)[None].astype(np.float32) / 255).permute(0, 3, 1, 2).unsqueeze(2)
            image_latents.append(self.vae.encode(value * 2 - 1)[0])
        return torch.cat(image_latents)

    def init_run(self):
        self.get_video_segment_num()
        self.scheduler.prepare(self.input_info)
        if not self.config.get("cpu_offload", False):
            logger.info("Qwen-Image-2.1 transformer is resident on the accelerator")
        elif self.config.get("offload_granularity", "model") == "model":
            logger.info("Moving the Qwen-Image-2.1 transformer to the accelerator")
            self.model.to_cuda()
        else:
            logger.info("Qwen-Image-2.1 block offload enabled; keeping source blocks on CPU and using two accelerator buffers")
            self.model.pre_weight.to_cuda()
            self.model.post_weight.to_cuda()
        torch_device_module.synchronize()

    @ProfilingContext4DebugL2("Offload DiT")
    def _offload_transformer(self):
        if not self.config.get("cpu_offload", False):
            return
        self.model.clear_condition_kv()
        if self.model.block_offload:
            self.model.pre_weight.to_cpu()
            self.model.post_weight.to_cpu()
        else:
            self.model.to_cpu()
        torch_device_module.synchronize()
        self.maybe_empty_cache(force=True, collect_garbage=True)

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        should_offload_transformer = self.config.get("cpu_offload", False)
        try:
            self.init_run()
            with ProfilingContext4DebugL1("Prefill condition KV"):
                self.model.prefill_condition_kv(self.inputs)
            result = self.run_segment()
        except BaseException:
            # Preserve an inference or partial-onload exception while making a
            # best-effort return of the large transformer to host memory.
            if should_offload_transformer:
                with suppress(Exception):
                    self._offload_transformer()
            raise
        if should_offload_transformer:
            self._offload_transformer()
        return result

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        return self.vae.decode(latents, self.input_info.size)

    def process_images_after_vae_decoder(self, value):
        input_info = self.input_info
        if input_info.save_result_path and not input_info.return_result_tensor and dist.is_initialized() and dist.get_rank() != 0:
            return {"images": None}
        pixels = (value / 2 + 0.5).clamp(0, 1).float().permute(0, 2, 3, 1).cpu().numpy()
        # OpenCV expects BGRA; keep alpha as the last channel.
        images = [cv2.cvtColor((p * 255).round().astype(np.uint8), cv2.COLOR_RGBA2BGRA) for p in pixels]
        if input_info.save_result_path and not input_info.return_result_tensor:
            path = Path(input_info.save_result_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), images[0]):
                raise RuntimeError(f"Failed to save image to: {path}")
            logger.info(f"✅ Image saved successfully to: {path} ✅")
        return {"images": images if input_info.return_result_tensor else None}

    def end_run(self):
        self.model.clear_condition_kv()
        self.inputs = None
        self.scheduler.clear()
        self.input_info = None

    @torch.inference_mode()
    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info
        try:
            self.inputs = self.run_input_encoder()
            latents = self.run_main()
            value = self.run_vae_decoder(latents)
            return self.process_images_after_vae_decoder(value)
        finally:
            self.end_run()
