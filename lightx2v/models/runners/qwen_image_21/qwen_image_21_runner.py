import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from loguru import logger

from lightx2v.models.input_encoders.hf.qwen_image_21.qwen3vl import QwenImage21TextEncoder
from lightx2v.models.networks.qwen_image_21.infer.pre_infer import build_token_layout
from lightx2v.models.networks.qwen_image_21.model import QwenImage21TransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.request_fields import IMAGE_REQUEST_FIELDS
from lightx2v.models.schedulers.qwen_image_21.scheduler import QwenImage21Scheduler
from lightx2v.models.video_encoders.hf.qwen_image_21.vae import QwenImage21VAE
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER


def image_dimensions(resolution, ratio):
    width = math.sqrt(resolution * resolution * ratio)
    height = width / ratio
    return max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)


@RUNNER_REGISTER("qwen_image_21")
class QwenImage21Runner(DefaultRunner):
    supported_request_fields_by_task = {"t2i": IMAGE_REQUEST_FIELDS, "i2i": IMAGE_REQUEST_FIELDS | {"image_path"}}

    def __init__(self, config):
        unsupported = (
            "cpu_offload",
            "text_encoder_cpu_offload",
            "vae_cpu_offload",
            "lazy_load",
            "unload_modules",
            "dit_quantized",
            "text_encoder_quantized",
            "shared_cpu_weights",
            "parallel",
            "seq_parallel",
            "cfg_parallel",
            "tensor_parallel",
            "pipefusion_parallel",
            "disagg_mode",
            "lora_configs",
            "vae_tiling",
            "use_compile",
        )
        for key in unsupported:
            if config.get(key):
                raise ValueError(f"qwen_image_21 does not yet support {key}")
        if config.get("feature_caching", "NoCaching") != "NoCaching":
            raise ValueError("qwen_image_21 supports exact condition KV caching, not feature caching")
        if not config["causal_condition"]:
            raise ValueError("Condition KV caching requires causal_condition=true")
        if config["enable_cfg"]:
            assert config["sample_guide_scale"] > 1, "enable_cfg=true requires sample_guide_scale > 1"
        if config["infer_steps"] < 1:
            raise ValueError("infer_steps must be positive")
        super().__init__(config)

    def init_scheduler(self):
        self.scheduler = QwenImage21Scheduler(self.config)

    def load_transformer(self):
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
        if self.config["task"] == "i2i":
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
        if height < 32 or width < 32 or height % 32 or width % 32:
            raise ValueError("Output height and width must be positive multiples of 32")
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
        for name, prompt in branches:
            branch = self.text_encoders[0].infer(prompt, images)
            branch["layout"] = build_token_layout(branch["image_mask"], shapes, self.config["axes_dims_rope"], rope=self.model.transformer_infer.rope)
            outputs[name] = branch
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

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.init_run()
        with ProfilingContext4DebugL1("Prefill condition KV"):
            self.model.prefill_condition_kv(self.inputs)
        return self.run_segment()

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        return self.vae.decode(latents, self.input_info.size)

    def process_images_after_vae_decoder(self, value):
        input_info = self.input_info
        pixels = (value / 2 + 0.5).clamp(0, 1).float().permute(0, 2, 3, 1).cpu().numpy()
        images = [Image.fromarray((p * 255).round().astype(np.uint8)) for p in pixels]
        if input_info.save_result_path and not input_info.return_result_tensor:
            path = Path(input_info.save_result_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(path)
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
