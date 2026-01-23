import json

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner  # noqa: F401
from lightx2v.models.runners.longcat_image.longcat_image_runner import LongCatImageRunner  # noqa: F401
from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner  # noqa: F401
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_animate_runner import WanAnimateRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, WanAudioRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_matrix_game2_runner import WanSFMtxg2Runner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_vace_runner import WanVaceRunner  # noqa: F401
from lightx2v.models.runners.z_image.z_image_runner import ZImageRunner  # noqa: F401
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import set_config, set_parallel_config
from lightx2v.utils.utils import seed_all, validate_config_paths


def dict_like(cls):
    cls.__getitem__ = lambda self, key: getattr(self, key)
    cls.__setitem__ = lambda self, key, value: setattr(self, key, value)
    cls.__delitem__ = lambda self, key: delattr(self, key)
    cls.__contains__ = lambda self, key: hasattr(self, key)

    def update(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                items = arg.items()
            else:
                items = arg
            for k, v in items:
                setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    cls.get = get
    cls.update = update

    return cls


@dict_like
class LightX2VPipeline:
    def __init__(
        self,
        task,
        model_path,
        model_cls,
        sf_model_path=None,
        dit_original_ckpt=None,
        low_noise_original_ckpt=None,
        high_noise_original_ckpt=None,
        transformer_model_name=None,
    ):
        self.task = task
        self.model_path = model_path
        self.model_cls = model_cls
        self.sf_model_path = sf_model_path
        self.dit_original_ckpt = dit_original_ckpt
        self.low_noise_original_ckpt = low_noise_original_ckpt
        self.high_noise_original_ckpt = high_noise_original_ckpt
        self.transformer_model_name = transformer_model_name

        if self.model_cls in [
            "wan2.1",
            "wan2.1_distill",
            "wan2.1_vace",
            "wan2.1_sf",
            "wan2.1_sf_mtxg2",
            "seko_talk",
            "wan2.2_moe",
            "wan2.2_moe_audio",
            "wan2.2_audio",
            "wan2.2_moe_distill",
            "wan2.2_animate",
        ]:
            self.vae_stride = (4, 8, 8)
            if self.model_cls.startswith("wan2.2"):
                self.use_image_encoder = False
        elif self.model_cls in ["wan2.2"]:
            self.vae_stride = (4, 16, 16)
            self.num_channels_latents = 48
        elif self.model_cls in ["hunyuan_video_1.5", "hunyuan_video_1.5_distill"]:
            self.vae_stride = (4, 16, 16)
            self.num_channels_latents = 32
        elif self.model_cls in ["ltx2"]:
            self.num_channels_latents = 128
            self.audio_mel_bins = 16

        if model_cls in ["qwen-image", "qwen-image-2512", "qwen-image-edit", "qwen-image-edit-2509", "qwen-image-edit-2511"]:
            self.CONDITION_IMAGE_SIZE = 147456
            self.USE_IMAGE_ID_IN_PROMPT = True
            if model_cls == "qwen-image-edit":
                self.CONDITION_IMAGE_SIZE = 1048576
                self.USE_IMAGE_ID_IN_PROMPT = False
            self.model_cls = "qwen_image"
            if self.task in ["i2i"]:
                self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
                self.prompt_template_encode_start_idx = 64
            elif self.task in ["t2i"]:
                self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
                self.prompt_template_encode_start_idx = 34
        elif self.model_cls in ["z_image"]:
            self.model_cls = "z_image"
        elif model_cls in ["longcat_image", "longcat-image"]:
            self.model_cls = "longcat_image"

        self.input_info = init_empty_input_info(self.task)

    def create_generator(
        self,
        attn_mode="flash_attn2",
        infer_steps=50,
        num_frames=81,
        height=480,
        width=832,
        guidance_scale=5.0,
        sample_shift=5.0,
        fps=16,
        aspect_ratio="16:9",
        boundary=0.900,
        boundary_step_index=2,
        denoising_step_list=[1000, 750, 500, 250],
        config_json=None,
        rope_type="torch",
        resize_mode=None,
        audio_fps=24000,
        double_precision_rope=True,
        norm_modulate_backend="torch",
    ):
        self.resize_mode = resize_mode
        if config_json is not None:
            self.set_infer_config_json(config_json)
        else:
            self.set_infer_config(
                attn_mode,
                rope_type,
                infer_steps,
                num_frames,
                height,
                width,
                guidance_scale,
                sample_shift,
                fps,
                aspect_ratio,
                boundary,
                boundary_step_index,
                denoising_step_list,
                audio_fps,
                double_precision_rope,
                norm_modulate_backend,
            )

        config = set_config(self)
        validate_config_paths(config)
        self.runner = self._init_runner(config)
        print(self.runner.config)
        logger.info(f"Initializing {self.model_cls} runner for {self.task} task...")
        logger.info(f"Model path: {self.model_path}")
        logger.info("LightGenerator initialized successfully!")

    def set_infer_config(
        self,
        attn_mode,
        rope_type,
        infer_steps,
        num_frames,
        height,
        width,
        guidance_scale,
        sample_shift,
        fps,
        aspect_ratio,
        boundary,
        boundary_step_index,
        denoising_step_list,
        audio_fps,
        double_precision_rope,
        norm_modulate_backend,
    ):
        self.infer_steps = infer_steps
        self.target_width = width
        self.target_height = height
        self.target_video_length = num_frames
        self.sample_guide_scale = guidance_scale
        self.sample_shift = sample_shift
        if self.sample_guide_scale == 1 or (self.model_cls == "z_image" and self.sample_guide_scale == 0):
            self.enable_cfg = False
        else:
            self.enable_cfg = True
        self.rope_type = rope_type
        self.fps = fps
        self.aspect_ratio = aspect_ratio
        self.boundary = boundary
        self.boundary_step_index = boundary_step_index
        self.denoising_step_list = denoising_step_list
        self.audio_fps = audio_fps
        self.double_precision_rope = double_precision_rope
        if self.model_cls.startswith("wan"):
            self.self_attn_1_type = attn_mode
            self.cross_attn_1_type = attn_mode
            self.cross_attn_2_type = attn_mode
        elif self.model_cls in ["hunyuan_video_1.5", "hunyuan_video_1.5_distill", "qwen_image", "longcat_image", "ltx2"]:
            self.attn_type = attn_mode
        self.norm_modulate_backend = norm_modulate_backend

    def set_infer_config_json(self, config_json):
        logger.info(f"Loading infer config from {config_json}")
        with open(config_json, "r") as f:
            config_json = json.load(f)
        self.update(config_json)

    def enable_lightvae(
        self,
        use_lightvae=False,
        use_tae=False,
        vae_path=None,
        tae_path=None,
    ):
        assert self.model_cls not in ["qwen_image", "longcat_image"]
        self.use_lightvae = use_lightvae
        self.use_tae = use_tae
        self.vae_path = vae_path
        self.tae_path = tae_path
        if self.use_tae and self.model_cls.startswith("wan") and "lighttae" in tae_path:
            self.need_scaled = True

    def enable_quantize(
        self,
        dit_quantized=False,
        text_encoder_quantized=False,
        image_encoder_quantized=False,
        dit_quantized_ckpt=None,
        low_noise_quantized_ckpt=None,
        high_noise_quantized_ckpt=None,
        text_encoder_quantized_ckpt=False,
        image_encoder_quantized_ckpt=False,
        quant_scheme="fp8-sgl",
        text_encoder_quant_scheme=None,
        skip_fp8_block_index=[0, 43, 44, 45, 46, 47],
    ):
        self.dit_quantized = dit_quantized
        self.dit_quant_scheme = quant_scheme
        self.dit_quantized_ckpt = dit_quantized_ckpt
        self.low_noise_quantized_ckpt = low_noise_quantized_ckpt
        self.high_noise_quantized_ckpt = high_noise_quantized_ckpt

        if self.model_cls.startswith("wan"):
            self.t5_quant_scheme = quant_scheme
            self.t5_quantized = text_encoder_quantized
            self.t5_quantized_ckpt = text_encoder_quantized_ckpt
            self.clip_quant_scheme = quant_scheme
            self.clip_quantized = image_encoder_quantized
            self.clip_quantized_ckpt = image_encoder_quantized_ckpt
        elif self.model_cls in ["hunyuan_video_1.5", "hunyuan_video_1.5_distill"]:
            self.qwen25vl_quantized = text_encoder_quantized
            self.qwen25vl_quantized_ckpt = text_encoder_quantized_ckpt
            self.qwen25vl_quant_scheme = quant_scheme
        elif self.model_cls in ["qwen_image"]:
            self.qwen25vl_quantized = text_encoder_quantized
            self.qwen25vl_quantized_ckpt = text_encoder_quantized_ckpt
            if text_encoder_quant_scheme is not None:
                self.qwen25vl_quant_scheme = text_encoder_quant_scheme
            else:
                self.qwen25vl_quant_scheme = quant_scheme
        elif self.model_cls in ["ltx2"]:
            self.skip_fp8_block_index = skip_fp8_block_index

    def enable_offload(
        self,
        cpu_offload=False,
        offload_granularity="block",
        text_encoder_offload=False,
        image_encoder_offload=False,
        vae_offload=False,
    ):
        self.cpu_offload = cpu_offload
        self.offload_granularity = offload_granularity
        self.vae_cpu_offload = vae_offload
        if self.model_cls in [
            "wan2.1",
            "wan2.1_distill",
            "wan2.1_vace",
            "wan2.1_sf",
            "wan2.1_sf_mtxg2",
            "seko_talk",
            "wan2.2_moe",
            "wan2.2",
            "wan2.2_moe_audio",
            "wan2.2_audio",
            "wan2.2_moe_distill",
            "wan2.2_animate",
        ]:
            self.t5_cpu_offload = text_encoder_offload
            self.clip_encoder_offload = image_encoder_offload

        elif self.model_cls in ["hunyuan_video_1.5", "hunyuan_video_1.5_distill"]:
            self.qwen25vl_cpu_offload = text_encoder_offload
            self.siglip_cpu_offload = image_encoder_offload
            self.byt5_cpu_offload = image_encoder_offload
        elif self.model_cls in ["qwen_image", "longcat_image"]:
            self.qwen25vl_cpu_offload = text_encoder_offload
        elif self.model_cls == "ltx2":
            self.gemma_cpu_offload = text_encoder_offload

    def enable_compile(
        self,
    ):
        self.compile = True
        self.compile_shapes = [
            [480, 832],
            [544, 960],
            [720, 1280],
            [832, 480],
            [960, 544],
            [1280, 720],
            [480, 480],
            [576, 576],
            [704, 704],
            [960, 960],
        ]

    def enable_lora(self, lora_configs, lora_dynamic_apply=False):
        self.lora_configs = lora_configs
        self.lora_dynamic_apply = lora_dynamic_apply

    def enable_cache(
        self,
        cache_method="Tea",
        coefficients=[],
        teacache_thresh=0.15,
        use_ret_steps=False,
        magcache_calibration=False,
        magcache_K=6,
        magcache_thresh=0.24,
        magcache_retention_ratio=0.2,
        magcache_ratios=[],
    ):
        self.feature_caching = cache_method
        if cache_method == "Tea":
            self.coefficients = coefficients
            self.teacache_thresh = teacache_thresh
            self.use_ret_steps = use_ret_steps
        elif cache_method == "Mag":
            self.magcache_calibration = magcache_calibration
            self.magcache_K = magcache_K
            self.magcache_thresh = magcache_thresh
            self.magcache_retention_ratio = magcache_retention_ratio
            self.magcache_ratios = magcache_ratios

    def enable_parallel(self, cfg_p_size=1, seq_p_size=1, seq_p_attn_type="ulysses"):
        self._init_parallel()
        self.parallel = {
            "cfg_p_size": cfg_p_size,
            "seq_p_size": seq_p_size,
            "seq_p_attn_type": seq_p_attn_type,
        }
        set_parallel_config(self)

    @torch.no_grad()
    def generate(
        self,
        seed,
        prompt,
        negative_prompt,
        save_result_path,
        image_path=None,
        image_strength=None,
        last_frame_path=None,
        audio_path=None,
        src_ref_images=None,
        src_video=None,
        src_mask=None,
        return_result_tensor=False,
        target_shape=[],
    ):
        # Run inference (following LightX2V pattern)
        # Note: image_path supports comma-separated paths for multiple images
        # image_strength can be a scalar (float/int) or a list matching the number of images
        self.seed = seed
        self.image_path = image_path
        self.last_frame_path = last_frame_path
        self.audio_path = audio_path
        self.src_ref_images = src_ref_images
        self.src_video = src_video
        self.src_mask = src_mask
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.save_result_path = save_result_path
        self.return_result_tensor = return_result_tensor
        self.target_shape = target_shape
        self.image_strength = image_strength

        input_info = init_empty_input_info(self.task)
        seed_all(self.seed)
        update_input_info_from_dict(input_info, self)
        self.runner.run_pipeline(input_info)
        logger.info("Video generated successfully!")
        logger.info(f"Video Saved in {save_result_path}")

    def _init_runner(self, config):
        torch.set_grad_enabled(False)
        runner = RUNNER_REGISTER[config["model_cls"]](config)
        runner.init_modules()
        return runner

    def _init_parallel(self):
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
