import os

os.environ.setdefault("PROFILING_DEBUG_LEVEL", "2")
os.environ.setdefault("DTYPE", "BF16")
os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "None")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from loguru import logger

from lightx2v.models.runners.runner_factory import build_runner
from lightx2v.utils.input_info import UNSET
from lightx2v.utils.set_config import build_startup_config, init_parallel
from lightx2v.utils.utils import validate_config_paths
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


class LightX2VPipeline:
    def __init__(
        self,
        task=None,
        model_path="",
        model_cls="",
        support_tasks=None,
        dit_original_ckpt=None,
        low_noise_original_ckpt=None,
        high_noise_original_ckpt=None,
        transformer_model_name=None,
        distill_method=None,
        model_variant=None,
    ):
        requested_model_cls = model_cls
        if model_cls in ["qwen-image", "qwen-image-2512", "qwen-image-edit", "qwen-image-edit-2509", "qwen-image-edit-2511"]:
            model_cls = "qwen_image"

        self.task = task
        # Select startup components without making support_tasks a request default.
        if not task and support_tasks:
            task = support_tasks[0]
        self.model_path = model_path
        self.model_cls = model_cls
        self.model_variant = model_variant
        self.runner = None
        self.startup_config = {
            key: value
            for key, value in {
                "task": task,
                "model_path": model_path,
                "model_cls": model_cls,
                "model_variant": model_variant,
                "dit_original_ckpt": dit_original_ckpt,
                "low_noise_original_ckpt": low_noise_original_ckpt,
                "high_noise_original_ckpt": high_noise_original_ckpt,
                "distill_method": distill_method,
                "transformer_model_name": transformer_model_name,
            }.items()
            if value is not None
        }

        if model_cls in [
            "wan2.1",
            "wan2.1_vace",
            "wan2.1_sf",
            "wan2.1_sf_mtxg2",
            "seko_talk",
            "seko_talk_ar",
            "wan2.2_moe",
            "wan2.2_animate",
            "wan2.2_animate2_distilled",
            "wan2.2_s2v",
        ]:
            self.startup_config["vae_stride"] = (4, 8, 8)
            if model_cls.startswith("wan2.2") and model_cls != "wan2.2_animate2_distilled":
                self.startup_config["use_image_encoder"] = False
        elif model_cls in ["wan2.2", "wan2.2_matrix_game3", "wan2.2_audio"]:
            self.startup_config.update(vae_stride=(4, 16, 16), num_channels_latents=48)
            if model_cls in ["wan2.2_matrix_game3", "wan2.2_audio"]:
                self.startup_config["use_image_encoder"] = False
        elif model_cls == "hunyuan_video_1.5":
            self.startup_config.update(vae_stride=(4, 16, 16), num_channels_latents=32)
        elif model_cls in ["ltx2", "ltx2_5"]:
            self.startup_config.update(num_channels_latents=128, audio_mel_bins=16)
        elif model_cls == "cosmos3":
            self.startup_config.update(vae_stride=(4, 16, 16), num_channels_latents=48)
        elif model_cls == "minimax_h3":
            self.startup_config.update(
                vae_spatial_scale_factor=16,
                vae_scale_factor=16,
                fps=24,
                audio_sampling_rate=32000,
                audio_flow_shift=3.0,
            )
        elif model_cls == "lingbot_video":
            self.startup_config.update(vae_stride=(4, 8, 8), num_channels_latents=16)

        if requested_model_cls in ["qwen-image", "qwen-image-2512", "qwen-image-edit", "qwen-image-edit-2509", "qwen-image-edit-2511"]:
            self.startup_config.update(CONDITION_IMAGE_SIZE=147456, USE_IMAGE_ID_IN_PROMPT=True)
            if requested_model_cls == "qwen-image-edit":
                self.startup_config.update(CONDITION_IMAGE_SIZE=1048576, USE_IMAGE_ID_IN_PROMPT=False)
            if task == "i2i":
                self.startup_config["prompt_template_encode"] = (
                    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
                )
                self.startup_config["prompt_template_encode_start_idx"] = 64
            elif task == "t2i":
                self.startup_config["prompt_template_encode"] = (
                    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
                )
                self.startup_config["prompt_template_encode_start_idx"] = 34

    def create_generator(
        self,
        attn_mode="flash_attn2",
        infer_steps=50,
        num_frames=81,
        size=(480, 832),
        guidance_scale=5.0,
        sample_shift=5.0,
        fps=None,
        aspect_ratio="16:9",
        boundary=0.900,
        boundary_step_index=2,
        denoising_step_list=(1000, 750, 500, 250),
        config_json=None,
        rope_type="torch_complex_rope",
        resize_mode=None,
        double_precision_rope=True,
        modulate_type=None,
        distilled_sigma_values=None,
    ):
        if self.runner is not None:
            raise RuntimeError("Generator has already been created for this pipeline")

        if config_json is None:
            self.set_infer_config(
                attn_mode,
                rope_type,
                infer_steps,
                num_frames,
                size,
                guidance_scale,
                sample_shift,
                aspect_ratio,
                boundary,
                boundary_step_index,
                denoising_step_list,
                double_precision_rope,
                modulate_type,
                distilled_sigma_values,
            )

        startup_config = dict(self.startup_config, config_json=config_json)
        if fps is not None:
            startup_config["fps"] = fps
        if resize_mode is not None:
            startup_config["resize_mode"] = resize_mode

        config = build_startup_config(startup_config)
        self.model_cls = config["model_cls"]
        self.model_variant = config.get("model_variant", self.model_variant)
        self.model_path = config["model_path"]
        validate_config_paths(config)

        if config["parallel"]:
            platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
            platform_device.init_parallel_env()
            init_parallel(config)

        self.runner = build_runner(config)
        print(self.runner.config)
        logger.info(f"Initialized {self.model_cls} runner; supported tasks: {', '.join(self.runner.supported_tasks)}")
        logger.info(f"Model path: {self.model_path}")
        logger.info("LightGenerator initialized successfully!")

    def modify_config(self, config_modify):
        """Update runtime configuration for embedding callers such as LightLLM."""
        with self.runner.config.temporarily_unlocked():
            self.runner.config.update(config_modify)

    def set_infer_config(
        self,
        attn_mode,
        rope_type,
        infer_steps,
        num_frames,
        size,
        guidance_scale,
        sample_shift,
        aspect_ratio,
        boundary,
        boundary_step_index,
        denoising_step_list,
        double_precision_rope,
        modulate_type,
        distilled_sigma_values,
    ):
        config = {
            "infer_steps": infer_steps,
            "size": list(size),
            "sample_guide_scale": guidance_scale,
            "sample_shift": sample_shift,
            "enable_cfg": guidance_scale != 1 and not (self.model_cls == "z_image" and guidance_scale == 0),
            "rope_type": rope_type,
            "aspect_ratio": aspect_ratio,
            "boundary": boundary,
            "boundary_step_index": boundary_step_index,
            "denoising_step_list": list(denoising_step_list),
            "double_precision_rope": double_precision_rope,
        }
        if modulate_type is not None:
            config["modulate_type"] = modulate_type
        if self.model_cls in ["ltx2", "ltx2_5"]:
            config.setdefault("modulate_type", self.startup_config.get("modulate_type", "torch"))
            if distilled_sigma_values is not None:
                config["distilled_sigma_values"] = distilled_sigma_values
                config["infer_steps"] = len(distilled_sigma_values) - 1
        if num_frames is not None:
            config["num_frames"] = num_frames
        elif self.model_cls == "ltx2_5" and self.startup_config.get("auto_duration", False):
            # ``None`` is meaningful for LTX-2.5: ask DurationHead to select
            # the request length. Preserve fixed JSON lengths for LTX-2/2.3
            # and for LTX-2.5 profiles with auto duration disabled.
            config["num_frames"] = None
        if self.model_cls.startswith("wan"):
            config.update(self_attn_1_type=attn_mode, cross_attn_1_type=attn_mode, cross_attn_2_type=attn_mode)
        elif self.model_cls in ["hunyuan_video_1.5", "qwen_image", "longcat_image", "ltx2", "ltx2_5", "z_image", "lingbot_video", "minimax_h3"]:
            config["attn_type"] = attn_mode
            if self.model_cls == "minimax_h3":
                config.update(
                    video_flow_shift=sample_shift,
                    audio_sampling_rate=32000,
                    audio_flow_shift=self.startup_config.get("audio_flow_shift", 3.0),
                    vae_spatial_scale_factor=16,
                    vae_scale_factor=16,
                )
        self.startup_config.update(config)

    def enable_lightvae(
        self,
        use_lightvae=False,
        use_tae=False,
        vae_path=None,
        tae_path=None,
    ):
        assert self.model_cls not in ["qwen_image", "longcat_image"]
        self.startup_config.update(use_lightvae=use_lightvae, use_tae=use_tae)
        if vae_path is not None:
            self.startup_config["vae_path"] = vae_path
        if tae_path is not None:
            self.startup_config["tae_path"] = tae_path
        if use_tae and self.model_cls.startswith("wan") and "lighttae" in tae_path:
            self.startup_config["need_scaled"] = True

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
        skip_fp8_block_index=(0, 43, 44, 45, 46, 47),
    ):
        self.startup_config.update(
            dit_quantized=dit_quantized,
            dit_quant_scheme=quant_scheme,
        )
        quantized_checkpoints = {
            "dit_quantized_ckpt": dit_quantized_ckpt,
            "low_noise_quantized_ckpt": low_noise_quantized_ckpt,
            "high_noise_quantized_ckpt": high_noise_quantized_ckpt,
        }
        self.startup_config.update({key: value for key, value in quantized_checkpoints.items() if value is not None})

        if self.model_cls.startswith("wan"):
            self.startup_config.update(
                t5_quant_scheme=quant_scheme,
                t5_quantized=text_encoder_quantized,
                t5_quantized_ckpt=text_encoder_quantized_ckpt,
                clip_quant_scheme=quant_scheme,
                clip_quantized=image_encoder_quantized,
                clip_quantized_ckpt=image_encoder_quantized_ckpt,
            )
        elif self.model_cls in ["hunyuan_video_1.5", "qwen_image"]:
            self.startup_config.update(
                qwen25vl_quantized=text_encoder_quantized,
                qwen25vl_quantized_ckpt=text_encoder_quantized_ckpt,
            )
            if text_encoder_quant_scheme is not None:
                self.startup_config["qwen25vl_quant_scheme"] = text_encoder_quant_scheme
        elif self.model_cls in ["ltx2", "ltx2_5"]:
            self.startup_config["skip_fp8_block_index"] = list(skip_fp8_block_index)
        elif self.model_cls == "z_image":
            self.startup_config.update(
                qwen3_quantized=text_encoder_quantized,
                qwen3_quantized_ckpt=text_encoder_quantized_ckpt,
            )
            if text_encoder_quant_scheme is not None:
                self.startup_config["qwen3_quant_scheme"] = text_encoder_quant_scheme

    def enable_offload(
        self,
        cpu_offload=False,
        offload_granularity="block",
        text_encoder_offload=False,
        image_encoder_offload=False,
        vae_offload=False,
    ):
        self.startup_config.update(cpu_offload=cpu_offload, offload_granularity=offload_granularity, vae_cpu_offload=vae_offload)
        if self.model_cls in [
            "wan2.1",
            "wan2.1_vace",
            "wan2.1_sf",
            "wan2.1_sf_mtxg2",
            "seko_talk",
            "seko_talk_ar",
            "wan2.2_moe",
            "wan2.2",
            "wan2.2_matrix_game3",
            "wan2.2_audio",
            "wan2.2_animate",
            "wan2.2_animate2_distilled",
            "wan2.2_s2v",
        ]:
            self.startup_config.update(t5_cpu_offload=text_encoder_offload, clip_cpu_offload=image_encoder_offload)

        elif self.model_cls == "hunyuan_video_1.5":
            self.startup_config.update(
                qwen25vl_cpu_offload=text_encoder_offload,
                siglip_cpu_offload=image_encoder_offload,
                byt5_cpu_offload=image_encoder_offload,
            )
        elif self.model_cls in ["qwen_image", "longcat_image"]:
            self.startup_config["qwen25vl_cpu_offload"] = text_encoder_offload
        elif self.model_cls in ["ltx2", "ltx2_5"]:
            self.startup_config["gemma_cpu_offload"] = text_encoder_offload
        elif self.model_cls == "z_image":
            self.startup_config["qwen3_cpu_offload"] = text_encoder_offload
        elif self.model_cls == "minimax_h3":
            self.startup_config["text_encoder_cpu_offload"] = text_encoder_offload

    def enable_lora(self, lora_configs, lora_dynamic_apply=False):
        self.startup_config.update(lora_configs=lora_configs, lora_dynamic_apply=lora_dynamic_apply)

    def switch_lora(self, lora_path: str, strength: float = 1.0):
        if lora_path == "":
            logger.info("Removing LoRA weights")
        else:
            logger.info(f"Switching LoRA to: {lora_path} with strength={strength}")
        if not self.runner.config.get("lora_dynamic_apply", False):
            logger.error("LoRA dynamic apply is not enabled. Please enable it first.")
            return
        self.runner.switch_lora(lora_path, strength)

    def enable_cache(
        self,
        cache_method="Tea",
        coefficients=(),
        teacache_thresh=0.15,
        use_ret_steps=False,
        magcache_calibration=False,
        magcache_K=6,
        magcache_thresh=0.24,
        magcache_retention_ratio=0.2,
        magcache_ratios=(),
    ):
        self.startup_config["feature_caching"] = cache_method
        if cache_method == "Tea":
            self.startup_config.update(coefficients=list(coefficients), teacache_thresh=teacache_thresh, use_ret_steps=use_ret_steps)
        elif cache_method == "Mag":
            self.startup_config.update(
                magcache_calibration=magcache_calibration,
                magcache_K=magcache_K,
                magcache_thresh=magcache_thresh,
                magcache_retention_ratio=magcache_retention_ratio,
                magcache_ratios=list(magcache_ratios),
            )

    def enable_parallel(self, cfg_p_size=1, seq_p_size=1, seq_p_attn_type="ulysses"):
        self.startup_config["parallel"] = {
            "cfg_p_size": cfg_p_size,
            "seq_p_size": seq_p_size,
            "seq_p_attn_type": seq_p_attn_type,
        }

    @torch.no_grad()
    def generate(
        self,
        seed=UNSET,
        prompt=None,
        negative_prompt=None,
        save_result_path=None,
        task=None,
        image_path=None,
        action_path=None,
        video_path=None,
        image_strength=None,
        i2i_denoise_strength=None,
        image_frame_indices=None,
        last_frame_path=None,
        audio_path=None,
        ref_image_paths=None,
        mask_path=None,
        return_result_tensor=None,
        size=None,
        num_frames=None,
        sr_ratio=None,
        ref_video_prompt=None,
        **task_inputs,
    ):
        """Generate one result, validating task-specific inputs in the runner.

        Size is (height, width) in pixels, subject to the model's sizing rules.
        Seed is a non-negative integer; omitted/None defaults to 42.
        NeoPP preserves explicit None for LightLLM session RNG continuation.
        An omitted output path is passed to the runner as None. Most runners skip saving;
        WorldMirror uses its default output directory.
        An omitted task uses the task explicitly set when creating the pipeline,
        or the runner's only supported task. Multi-task runners require a task.
        """
        if task is None:
            task = self.task
        request_data = {
            "task": task,
            "seed": seed,
            "prompt": prompt,
            "ref_video_prompt": ref_video_prompt,
            "negative_prompt": negative_prompt,
            "save_result_path": save_result_path,
            "image_path": image_path,
            "action_path": action_path,
            "video_path": video_path,
            "last_frame_path": last_frame_path,
            "audio_path": audio_path,
            "ref_image_paths": ref_image_paths,
            "mask_path": mask_path,
            "return_result_tensor": return_result_tensor,
            "size": size,
            "num_frames": num_frames,
            "sr_ratio": sr_ratio,
            "image_strength": image_strength,
            "i2i_denoise_strength": i2i_denoise_strength,
            "image_frame_indices": image_frame_indices,
        }
        request_data.update(task_inputs)

        input_info = self.runner.prepare_request(request_data)
        gen_result = self.runner.run_request(input_info)
        logger.info("Generated successfully!")
        return gen_result
