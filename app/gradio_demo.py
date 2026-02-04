"""
重构后的 Gradio Demo 主入口文件
整合了所有模块，支持中英文切换
"""

import argparse
import gc
import json
import logging
import os
import warnings

import torch
from loguru import logger
from utils.i18n import DEFAULT_LANG, set_language
from utils.model_utils import cleanup_memory, extract_op_name, get_model_configs
from utils.ui_builder import build_ui, generate_unique_filename, get_auto_config_dict

from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.set_config import get_default_config

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.utils")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

os.environ["PROFILING_DEBUG_LEVEL"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DTYPE"] = "BF16"


logger.add(
    "inference_logs.log",
    rotation="100 MB",
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)


global_runner = None
current_config = None
cur_dit_path = None
cur_use_lora = None
cur_lora_path = None
cur_high_lora_path = None
cur_low_lora_path = None


def run_inference(
    prompt="",
    negative_prompt="",
    save_result_path="",
    infer_steps=4,
    num_frames=81,
    resolution="480p",
    seed=42,
    sample_shift=5,
    cfg_scale=1,
    fps=16,
    model_path_input=None,
    model_type_input="wan2.1",
    task_type_input="i2v",
    dit_path_input=None,
    high_noise_path_input=None,
    low_noise_path_input=None,
    t5_path_input=None,
    clip_path_input="",
    image_path=None,
    vae_path=None,
    qwen_image_dit_path_input=None,
    qwen_image_vae_path_input=None,
    qwen_image_scheduler_path_input=None,
    qwen25vl_encoder_path_input=None,
    z_image_dit_path_input=None,
    z_image_vae_path_input=None,
    z_image_scheduler_path_input=None,
    qwen3_encoder_path_input=None,
    aspect_ratio="1:1",
    use_lora=None,
    lora_path=None,
    lora_strength=None,
    high_noise_lora_path=None,
    low_noise_lora_path=None,
    high_noise_lora_strength=None,
    low_noise_lora_strength=None,
):
    cleanup_memory()

    auto_config = get_auto_config_dict(model_type=model_type_input, resolution=resolution, num_frames=num_frames, task_type=task_type_input)

    # 从 auto_config 中获取 offload 和 rope 相关配置
    rope_chunk = auto_config["rope_chunk_val"]
    rope_chunk_size = auto_config["rope_chunk_size_val"]
    cpu_offload = auto_config["cpu_offload_val"]
    offload_granularity = auto_config["offload_granularity_val"]
    lazy_load = auto_config["lazy_load_val"]
    t5_cpu_offload = auto_config["t5_cpu_offload_val"]
    clip_cpu_offload = auto_config["clip_cpu_offload_val"]
    vae_cpu_offload = auto_config["vae_cpu_offload_val"]
    unload_modules = auto_config["unload_modules_val"]
    attention_type = auto_config["attention_type_val"]
    quant_op = auto_config["quant_op_val"]
    use_tiling_vae = auto_config["use_tiling_vae_val"]
    clean_cuda_cache = auto_config["clean_cuda_cache_val"]
    quant_op = extract_op_name(quant_op)
    attention_type = extract_op_name(attention_type)
    task = task_type_input

    is_image_output = task in ["i2i", "t2i"]
    save_result_path = generate_unique_filename(output_dir, is_image=is_image_output)

    if cfg_scale == 1:
        enable_cfg = False
    else:
        enable_cfg = True

    model_config = get_model_configs(
        model_type_input,
        model_path_input,
        dit_path_input,
        high_noise_path_input,
        low_noise_path_input,
        t5_path_input,
        clip_path_input,
        vae_path,
        qwen_image_dit_path_input,
        qwen_image_vae_path_input,
        qwen_image_scheduler_path_input,
        qwen25vl_encoder_path_input,
        z_image_dit_path_input,
        z_image_vae_path_input,
        z_image_scheduler_path_input,
        qwen3_encoder_path_input,
        quant_op,
        use_lora=use_lora,
        lora_path=lora_path,
        lora_strength=lora_strength,
        high_noise_lora_path=high_noise_lora_path,
        low_noise_lora_path=low_noise_lora_path,
        high_noise_lora_strength=high_noise_lora_strength,
        low_noise_lora_strength=low_noise_lora_strength,
    )
    model_cls = model_config["model_cls"]
    model_path = model_config["model_path"]

    global global_runner, current_config, cur_dit_path, cur_use_lora, cur_lora_path, cur_high_lora_path, cur_low_lora_path

    logger.info(f"Auto-determined model_cls: {model_cls} (model type: {model_type_input})")

    if model_cls.startswith("wan2.2"):
        current_dit_path = f"{high_noise_path_input}|{low_noise_path_input}" if high_noise_path_input and low_noise_path_input else None
    else:
        current_dit_path = dit_path_input

    needs_reinit = lazy_load or unload_modules or global_runner is None or cur_dit_path != current_dit_path or cur_use_lora != use_lora

    config_graio = {
        "infer_steps": infer_steps,
        "target_video_length": num_frames,
        "resolution": resolution,
        "resize_mode": "adaptive",
        "self_attn_1_type": attention_type,
        "cross_attn_1_type": attention_type,
        "cross_attn_2_type": attention_type,
        "attn_type": attention_type,
        "enable_cfg": enable_cfg,
        "sample_guide_scale": cfg_scale,
        "sample_shift": sample_shift,
        "fps": fps,
        "feature_caching": "NoCaching",
        "do_mm_calib": False,
        "parallel_attn_type": None,
        "parallel_vae": False,
        "max_area": False,
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "lora_path": None,
        "strength_model": 1.0,
        "use_prompt_enhancer": False,
        "text_len": 512,
        "denoising_step_list": [1000, 750, 500, 250],
        "cpu_offload": True if "wan2.2" in model_cls else cpu_offload,
        "offload_granularity": ("phase" if "wan2.2" in model_cls else offload_granularity),
        "t5_cpu_offload": t5_cpu_offload,
        "clip_cpu_offload": clip_cpu_offload,
        "vae_cpu_offload": vae_cpu_offload,
        "use_tiling_vae": use_tiling_vae,
        "lazy_load": lazy_load,
        "rope_chunk": rope_chunk,
        "rope_chunk_size": rope_chunk_size,
        "clean_cuda_cache": clean_cuda_cache,
        "unload_modules": unload_modules,
        "seq_parallel": False,
        "warm_up_cpu_buffers": False,
        "boundary_step_index": 2,
        "boundary": 0.900,
        "use_image_encoder": False if "wan2.2" in model_cls else True,
        "rope_type": "torch",
        "t5_lazy_load": lazy_load,
        "bucket_shape": {
            "0.667": [[480, 832], [544, 960], [720, 960]],
            "1.500": [[832, 480], [960, 544], [960, 720]],
            "1.000": [[480, 480], [576, 576], [720, 720]],
        },
        "aspect_ratio": aspect_ratio,
    }

    args = argparse.Namespace(
        model_cls=model_cls,
        seed=seed,
        task=task,
        model_path=model_path,
        prompt_enhancer=None,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path=image_path,
        save_result_path=save_result_path,
        return_result_tensor=False,
        aspect_ratio=aspect_ratio,
        target_shape=[],
    )
    input_info = init_empty_input_info(args.task)
    config = get_default_config()
    config.update({k: v for k, v in vars(args).items()})
    config.update(config_graio)
    config.update(model_config)

    # 如果 model_config 中包含 lora_configs，设置 lora_dynamic_apply
    if config.get("lora_configs"):
        config["lora_dynamic_apply"] = True

    logger.info(f"Using model: {model_path}")
    logger.info(f"Inference config:\n{json.dumps(config, indent=4, ensure_ascii=False)}")

    # 初始化或重用 runner
    runner = global_runner
    if needs_reinit:
        if runner is not None:
            del runner
            torch.cuda.empty_cache()
            gc.collect()

        from lightx2v.infer import init_runner

        runner = init_runner(config)

        data = args.__dict__
        update_input_info_from_dict(input_info, data)

        current_config = config
        cur_dit_path = current_dit_path
        cur_use_lora = use_lora
        cur_lora_path = lora_path

        # 保存 Wan2.2 的 LoRA 路径
        if model_cls.startswith("wan2.2"):
            lora_configs = config.get("lora_configs")
            if lora_configs:
                lora_name_to_info = {item["name"]: item for item in lora_configs}
                cur_high_lora_path = lora_name_to_info.get("high_noise_model", {}).get("path")
                cur_low_lora_path = lora_name_to_info.get("low_noise_model", {}).get("path")
            else:
                cur_high_lora_path = None
                cur_low_lora_path = None

        if not lazy_load:
            global_runner = runner
    else:
        runner.config = config
        data = args.__dict__
        update_input_info_from_dict(input_info, data)

        # 如果 use_lora 为 True 且 lora_path 变化了，调用 switch_lora
        if use_lora:
            lora_configs = config.get("lora_configs")
            if model_cls.startswith("wan2.2") and lora_configs:
                # 对于 Wan2.2 模型，从 lora_configs 中获取 high_noise 和 low_noise 的 LoRA 路径
                lora_name_to_info = {item["name"]: item for item in lora_configs}
                high_lora_path = None
                high_lora_strength = 1.0
                low_lora_path = None
                low_lora_strength = 1.0

                if "high_noise_model" in lora_name_to_info:
                    high_lora_info = lora_name_to_info["high_noise_model"]
                    high_lora_path = high_lora_info["path"]
                    high_lora_strength = high_lora_info.get("strength", 1.0)

                if "low_noise_model" in lora_name_to_info:
                    low_lora_info = lora_name_to_info["low_noise_model"]
                    low_lora_path = low_lora_info["path"]
                    low_lora_strength = low_lora_info.get("strength", 1.0)

                # 检查 high_lora_path 和 low_lora_path 是否变化
                high_lora_changed = high_lora_path != cur_high_lora_path
                low_lora_changed = low_lora_path != cur_low_lora_path

                if high_lora_changed or low_lora_changed:
                    if hasattr(runner, "switch_lora"):
                        runner.switch_lora(
                            high_lora_path=high_lora_path,
                            high_lora_strength=high_lora_strength,
                            low_lora_path=low_lora_path,
                            low_lora_strength=low_lora_strength,
                        )
                        logger.info(f"Switched LoRA for Wan2.2: high={high_lora_path}, low={low_lora_path}")
                        cur_high_lora_path = high_lora_path
                        cur_low_lora_path = low_lora_path
                    else:
                        logger.warning("Runner does not support switch_lora method")
            elif lora_path and lora_path != cur_lora_path:
                lora_strength_val = float(lora_strength) if lora_strength is not None else 1.0
                if hasattr(runner, "switch_lora"):
                    runner.switch_lora(lora_path, lora_strength_val)
                    logger.info(f"Switched LoRA to: {lora_path} with strength={lora_strength_val}")
                else:
                    logger.warning("Runner does not support switch_lora method")
                cur_lora_path = lora_path

    runner.run_pipeline(input_info)
    cleanup_memory()

    return save_result_path


def main(lang=DEFAULT_LANG):
    """主函数"""
    set_language(lang)
    demo = build_ui(
        model_path=model_path,
        output_dir=output_dir,
        run_inference=run_inference,
        lang=lang,
    )

    # 启动 Gradio 应用
    demo.launch(
        share=True,
        server_port=args.server_port,
        server_name=args.server_name,
        inbrowser=True,
        allowed_paths=[output_dir],
        max_file_size="1gb",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="轻量级视频生成")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件夹路径")
    parser.add_argument("--server_port", type=int, default=7862, help="服务器端口")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="服务器IP")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出视频保存目录")
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="界面语言")
    args = parser.parse_args()

    global model_path, model_cls, output_dir
    model_path = args.model_path
    model_cls = "wan2.1"
    output_dir = args.output_dir

    main(lang=args.lang)
