import random
from datetime import datetime

import gradio as gr
from loguru import logger
from utils.i18n import DEFAULT_LANG, t
from utils.image_page import build_image_page
from utils.model_utils import (
    get_cpu_memory,
    get_gpu_generation,
    get_gpu_memory,
)
from utils.video_page import build_video_page


def get_gpu_rules(resolution):
    """根据分辨率获取 GPU 规则

    Args:
        resolution: 分辨率 ("480p", "540p", "720p")

    Returns:
        list: GPU 规则列表，每个元素为 (threshold, config_dict)
    """
    if resolution in ["540p", "720p"]:
        return [
            (80, {}),
            (40, {"cpu_offload_val": False, "t5_cpu_offload_val": True, "vae_cpu_offload_val": True, "clip_cpu_offload_val": True}),
            (32, {"cpu_offload_val": True, "t5_cpu_offload_val": False, "vae_cpu_offload_val": False, "clip_cpu_offload_val": False}),
            (
                24,
                {
                    "cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                    "t5_cpu_offload_val": True,
                    "vae_cpu_offload_val": True,
                    "clip_cpu_offload_val": True,
                },
            ),
            (
                16,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "vae_cpu_offload_val": True,
                    "clip_cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                    "rope_chunk_val": True,
                    "rope_chunk_size_val": 100,
                },
            ),
            (
                8,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "vae_cpu_offload_val": True,
                    "clip_cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                    "rope_chunk_val": True,
                    "rope_chunk_size_val": 100,
                    "clean_cuda_cache_val": True,
                },
            ),
            (
                -1,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "vae_cpu_offload_val": True,
                    "clip_cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                    "rope_chunk_val": True,
                    "rope_chunk_size_val": 100,
                    "clean_cuda_cache_val": True,
                },
            ),
        ]
    else:
        return [
            (80, {}),
            (40, {"cpu_offload_val": False, "t5_cpu_offload_val": True, "vae_cpu_offload_val": True, "clip_cpu_offload_val": True}),
            (32, {"cpu_offload_val": True, "t5_cpu_offload_val": False, "vae_cpu_offload_val": False, "clip_cpu_offload_val": False}),
            (
                24,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "vae_cpu_offload_val": True,
                    "clip_cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                },
            ),
            (
                16,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "vae_cpu_offload_val": True,
                    "clip_cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                },
            ),
            (
                8,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "vae_cpu_offload_val": True,
                    "clip_cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                },
            ),
            (
                -1,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "vae_cpu_offload_val": True,
                    "clip_cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                },
            ),
        ]


def get_cpu_rules():
    """获取 CPU 规则

    Returns:
        list: CPU 规则列表，每个元素为 (threshold, config_dict)
    """
    return [
        (128, {}),
        (64, {}),
        (32, {"unload_modules_val": True}),
        (
            16,
            {
                "lazy_load_val": True,
                "unload_modules_val": True,
            },
        ),
        (
            -1,
            {
                "t5_lazy_load": True,
                "lazy_load_val": True,
                "unload_modules_val": True,
            },
        ),
    ]


css = """
        .main-content { max-width: 1600px; margin: auto; padding: 20px; }
        .warning { color: #ff6b6b; font-weight: bold; }

        /* 模型状态样式 */
        .model-status {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 12px !important;
            line-height: 1.2 !important;
            min-height: 20px !important;
        }

        /* 模型配置区域样式 */
        .model-config {
            margin-bottom: 20px !important;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 15px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        /* 输入参数区域样式 */
        .input-params {
            margin-bottom: 20px !important;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 15px;
            background: linear-gradient(135deg, #fff5f5 0%, #ffeef0 100%);
        }

        /* 输出视频区域样式 */
        .output-video {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            min-height: 400px;
        }

        /* 生成按钮样式 */
        .generate-btn {
            width: 100%;
            margin-top: 20px;
            padding: 15px 30px !important;
            font-size: 18px !important;
            font-weight: bold !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        }

        /* Accordion 标题样式 */
        .model-config .gr-accordion-header,
        .input-params .gr-accordion-header,
        .output-video .gr-accordion-header {
            font-size: 20px !important;
            font-weight: bold !important;
            padding: 15px !important;
        }

        /* 优化间距 */
        .gr-row {
            margin-bottom: 15px;
        }

        /* 视频播放器样式 */
        .output-video video {
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        /* Diffusion模型容器 */
        .diffusion-model-group {
            margin-bottom: 20px !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /* 移除 Gradio 组件的默认边框 */
        .diffusion-model-group > div,
        .diffusion-model-group .gr-group,
        .diffusion-model-group .gr-box {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /* 编码器组容器（文本编码器、图像编码器） */
        .encoder-group {
            margin-bottom: 20px !important;
        }

        /* VAE组容器 */
        .vae-group {
            margin-bottom: 20px !important;
        }

        /* 模型组标题样式 */
        .model-group-title {
            font-size: 16px !important;
            font-weight: 600 !important;
            margin-bottom: 12px !important;
            color: #24292f !important;
        }

        /* 下载按钮样式 */
        .download-btn {
            width: 100% !important;
            margin-top: 8px !important;
            border-radius: 6px !important;
            transition: all 0.2s ease !important;
        }
        .download-btn:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }

        /* 水平排列的Radio按钮 */
        .horizontal-radio .form-radio {
            display: flex !important;
            flex-direction: row !important;
            gap: 20px !important;
        }
        .horizontal-radio .form-radio > label {
            margin-right: 20px !important;
        }

        /* wan2.2 行样式 - 去掉上边框和分隔线 */
        .wan22-row {
            border-top: none !important;
            border-bottom: none !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        .wan22-row > div {
            border-top: none !important;
            border-bottom: none !important;
        }
        .wan22-row .gr-column {
            border-top: none !important;
            border-bottom: none !important;
            border-left: none !important;
            border-right: none !important;
        }
        .wan22-row .gr-column:first-child {
            border-right: none !important;
        }
        .wan22-row .gr-column:last-child {
            border-left: none !important;
        }
    """


MAX_NUMPY_SEED = 2**32 - 1


def generate_random_seed():
    """生成随机种子"""
    return random.randint(0, MAX_NUMPY_SEED)


def generate_unique_filename(output_dir, is_image=False):
    """生成唯一文件名"""
    import os

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = ".png" if is_image else ".mp4"
    return os.path.join(output_dir, f"{timestamp}{ext}")


def get_auto_config_dict(
    model_type,
    resolution,
    num_frames=81,
    task_type=None,
):
    if task_type in ["i2i", "t2i"]:
        attention_type_val = "torch_sdpa"
    else:
        attention_type_val = "sage_attn2"

    gpu_gen = get_gpu_generation()
    if gpu_gen == "40":
        quant_op_val = "q8f"
    elif gpu_gen == "30":
        quant_op_val = "vllm"
    else:
        quant_op_val = "triton"

    default_config = {
        "lazy_load_val": False,
        "rope_chunk_val": False,
        "rope_chunk_size_val": 100,
        "clean_cuda_cache_val": False,
        "cpu_offload_val": False,
        "offload_granularity_val": "block",
        "t5_cpu_offload_val": False,
        "clip_cpu_offload_val": False,
        "vae_cpu_offload_val": False,
        "unload_modules_val": False,
        "attention_type_val": attention_type_val,
        "quant_op_val": quant_op_val,
        "use_tiling_vae_val": False,
    }

    # If num_frames > 81, set rope_chunk to True regardless of resolution
    if num_frames is not None and num_frames > 81:
        default_config["rope_chunk_val"] = True

    gpu_memory = round(get_gpu_memory())
    cpu_memory = round(get_cpu_memory())

    gpu_rules = get_gpu_rules(resolution)
    cpu_rules = get_cpu_rules()

    for threshold, updates in gpu_rules:
        if gpu_memory >= threshold:
            default_config.update(updates)
            break

    for threshold, updates in cpu_rules:
        if cpu_memory >= threshold:
            default_config.update(updates)
            break

    if model_type == "Z-Image-Turbo":
        default_config["lazy_load_val"] = False
        if default_config["cpu_offload_val"]:
            default_config["offload_granularity_val"] = "model"
        default_config["quant_op_val"] = "triton"

    return default_config


def build_ui(
    model_path,
    output_dir,
    run_inference,
    lang=DEFAULT_LANG,
):
    # 在启动时加载 Hugging Face 模型列表缓存
    logger.info(t("loading_models", lang))
    from utils.model_utils import load_hf_models_cache

    load_hf_models_cache()
    logger.info(t("models_loaded", lang))

    with gr.Blocks(title="Lightx2v (轻量级视频推理和生成引擎)") as demo:
        gr.Markdown(f"# 🎬 LightX2V 图片/视频生成器")
        gr.HTML(f"<style>{css}</style>")

        # 使用 Tabs 分成两个页面
        with gr.Tabs():
            # 图片生成页面
            with gr.Tab("🖼️ 图片生成"):
                build_image_page(
                    model_path,
                    output_dir,
                    run_inference,
                    lang,
                )
            # 视频生成页面
            with gr.Tab("🎬 视频生成"):
                build_video_page(
                    model_path,
                    output_dir,
                    run_inference,
                    lang,
                )

    # 返回 demo 对象，由调用者负责 launch
    return demo
