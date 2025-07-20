import os
import gradio as gr
import argparse
import json
import torch
import gc
from easydict import EasyDict
from datetime import datetime
from loguru import logger

import importlib.util
import psutil
import random

logger.add(
    "inference_logs.log",
    rotation="100 MB",
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

MAX_NUMPY_SEED = 2**32 - 1


def generate_random_seed():
    return random.randint(0, MAX_NUMPY_SEED)


def is_module_installed(module_name):
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ModuleNotFoundError:
        return False


def get_available_quant_ops():
    available_ops = []

    vllm_installed = is_module_installed("vllm")
    if vllm_installed:
        available_ops.append(("vllm", True))
    else:
        available_ops.append(("vllm", False))

    sgl_installed = is_module_installed("sgl_kernel")
    if sgl_installed:
        available_ops.append(("sgl", True))
    else:
        available_ops.append(("sgl", False))

    q8f_installed = is_module_installed("q8_kernels")
    if q8f_installed:
        available_ops.append(("q8f", True))
    else:
        available_ops.append(("q8f", False))

    return available_ops


def get_available_attn_ops():
    available_ops = []

    vllm_installed = is_module_installed("flash_attn")
    if vllm_installed:
        available_ops.append(("flash_attn2", True))
    else:
        available_ops.append(("flash_attn2", False))

    sgl_installed = is_module_installed("flash_attn_interface")
    if sgl_installed:
        available_ops.append(("flash_attn3", True))
    else:
        available_ops.append(("flash_attn3", False))

    q8f_installed = is_module_installed("sageattention")
    if q8f_installed:
        available_ops.append(("sage_attn2", True))
    else:
        available_ops.append(("sage_attn2", False))

    torch_installed = is_module_installed("torch")
    if torch_installed:
        available_ops.append(("torch_sdpa", True))
    else:
        available_ops.append(("torch_sdpa", False))

    return available_ops


def get_gpu_memory(gpu_idx=0):
    if not torch.cuda.is_available():
        return 0
    try:
        with torch.cuda.device(gpu_idx):
            memory_info = torch.cuda.mem_get_info()
            total_memory = memory_info[1] / (1024**3)  # Convert bytes to GB
            return total_memory
    except Exception as e:
        logger.warning(f"Failed to get GPU memory: {e}")
        return 0


def get_cpu_memory():
    available_bytes = psutil.virtual_memory().available
    return available_bytes / 1024**3


def cleanup_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        if hasattr(psutil, "virtual_memory"):
            if os.name == "posix":
                try:
                    os.system("sync")
                except:  # noqa
                    pass
    except:  # noqa
        pass


def generate_unique_filename(base_dir="./saved_videos"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{model_cls}_{timestamp}.mp4")


def is_fp8_supported_gpu():
    if not torch.cuda.is_available():
        return False
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    return (major == 8 and minor == 9) or (major >= 9)


global_runner = None
current_config = None
cur_dit_quant_scheme = None
cur_clip_quant_scheme = None
cur_t5_quant_scheme = None
cur_precision_mode = None
cur_enable_teacache = None

available_quant_ops = get_available_quant_ops()
quant_op_choices = []
for op_name, is_installed in available_quant_ops:
    status_text = "✅ Installed" if is_installed else "❌ Not Installed"
    display_text = f"{op_name} ({status_text})"
    quant_op_choices.append((op_name, display_text))

available_attn_ops = get_available_attn_ops()
attn_op_choices = []
for op_name, is_installed in available_attn_ops:
    status_text = "✅ Installed" if is_installed else "❌ Not Installed"
    display_text = f"{op_name} ({status_text})"
    attn_op_choices.append((op_name, display_text))


def run_inference(
    prompt,
    negative_prompt,
    save_video_path,
    torch_compile,
    infer_steps,
    num_frames,
    resolution,
    seed,
    sample_shift,
    enable_teacache,
    teacache_thresh,
    use_ret_steps,
    enable_cfg,
    cfg_scale,
    dit_quant_scheme,
    t5_quant_scheme,
    clip_quant_scheme,
    fps,
    use_tiny_vae,
    use_tiling_vae,
    lazy_load,
    precision_mode,
    cpu_offload,
    offload_granularity,
    offload_ratio,
    t5_cpu_offload,
    unload_modules,
    t5_offload_granularity,
    attention_type,
    quant_op,
    rotary_chunk,
    rotary_chunk_size,
    clean_cuda_cache,
    image_path=None,
):
    cleanup_memory()

    quant_op = quant_op.split("(")[0].strip()
    attention_type = attention_type.split("(")[0].strip()

    global global_runner, current_config, model_path, task
    global cur_dit_quant_scheme, cur_clip_quant_scheme, cur_t5_quant_scheme, cur_precision_mode, cur_enable_teacache

    if os.path.exists(os.path.join(model_path, "config.json")):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            model_config = json.load(f)

    if task == "t2v":
        if model_size == "1.3b":
            # 1.3B
            coefficient = [
                [
                    -5.21862437e04,
                    9.23041404e03,
                    -5.28275948e02,
                    1.36987616e01,
                    -4.99875664e-02,
                ],
                [
                    2.39676752e03,
                    -1.31110545e03,
                    2.01331979e02,
                    -8.29855975e00,
                    1.37887774e-01,
                ],
            ]
        else:
            # 14B
            coefficient = [
                [
                    -3.03318725e05,
                    4.90537029e04,
                    -2.65530556e03,
                    5.87365115e01,
                    -3.15583525e-01,
                ],
                [
                    -5784.54975374,
                    5449.50911966,
                    -1811.16591783,
                    256.27178429,
                    -13.02252404,
                ],
            ]
    elif task == "i2v":
        if resolution in [
            "1280x720",
            "720x1280",
            "1280x544",
            "544x1280",
            "1104x832",
            "832x1104",
            "960x960",
        ]:
            # 720p
            coefficient = [
                [
                    8.10705460e03,
                    2.13393892e03,
                    -3.72934672e02,
                    1.66203073e01,
                    -4.17769401e-02,
                ],
                [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
            ]
        else:
            # 480p
            coefficient = [
                [
                    2.57151496e05,
                    -3.54229917e04,
                    1.40286849e03,
                    -1.35890334e01,
                    1.32517977e-01,
                ],
                [
                    -3.02331670e02,
                    2.23948934e02,
                    -5.25463970e01,
                    5.87348440e00,
                    -2.01973289e-01,
                ],
            ]

    save_video_path = generate_unique_filename()

    is_dit_quant = dit_quant_scheme != "bf16"
    is_t5_quant = t5_quant_scheme != "bf16"
    if is_t5_quant:
        t5_path = os.path.join(model_path, t5_quant_scheme)
        t5_quant_ckpt = os.path.join(t5_path, f"models_t5_umt5-xxl-enc-{t5_quant_scheme}.pth")
    else:
        t5_quant_ckpt = None

    is_clip_quant = clip_quant_scheme != "fp16"
    if is_clip_quant:
        clip_path = os.path.join(model_path, clip_quant_scheme)
        clip_quant_ckpt = os.path.join(clip_path, f"clip-{clip_quant_scheme}.pth")
    else:
        clip_quant_ckpt = None

    needs_reinit = (
        lazy_load
        or unload_modules
        or global_runner is None
        or current_config is None
        or cur_dit_quant_scheme is None
        or cur_dit_quant_scheme != dit_quant_scheme
        or cur_clip_quant_scheme is None
        or cur_clip_quant_scheme != clip_quant_scheme
        or cur_t5_quant_scheme is None
        or cur_t5_quant_scheme != t5_quant_scheme
        or cur_precision_mode is None
        or cur_precision_mode != precision_mode
        or cur_enable_teacache is None
        or cur_enable_teacache != enable_teacache
    )

    if torch_compile:
        os.environ["ENABLE_GRAPH_MODE"] = "true"
    else:
        os.environ["ENABLE_GRAPH_MODE"] = "false"
    if precision_mode == "bf16":
        os.environ["DTYPE"] = "BF16"
    else:
        os.environ.pop("DTYPE", None)

    if is_dit_quant:
        if quant_op == "vllm":
            mm_type = f"W-{dit_quant_scheme}-channel-sym-A-{dit_quant_scheme}-channel-sym-dynamic-Vllm"
        elif quant_op == "sgl":
            if dit_quant_scheme == "int8":
                mm_type = f"W-{dit_quant_scheme}-channel-sym-A-{dit_quant_scheme}-channel-sym-dynamic-Sgl-ActVllm"
            else:
                mm_type = f"W-{dit_quant_scheme}-channel-sym-A-{dit_quant_scheme}-channel-sym-dynamic-Sgl"
        elif quant_op == "q8f":
            mm_type = f"W-{dit_quant_scheme}-channel-sym-A-{dit_quant_scheme}-channel-sym-dynamic-Q8F"

        dit_quantized_ckpt = os.path.join(model_path, dit_quant_scheme)
        if os.path.exists(os.path.join(dit_quantized_ckpt, "config.json")):
            with open(os.path.join(dit_quantized_ckpt, "config.json"), "r") as f:
                quant_model_config = json.load(f)
        else:
            quant_model_config = {}
    else:
        mm_type = "Default"
        dit_quantized_ckpt = None
        quant_model_config = {}

    config = {
        "infer_steps": infer_steps,
        "target_video_length": num_frames,
        "target_width": int(resolution.split("x")[0]),
        "target_height": int(resolution.split("x")[1]),
        "self_attn_1_type": attention_type,
        "cross_attn_1_type": attention_type,
        "cross_attn_2_type": attention_type,
        "seed": seed,
        "enable_cfg": enable_cfg,
        "sample_guide_scale": cfg_scale,
        "sample_shift": sample_shift,
        "cpu_offload": cpu_offload,
        "offload_granularity": offload_granularity,
        "offload_ratio": offload_ratio,
        "t5_offload_granularity": t5_offload_granularity,
        "dit_quantized_ckpt": dit_quantized_ckpt,
        "mm_config": {
            "mm_type": mm_type,
        },
        "fps": fps,
        "feature_caching": "Tea" if enable_teacache else "NoCaching",
        "coefficients": coefficient[0] if use_ret_steps else coefficient[1],
        "use_ret_steps": use_ret_steps,
        "teacache_thresh": teacache_thresh,
        "t5_cpu_offload": t5_cpu_offload,
        "unload_modules": unload_modules,
        "t5_quantized": is_t5_quant,
        "t5_quantized_ckpt": t5_quant_ckpt,
        "t5_quant_scheme": t5_quant_scheme,
        "clip_quantized": is_clip_quant,
        "clip_quantized_ckpt": clip_quant_ckpt,
        "clip_quant_scheme": clip_quant_scheme,
        "use_tiling_vae": use_tiling_vae,
        "tiny_vae": use_tiny_vae,
        "tiny_vae_path": (os.path.join(model_path, "taew2_1.pth") if use_tiny_vae else None),
        "lazy_load": lazy_load,
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
        "rotary_chunk": rotary_chunk,
        "rotary_chunk_size": rotary_chunk_size,
        "clean_cuda_cache": clean_cuda_cache,
    }

    args = argparse.Namespace(
        model_cls=model_cls,
        task=task,
        model_path=model_path,
        prompt_enhancer=None,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path=image_path,
        save_video_path=save_video_path,
    )

    config.update({k: v for k, v in vars(args).items()})
    config = EasyDict(config)
    config.update(model_config)
    config.update(quant_model_config)

    logger.info(f"Using model: {model_path}")
    logger.info(f"Inference configuration:\n{json.dumps(config, indent=4, ensure_ascii=False)}")

    # Initialize or reuse the runner
    runner = global_runner
    if needs_reinit:
        if runner is not None:
            del runner
            torch.cuda.empty_cache()
            gc.collect()

        from lightx2v.infer import init_runner  # noqa

        runner = init_runner(config)
        current_config = config
        cur_dit_quant_scheme = dit_quant_scheme
        cur_clip_quant_scheme = clip_quant_scheme
        cur_t5_quant_scheme = t5_quant_scheme
        cur_precision_mode = precision_mode
        cur_enable_teacache = enable_teacache

        if not lazy_load:
            global_runner = runner
    else:
        runner.config = config

    runner.run_pipeline()

    del config, args, model_config, quant_model_config
    if "dit_quantized_ckpt" in locals():
        del dit_quantized_ckpt
    if "t5_quant_ckpt" in locals():
        del t5_quant_ckpt
    if "clip_quant_ckpt" in locals():
        del clip_quant_ckpt

    cleanup_memory()

    return save_video_path


def handle_lazy_load_change(lazy_load_enabled):
    """Handle lazy_load checkbox change to automatically enable unload_modules"""
    return gr.update(value=lazy_load_enabled)


def auto_configure(enable_auto_config, resolution):
    default_config = {
        "torch_compile_val": False,
        "lazy_load_val": False,
        "rotary_chunk_val": False,
        "rotary_chunk_size_val": 100,
        "clean_cuda_cache_val": False,
        "cpu_offload_val": False,
        "offload_granularity_val": "block",
        "offload_ratio_val": 1,
        "t5_cpu_offload_val": False,
        "unload_modules_val": False,
        "t5_offload_granularity_val": "model",
        "attention_type_val": attn_op_choices[0][1],
        "quant_op_val": quant_op_choices[0][1],
        "dit_quant_scheme_val": "bf16",
        "t5_quant_scheme_val": "bf16",
        "clip_quant_scheme_val": "fp16",
        "precision_mode_val": "fp32",
        "use_tiny_vae_val": False,
        "use_tiling_vae_val": False,
        "enable_teacache_val": False,
        "teacache_thresh_val": 0.26,
        "use_ret_steps_val": False,
    }

    if not enable_auto_config:
        return tuple(gr.update(value=default_config[key]) for key in default_config)

    gpu_memory = round(get_gpu_memory())
    cpu_memory = round(get_cpu_memory())

    if is_fp8_supported_gpu():
        quant_type = "fp8"
    else:
        quant_type = "int8"

    attn_priority = ["sage_attn2", "flash_attn3", "flash_attn2", "torch_sdpa"]
    quant_op_priority = ["sgl", "vllm", "q8f"]

    for op in attn_priority:
        if dict(available_attn_ops).get(op):
            default_config["attention_type_val"] = dict(attn_op_choices)[op]
            break

    for op in quant_op_priority:
        if dict(available_quant_ops).get(op):
            default_config["quant_op_val"] = dict(quant_op_choices)[op]
            break

    if resolution in [
        "1280x720",
        "720x1280",
        "1280x544",
        "544x1280",
        "1104x832",
        "832x1104",
        "960x960",
    ]:
        res = "720p"
    elif resolution in [
        "960x544",
        "544x960",
    ]:
        res = "540p"
    else:
        res = "480p"

    if model_size == "14b":
        is_14b = True
    else:
        is_14b = False

    if res == "720p" and is_14b:
        gpu_rules = [
            (80, {}),
            (48, {"cpu_offload_val": True, "offload_ratio_val": 0.5, "t5_cpu_offload_val": True}),
            (40, {"cpu_offload_val": True, "offload_ratio_val": 0.8, "t5_cpu_offload_val": True}),
            (32, {"cpu_offload_val": True, "offload_ratio_val": 1, "t5_cpu_offload_val": True}),
            (
                24,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "offload_ratio_val": 1,
                    "t5_offload_granularity_val": "block",
                    "precision_mode_val": "bf16",
                    "use_tiling_vae_val": True,
                },
            ),
            (
                16,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "offload_ratio_val": 1,
                    "t5_offload_granularity_val": "block",
                    "precision_mode_val": "bf16",
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                    "rotary_chunk_val": True,
                    "rotary_chunk_size_val": 100,
                },
            ),
            (
                12,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "offload_ratio_val": 1,
                    "t5_offload_granularity_val": "block",
                    "precision_mode_val": "bf16",
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                    "rotary_chunk_val": True,
                    "rotary_chunk_size_val": 100,
                    "clean_cuda_cache_val": True,
                    "use_tiny_vae_val": True,
                },
            ),
            (
                8,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "offload_ratio_val": 1,
                    "t5_offload_granularity_val": "block",
                    "precision_mode_val": "bf16",
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                    "rotary_chunk_val": True,
                    "rotary_chunk_size_val": 100,
                    "clean_cuda_cache_val": True,
                    "t5_quant_scheme_val": quant_type,
                    "clip_quant_scheme_val": quant_type,
                    "dit_quant_scheme_val": quant_type,
                    "lazy_load_val": True,
                    "unload_modules_val": True,
                    "use_tiny_vae_val": True,
                },
            ),
        ]

    elif is_14b:
        gpu_rules = [
            (80, {}),
            (48, {"cpu_offload_val": True, "offload_ratio_val": 0.2, "t5_cpu_offload_val": True}),
            (40, {"cpu_offload_val": True, "offload_ratio_val": 0.5, "t5_cpu_offload_val": True}),
            (24, {"cpu_offload_val": True, "offload_ratio_val": 0.8, "t5_cpu_offload_val": True}),
            (
                16,
                {
                    "cpu_offload_val": True,
                    "t5_cpu_offload_val": True,
                    "offload_ratio_val": 1,
                    "t5_offload_granularity_val": "block",
                    "precision_mode_val": "bf16",
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "block",
                },
            ),
            (
                8,
                (
                    {
                        "cpu_offload_val": True,
                        "t5_cpu_offload_val": True,
                        "offload_ratio_val": 1,
                        "t5_offload_granularity_val": "block",
                        "precision_mode_val": "bf16",
                        "use_tiling_vae_val": True,
                        "offload_granularity_val": "phase",
                        "t5_quant_scheme_val": quant_type,
                        "clip_quant_scheme_val": quant_type,
                        "dit_quant_scheme_val": quant_type,
                        "lazy_load_val": True,
                        "unload_modules_val": True,
                        "rotary_chunk_val": True,
                        "rotary_chunk_size_val": 10000,
                        "use_tiny_vae_val": True,
                    }
                    if res == "540p"
                    else {
                        "cpu_offload_val": True,
                        "t5_cpu_offload_val": True,
                        "offload_ratio_val": 1,
                        "t5_offload_granularity_val": "block",
                        "precision_mode_val": "bf16",
                        "use_tiling_vae_val": True,
                        "offload_granularity_val": "phase",
                        "t5_quant_scheme_val": quant_type,
                        "clip_quant_scheme_val": quant_type,
                        "dit_quant_scheme_val": quant_type,
                        "lazy_load_val": True,
                        "unload_modules_val": True,
                        "use_tiny_vae_val": True,
                    }
                ),
            ),
        ]

    else:
        gpu_rules = [
            (24, {}),
            (
                8,
                {
                    "t5_cpu_offload_val": True,
                    "t5_offload_granularity_val": "block",
                    "t5_quant_scheme_val": quant_type,
                },
            ),
        ]

    if is_14b:
        cpu_rules = [
            (128, {}),
            (64, {"dit_quant_scheme_val": quant_type}),
            (32, {"dit_quant_scheme_val": quant_type, "lazy_load_val": True}),
            (
                16,
                {
                    "dit_quant_scheme_val": quant_type,
                    "t5_quant_scheme_val": quant_type,
                    "clip_quant_scheme_val": quant_type,
                    "lazy_load_val": True,
                    "unload_modules_val": True,
                },
            ),
        ]
    else:
        cpu_rules = [
            (64, {}),
            (
                16,
                {
                    "t5_quant_scheme_val": quant_type,
                    "unload_modules_val": True,
                    "use_tiny_vae_val": True,
                },
            ),
        ]

    for threshold, updates in gpu_rules:
        if gpu_memory >= threshold:
            default_config.update(updates)
            break

    for threshold, updates in cpu_rules:
        if cpu_memory >= threshold:
            default_config.update(updates)
            break

    return tuple(gr.update(value=default_config[key]) for key in default_config)


def main():
    def toggle_image_input(task):
        return gr.update(visible=(task == "Image to Video"))

    with gr.Blocks(
        title="Lightx2v (Lightweight Video Inference and Generation Engine)",
        css="""
        .main-content { max-width: 1400px; margin: auto; }
        .output-video { max-height: 650px; }
        .warning { color: #ff6b6b; font-weight: bold; }
        .advanced-options { background: #f9f9ff; border-radius: 10px; padding: 15px; }
        .tab-button { font-size: 16px; padding: 10px 20px; }
    """,
    ) as demo:
        gr.Markdown(f"# 🎬 {model_cls} Video Generator")
        gr.Markdown(f"### Using Model: {model_path}")

        with gr.Tabs() as tabs:
            with gr.Tab("Basic Settings", id=1):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Group():
                            gr.Markdown("## 📥 Input Parameters")

                            if task == "i2v":
                                with gr.Row():
                                    image_path = gr.Image(
                                        label="Input Image",
                                        type="filepath",
                                        height=300,
                                        interactive=True,
                                        visible=True,
                                    )

                            with gr.Row():
                                with gr.Column():
                                    prompt = gr.Textbox(
                                        label="Prompt",
                                        lines=3,
                                        placeholder="Describe the video content...",
                                        max_lines=5,
                                    )
                                with gr.Column():
                                    negative_prompt = gr.Textbox(
                                        label="Negative Prompt",
                                        lines=3,
                                        placeholder="What you don't want to appear in the video...",
                                        max_lines=5,
                                        value="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                                    )
                                with gr.Column():
                                    resolution = gr.Dropdown(
                                        choices=[
                                            # 720p
                                            ("1280x720 (16:9, 720p)", "1280x720"),
                                            ("720x1280 (9:16, 720p)", "720x1280"),
                                            ("1280x544 (21:9, 720p)", "1280x544"),
                                            ("544x1280 (9:21, 720p)", "544x1280"),
                                            ("1104x832 (4:3, 720p)", "1104x832"),
                                            ("832x1104 (3:4, 720p)", "832x1104"),
                                            ("960x960 (1:1, 720p)", "960x960"),
                                            # 480p
                                            ("960x544 (16:9, 540p)", "960x544"),
                                            ("544x960 (9:16, 540p)", "544x960"),
                                            ("832x480 (16:9, 480p)", "832x480"),
                                            ("480x832 (9:16, 480p)", "480x832"),
                                            ("832x624 (4:3, 480p)", "832x624"),
                                            ("624x832 (3:4, 480p)", "624x832"),
                                            ("720x720 (1:1, 480p)", "720x720"),
                                            ("512x512 (1:1, 480p)", "512x512"),
                                        ],
                                        value="832x480",
                                        label="Maximum Resolution",
                                    )

                                with gr.Column():
                                    enable_auto_config = gr.Checkbox(
                                        label="Auto-configure Inference Options",
                                        value=False,
                                        info="Automatically optimize GPU settings to match the current resolution. After changing the resolution, please re-check this option to prevent potential performance degradation or runtime errors.",
                                    )
                                with gr.Column(scale=9):
                                    seed = gr.Slider(
                                        label="Random Seed",
                                        minimum=0,
                                        maximum=MAX_NUMPY_SEED,
                                        step=1,
                                        value=generate_random_seed(),
                                    )
                                with gr.Column(scale=1):
                                    randomize_btn = gr.Button("🎲 Randomize", variant="secondary")

                                randomize_btn.click(fn=generate_random_seed, inputs=None, outputs=seed)

                                with gr.Column():
                                    infer_steps = gr.Slider(
                                        label="Inference Steps",
                                        minimum=1,
                                        maximum=100,
                                        step=1,
                                        value=40,
                                        info="Number of inference steps for video generation. Increasing steps may improve quality but reduce speed.",
                                    )

                            enable_cfg = gr.Checkbox(
                                label="Enable Classifier-Free Guidance",
                                value=True,
                                info="Enable classifier-free guidance to control prompt strength",
                            )
                            cfg_scale = gr.Slider(
                                label="CFG Scale Factor",
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=5,
                                info="Controls the influence strength of the prompt. Higher values give more influence to the prompt.",
                            )
                            sample_shift = gr.Slider(
                                label="Distribution Shift",
                                value=5,
                                minimum=0,
                                maximum=10,
                                step=1,
                                info="Controls the degree of distribution shift for samples. Larger values indicate more significant shifts.",
                            )

                            fps = gr.Slider(
                                label="Frames Per Second (FPS)",
                                minimum=8,
                                maximum=30,
                                step=1,
                                value=16,
                                info="Frames per second of the video. Higher FPS results in smoother videos.",
                            )
                            num_frames = gr.Slider(
                                label="Total Frames",
                                minimum=16,
                                maximum=120,
                                step=1,
                                value=81,
                                info="Total number of frames in the video. More frames result in longer videos.",
                            )

                        save_video_path = gr.Textbox(
                            label="Output Video Path",
                            value=generate_unique_filename(),
                            info="Must include .mp4 extension. If left blank or using the default value, a unique filename will be automatically generated.",
                        )
                    with gr.Column(scale=6):
                        gr.Markdown("## 📤 Generated Video")
                        output_video = gr.Video(
                            label="Result",
                            height=624,
                            width=360,
                            autoplay=True,
                            elem_classes=["output-video"],
                        )

                        infer_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Tab("⚙️ Advanced Options", id=2):
                with gr.Group(elem_classes="advanced-options"):
                    gr.Markdown("### GPU Memory Optimization")
                    with gr.Row():
                        rotary_chunk = gr.Checkbox(
                            label="Chunked Rotary Position Embedding",
                            value=False,
                            info="When enabled, processes rotary position embeddings in chunks to save GPU memory.",
                        )

                        rotary_chunk_size = gr.Slider(
                            label="Rotary Embedding Chunk Size",
                            value=100,
                            minimum=100,
                            maximum=10000,
                            step=100,
                            info="Controls the chunk size for applying rotary embeddings. Larger values may improve performance but increase memory usage. Only effective if 'rotary_chunk' is checked.",
                        )

                        unload_modules = gr.Checkbox(
                            label="Unload Modules",
                            value=False,
                            info="Unload modules (T5, CLIP, DIT, etc.) after inference to reduce GPU/CPU memory usage",
                        )
                        clean_cuda_cache = gr.Checkbox(
                            label="Clean CUDA Memory Cache",
                            value=False,
                            info="When enabled, frees up GPU memory promptly but slows down inference.",
                        )

                    gr.Markdown("### Asynchronous Offloading")
                    with gr.Row():
                        cpu_offload = gr.Checkbox(
                            label="CPU Offloading",
                            value=False,
                            info="Offload parts of the model computation from GPU to CPU to reduce GPU memory usage",
                        )

                        lazy_load = gr.Checkbox(
                            label="Enable Lazy Loading",
                            value=False,
                            info="Lazy load model components during inference. Requires CPU loading and DIT quantization.",
                        )

                        offload_granularity = gr.Dropdown(
                            label="Dit Offload Granularity",
                            choices=["block", "phase"],
                            value="phase",
                            info="Sets Dit model offloading granularity: blocks or computational phases",
                        )
                        offload_ratio = gr.Slider(
                            label="Offload ratio for Dit model",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=1.0,
                            info="Controls how much of the Dit model is offloaded to the CPU",
                        )
                        t5_cpu_offload = gr.Checkbox(
                            label="T5 CPU Offloading",
                            value=False,
                            info="Offload the T5 Encoder model to CPU to reduce GPU memory usage",
                        )

                        t5_offload_granularity = gr.Dropdown(
                            label="T5 Encoder Offload Granularity",
                            choices=["model", "block"],
                            value="model",
                            info="Controls the granularity when offloading the T5 Encoder model to CPU",
                        )

                    gr.Markdown("### Low-Precision Quantization")
                    with gr.Row():
                        torch_compile = gr.Checkbox(
                            label="Torch Compile",
                            value=False,
                            info="Use torch.compile to accelerate the inference process",
                        )

                        attention_type = gr.Dropdown(
                            label="Attention Operator",
                            choices=[op[1] for op in attn_op_choices],
                            value=attn_op_choices[0][1],
                            info="Use appropriate attention operators to accelerate inference",
                        )
                        quant_op = gr.Dropdown(
                            label="Quantization Matmul Operator",
                            choices=[op[1] for op in quant_op_choices],
                            value=quant_op_choices[0][1],
                            info="Select the quantization matrix multiplication operator to accelerate inference",
                            interactive=True,
                        )
                        dit_quant_scheme = gr.Dropdown(
                            label="Dit",
                            choices=["fp8", "int8", "bf16"],
                            value="bf16",
                            info="Quantization precision for the Dit model",
                        )
                        t5_quant_scheme = gr.Dropdown(
                            label="T5 Encoder",
                            choices=["fp8", "int8", "bf16"],
                            value="bf16",
                            info="Quantization precision for the T5 Encoder model",
                        )
                        clip_quant_scheme = gr.Dropdown(
                            label="Clip Encoder",
                            choices=["fp8", "int8", "fp16"],
                            value="fp16",
                            info="Quantization precision for the Clip Encoder",
                        )
                        precision_mode = gr.Dropdown(
                            label="Precision Mode for Sensitive Layers",
                            choices=["fp32", "bf16"],
                            value="fp32",
                            info="Select the numerical precision for critical model components like normalization and embedding layers. FP32 offers higher accuracy, while BF16 improves performance on compatible hardware.",
                        )

                    gr.Markdown("### Variational Autoencoder (VAE)")
                    with gr.Row():
                        use_tiny_vae = gr.Checkbox(
                            label="Use Tiny VAE",
                            value=False,
                            info="Use a lightweight VAE model to accelerate the decoding process",
                        )
                        use_tiling_vae = gr.Checkbox(
                            label="VAE Tiling Inference",
                            value=False,
                            info="Use VAE tiling inference to reduce GPU memory usage",
                        )

                    gr.Markdown("### Feature Caching")
                    with gr.Row():
                        enable_teacache = gr.Checkbox(
                            label="Tea Cache",
                            value=False,
                            info="Cache features during inference to reduce the number of inference steps",
                        )
                        teacache_thresh = gr.Slider(
                            label="Tea Cache Threshold",
                            value=0.26,
                            minimum=0,
                            maximum=1,
                            info="Higher acceleration may result in lower quality —— Setting to 0.1 provides ~2.0x acceleration, setting to 0.2 provides ~3.0x acceleration",
                        )
                        use_ret_steps = gr.Checkbox(
                            label="Cache Only Key Steps",
                            value=False,
                            info="When checked, cache is written only at key steps where the scheduler returns results; when unchecked, cache is written at all steps to ensure the highest quality",
                        )

                enable_auto_config.change(
                    fn=auto_configure,
                    inputs=[enable_auto_config, resolution],
                    outputs=[
                        torch_compile,
                        lazy_load,
                        rotary_chunk,
                        rotary_chunk_size,
                        clean_cuda_cache,
                        cpu_offload,
                        offload_granularity,
                        offload_ratio,
                        t5_cpu_offload,
                        unload_modules,
                        t5_offload_granularity,
                        attention_type,
                        quant_op,
                        dit_quant_scheme,
                        t5_quant_scheme,
                        clip_quant_scheme,
                        precision_mode,
                        use_tiny_vae,
                        use_tiling_vae,
                        enable_teacache,
                        teacache_thresh,
                        use_ret_steps,
                    ],
                )

                lazy_load.change(
                    fn=handle_lazy_load_change,
                    inputs=[lazy_load],
                    outputs=[unload_modules],
                )
        if task == "i2v":
            infer_btn.click(
                fn=run_inference,
                inputs=[
                    prompt,
                    negative_prompt,
                    save_video_path,
                    torch_compile,
                    infer_steps,
                    num_frames,
                    resolution,
                    seed,
                    sample_shift,
                    enable_teacache,
                    teacache_thresh,
                    use_ret_steps,
                    enable_cfg,
                    cfg_scale,
                    dit_quant_scheme,
                    t5_quant_scheme,
                    clip_quant_scheme,
                    fps,
                    use_tiny_vae,
                    use_tiling_vae,
                    lazy_load,
                    precision_mode,
                    cpu_offload,
                    offload_granularity,
                    offload_ratio,
                    t5_cpu_offload,
                    unload_modules,
                    t5_offload_granularity,
                    attention_type,
                    quant_op,
                    rotary_chunk,
                    rotary_chunk_size,
                    clean_cuda_cache,
                    image_path,
                ],
                outputs=output_video,
            )
        else:
            infer_btn.click(
                fn=run_inference,
                inputs=[
                    prompt,
                    negative_prompt,
                    save_video_path,
                    torch_compile,
                    infer_steps,
                    num_frames,
                    resolution,
                    seed,
                    sample_shift,
                    enable_teacache,
                    teacache_thresh,
                    use_ret_steps,
                    enable_cfg,
                    cfg_scale,
                    dit_quant_scheme,
                    t5_quant_scheme,
                    clip_quant_scheme,
                    fps,
                    use_tiny_vae,
                    use_tiling_vae,
                    lazy_load,
                    precision_mode,
                    cpu_offload,
                    offload_granularity,
                    offload_ratio,
                    t5_cpu_offload,
                    unload_modules,
                    t5_offload_granularity,
                    attention_type,
                    quant_op,
                    rotary_chunk,
                    rotary_chunk_size,
                    clean_cuda_cache,
                ],
                outputs=output_video,
            )

    demo.launch(share=True, server_port=args.server_port, server_name=args.server_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Light Video Generation")
    parser.add_argument("--model_path", type=str, required=True, help="Model folder path")
    parser.add_argument(
        "--model_cls",
        type=str,
        choices=["wan2.1"],
        default="wan2.1",
        help="Model class to use",
    )
    parser.add_argument("--model_size", type=str, required=True, choices=["14b", "1.3b"], help="Model type to use")
    parser.add_argument("--task", type=str, required=True, choices=["i2v", "t2v"], help="Specify the task type. 'i2v' for image-to-video translation, 't2v' for text-to-video generation.")
    parser.add_argument("--server_port", type=int, default=7862, help="Server port")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server ip")
    args = parser.parse_args()

    global model_path, model_cls, model_size
    model_path = args.model_path
    model_cls = args.model_cls
    model_size = args.model_size
    task = args.task

    main()
