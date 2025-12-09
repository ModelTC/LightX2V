import argparse
import gc
import glob
import importlib.util
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DTYPE"] = "BF16"
import random
from datetime import datetime

import gradio as gr
import psutil
import torch
from loguru import logger

from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.set_config import get_default_config

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
except ImportError:
    apply_rope_with_cos_sin_cache_inplace = None


logger.add(
    "inference_logs.log",
    rotation="100 MB",
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

MAX_NUMPY_SEED = 2**32 - 1


def scan_model_path_contents(model_path):
    """扫描 model_path 目录，返回可用的文件和子目录"""
    if not model_path or not os.path.exists(model_path):
        return {"dirs": [], "files": [], "safetensors_dirs": [], "pth_files": []}

    dirs = []
    files = []
    safetensors_dirs = []
    pth_files = []

    try:
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if os.path.isdir(item_path):
                dirs.append(item)
                # 检查目录是否包含 safetensors 文件
                if glob.glob(os.path.join(item_path, "*.safetensors")):
                    safetensors_dirs.append(item)
            elif os.path.isfile(item_path):
                files.append(item)
                if item.endswith(".pth"):
                    pth_files.append(item)
    except Exception as e:
        logger.warning(f"扫描目录失败: {e}")

    return {
        "dirs": sorted(dirs),
        "files": sorted(files),
        "safetensors_dirs": sorted(safetensors_dirs),
        "pth_files": sorted(pth_files),
    }


def get_dit_choices(model_path, model_type="wan2.1"):
    """获取 Diffusion 模型可选项（根据模型类型筛选）"""
    contents = scan_model_path_contents(model_path)
    excluded_keywords = ["vae", "tae", "clip", "t5", "high_noise", "low_noise"]
    fp8_supported = is_fp8_supported_gpu()

    if model_type == "wan2.1":
        # wan2.1: 筛选包含 wan2.1 或 Wan2.1 的文件/目录
        def is_valid(name):
            name_lower = name.lower()
            if "wan2.1" not in name_lower:
                return False
            if not fp8_supported and "fp8" in name_lower:
                return False
            return not any(kw in name_lower for kw in excluded_keywords)
    else:
        # wan2.2: 筛选包含 wan2.2 或 Wan2.2 的文件/目录
        def is_valid(name):
            name_lower = name.lower()
            if "wan2.2" not in name_lower:
                return False
            if not fp8_supported and "fp8" in name_lower:
                return False
            return not any(kw in name_lower for kw in excluded_keywords)

    # 筛选符合条件的目录和文件
    dir_choices = [d for d in contents["dirs"] if is_valid(d)]
    file_choices = [f for f in contents["files"] if is_valid(f)]
    choices = dir_choices + file_choices
    return choices if choices else [""]


def get_high_noise_choices(model_path):
    """获取高噪模型可选项（包含 high_noise 的文件/目录）"""
    contents = scan_model_path_contents(model_path)
    fp8_supported = is_fp8_supported_gpu()

    def is_valid(name):
        name_lower = name.lower()
        if not fp8_supported and "fp8" in name_lower:
            return False
        return "high_noise" in name_lower or "high-noise" in name_lower

    dir_choices = [d for d in contents["dirs"] if is_valid(d)]
    file_choices = [f for f in contents["files"] if is_valid(f)]
    choices = dir_choices + file_choices
    return choices if choices else [""]


def get_low_noise_choices(model_path):
    """获取低噪模型可选项（包含 low_noise 的文件/目录）"""
    contents = scan_model_path_contents(model_path)
    fp8_supported = is_fp8_supported_gpu()

    def is_valid(name):
        name_lower = name.lower()
        if not fp8_supported and "fp8" in name_lower:
            return False
        return "low_noise" in name_lower or "low-noise" in name_lower

    dir_choices = [d for d in contents["dirs"] if is_valid(d)]
    file_choices = [f for f in contents["files"] if is_valid(f)]
    choices = dir_choices + file_choices
    return choices if choices else [""]


def get_t5_choices(model_path):
    """获取 T5 模型可选项（.pth 或 .safetensors 文件，包含 t5 关键字）"""
    contents = scan_model_path_contents(model_path)
    fp8_supported = is_fp8_supported_gpu()

    # 从 .pth 文件中筛选
    pth_choices = [f for f in contents["pth_files"] if "t5" in f.lower() and (fp8_supported or "fp8" not in f.lower())]

    # 从 .safetensors 文件中筛选
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and "t5" in f.lower() and (fp8_supported or "fp8" not in f.lower())]

    # 从包含 safetensors 的目录中筛选
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if "t5" in d.lower() and (fp8_supported or "fp8" not in d.lower())]

    choices = pth_choices + safetensors_choices + safetensors_dir_choices
    return choices if choices else [""]


def get_clip_choices(model_path):
    """获取 CLIP 模型可选项（.pth 或 .safetensors 文件，包含 clip 关键字）"""
    contents = scan_model_path_contents(model_path)
    fp8_supported = is_fp8_supported_gpu()

    # 从 .pth 文件中筛选
    pth_choices = [f for f in contents["pth_files"] if "clip" in f.lower() and (fp8_supported or "fp8" not in f.lower())]

    # 从 .safetensors 文件中筛选
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and "clip" in f.lower() and (fp8_supported or "fp8" not in f.lower())]

    # 从包含 safetensors 的目录中筛选
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if "clip" in d.lower() and (fp8_supported or "fp8" not in d.lower())]

    choices = pth_choices + safetensors_choices + safetensors_dir_choices
    return choices if choices else [""]


def get_vae_choices(model_path):
    """获取 VAE 模型可选项（.pth 或 .safetensors 文件，包含 vae/VAE/tae 关键字）"""
    contents = scan_model_path_contents(model_path)
    fp8_supported = is_fp8_supported_gpu()

    # 从 .pth 文件中筛选
    pth_choices = [f for f in contents["pth_files"] if any(kw in f.lower() for kw in ["vae", "tae"]) and (fp8_supported or "fp8" not in f.lower())]

    # 从 .safetensors 文件中筛选
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and any(kw in f.lower() for kw in ["vae", "tae"]) and (fp8_supported or "fp8" not in f.lower())]

    # 从包含 safetensors 的目录中筛选
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if any(kw in d.lower() for kw in ["vae", "tae"]) and (fp8_supported or "fp8" not in d.lower())]

    choices = pth_choices + safetensors_choices + safetensors_dir_choices
    return choices if choices else [""]


def detect_quant_scheme(model_name):
    """根据模型名字自动检测量化精度
    - 如果模型名字包含 "int8" → "int8"
    - 如果模型名字包含 "fp8" 且设备支持 → "fp8"
    - 否则返回 None（表示不使用量化）
    """
    if not model_name:
        return None
    name_lower = model_name.lower()
    if "int8" in name_lower:
        return "int8"
    elif "fp8" in name_lower:
        if is_fp8_supported_gpu():
            return "fp8"
        else:
            # 设备不支持fp8，返回None（使用默认精度）
            return None
    return None


def update_model_path_options(model_path, model_type="wan2.1"):
    """当 model_path 或 model_type 改变时，更新所有模型路径选择器"""
    dit_choices = get_dit_choices(model_path, model_type)
    high_noise_choices = get_high_noise_choices(model_path)
    low_noise_choices = get_low_noise_choices(model_path)
    t5_choices = get_t5_choices(model_path)
    clip_choices = get_clip_choices(model_path)
    vae_choices = get_vae_choices(model_path)

    return (
        gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else ""),
        gr.update(choices=high_noise_choices, value=high_noise_choices[0] if high_noise_choices else ""),
        gr.update(choices=low_noise_choices, value=low_noise_choices[0] if low_noise_choices else ""),
        gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),
        gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else ""),
        gr.update(choices=vae_choices, value=vae_choices[0] if vae_choices else ""),
    )


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

    sage_installed = is_module_installed("sageattention")
    if sage_installed:
        available_ops.append(("sage_attn2", True))
    else:
        available_ops.append(("sage_attn2", False))

    sage3_installed = is_module_installed("sageattn3")
    if sage3_installed:
        available_ops.append(("sage_attn3", True))
    else:
        available_ops.append(("sage_attn3", False))

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
        logger.warning(f"获取GPU内存失败: {e}")
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
        import psutil

        if hasattr(psutil, "virtual_memory"):
            if os.name == "posix":
                try:
                    os.system("sync")
                except:  # noqa
                    pass
    except:  # noqa
        pass


def generate_unique_filename(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{timestamp}.mp4")


def is_fp8_supported_gpu():
    if not torch.cuda.is_available():
        return False
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    return (major == 8 and minor == 9) or (major >= 9)


def is_ada_architecture_gpu():
    if not torch.cuda.is_available():
        return False
    try:
        gpu_name = torch.cuda.get_device_name(0).upper()
        ada_keywords = ["RTX 40", "RTX40", "4090", "4080", "4070", "4060"]
        return any(keyword in gpu_name for keyword in ada_keywords)
    except Exception as e:
        logger.warning(f"Failed to get GPU name: {e}")
        return False


def get_quantization_options(model_path):
    """根据model_path动态获取量化选项"""
    import os

    # 检查子目录
    subdirs = ["original", "fp8", "int8"]
    has_subdirs = {subdir: os.path.exists(os.path.join(model_path, subdir)) for subdir in subdirs}

    # 检查根目录下的原始文件
    t5_bf16_exists = os.path.exists(os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth"))
    clip_fp16_exists = os.path.exists(os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))

    # 生成选项
    def get_choices(has_subdirs, original_type, fp8_type, int8_type, fallback_type, has_original_file=False):
        choices = []
        if has_subdirs["original"]:
            choices.append(original_type)
        if has_subdirs["fp8"]:
            choices.append(fp8_type)
        if has_subdirs["int8"]:
            choices.append(int8_type)

        # 如果没有子目录但有原始文件，添加原始类型
        if has_original_file:
            if not choices or "original" not in choices:
                choices.append(original_type)

        # 如果没有任何选项，使用默认值
        if not choices:
            choices = [fallback_type]

        return choices, choices[0]

    # DIT选项
    dit_choices, dit_default = get_choices(has_subdirs, "bf16", "fp8", "int8", "bf16")

    # T5选项 - 检查是否有原始文件
    t5_choices, t5_default = get_choices(has_subdirs, "bf16", "fp8", "int8", "bf16", t5_bf16_exists)

    # CLIP选项 - 检查是否有原始文件
    clip_choices, clip_default = get_choices(has_subdirs, "fp16", "fp8", "int8", "fp16", clip_fp16_exists)

    return {"dit_choices": dit_choices, "dit_default": dit_default, "t5_choices": t5_choices, "t5_default": t5_default, "clip_choices": clip_choices, "clip_default": clip_default}


def determine_model_cls(model_type, dit_name, high_noise_name):
    """根据模型类型和文件名确定 model_cls"""
    # 确定要检查的文件名
    if model_type == "wan2.1":
        check_name = dit_name.lower() if dit_name else ""
        is_distill = "4step" in check_name
        return "wan2.1_distill" if is_distill else "wan2.1"
    else:
        # wan2.2
        check_name = high_noise_name.lower() if high_noise_name else ""
        is_distill = "4step" in check_name
        return "wan2.2_moe_distill" if is_distill else "wan2.2_moe"


global_runner = None
current_config = None
cur_dit_path = None
cur_t5_path = None
cur_clip_path = None

available_quant_ops = get_available_quant_ops()
quant_op_choices = []
for op_name, is_installed in available_quant_ops:
    status_text = "✅ 已安装" if is_installed else "❌ 未安装"
    display_text = f"{op_name} ({status_text})"
    quant_op_choices.append((op_name, display_text))

available_attn_ops = get_available_attn_ops()
# 优先级顺序
attn_priority = ["sage_attn2", "flash_attn3", "flash_attn2", "torch_sdpa"]
# 按优先级排序，已安装的在前，未安装的在后
attn_op_choices = []
attn_op_dict = dict(available_attn_ops)

# 先添加已安装的（按优先级）
for op_name in attn_priority:
    if op_name in attn_op_dict and attn_op_dict[op_name]:
        status_text = "✅ 已安装"
        display_text = f"{op_name} ({status_text})"
        attn_op_choices.append((op_name, display_text))

# 再添加未安装的（按优先级）
for op_name in attn_priority:
    if op_name in attn_op_dict and not attn_op_dict[op_name]:
        status_text = "❌ 未安装"
        display_text = f"{op_name} ({status_text})"
        attn_op_choices.append((op_name, display_text))

# 添加其他不在优先级列表中的算子（已安装的在前）
other_ops = [(op_name, is_installed) for op_name, is_installed in available_attn_ops if op_name not in attn_priority]
for op_name, is_installed in sorted(other_ops, key=lambda x: not x[1]):  # 已安装的在前
    status_text = "✅ 已安装" if is_installed else "❌ 未安装"
    display_text = f"{op_name} ({status_text})"
    attn_op_choices.append((op_name, display_text))


def run_inference(
    prompt,
    negative_prompt,
    save_result_path,
    infer_steps,
    num_frames,
    resolution,
    seed,
    sample_shift,
    enable_cfg,
    cfg_scale,
    fps,
    use_tiling_vae,
    lazy_load,
    cpu_offload,
    offload_granularity,
    t5_cpu_offload,
    clip_cpu_offload,
    vae_cpu_offload,
    unload_modules,
    attention_type,
    quant_op,
    rope_chunk,
    rope_chunk_size,
    clean_cuda_cache,
    model_path_input,
    model_type_input,
    task_type_input,
    dit_path_input,
    high_noise_path_input,
    low_noise_path_input,
    t5_path_input,
    clip_path_input,
    vae_path_input,
    image_path=None,
):
    cleanup_memory()

    quant_op = quant_op.split("(")[0].strip()
    attention_type = attention_type.split("(")[0].strip()

    global global_runner, current_config, model_path, model_cls
    global cur_dit_path, cur_t5_path, cur_clip_path

    task = task_type_input
    model_cls = determine_model_cls(model_type_input, dit_path_input, high_noise_path_input)
    logger.info(f"自动确定 model_cls: {model_cls} (模型类型: {model_type_input})")

    if model_type_input == "wan2.1":
        dit_quant_detected = detect_quant_scheme(dit_path_input)
    else:
        dit_quant_detected = detect_quant_scheme(high_noise_path_input)
    t5_quant_detected = detect_quant_scheme(t5_path_input)
    clip_quant_detected = detect_quant_scheme(clip_path_input)
    logger.info(f"自动检测量化精度 - DIT: {dit_quant_detected}, T5: {t5_quant_detected}, CLIP: {clip_quant_detected}")

    if model_path_input and model_path_input.strip():
        model_path = model_path_input.strip()

    if os.path.exists(os.path.join(model_path, "config.json")):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            model_config = json.load(f)
    else:
        model_config = {}

    save_result_path = generate_unique_filename(output_dir)

    is_dit_quant = dit_quant_detected != "bf16"
    is_t5_quant = t5_quant_detected != "bf16"
    is_clip_quant = clip_quant_detected != "fp16"

    dit_quantized_ckpt = None
    dit_original_ckpt = None
    high_noise_quantized_ckpt = None
    low_noise_quantized_ckpt = None
    high_noise_original_ckpt = None
    low_noise_original_ckpt = None

    if is_dit_quant:
        dit_quant_scheme = f"{dit_quant_detected}-{quant_op}"
        if "wan2.1" in model_cls:
            dit_quantized_ckpt = os.path.join(model_path, dit_path_input)
        else:
            high_noise_quantized_ckpt = os.path.join(model_path, high_noise_path_input)
            low_noise_quantized_ckpt = os.path.join(model_path, low_noise_path_input)
    else:
        dit_quantized_ckpt = "Default"
        if "wan2.1" in model_cls:
            dit_original_ckpt = os.path.join(model_path, dit_path_input)
        else:
            high_noise_original_ckpt = os.path.join(model_path, high_noise_path_input)
            low_noise_original_ckpt = os.path.join(model_path, low_noise_path_input)

    # 使用前端选择的 T5 路径
    if is_t5_quant:
        t5_quantized_ckpt = os.path.join(model_path, t5_path_input)
        t5_quant_scheme = f"{t5_quant_detected}-{quant_op}"
        t5_original_ckpt = None
    else:
        t5_quantized_ckpt = None
        t5_quant_scheme = None
        t5_original_ckpt = os.path.join(model_path, t5_path_input)

    # 使用前端选择的 CLIP 路径
    if is_clip_quant:
        clip_quantized_ckpt = os.path.join(model_path, clip_path_input)
        clip_quant_scheme = f"{clip_quant_detected}-{quant_op}"
        clip_original_ckpt = None
    else:
        clip_quantized_ckpt = None
        clip_quant_scheme = None
        clip_original_ckpt = os.path.join(model_path, clip_path_input)

    if model_type_input == "wan2.1":
        current_dit_path = dit_path_input
    else:
        current_dit_path = f"{high_noise_path_input}|{low_noise_path_input}" if high_noise_path_input and low_noise_path_input else None

    current_t5_path = t5_path_input
    current_clip_path = clip_path_input

    needs_reinit = (
        lazy_load
        or unload_modules
        or global_runner is None
        or current_config is None
        or cur_dit_path is None
        or cur_dit_path != current_dit_path
        or cur_t5_path is None
        or cur_t5_path != current_t5_path
        or cur_clip_path is None
        or cur_clip_path != current_clip_path
    )

    if cfg_scale == 1:
        enable_cfg = False
    else:
        enable_cfg = True

    vae_name_lower = vae_path_input.lower() if vae_path_input else ""
    use_tae = "tae" in vae_name_lower or "lighttae" in vae_name_lower
    use_lightvae = "lightvae" in vae_name_lower
    need_scaled = "lighttae" in vae_name_lower

    logger.info(f"VAE 配置 - use_tae: {use_tae}, use_lightvae: {use_lightvae}, need_scaled: {need_scaled} (VAE: {vae_path_input})")

    config_graio = {
        "infer_steps": infer_steps,
        "target_video_length": num_frames,
        "target_width": int(resolution.split("x")[0]),
        "target_height": int(resolution.split("x")[1]),
        "self_attn_1_type": attention_type,
        "cross_attn_1_type": attention_type,
        "cross_attn_2_type": attention_type,
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
        "offload_granularity": "phase" if "wan2.2" in model_cls else offload_granularity,
        "t5_cpu_offload": t5_cpu_offload,
        "clip_cpu_offload": clip_cpu_offload,
        "vae_cpu_offload": vae_cpu_offload,
        "dit_quantized": is_dit_quant,
        "dit_quant_scheme": dit_quant_scheme,
        "dit_quantized_ckpt": dit_quantized_ckpt,
        "dit_original_ckpt": dit_original_ckpt,
        "high_noise_quantized_ckpt": high_noise_quantized_ckpt,
        "low_noise_quantized_ckpt": low_noise_quantized_ckpt,
        "high_noise_original_ckpt": high_noise_original_ckpt,
        "low_noise_original_ckpt": low_noise_original_ckpt,
        "t5_original_ckpt": t5_original_ckpt,
        "t5_quantized": is_t5_quant,
        "t5_quantized_ckpt": t5_quantized_ckpt,
        "t5_quant_scheme": t5_quant_scheme,
        "clip_original_ckpt": clip_original_ckpt,
        "clip_quantized": is_clip_quant,
        "clip_quantized_ckpt": clip_quantized_ckpt,
        "clip_quant_scheme": clip_quant_scheme,
        "vae_path": os.path.join(model_path, vae_path_input),
        "use_tiling_vae": use_tiling_vae,
        "use_tae": use_tae,
        "use_lightvae": use_lightvae,
        "need_scaled": need_scaled,
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
        "rope_type": "flashinfer" if apply_rope_with_cos_sin_cache_inplace else "torch",
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
    )

    config = get_default_config()
    config.update({k: v for k, v in vars(args).items()})
    config.update(model_config)
    config.update(config_graio)

    logger.info(f"使用模型: {model_path}")
    logger.info(f"推理配置:\n{json.dumps(config, indent=4, ensure_ascii=False)}")

    # Initialize or reuse the runner
    runner = global_runner
    if needs_reinit:
        if runner is not None:
            del runner
            torch.cuda.empty_cache()
            gc.collect()

        from lightx2v.infer import init_runner  # noqa

        runner = init_runner(config)
        input_info = set_input_info(args)

        current_config = config
        cur_dit_path = current_dit_path
        cur_t5_path = current_t5_path
        cur_clip_path = current_clip_path

        if not lazy_load:
            global_runner = runner
    else:
        runner.config = config

    runner.run_pipeline(input_info)
    cleanup_memory()

    return save_result_path


def handle_lazy_load_change(lazy_load_enabled):
    """Handle lazy_load checkbox change to automatically enable unload_modules"""
    return gr.update(value=lazy_load_enabled)


def auto_configure(resolution):
    """根据机器配置和分辨率自动设置推理选项"""
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
        "attention_type_val": attn_op_choices[0][1],
        "quant_op_val": quant_op_choices[0][1],
        "use_tiling_vae_val": False,
    }

    gpu_memory = round(get_gpu_memory())
    cpu_memory = round(get_cpu_memory())

    attn_priority = ["sage_attn3", "sage_attn2", "flash_attn3", "flash_attn2", "torch_sdpa"]

    if is_ada_architecture_gpu():
        quant_op_priority = ["q8f", "vllm", "sgl"]
    else:
        quant_op_priority = ["vllm", "sgl", "q8f"]

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

    if res == "720p":
        gpu_rules = [
            (80, {}),
            (40, {"cpu_offload_val": False, "t5_cpu_offload_val": True, "vae_cpu_offload_val": True, "clip_cpu_offload_val": True}),
            (32, {"cpu_offload_val": True, "t5_cpu_offload_val": False, "vae_cpu_offload_val": False, "clip_cpu_offload_val": False}),
            (
                24,
                {
                    "cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                },
            ),
            (
                16,
                {
                    "cpu_offload_val": True,
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
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                    "rope_chunk_val": True,
                    "rope_chunk_size_val": 100,
                    "clean_cuda_cache_val": True,
                },
            ),
        ]

    else:
        gpu_rules = [
            (80, {}),
            (40, {"cpu_offload_val": False, "t5_cpu_offload_val": True, "vae_cpu_offload_val": True, "clip_cpu_offload_val": True}),
            (32, {"cpu_offload_val": True, "t5_cpu_offload_val": False, "vae_cpu_offload_val": False, "clip_cpu_offload_val": False}),
            (
                24,
                {
                    "cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                },
            ),
            (
                16,
                {
                    "cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                },
            ),
            (
                8,
                {
                    "cpu_offload_val": True,
                    "use_tiling_vae_val": True,
                    "offload_granularity_val": "phase",
                },
            ),
        ]

    cpu_rules = [
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
    ]

    for threshold, updates in gpu_rules:
        if gpu_memory >= threshold:
            default_config.update(updates)
            break

    for threshold, updates in cpu_rules:
        if cpu_memory >= threshold:
            default_config.update(updates)
            break

    return (
        gr.update(value=default_config["lazy_load_val"]),
        gr.update(value=default_config["rope_chunk_val"]),
        gr.update(value=default_config["rope_chunk_size_val"]),
        gr.update(value=default_config["clean_cuda_cache_val"]),
        gr.update(value=default_config["cpu_offload_val"]),
        gr.update(value=default_config["offload_granularity_val"]),
        gr.update(value=default_config["t5_cpu_offload_val"]),
        gr.update(value=default_config["clip_cpu_offload_val"]),
        gr.update(value=default_config["vae_cpu_offload_val"]),
        gr.update(value=default_config["unload_modules_val"]),
        gr.update(value=default_config["attention_type_val"]),
        gr.update(value=default_config["quant_op_val"]),
        gr.update(value=default_config["use_tiling_vae_val"]),
    )


css = """
        .main-content { max-width: 1600px; margin: auto; padding: 20px; }
        .warning { color: #ff6b6b; font-weight: bold; }

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
    """


def main():
    with gr.Blocks(title="Lightx2v (轻量级视频推理和生成引擎)") as demo:
        gr.Markdown(f"# 🎬 LightX2V 视频生成器")
        gr.HTML(f"<style>{css}</style>")
        # 主布局：左右分栏
        with gr.Row():
            # 左侧：配置和输入区域
            with gr.Column(scale=5):
                # 模型配置区域
                with gr.Accordion("🗂️ 模型配置", open=True, elem_classes=["model-config"]):
                    # FP8 支持提示
                    if not is_fp8_supported_gpu():
                        gr.Markdown("⚠️ **您的设备不支持fp8推理**，已自动隐藏包含fp8的模型选项。")

                    # 隐藏的状态组件
                    model_path_input = gr.Textbox(value=model_path, visible=False)

                    # 模型类型 + 任务类型
                    with gr.Row():
                        model_type_input = gr.Radio(
                            label="模型类型",
                            choices=["wan2.1", "wan2.2"],
                            value="wan2.1",
                            info="wan2.2 需要分别指定高噪模型和低噪模型",
                        )
                        task_type_input = gr.Radio(
                            label="任务类型",
                            choices=["i2v", "t2v"],
                            value="i2v",
                            info="i2v: 图生视频, t2v: 文生视频",
                        )

                    # wan2.1：Diffusion模型（单独一行）
                    with gr.Row() as wan21_row:
                        dit_path_input = gr.Dropdown(
                            label="🎨 Diffusion模型",
                            choices=get_dit_choices(model_path, "wan2.1"),
                            value=get_dit_choices(model_path, "wan2.1")[0] if get_dit_choices(model_path, "wan2.1") else "",
                            allow_custom_value=True,
                            visible=True,
                        )

                    # wan2.2 专用：高噪模型 + 低噪模型（默认隐藏）
                    with gr.Row(visible=False) as wan22_row:
                        high_noise_path_input = gr.Dropdown(
                            label="🔊 高噪模型",
                            choices=get_high_noise_choices(model_path),
                            value=get_high_noise_choices(model_path)[0] if get_high_noise_choices(model_path) else "",
                            allow_custom_value=True,
                        )
                        low_noise_path_input = gr.Dropdown(
                            label="🔇 低噪模型",
                            choices=get_low_noise_choices(model_path),
                            value=get_low_noise_choices(model_path)[0] if get_low_noise_choices(model_path) else "",
                            allow_custom_value=True,
                        )

                    # 文本编码器（单独一行）
                    with gr.Row():
                        t5_path_input = gr.Dropdown(
                            label="📝 文本编码器",
                            choices=get_t5_choices(model_path),
                            value=get_t5_choices(model_path)[0] if get_t5_choices(model_path) else "",
                            allow_custom_value=True,
                        )

                    # 图像编码器 + VAE解码器
                    with gr.Row():
                        clip_path_input = gr.Dropdown(
                            label="🖼️ 图像编码器",
                            choices=get_clip_choices(model_path),
                            value=get_clip_choices(model_path)[0] if get_clip_choices(model_path) else "",
                            allow_custom_value=True,
                        )
                        vae_path_input = gr.Dropdown(
                            label="🎞️ VAE解码器",
                            choices=get_vae_choices(model_path),
                            value=get_vae_choices(model_path)[0] if get_vae_choices(model_path) else "",
                            allow_custom_value=True,
                        )

                    # 注意力算子和量化矩阵乘法算子
                    with gr.Row():
                        attention_type = gr.Dropdown(
                            label="⚡ 注意力算子",
                            choices=[op[1] for op in attn_op_choices],
                            value=attn_op_choices[0][1] if attn_op_choices else "",
                            info="使用适当的注意力算子加速推理",
                        )
                        quant_op = gr.Dropdown(
                            label="量化矩阵乘法算子",
                            choices=[op[1] for op in quant_op_choices],
                            value=quant_op_choices[0][1],
                            info="选择量化矩阵乘法算子以加速推理",
                            interactive=True,
                        )

                    # 判断模型是否是 distill 版本
                    def is_distill_model(model_type, dit_path, high_noise_path):
                        """根据模型类型和路径判断是否是 distill 版本"""
                        if model_type == "wan2.1":
                            check_name = dit_path.lower() if dit_path else ""
                        else:
                            check_name = high_noise_path.lower() if high_noise_path else ""
                        return "4step" in check_name

                    # 模型类型切换事件
                    def on_model_type_change(model_type, model_path_val):
                        if model_type == "wan2.2":
                            return gr.update(visible=False), gr.update(visible=True), gr.update()
                        else:
                            # 更新 wan2.1 的 Diffusion 模型选项
                            dit_choices = get_dit_choices(model_path_val, "wan2.1")
                            return (
                                gr.update(visible=True),
                                gr.update(visible=False),
                                gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else ""),
                            )

                    model_type_input.change(
                        fn=on_model_type_change,
                        inputs=[model_type_input, model_path_input],
                        outputs=[wan21_row, wan22_row, dit_path_input],
                    )

                # 输入参数区域
                with gr.Accordion("📥 输入参数", open=True, elem_classes=["input-params"]):
                    # 图片输入（i2v 时显示）
                    with gr.Row(visible=True) as image_input_row:
                        image_path = gr.Image(
                            label="输入图像",
                            type="filepath",
                            height=300,
                            interactive=True,
                        )

                    # 任务类型切换事件
                    def on_task_type_change(task_type):
                        return gr.update(visible=(task_type == "i2v"))

                    task_type_input.change(
                        fn=on_task_type_change,
                        inputs=[task_type_input],
                        outputs=[image_input_row],
                    )

                    with gr.Row():
                        with gr.Column():
                            prompt = gr.Textbox(
                                label="提示词",
                                lines=3,
                                placeholder="描述视频内容...",
                                max_lines=5,
                            )
                        with gr.Column():
                            negative_prompt = gr.Textbox(
                                label="负向提示词",
                                lines=3,
                                placeholder="不希望出现在视频中的内容...",
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
                                label="最大分辨率",
                            )

                        with gr.Column(scale=9):
                            seed = gr.Slider(
                                label="随机种子",
                                minimum=0,
                                maximum=MAX_NUMPY_SEED,
                                step=1,
                                value=generate_random_seed(),
                            )
                        with gr.Column():
                            default_dit = get_dit_choices(model_path, "wan2.1")[0] if get_dit_choices(model_path, "wan2.1") else ""
                            default_high_noise = get_high_noise_choices(model_path)[0] if get_high_noise_choices(model_path) else ""
                            default_is_distill = is_distill_model("wan2.1", default_dit, default_high_noise)

                            if default_is_distill:
                                infer_steps = gr.Slider(
                                    label="推理步数",
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=4,
                                    info="蒸馏模型推理步数默认为4。",
                                )
                            else:
                                infer_steps = gr.Slider(
                                    label="推理步数",
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=40,
                                    info="视频生成的推理步数。增加步数可能提高质量但降低速度。",
                                )

                            # 当模型路径改变时，动态更新推理步数
                            def update_infer_steps(model_type, dit_path, high_noise_path):
                                is_distill = is_distill_model(model_type, dit_path, high_noise_path)
                                if is_distill:
                                    return gr.update(minimum=1, maximum=100, value=4, interactive=True)
                                else:
                                    return gr.update(minimum=1, maximum=100, value=40, interactive=True)

                            # 监听模型路径变化
                            dit_path_input.change(
                                fn=lambda mt, dp, hnp: update_infer_steps(mt, dp, hnp),
                                inputs=[model_type_input, dit_path_input, high_noise_path_input],
                                outputs=[infer_steps],
                            )
                            high_noise_path_input.change(
                                fn=lambda mt, dp, hnp: update_infer_steps(mt, dp, hnp),
                                inputs=[model_type_input, dit_path_input, high_noise_path_input],
                                outputs=[infer_steps],
                            )
                            model_type_input.change(
                                fn=lambda mt, dp, hnp: update_infer_steps(mt, dp, hnp),
                                inputs=[model_type_input, dit_path_input, high_noise_path_input],
                                outputs=[infer_steps],
                            )

                    # 根据模型类别设置默认CFG
                    # CFG缩放因子：distill 时默认为 1，否则默认为 5
                    default_cfg_scale = 1 if default_is_distill else 5
                    # enable_cfg 不暴露到前端，根据 cfg_scale 自动设置
                    # 如果 cfg_scale == 1，则 enable_cfg = False，否则 enable_cfg = True
                    default_enable_cfg = False if default_cfg_scale == 1 else True
                    enable_cfg = gr.Checkbox(
                        label="启用无分类器引导",
                        value=default_enable_cfg,
                        visible=False,  # 隐藏，不暴露到前端
                    )

                    with gr.Row():
                        sample_shift = gr.Slider(
                            label="分布偏移",
                            value=5,
                            minimum=0,
                            maximum=10,
                            step=1,
                            info="控制样本分布偏移的程度。值越大表示偏移越明显。",
                        )
                        cfg_scale = gr.Slider(
                            label="CFG缩放因子",
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=default_cfg_scale,
                            info="控制提示词的影响强度。值越高，提示词的影响越大。当值为1时，自动禁用CFG。",
                        )

                    # 根据 cfg_scale 更新 enable_cfg
                    def update_enable_cfg(cfg_scale_val):
                        """根据 cfg_scale 的值自动设置 enable_cfg"""
                        if cfg_scale_val == 1:
                            return gr.update(value=False)
                        else:
                            return gr.update(value=True)

                    # 当模型路径改变时，动态更新 CFG 缩放因子和 enable_cfg
                    def update_cfg_scale(model_type, dit_path, high_noise_path):
                        is_distill = is_distill_model(model_type, dit_path, high_noise_path)
                        if is_distill:
                            new_cfg_scale = 1
                        else:
                            new_cfg_scale = 5
                        new_enable_cfg = False if new_cfg_scale == 1 else True
                        return gr.update(value=new_cfg_scale), gr.update(value=new_enable_cfg)

                    dit_path_input.change(
                        fn=lambda mt, dp, hnp: update_cfg_scale(mt, dp, hnp),
                        inputs=[model_type_input, dit_path_input, high_noise_path_input],
                        outputs=[cfg_scale, enable_cfg],
                    )
                    high_noise_path_input.change(
                        fn=lambda mt, dp, hnp: update_cfg_scale(mt, dp, hnp),
                        inputs=[model_type_input, dit_path_input, high_noise_path_input],
                        outputs=[cfg_scale, enable_cfg],
                    )
                    model_type_input.change(
                        fn=lambda mt, dp, hnp: update_cfg_scale(mt, dp, hnp),
                        inputs=[model_type_input, dit_path_input, high_noise_path_input],
                        outputs=[cfg_scale, enable_cfg],
                    )

                    cfg_scale.change(
                        fn=update_enable_cfg,
                        inputs=[cfg_scale],
                        outputs=[enable_cfg],
                    )

                    with gr.Row():
                        fps = gr.Slider(
                            label="每秒帧数(FPS)",
                            minimum=8,
                            maximum=30,
                            step=1,
                            value=16,
                            info="视频的每秒帧数。较高的FPS会产生更流畅的视频。",
                        )
                        num_frames = gr.Slider(
                            label="总帧数",
                            minimum=16,
                            maximum=120,
                            step=1,
                            value=81,
                            info="视频中的总帧数。更多帧数会产生更长的视频。",
                        )

                    save_result_path = gr.Textbox(
                        label="输出视频路径",
                        value=generate_unique_filename(output_dir),
                        info="必须包含.mp4扩展名。如果留空或使用默认值，将自动生成唯一文件名。",
                        visible=False,  # 隐藏输出路径，自动生成
                    )

            with gr.Column(scale=4):
                with gr.Accordion("📤 生成的视频", open=True, elem_classes=["output-video"]):
                    output_video = gr.Video(
                        label="",
                        height=600,
                        autoplay=True,
                        show_label=False,
                    )

                    infer_btn = gr.Button("🎬 生成视频", variant="primary", size="lg", elem_classes=["generate-btn"])

            rope_chunk = gr.Checkbox(label="分块旋转位置编码", value=False, visible=False)
            rope_chunk_size = gr.Slider(label="旋转编码块大小", value=100, minimum=100, maximum=10000, step=100, visible=False)
            unload_modules = gr.Checkbox(label="卸载模块", value=False, visible=False)
            clean_cuda_cache = gr.Checkbox(label="清理CUDA内存缓存", value=False, visible=False)
            cpu_offload = gr.Checkbox(label="CPU卸载", value=False, visible=False)
            lazy_load = gr.Checkbox(label="启用延迟加载", value=False, visible=False)
            offload_granularity = gr.Dropdown(label="Dit卸载粒度", choices=["block", "phase"], value="phase", visible=False)
            t5_cpu_offload = gr.Checkbox(label="T5 CPU卸载", value=False, visible=False)
            clip_cpu_offload = gr.Checkbox(label="CLIP CPU卸载", value=False, visible=False)
            vae_cpu_offload = gr.Checkbox(label="VAE CPU卸载", value=False, visible=False)
            use_tiling_vae = gr.Checkbox(label="VAE分块推理", value=False, visible=False)

        resolution.change(
            fn=auto_configure,
            inputs=[resolution],
            outputs=[
                lazy_load,
                rope_chunk,
                rope_chunk_size,
                clean_cuda_cache,
                cpu_offload,
                offload_granularity,
                t5_cpu_offload,
                clip_cpu_offload,
                vae_cpu_offload,
                unload_modules,
                attention_type,
                quant_op,
                use_tiling_vae,
            ],
        )

        demo.load(
            fn=lambda res: auto_configure(res),
            inputs=[resolution],
            outputs=[
                lazy_load,
                rope_chunk,
                rope_chunk_size,
                clean_cuda_cache,
                cpu_offload,
                offload_granularity,
                t5_cpu_offload,
                clip_cpu_offload,
                vae_cpu_offload,
                unload_modules,
                attention_type,
                quant_op,
                use_tiling_vae,
            ],
        )

        infer_btn.click(
            fn=run_inference,
            inputs=[
                prompt,
                negative_prompt,
                save_result_path,
                infer_steps,
                num_frames,
                resolution,
                seed,
                sample_shift,
                enable_cfg,
                cfg_scale,
                fps,
                use_tiling_vae,
                lazy_load,
                cpu_offload,
                offload_granularity,
                t5_cpu_offload,
                clip_cpu_offload,
                vae_cpu_offload,
                unload_modules,
                attention_type,
                quant_op,
                rope_chunk,
                rope_chunk_size,
                clean_cuda_cache,
                model_path_input,
                model_type_input,
                task_type_input,
                dit_path_input,
                high_noise_path_input,
                low_noise_path_input,
                t5_path_input,
                clip_path_input,
                vae_path_input,
                image_path,
            ],
            outputs=output_video,
        )

    demo.launch(share=True, server_port=args.server_port, server_name=args.server_name, inbrowser=True, allowed_paths=[output_dir])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="轻量级视频生成")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件夹路径")
    parser.add_argument("--server_port", type=int, default=7862, help="服务器端口")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="服务器IP")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出视频保存目录")
    args = parser.parse_args()

    global model_path, model_cls, output_dir
    model_path = args.model_path
    model_cls = "wan2.1"
    output_dir = args.output_dir

    main()
