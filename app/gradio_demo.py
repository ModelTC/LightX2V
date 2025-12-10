import argparse
import gc
import glob
import importlib.util
import json
import os

os.environ["PROFILING_DEBUG_LEVEL"] = "2"
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
    """Scan model_path directory and return available files and subdirectories"""
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
                # Check if directory contains safetensors files
                if glob.glob(os.path.join(item_path, "*.safetensors")):
                    safetensors_dirs.append(item)
            elif os.path.isfile(item_path):
                files.append(item)
                if item.endswith(".pth"):
                    pth_files.append(item)
    except Exception as e:
        logger.warning(f"Failed to scan directory: {e}")

    return {
        "dirs": sorted(dirs),
        "files": sorted(files),
        "safetensors_dirs": sorted(safetensors_dirs),
        "pth_files": sorted(pth_files),
    }


def get_dit_choices(model_path, model_type="wan2.1"):
    """Get Diffusion model options (filtered by model type)"""
    contents = scan_model_path_contents(model_path)
    excluded_keywords = ["vae", "tae", "clip", "t5", "high_noise", "low_noise"]
    fp8_supported = is_fp8_supported_gpu()

    if model_type == "wan2.1":
        # wan2.1: filter files/dirs containing wan2.1 or Wan2.1
        def is_valid(name):
            name_lower = name.lower()
            if "wan2.1" not in name_lower:
                return False
            if not fp8_supported and "fp8" in name_lower:
                return False
            return not any(kw in name_lower for kw in excluded_keywords)
    else:
        # wan2.2: filter files/dirs containing wan2.2 or Wan2.2
        def is_valid(name):
            name_lower = name.lower()
            if "wan2.2" not in name_lower:
                return False
            if not fp8_supported and "fp8" in name_lower:
                return False
            return not any(kw in name_lower for kw in excluded_keywords)

    # Filter matching directories and files
    dir_choices = [d for d in contents["dirs"] if is_valid(d)]
    file_choices = [f for f in contents["files"] if is_valid(f)]
    choices = dir_choices + file_choices
    return choices if choices else [""]


def get_high_noise_choices(model_path):
    """Get high noise model options (files/dirs containing high_noise)"""
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
    """Get low noise model options (files/dirs containing low_noise)"""
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
    """Get T5 model options (.pth or .safetensors files containing t5 keyword)"""
    contents = scan_model_path_contents(model_path)
    fp8_supported = is_fp8_supported_gpu()

    # Filter from .pth files
    pth_choices = [f for f in contents["pth_files"] if "t5" in f.lower() and (fp8_supported or "fp8" not in f.lower())]

    # Filter from .safetensors files
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and "t5" in f.lower() and (fp8_supported or "fp8" not in f.lower())]

    # Filter from directories containing safetensors
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if "t5" in d.lower() and (fp8_supported or "fp8" not in d.lower())]

    choices = pth_choices + safetensors_choices + safetensors_dir_choices
    return choices if choices else [""]


def get_clip_choices(model_path):
    """Get CLIP model options (.pth or .safetensors files containing clip keyword)"""
    contents = scan_model_path_contents(model_path)
    fp8_supported = is_fp8_supported_gpu()

    # Filter from .pth files
    pth_choices = [f for f in contents["pth_files"] if "clip" in f.lower() and (fp8_supported or "fp8" not in f.lower())]

    # Filter from .safetensors files
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and "clip" in f.lower() and (fp8_supported or "fp8" not in f.lower())]

    # Filter from directories containing safetensors
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if "clip" in d.lower() and (fp8_supported or "fp8" not in d.lower())]

    choices = pth_choices + safetensors_choices + safetensors_dir_choices
    return choices if choices else [""]


def get_vae_choices(model_path):
    """Get VAE model options (.pth or .safetensors files containing vae/VAE/tae keyword)"""
    contents = scan_model_path_contents(model_path)
    fp8_supported = is_fp8_supported_gpu()

    # Filter from .pth files
    pth_choices = [f for f in contents["pth_files"] if any(kw in f.lower() for kw in ["vae", "tae"]) and (fp8_supported or "fp8" not in f.lower())]

    # Filter from .safetensors files
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and any(kw in f.lower() for kw in ["vae", "tae"]) and (fp8_supported or "fp8" not in f.lower())]

    # Filter from directories containing safetensors
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if any(kw in d.lower() for kw in ["vae", "tae"]) and (fp8_supported or "fp8" not in d.lower())]

    choices = pth_choices + safetensors_choices + safetensors_dir_choices
    return choices if choices else [""]


def detect_quant_scheme(model_name):
    """Automatically detect quantization scheme from model name
    - If model name contains "int8" → "int8"
    - If model name contains "fp8" and device supports → "fp8"
    - Otherwise return None (no quantization)
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
            # Device doesn't support fp8, return None (use default precision)
            return None
    return None


def update_model_path_options(model_path, model_type="wan2.1"):
    """Update all model path selectors when model_path or model_type changes"""
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
    """Get quantization options dynamically based on model_path"""
    import os

    # Check subdirectories
    subdirs = ["original", "fp8", "int8"]
    has_subdirs = {subdir: os.path.exists(os.path.join(model_path, subdir)) for subdir in subdirs}

    # Check original files in root directory
    t5_bf16_exists = os.path.exists(os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth"))
    clip_fp16_exists = os.path.exists(os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))

    # Generate options
    def get_choices(has_subdirs, original_type, fp8_type, int8_type, fallback_type, has_original_file=False):
        choices = []
        if has_subdirs["original"]:
            choices.append(original_type)
        if has_subdirs["fp8"]:
            choices.append(fp8_type)
        if has_subdirs["int8"]:
            choices.append(int8_type)

        # If no subdirectories but original file exists, add original type
        if has_original_file:
            if not choices or "original" not in choices:
                choices.append(original_type)

        # If no options at all, use default value
        if not choices:
            choices = [fallback_type]

        return choices, choices[0]

    # DIT options
    dit_choices, dit_default = get_choices(has_subdirs, "bf16", "fp8", "int8", "bf16")

    # T5 options - check if original file exists
    t5_choices, t5_default = get_choices(has_subdirs, "bf16", "fp8", "int8", "bf16", t5_bf16_exists)

    # CLIP options - check if original file exists
    clip_choices, clip_default = get_choices(has_subdirs, "fp16", "fp8", "int8", "fp16", clip_fp16_exists)

    return {"dit_choices": dit_choices, "dit_default": dit_default, "t5_choices": t5_choices, "t5_default": t5_default, "clip_choices": clip_choices, "clip_default": clip_default}


def determine_model_cls(model_type, dit_name, high_noise_name):
    """Determine model_cls based on model type and file name"""
    # Determine file name to check
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
    status_text = "✅ Installed" if is_installed else "❌ Not Installed"
    display_text = f"{op_name} ({status_text})"
    quant_op_choices.append((op_name, display_text))

available_attn_ops = get_available_attn_ops()
# Priority order
attn_priority = ["sage_attn3", "sage_attn2", "flash_attn3", "flash_attn2", "torch_sdpa"]
# Sort by priority, installed ones first, uninstalled ones last
attn_op_choices = []
attn_op_dict = dict(available_attn_ops)

# Add installed ones first (by priority)
for op_name in attn_priority:
    if op_name in attn_op_dict and attn_op_dict[op_name]:
        status_text = "✅ Installed"
        display_text = f"{op_name} ({status_text})"
        attn_op_choices.append((op_name, display_text))

# Add uninstalled ones (by priority)
for op_name in attn_priority:
    if op_name in attn_op_dict and not attn_op_dict[op_name]:
        status_text = "❌ Not Installed"
        display_text = f"{op_name} ({status_text})"
        attn_op_choices.append((op_name, display_text))

# Add other operators not in priority list (installed ones first)
other_ops = [(op_name, is_installed) for op_name, is_installed in available_attn_ops if op_name not in attn_priority]
for op_name, is_installed in sorted(other_ops, key=lambda x: not x[1]):  # Installed ones first
    status_text = "✅ Installed" if is_installed else "❌ Not Installed"
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
    logger.info(f"Auto-determined model_cls: {model_cls} (Model type: {model_type_input})")

    if model_type_input == "wan2.1":
        dit_quant_detected = detect_quant_scheme(dit_path_input)
    else:
        dit_quant_detected = detect_quant_scheme(high_noise_path_input)
    t5_quant_detected = detect_quant_scheme(t5_path_input)
    clip_quant_detected = detect_quant_scheme(clip_path_input)
    logger.info(f"Auto-detected quantization scheme - DIT: {dit_quant_detected}, T5: {t5_quant_detected}, CLIP: {clip_quant_detected}")

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

    # Use frontend-selected T5 path
    if is_t5_quant:
        t5_quantized_ckpt = os.path.join(model_path, t5_path_input)
        t5_quant_scheme = f"{t5_quant_detected}-{quant_op}"
        t5_original_ckpt = None
    else:
        t5_quantized_ckpt = None
        t5_quant_scheme = None
        t5_original_ckpt = os.path.join(model_path, t5_path_input)

    # Use frontend-selected CLIP path
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

    logger.info(f"VAE configuration - use_tae: {use_tae}, use_lightvae: {use_lightvae}, need_scaled: {need_scaled} (VAE: {vae_path_input})")

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
    """Auto-configure inference options based on machine configuration and resolution"""
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

        /* Model configuration area styles */
        .model-config {
            margin-bottom: 20px !important;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 15px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        /* Input parameters area styles */
        .input-params {
            margin-bottom: 20px !important;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 15px;
            background: linear-gradient(135deg, #fff5f5 0%, #ffeef0 100%);
        }

        /* Output video area styles */
        .output-video {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            min-height: 400px;
        }

        /* Generate button styles */
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

        /* Accordion header styles */
        .model-config .gr-accordion-header,
        .input-params .gr-accordion-header,
        .output-video .gr-accordion-header {
            font-size: 20px !important;
            font-weight: bold !important;
            padding: 15px !important;
        }

        /* Optimize spacing */
        .gr-row {
            margin-bottom: 15px;
        }

        /* Video player styles */
        .output-video video {
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
    """


def main():
    with gr.Blocks(title="Lightx2v (Lightweight Video Inference and Generation Engine)") as demo:
        gr.Markdown(f"# 🎬 LightX2V Video Generator")
        gr.HTML(f"<style>{css}</style>")
        # Main layout: left and right columns
        with gr.Row():
            # Left: configuration and input area
            with gr.Column(scale=5):
                # Model configuration area
                with gr.Accordion("🗂️ Model Configuration", open=True, elem_classes=["model-config"]):
                    # FP8 support notice
                    if not is_fp8_supported_gpu():
                        gr.Markdown("⚠️ **Your device does not support FP8 inference**. Models containing FP8 have been automatically hidden.")

                    # Hidden state components
                    model_path_input = gr.Textbox(value=model_path, visible=False)

                    # Model type + Task type
                    with gr.Row():
                        model_type_input = gr.Radio(
                            label="Model Type",
                            choices=["wan2.1", "wan2.2"],
                            value="wan2.1",
                            info="wan2.2 requires separate high noise and low noise models",
                        )
                        task_type_input = gr.Radio(
                            label="Task Type",
                            choices=["i2v", "t2v"],
                            value="i2v",
                            info="i2v: Image-to-video, t2v: Text-to-video",
                        )

                    # wan2.1: Diffusion model (single row)
                    with gr.Row() as wan21_row:
                        dit_path_input = gr.Dropdown(
                            label="🎨 Diffusion Model",
                            choices=get_dit_choices(model_path, "wan2.1"),
                            value=get_dit_choices(model_path, "wan2.1")[0] if get_dit_choices(model_path, "wan2.1") else "",
                            allow_custom_value=True,
                            visible=True,
                        )

                    # wan2.2 specific: high noise model + low noise model (hidden by default)
                    with gr.Row(visible=False) as wan22_row:
                        high_noise_path_input = gr.Dropdown(
                            label="🔊 High Noise Model",
                            choices=get_high_noise_choices(model_path),
                            value=get_high_noise_choices(model_path)[0] if get_high_noise_choices(model_path) else "",
                            allow_custom_value=True,
                        )
                        low_noise_path_input = gr.Dropdown(
                            label="🔇 Low Noise Model",
                            choices=get_low_noise_choices(model_path),
                            value=get_low_noise_choices(model_path)[0] if get_low_noise_choices(model_path) else "",
                            allow_custom_value=True,
                        )

                    # Text encoder (single row)
                    with gr.Row():
                        t5_path_input = gr.Dropdown(
                            label="📝 Text Encoder",
                            choices=get_t5_choices(model_path),
                            value=get_t5_choices(model_path)[0] if get_t5_choices(model_path) else "",
                            allow_custom_value=True,
                        )

                    # Image encoder + VAE decoder
                    with gr.Row():
                        clip_path_input = gr.Dropdown(
                            label="🖼️ Image Encoder",
                            choices=get_clip_choices(model_path),
                            value=get_clip_choices(model_path)[0] if get_clip_choices(model_path) else "",
                            allow_custom_value=True,
                        )
                        vae_path_input = gr.Dropdown(
                            label="🎞️ VAE Decoder",
                            choices=get_vae_choices(model_path),
                            value=get_vae_choices(model_path)[0] if get_vae_choices(model_path) else "",
                            allow_custom_value=True,
                        )

                    # Attention operator and quantization matrix multiplication operator
                    with gr.Row():
                        attention_type = gr.Dropdown(
                            label="⚡ Attention Operator",
                            choices=[op[1] for op in attn_op_choices],
                            value=attn_op_choices[0][1] if attn_op_choices else "",
                            info="Use appropriate attention operators to accelerate inference",
                        )
                        quant_op = gr.Dropdown(
                            label="Quantization Matmul Operator",
                            choices=[op[1] for op in quant_op_choices],
                            value=quant_op_choices[0][1],
                            info="Select quantization matrix multiplication operator to accelerate inference",
                            interactive=True,
                        )

                    # Determine if model is distill version
                    def is_distill_model(model_type, dit_path, high_noise_path):
                        """Determine if model is distill version based on model type and path"""
                        if model_type == "wan2.1":
                            check_name = dit_path.lower() if dit_path else ""
                        else:
                            check_name = high_noise_path.lower() if high_noise_path else ""
                        return "4step" in check_name

                    # Model type change event
                    def on_model_type_change(model_type, model_path_val):
                        if model_type == "wan2.2":
                            return gr.update(visible=False), gr.update(visible=True), gr.update()
                        else:
                            # Update wan2.1 Diffusion model options
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

                # Input parameters area
                with gr.Accordion("📥 Input Parameters", open=True, elem_classes=["input-params"]):
                    # Image input (shown for i2v)
                    with gr.Row(visible=True) as image_input_row:
                        image_path = gr.Image(
                            label="Input Image",
                            type="filepath",
                            height=300,
                            interactive=True,
                        )

                    # Task type change event
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
                                value="Camera shake, bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
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

                        with gr.Column(scale=9):
                            seed = gr.Slider(
                                label="Random Seed",
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
                                    label="Inference Steps",
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=4,
                                    info="Distill model inference steps default to 4.",
                                )
                            else:
                                infer_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=40,
                                    info="Number of inference steps for video generation. Increasing steps may improve quality but reduce speed.",
                                )

                            # Dynamically update inference steps when model path changes
                            def update_infer_steps(model_type, dit_path, high_noise_path):
                                is_distill = is_distill_model(model_type, dit_path, high_noise_path)
                                if is_distill:
                                    return gr.update(minimum=1, maximum=100, value=4, interactive=True)
                                else:
                                    return gr.update(minimum=1, maximum=100, value=40, interactive=True)

                            # Listen to model path changes
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

                    # Set default CFG based on model class
                    # CFG scale factor: default to 1 for distill, otherwise 5
                    default_cfg_scale = 1 if default_is_distill else 5
                    # enable_cfg is not exposed to frontend, automatically set based on cfg_scale
                    # If cfg_scale == 1, then enable_cfg = False, otherwise enable_cfg = True
                    default_enable_cfg = False if default_cfg_scale == 1 else True
                    enable_cfg = gr.Checkbox(
                        label="Enable Classifier-Free Guidance",
                        value=default_enable_cfg,
                        visible=False,  # Hidden, not exposed to frontend
                    )

                    with gr.Row():
                        sample_shift = gr.Slider(
                            label="Distribution Shift",
                            value=5,
                            minimum=0,
                            maximum=10,
                            step=1,
                            info="Controls the degree of distribution shift for samples. Larger values indicate more significant shifts.",
                        )
                        cfg_scale = gr.Slider(
                            label="CFG Scale Factor",
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=default_cfg_scale,
                            info="Controls the influence strength of the prompt. Higher values give more influence to the prompt. When value is 1, CFG is automatically disabled.",
                        )

                    # Update enable_cfg based on cfg_scale
                    def update_enable_cfg(cfg_scale_val):
                        """Automatically set enable_cfg based on cfg_scale value"""
                        if cfg_scale_val == 1:
                            return gr.update(value=False)
                        else:
                            return gr.update(value=True)

                    # Dynamically update CFG scale factor and enable_cfg when model path changes
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

                    save_result_path = gr.Textbox(
                        label="Output Video Path",
                        value=generate_unique_filename(output_dir),
                        info="Must include .mp4 extension. If left blank or using the default value, a unique filename will be automatically generated.",
                        visible=False,  # Hide output path, auto-generated
                    )

            with gr.Column(scale=4):
                with gr.Accordion("📤 Generated Video", open=True, elem_classes=["output-video"]):
                    output_video = gr.Video(
                        label="",
                        height=600,
                        autoplay=True,
                        show_label=False,
                    )

                    infer_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg", elem_classes=["generate-btn"])

            rope_chunk = gr.Checkbox(label="Chunked Rotary Position Embedding", value=False, visible=False)
            rope_chunk_size = gr.Slider(label="Rotary Embedding Chunk Size", value=100, minimum=100, maximum=10000, step=100, visible=False)
            unload_modules = gr.Checkbox(label="Unload Modules", value=False, visible=False)
            clean_cuda_cache = gr.Checkbox(label="Clean CUDA Memory Cache", value=False, visible=False)
            cpu_offload = gr.Checkbox(label="CPU Offloading", value=False, visible=False)
            lazy_load = gr.Checkbox(label="Enable Lazy Loading", value=False, visible=False)
            offload_granularity = gr.Dropdown(label="Dit Offload Granularity", choices=["block", "phase"], value="phase", visible=False)
            t5_cpu_offload = gr.Checkbox(label="T5 CPU Offloading", value=False, visible=False)
            clip_cpu_offload = gr.Checkbox(label="CLIP CPU Offloading", value=False, visible=False)
            vae_cpu_offload = gr.Checkbox(label="VAE CPU Offloading", value=False, visible=False)
            use_tiling_vae = gr.Checkbox(label="VAE Tiling Inference", value=False, visible=False)

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
    parser = argparse.ArgumentParser(description="Lightweight Video Generation")
    parser.add_argument("--model_path", type=str, required=True, help="Model folder path")
    parser.add_argument("--server_port", type=int, default=7862, help="Server port")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server IP")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output video save directory")
    args = parser.parse_args()

    global model_path, model_cls, output_dir
    model_path = args.model_path
    model_cls = "wan2.1"
    output_dir = args.output_dir

    main()
