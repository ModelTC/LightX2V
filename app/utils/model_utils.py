import concurrent.futures
import glob
import os

from loguru import logger

try:
    from huggingface_hub import HfApi, list_repo_files

    HF_AVAILABLE = True
except ImportError:
    HfApi = None
    list_repo_files = None
    HF_AVAILABLE = False

try:
    from modelscope.hub.api import HubApi

    MS_AVAILABLE = True
except ImportError:
    HubApi = None
    MS_AVAILABLE = False
import gc
import importlib.util
import re

import psutil
import torch
from loguru import logger


def is_module_installed(module_name):
    """检查模块是否已安装"""
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def get_available_quant_ops():
    """获取可用的量化算子"""
    available_ops = []

    triton_installed = is_module_installed("triton")
    if triton_installed:
        available_ops.append(("triton", True))
    else:
        available_ops.append(("triton", False))

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

    # 检测 torch 选项：需要同时满足 hasattr(torch, "_scaled_mm") 和安装了 torchao
    torch_available = hasattr(torch, "_scaled_mm") and is_module_installed("torchao")
    if torch_available:
        available_ops.append(("torch", True))
    else:
        available_ops.append(("torch", False))

    return available_ops


def get_available_attn_ops():
    """获取可用的注意力算子"""
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
    """获取GPU显存（GB）"""
    if not torch.cuda.is_available():
        return 0
    with torch.cuda.device(gpu_idx):
        memory_info = torch.cuda.mem_get_info()
        total_memory = memory_info[1] / (1024**3)  # Convert bytes to GB
        return total_memory


def get_cpu_memory():
    """获取CPU可用内存（GB）"""
    available_bytes = psutil.virtual_memory().available
    return available_bytes / 1024**3


def cleanup_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def is_fp8_supported_gpu():
    """检查GPU是否支持fp8"""
    if not torch.cuda.is_available():
        return False
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    return (major == 8 and minor == 9) or (major >= 9)


def is_sm_greater_than_90():
    """检测计算能力是否大于 (9,0)"""
    if not torch.cuda.is_available():
        return False
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    return (major, minor) > (9, 0)


def get_gpu_generation():
    """检测GPU系列，返回 '40' 表示40系，'30' 表示30系，None 表示其他"""
    if not torch.cuda.is_available():
        return None
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_name_lower = gpu_name.lower()

        # 检测40系显卡 (RTX 40xx, RTX 4060, RTX 4070, RTX 4080, RTX 4090等)
        if any(keyword in gpu_name_lower for keyword in ["rtx 40", "rtx40", "geforce rtx 40"]):
            # 进一步检查是40xx系列
            match = re.search(r"rtx\s*40\d+|40\d+", gpu_name_lower)
            if match:
                return "40"

        # 检测30系显卡 (RTX 30xx, RTX 3060, RTX 3070, RTX 3080, RTX 3090等)
        if any(keyword in gpu_name_lower for keyword in ["rtx 30", "rtx30", "geforce rtx 30"]):
            # 进一步检查是30xx系列
            match = re.search(r"rtx\s*30\d+|30\d+", gpu_name_lower)
            if match:
                return "30"

        return None
    except Exception as e:
        logger.warning(f"无法检测GPU系列: {e}")
        return None


# 模型列表缓存（避免每次从 HF 获取）
HF_MODELS_CACHE = {
    "lightx2v/wan2.1-Distill-Models": [],
    "lightx2v/wan2.1-Official-Models": [],
    "lightx2v/wan2.2-Distill-Models": [],
    "lightx2v/wan2.2-Official-Models": [],
    "lightx2v/Encoders": [],
    "lightx2v/Autoencoders": [],
    "lightx2v/Qwen-Image-Edit-2511-Lightning": [],
    "Qwen/Qwen-Image-Edit-2511": [],
    "lightx2v/Qwen-Image-2512-Lightning": [],
    "Qwen/Qwen-Image-2512": [],
    "lightx2v/Z-Image-Turbo-Quantized": [],
    "Tongyi-MAI/Z-Image-Turbo": [],
    "JunHowie/Qwen3-4B-GPTQ-Int4": [],
}


def process_files(files, repo_id=None):
    """处理文件列表，提取模型名称"""
    model_names = []
    seen_dirs = set()

    # 对于 Qwen/Qwen-Image-Edit-2511、Qwen/Qwen-Image-2512 和 Tongyi-MAI/Z-Image-Turbo 仓库，保留 vae 和 scheduler 目录
    is_qwen_image_repo = repo_id in ["Qwen/Qwen-Image-Edit-2511", "Qwen/Qwen-Image-2512", "Tongyi-MAI/Z-Image-Turbo"]
    # 对于 Qwen3 编码器仓库，整个仓库就是一个模型目录
    is_qwen3_encoder_repo = repo_id == "JunHowie/Qwen3-4B-GPTQ-Int4"

    for file in files:
        # 排除包含comfyui的文件
        if "comfyui" in file.lower():
            continue

        # 如果是顶层文件（不包含路径分隔符）
        if "/" not in file:
            # 对于 Qwen3 编码器仓库，不添加单个文件，只添加目录
            # 因为 Qwen3 编码器应该下载整个仓库目录
            if not is_qwen3_encoder_repo and file.endswith(".safetensors"):
                model_names.append(file)
        else:
            # 提取顶层目录名（支持_split目录）
            top_dir = file.split("/")[0]
            if top_dir not in seen_dirs:
                seen_dirs.add(top_dir)
                # 对于 Qwen 仓库，保留 vae 和 scheduler 目录
                if is_qwen_image_repo and top_dir.lower() in ["vae", "scheduler"]:
                    model_names.append(top_dir)
                # 对于 Qwen3 编码器仓库，保留所有顶层目录（排除 comfyui）
                elif is_qwen3_encoder_repo:
                    model_names.append(top_dir)
                # 支持safetensors文件目录和_split分block存储目录
                elif "_split" in top_dir or any(f.startswith(f"{top_dir}/") and f.endswith(".safetensors") for f in files):
                    model_names.append(top_dir)
    return sorted(set(model_names))


def load_hf_models_cache():
    """从 Hugging Face 加载模型列表并缓存，如果 HF 超时或失败，则尝试使用 ModelScope"""
    # 超时时间（秒）
    HF_TIMEOUT = 30

    for repo_id in HF_MODELS_CACHE.keys():
        files = None
        source = None

        # 首先尝试从 ModelScope 获取
        try:
            if MS_AVAILABLE:
                logger.info(f"Loading models from ModelScope {repo_id}...")
                api = HubApi()
                # ModelScope API 获取文件列表
                model_files = api.get_model_files(model_id=repo_id, recursive=True)
                # 提取文件路径
                files = [file["Path"] for file in model_files if file.get("Type") == "blob"]
                source = "ModelScope"
                logger.info(f"Successfully loaded models from ModelScope {repo_id}")
        except:  # noqa E722
            # 如果 ModelScope 失败，尝试从 Hugging Face 获取（带超时）
            if files is None and HF_AVAILABLE:
                logger.info(f"Loading models from Hugging Face {repo_id}...")
                api = HfApi()

                # 使用线程池执行器设置超时
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(list_repo_files, repo_id=repo_id, repo_type="model")
                    files = future.result(timeout=HF_TIMEOUT)
                    source = "Hugging Face"

        # 处理文件列表
        if files:
            model_names = process_files(files, repo_id)
            HF_MODELS_CACHE[repo_id] = model_names
            logger.info(f"Loaded {len(HF_MODELS_CACHE[repo_id])} models from {source} {repo_id}")
        else:
            logger.warning(f"No files retrieved from {repo_id}, setting empty cache")
            HF_MODELS_CACHE[repo_id] = []


def get_hf_models(repo_id, prefix_filter=None, keyword_filter=None):
    """从缓存的模型列表中获取模型（不再实时从 HF 获取）"""
    if repo_id not in HF_MODELS_CACHE:
        return []

    models = HF_MODELS_CACHE[repo_id]

    if prefix_filter:
        models = [m for m in models if m.lower().startswith(prefix_filter.lower())]

    if keyword_filter:
        models = [m for m in models if keyword_filter.lower() in m.lower()]

    return models


def extract_op_name(op_str):
    """从格式化的操作符名称中提取原始名称"""
    if not op_str:
        return ""
    op_str = op_str.strip()
    if op_str.startswith("✅"):
        op_str = op_str[1:].strip()
    elif op_str.startswith("❌"):
        op_str = op_str[1:].strip()
    if "(" in op_str:
        op_str = op_str.split("(")[0].strip()
    return op_str


def scan_model_path_contents(model_path):
    """扫描 model_path 目录，返回可用的文件和子目录"""
    if not model_path or not os.path.exists(model_path):
        return {"dirs": [], "files": [], "safetensors_dirs": [], "pth_files": []}

    dirs = []
    files = []
    safetensors_dirs = []
    pth_files = []

    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if os.path.isdir(item_path):
            dirs.append(item)
            if glob.glob(os.path.join(item_path, "*.safetensors")):
                safetensors_dirs.append(item)
        elif os.path.isfile(item_path):
            files.append(item)
            if item.endswith(".pth"):
                pth_files.append(item)

    return {
        "dirs": sorted(dirs),
        "files": sorted(files),
        "safetensors_dirs": sorted(safetensors_dirs),
        "pth_files": sorted(pth_files),
    }


def check_model_exists(model_path, model_name):
    """检查模型是否已下载
    支持子目录路径，如 Qwen-Image-2512/qwen_image_2512_fp8_e4m3fn_scaled.safetensors
    """
    if not model_path or not os.path.exists(model_path):
        return False

    # 处理包含子目录路径的模型名（如 Qwen-Image-2512/qwen_image_2512_fp8_e4m3fn_scaled.safetensors）
    model_path_full = os.path.join(model_path, model_name)
    # 检查是否存在（文件或目录）
    if os.path.exists(model_path_full):
        return True

    # 额外检查：如果是 safetensors 文件，也检查同名目录（_split 目录）
    if model_name.endswith(".safetensors"):
        # 如果模型名包含子目录路径，提取文件名
        if "/" in model_name:
            base_name_with_subdir = model_name.replace(".safetensors", "")
            # 检查子目录下的 _split 目录
            split_dir = os.path.join(model_path, base_name_with_subdir + "_split")
            if os.path.exists(split_dir):
                return True
        else:
            # 检查同名目录（可能是分块存储）
            base_name = model_name.replace(".safetensors", "")
            split_dir = os.path.join(model_path, base_name + "_split")
            if os.path.exists(split_dir):
                return True

    return False


def format_model_choice(model_name, model_path, status_emoji=None):
    """格式化模型选项，添加下载状态标识"""
    if not model_name:
        return ""

    # 如果提供了状态 emoji，直接使用
    if status_emoji is not None:
        return f"{status_emoji} {model_name}"

    # 否则检查本地是否存在
    exists = check_model_exists(model_path, model_name)
    emoji = "✅" if exists else "❌"
    return f"{emoji} {model_name}"


def extract_model_name(formatted_name):
    """从格式化的选项名称中提取原始模型名称"""
    if not formatted_name:
        return ""
    # 移除开头的 emoji 和空格
    if formatted_name.startswith("✅ ") or formatted_name.startswith("❌ "):
        return formatted_name[2:].strip()
    return formatted_name.strip()


def sort_model_choices(models):
    """根据设备能力和模型类型对模型列表排序

    排序规则：
    - 如果设备支持 fp8：fp8+split > int8+split > fp8 > int8 > 其他
    - 如果设备不支持 fp8：int8+split > int8 > 其他
    """
    # 延迟导入避免循环依赖
    from utils.model_utils import is_fp8_supported_gpu

    fp8_supported = is_fp8_supported_gpu()

    def get_priority(name):
        name_lower = name.lower()
        if fp8_supported:
            # fp8 设备：fp8+split > int8+split > fp8 > int8 > 其他
            if "fp8" in name_lower and "_split" in name_lower:
                return 0  # 最高优先级
            elif "int8" in name_lower and "_split" in name_lower:
                return 1
            elif "fp8" in name_lower:
                return 2
            elif "int8" in name_lower:
                return 3
            else:
                return 4  # 其他
        else:
            # 非 fp8 设备：优先 int8+split，其次 int8
            if "int8" in name_lower and "_split" in name_lower:
                return 0  # 最高优先级
            elif "int8" in name_lower:
                return 1
            else:
                return 2  # 其他（fp8 已经被过滤掉了）

    return sorted(models, key=lambda x: (get_priority(x), x.lower()))


def detect_quant_scheme(model_name):
    """根据模型名字自动检测量化精度
    - 如果模型名字包含 "int8" → "int8"
    - 如果模型名字包含 "fp8" 且设备支持 → "fp8"
    - 否则返回 None（表示不使用量化）
    """
    if not model_name:
        return None
    # 延迟导入避免循环依赖
    from utils.model_utils import is_fp8_supported_gpu

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


def is_distill_model_from_name(model_name):
    """根据模型名称判断是否是 distill 模型"""
    if not model_name:
        return None
    return "4step" in model_name.lower() or "8step" in model_name.lower()


def get_quant_scheme(quant_detected, quant_op_val):
    if quant_op_val == "torch":
        return f"{quant_detected}-torchao"
    elif quant_op_val == "triton":
        return f"{quant_detected}-triton"
    else:
        return f"{quant_detected}-{quant_op_val}"


def build_wan21(
    model_path_input,
    dit_path_input,
    t5_path_input,
    clip_path_input,
    vae_path_input,
    quant_op,
):
    wan21_config = {
        "dim": 5120,
        "eps": 1e-06,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_dim": 36,
        "num_heads": 40,
        "num_layers": 40,
        "out_dim": 16,
        "text_len": 512,
    }

    dit_path_input = extract_model_name(dit_path_input) if dit_path_input else ""
    t5_path_input = extract_model_name(t5_path_input) if t5_path_input else ""
    clip_path_input = extract_model_name(clip_path_input) if clip_path_input else ""

    is_distill = is_distill_model_from_name(dit_path_input)
    if is_distill:
        wan21_config["model_cls"] = "wan2.1_distill"
    else:
        wan21_config["model_cls"] = "wan2.1"

    wan21_config["dit_quantized_ckpt"] = None
    wan21_config["dit_original_ckpt"] = None
    dit_quant_detected = detect_quant_scheme(dit_path_input)
    is_dit_quant = dit_quant_detected in ["fp8", "int8"]
    if is_dit_quant:
        wan21_config["dit_quantized"] = True
        wan21_config["dit_quant_scheme"] = get_quant_scheme(dit_quant_detected, quant_op)
        wan21_config["dit_quantized_ckpt"] = os.path.join(model_path_input, dit_path_input)
    else:
        wan21_config["dit_quantized"] = False
        wan21_config["dit_quant_scheme"] = "Default"
        wan21_config["dit_original_ckpt"] = os.path.join(model_path_input, dit_path_input)

    wan21_config["t5_original_ckpt"] = None
    wan21_config["t5_quantized_ckpt"] = None
    t5_quant_detected = detect_quant_scheme(t5_path_input)
    is_t5_quant = t5_quant_detected in ["fp8", "int8"]
    if is_t5_quant:
        wan21_config["t5_quantized"] = True
        wan21_config["t5_quant_scheme"] = get_quant_scheme(t5_quant_detected, quant_op)
        wan21_config["t5_quantized_ckpt"] = os.path.join(model_path_input, t5_path_input)
    else:
        wan21_config["t5_quantized"] = False
        wan21_config["t5_quant_scheme"] = "Default"
        wan21_config["t5_original_ckpt"] = os.path.join(model_path_input, t5_path_input)

    wan21_config["clip_original_ckpt"] = None
    wan21_config["clip_quantized_ckpt"] = None
    clip_quant_detected = detect_quant_scheme(clip_path_input)
    is_clip_quant = clip_quant_detected in ["fp8", "int8"]
    if is_clip_quant:
        wan21_config["clip_quantized"] = True  # 启用量化
        wan21_config["clip_quant_scheme"] = get_quant_scheme(clip_quant_detected, quant_op)
        wan21_config["clip_quantized_ckpt"] = os.path.join(model_path_input, clip_path_input)
    else:
        wan21_config["clip_quantized"] = False  # 禁用量化
        wan21_config["clip_quant_scheme"] = "Default"
        wan21_config["clip_original_ckpt"] = os.path.join(model_path_input, clip_path_input)

    # 提取 VAE 路径，移除状态符号
    vae_path_input = extract_model_name(vae_path_input) if vae_path_input else ""
    wan21_config["vae_path"] = os.path.join(model_path_input, vae_path_input) if vae_path_input else None
    wan21_config["model_path"] = model_path_input
    return wan21_config


def build_wan22(
    model_path_input,
    high_noise_path_input,
    low_noise_path_input,
    t5_path_input,
    vae_path_input,
    quant_op,
):
    wan22_config = {
        "dim": 5120,
        "eps": 1e-06,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_dim": 36,
        "num_heads": 40,
        "num_layers": 40,
        "out_dim": 16,
        "text_len": 512,
    }
    """构建 Wan2.2 模型配置"""
    high_noise_path_input = extract_model_name(high_noise_path_input) if high_noise_path_input else ""
    low_noise_path_input = extract_model_name(low_noise_path_input) if low_noise_path_input else ""
    t5_path_input = extract_model_name(t5_path_input) if t5_path_input else ""

    is_distill = is_distill_model_from_name(high_noise_path_input)
    if is_distill:
        wan22_config["model_cls"] = "wan2.2_moe_distill"
    else:
        wan22_config["model_cls"] = "wan2.2_moe"

    wan22_config["high_noise_quantized_ckpt"] = None
    wan22_config["low_noise_quantized_ckpt"] = None
    wan22_config["high_noise_original_ckpt"] = None
    wan22_config["low_noise_original_ckpt"] = None
    dit_quant_detected = detect_quant_scheme(high_noise_path_input)
    is_dit_quant = dit_quant_detected in ["fp8", "int8"]
    if is_dit_quant:
        wan22_config["dit_quantized"] = True
        wan22_config["dit_quant_scheme"] = get_quant_scheme(dit_quant_detected, quant_op)
        wan22_config["high_noise_quant_scheme"] = get_quant_scheme(dit_quant_detected, quant_op)
        wan22_config["high_noise_quantized_ckpt"] = os.path.join(model_path_input, high_noise_path_input)
        wan22_config["low_noise_quantized_ckpt"] = os.path.join(model_path_input, low_noise_path_input)
    else:
        wan22_config["dit_quantized"] = False
        wan22_config["dit_quant_scheme"] = "Default"
        wan22_config["high_noise_quant_scheme"] = "Default"
        wan22_config["high_noise_original_ckpt"] = os.path.join(model_path_input, high_noise_path_input)
        wan22_config["low_noise_original_ckpt"] = os.path.join(model_path_input, low_noise_path_input)

    wan22_config["t5_original_ckpt"] = None
    wan22_config["t5_quantized_ckpt"] = None
    t5_quant_detected = detect_quant_scheme(t5_path_input)
    is_t5_quant = t5_quant_detected in ["fp8", "int8"]
    if is_t5_quant:
        wan22_config["t5_quantized"] = True
        wan22_config["t5_quant_scheme"] = get_quant_scheme(t5_quant_detected, quant_op)
        wan22_config["t5_quantized_ckpt"] = os.path.join(model_path_input, t5_path_input)
    else:
        wan22_config["t5_quantized"] = False
        wan22_config["t5_quant_scheme"] = "Default"
        wan22_config["t5_original_ckpt"] = os.path.join(model_path_input, t5_path_input)

    # 提取 VAE 路径，移除状态符号
    vae_path_input = extract_model_name(vae_path_input) if vae_path_input else ""
    wan22_config["vae_path"] = os.path.join(model_path_input, vae_path_input) if vae_path_input else None
    wan22_config["model_path"] = model_path_input
    return wan22_config


def build_qwen_image(
    model_type_input,
    model_path_input,
    qwen_image_dit_path_input,
    qwen_image_vae_path_input,
    qwen_image_scheduler_path_input,
    qwen25vl_encoder_path_input,
    quant_op,
):
    # 共同配置
    qwen_image_config = {
        "model_cls": "qwen_image",
        "attention_head_dim": 128,
        "axes_dims_rope": [16, 56, 56],
        "guidance_embeds": False,
        "in_channels": 64,
        "joint_attention_dim": 3584,
        "num_attention_heads": 24,
        "num_layers": 60,
        "out_channels": 16,
        "patch_size": 2,
        "attention_out_dim": 3072,
        "attention_dim_head": 128,
        "transformer_in_channels": 64,
    }

    # 根据模型类型设置不同配置
    if model_type_input == "Qwen-Image-Edit-2511":
        qwen_image_config.update(
            {
                "CONDITION_IMAGE_SIZE": 147456,
                "USE_IMAGE_ID_IN_PROMPT": True,
                "prompt_template_encode": "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                "prompt_template_encode_start_idx": 64,
                "vae_scale_factor": 8,
                "zero_cond_t": True,
            }
        )
    elif model_type_input == "Qwen-Image-2512":
        qwen_image_config.update(
            {
                "prompt_template_encode": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                "prompt_template_encode_start_idx": 34,
                "zero_cond_t": False,
            }
        )
    else:
        raise ValueError(f"Invalid model type: {model_type_input}")

    """构建 Qwen-Image 模型配置"""
    qwen_image_dit_path_input = extract_model_name(qwen_image_dit_path_input) if qwen_image_dit_path_input else ""
    qwen_image_vae_path_input = extract_model_name(qwen_image_vae_path_input) if qwen_image_vae_path_input else ""
    qwen_image_scheduler_path_input = extract_model_name(qwen_image_scheduler_path_input) if qwen_image_scheduler_path_input else ""
    qwen25vl_encoder_path_input = extract_model_name(qwen25vl_encoder_path_input) if qwen25vl_encoder_path_input else ""

    # 确定子目录路径
    if model_type_input == "Qwen-Image-Edit-2511":
        model_subdir = "Qwen-Image-Edit-2511"
    elif model_type_input == "Qwen-Image-2512":
        model_subdir = "Qwen-Image-2512"
    else:
        model_subdir = ""

    # 处理包含子目录路径的输入（去掉子目录前缀，只保留文件名）
    if qwen_image_dit_path_input and "/" in qwen_image_dit_path_input:
        qwen_image_dit_path_input = qwen_image_dit_path_input.split("/")[-1]
    if qwen_image_vae_path_input and "/" in qwen_image_vae_path_input:
        qwen_image_vae_path_input = qwen_image_vae_path_input.split("/")[-1]
    if qwen_image_scheduler_path_input and "/" in qwen_image_scheduler_path_input:
        qwen_image_scheduler_path_input = qwen_image_scheduler_path_input.split("/")[-1]

    qwen_dit_quant_detected = detect_quant_scheme(qwen_image_dit_path_input)
    is_qwen_dit_quant = qwen_dit_quant_detected in ["fp8", "int8"]
    if is_qwen_dit_quant:
        qwen_image_config["dit_quantized"] = True
        qwen_image_config["dit_quant_scheme"] = get_quant_scheme(qwen_dit_quant_detected, quant_op)
        # 使用子目录路径构建完整路径
        dit_path = os.path.join(model_path_input, model_subdir, qwen_image_dit_path_input) if qwen_image_dit_path_input else None
        qwen_image_config["dit_quantized_ckpt"] = dit_path
        qwen_image_config["dit_original_ckpt"] = None
    else:
        qwen_image_config["dit_quantized"] = False
        qwen_image_config["dit_quant_scheme"] = "Default"
        # 使用子目录路径构建完整路径
        dit_path = os.path.join(model_path_input, model_subdir, qwen_image_dit_path_input) if qwen_image_dit_path_input else None
        qwen_image_config["dit_original_ckpt"] = dit_path
        qwen_image_config["dit_quantized_ckpt"] = None

    # VAE 和 Scheduler 路径也使用子目录
    vae_path = os.path.join(model_path_input, model_subdir, "vae") if qwen_image_vae_path_input else None
    scheduler_path = os.path.join(model_path_input, model_subdir, "scheduler") if qwen_image_scheduler_path_input else None
    qwen_image_config["vae_path"] = vae_path
    qwen_image_config["scheduler_path"] = scheduler_path
    qwen_image_config["qwen25vl_quantized"] = True
    qwen_image_config["qwen25vl_quant_scheme"] = "int4"
    qwen_image_config["qwen25vl_quantized_ckpt"] = os.path.join(model_path_input, qwen25vl_encoder_path_input) if qwen25vl_encoder_path_input else None
    qwen_image_config["qwen25vl_tokenizer_path"] = os.path.join(model_path_input, qwen25vl_encoder_path_input) if qwen25vl_encoder_path_input else None
    qwen_image_config["qwen25vl_processor_path"] = os.path.join(model_path_input, qwen25vl_encoder_path_input) if qwen25vl_encoder_path_input else None
    # 使用子目录路径作为 model_path
    if model_type_input == "Qwen-Image-Edit-2511":
        qwen_image_config["model_path"] = os.path.join(model_path_input, "Qwen-Image-Edit-2511")
    elif model_type_input == "Qwen-Image-2512":
        qwen_image_config["model_path"] = os.path.join(model_path_input, "Qwen-Image-2512")
    qwen_image_config["vae_scale_factor"] = 8
    return qwen_image_config


def build_z_image(
    model_path_input,
    z_image_dit_path_input,
    z_image_vae_path_input,
    z_image_scheduler_path_input,
    qwen3_encoder_path_input,
    quant_op,
):
    """构建 Z-Image-Turbo 模型配置"""
    # 共同配置
    z_image_config = {
        "all_f_patch_size": [1],
        "all_patch_size": [2],
        "axes_dims": [32, 48, 48],
        "axes_lens": [1536, 512, 512],
        "cap_feat_dim": 2560,
        "dim": 3840,
        "in_channels": 16,
        "n_heads": 30,
        "n_kv_heads": 30,
        "n_layers": 30,
        "n_refiner_layers": 2,
        "norm_eps": 1e-05,
        "qk_norm": True,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
    }
    z_image_dit_path_input = extract_model_name(z_image_dit_path_input) if z_image_dit_path_input else ""
    z_image_vae_path_input = extract_model_name(z_image_vae_path_input) if z_image_vae_path_input else ""
    z_image_scheduler_path_input = extract_model_name(z_image_scheduler_path_input) if z_image_scheduler_path_input else ""
    qwen3_encoder_path_input = extract_model_name(qwen3_encoder_path_input) if qwen3_encoder_path_input else ""

    model_subdir = "Z-Image-Turbo"

    # 处理包含子目录路径的输入（去掉子目录前缀，只保留文件名）
    if z_image_dit_path_input and "/" in z_image_dit_path_input:
        z_image_dit_path_input = z_image_dit_path_input.split("/")[-1]
    if z_image_vae_path_input and "/" in z_image_vae_path_input:
        z_image_vae_path_input = z_image_vae_path_input.split("/")[-1]
    if z_image_scheduler_path_input and "/" in z_image_scheduler_path_input:
        z_image_scheduler_path_input = z_image_scheduler_path_input.split("/")[-1]

    z_image_dit_quant_detected = detect_quant_scheme(z_image_dit_path_input)
    is_z_image_dit_quant = z_image_dit_quant_detected in ["fp8", "int8"]
    if is_z_image_dit_quant:
        z_image_config["dit_quantized"] = True
        z_image_config["dit_quant_scheme"] = get_quant_scheme(z_image_dit_quant_detected, quant_op)
        # 使用子目录路径构建完整路径
        dit_path = os.path.join(model_path_input, model_subdir, z_image_dit_path_input) if z_image_dit_path_input else None
        z_image_config["dit_quantized_ckpt"] = dit_path
        z_image_config["dit_original_ckpt"] = None
    else:
        z_image_config["dit_quantized"] = False
        z_image_config["dit_quant_scheme"] = "Default"
        # 使用子目录路径构建完整路径
        dit_path = os.path.join(model_path_input, model_subdir, z_image_dit_path_input) if z_image_dit_path_input else None
        z_image_config["dit_original_ckpt"] = dit_path
        z_image_config["dit_quantized_ckpt"] = None

    # VAE 和 Scheduler 路径也使用子目录
    vae_path = os.path.join(model_path_input, model_subdir, "vae") if z_image_vae_path_input else None
    scheduler_path = os.path.join(model_path_input, model_subdir, "scheduler") if z_image_scheduler_path_input else None
    z_image_config["vae_path"] = vae_path
    z_image_config["scheduler_path"] = scheduler_path
    z_image_config["qwen3_quantized"] = True
    z_image_config["qwen3_quant_scheme"] = "int4"
    z_image_config["qwen3_quantized_ckpt"] = os.path.join(model_path_input, qwen3_encoder_path_input) if qwen3_encoder_path_input else None
    z_image_config["qwen3_tokenizer_path"] = os.path.join(model_path_input, qwen3_encoder_path_input) if qwen3_encoder_path_input else None
    z_image_config["qwen3_processor_path"] = os.path.join(model_path_input, qwen3_encoder_path_input) if qwen3_encoder_path_input else None
    # 使用子目录路径作为 model_path
    z_image_config["model_path"] = os.path.join(model_path_input, "Z-Image-Turbo")
    z_image_config["model_cls"] = "z_image"
    z_image_config["vae_scale_factor"] = 8
    z_image_config["patch_size"] = 2
    return z_image_config


def get_model_configs(
    model_type_input,
    model_path_input,
    dit_path_input,
    high_noise_path_input,
    low_noise_path_input,
    t5_path_input,
    clip_path_input,
    vae_path_input,
    qwen_image_dit_path_input,
    qwen_image_vae_path_input,
    qwen_image_scheduler_path_input,
    qwen25vl_encoder_path_input,
    z_image_dit_path_input,
    z_image_vae_path_input,
    z_image_scheduler_path_input,
    qwen3_encoder_path_input,
    quant_op,
    use_lora=None,
    lora_path=None,
    lora_strength=None,
    high_noise_lora_path=None,
    low_noise_lora_path=None,
    high_noise_lora_strength=None,
    low_noise_lora_strength=None,
):
    if model_path_input and model_path_input.strip():
        model_path_input = model_path_input.strip()

    # 构建 LoRA 配置（如果提供）
    lora_configs = None
    if use_lora:
        if model_type_input == "Wan2.2":
            lora_configs = []

            # 处理 high_noise LoRA
            if high_noise_lora_path and high_noise_lora_path.strip():
                high_noise_lora_full_path = os.path.join(model_path_input, "loras", high_noise_lora_path.strip())
                high_noise_strength = float(high_noise_lora_strength) if high_noise_lora_strength is not None else 1.0
                lora_configs.append(
                    {
                        "name": "high_noise_model",
                        "path": high_noise_lora_full_path,
                        "strength": high_noise_strength,
                    }
                )
            elif lora_path and lora_path.strip():
                # 如果没有分别提供，使用统一的 lora_path
                high_noise_lora_full_path = os.path.join(model_path_input, "loras", lora_path.strip())
                high_noise_strength = float(lora_strength) if lora_strength is not None else 1.0
                lora_configs.append(
                    {
                        "name": "high_noise_model",
                        "path": high_noise_lora_full_path,
                        "strength": high_noise_strength,
                    }
                )

            # 处理 low_noise LoRA
            if low_noise_lora_path and low_noise_lora_path.strip():
                low_noise_lora_full_path = os.path.join(model_path_input, "loras", low_noise_lora_path.strip())
                low_noise_strength = float(low_noise_lora_strength) if low_noise_lora_strength is not None else 1.0
                lora_configs.append(
                    {
                        "name": "low_noise_model",
                        "path": low_noise_lora_full_path,
                        "strength": low_noise_strength,
                    }
                )
            elif lora_path and lora_path.strip():
                # 如果没有分别提供，使用统一的 lora_path
                low_noise_lora_full_path = os.path.join(model_path_input, "loras", lora_path.strip())
                low_noise_strength = float(lora_strength) if lora_strength is not None else 1.0
                lora_configs.append(
                    {
                        "name": "low_noise_model",
                        "path": low_noise_lora_full_path,
                        "strength": low_noise_strength,
                    }
                )

            # 如果没有任何 LoRA 配置，设置为 None
            if not lora_configs:
                lora_configs = None
        else:
            # 其他模型类型（Wan2.1, Qwen-Image, Z-Image-Turbo）
            if lora_path and lora_path.strip():
                lora_full_path = os.path.join(model_path_input, "loras", lora_path.strip())
                lora_configs = [
                    {
                        "path": lora_full_path,
                        "strength": float(lora_strength) if lora_strength is not None else 1.0,
                    }
                ]

    if model_type_input == "Wan2.1":
        config = build_wan21(
            model_path_input,
            dit_path_input,
            t5_path_input,
            clip_path_input,
            vae_path_input,
            quant_op,
        )
        if lora_configs:
            config["lora_configs"] = lora_configs
        return config
    elif model_type_input == "Wan2.2":
        config = build_wan22(
            model_path_input,
            high_noise_path_input,
            low_noise_path_input,
            t5_path_input,
            vae_path_input,
            quant_op,
        )
        if lora_configs:
            config["lora_configs"] = lora_configs
        return config

    elif model_type_input in ["Qwen-Image-Edit-2511", "Qwen-Image-2512"]:
        config = build_qwen_image(
            model_type_input,
            model_path_input,
            qwen_image_dit_path_input,
            qwen_image_vae_path_input,
            qwen_image_scheduler_path_input,
            qwen25vl_encoder_path_input,
            quant_op,
        )
        if lora_configs:
            config["lora_configs"] = lora_configs
        return config
    elif model_type_input == "Z-Image-Turbo":
        config = build_z_image(
            model_path_input,
            z_image_dit_path_input,
            z_image_vae_path_input,
            z_image_scheduler_path_input,
            qwen3_encoder_path_input,
            quant_op,
        )
        if lora_configs:
            config["lora_configs"] = lora_configs
        return config
