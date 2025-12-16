import argparse
import gc
import glob
import importlib.util
import json
import logging
import os
import warnings

# 抑制 Hugging Face 下载时的网络重试警告（这些是正常的重试行为）
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.utils")
# 抑制 reqwest 的重试警告（这些是 JSON 日志输出，不是真正的错误）
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

os.environ["PROFILING_DEBUG_LEVEL"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DTYPE"] = "BF16"
import random  # noqa E402
from datetime import datetime  # noqa E402

import gradio as gr  # noqa E402
import psutil  # noqa E402
import torch  # noqa E402
from loguru import logger  # noqa E402

from lightx2v.utils.input_info import set_input_info  # noqa E402
from lightx2v.utils.set_config import get_default_config  # noqa E402

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace  # noqa E402
except ImportError:
    apply_rope_with_cos_sin_cache_inplace = None  # noqa E402

from huggingface_hub import HfApi, hf_hub_download, list_repo_files  # noqa E402
from huggingface_hub import snapshot_download as hf_snapshot_download  # noqa E402

HF_AVAILABLE = True

from modelscope.hub.api import HubApi  # noqa E402
from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download  # noqa E402

MS_AVAILABLE = True


logger.add(
    "inference_logs.log",
    rotation="100 MB",
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

MAX_NUMPY_SEED = 2**32 - 1

MODEL_CONFIG = {
    "Wan_14b": {
        "_class_name": "WanModel",
        "_diffusers_version": "0.33.0",
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
}
# 模型列表缓存（避免每次从 HF 获取）
HF_MODELS_CACHE = {
    "lightx2v/Wan2.1-Distill-Models": [],
    "lightx2v/Wan2.1-Official-Models": [],
    "lightx2v/Wan2.2-Distill-Models": [],
    "lightx2v/Wan2.2-Official-Models": [],
    "lightx2v/Encoders": [],
    "lightx2v/Autoencoders": [],
}


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


def load_hf_models_cache():
    """从 Hugging Face 加载模型列表并缓存，如果 HF 超时或失败，则尝试使用 ModelScope"""
    import concurrent.futures

    def process_files(files):
        """处理文件列表，提取模型名称"""
        model_names = []
        seen_dirs = set()
        for file in files:
            # 排除包含comfyui的文件
            if "comfyui" in file.lower():
                continue

            # 如果是顶层文件（不包含路径分隔符）
            if "/" not in file:
                # 只保留safetensors文件
                if file.endswith(".safetensors"):
                    model_names.append(file)
            else:
                # 提取顶层目录名（支持_split目录）
                top_dir = file.split("/")[0]
                if top_dir not in seen_dirs:
                    seen_dirs.add(top_dir)
                    # 支持safetensors文件目录和_split分block存储目录
                    if "_split" in top_dir or any(f.startswith(f"{top_dir}/") and f.endswith(".safetensors") for f in files):
                        model_names.append(top_dir)
        return sorted(set(model_names))

    # 超时时间（秒）
    HF_TIMEOUT = 30

    for repo_id in HF_MODELS_CACHE.keys():
        files = None
        source = None

        # 首先尝试从 Hugging Face 获取（带超时）
        try:
            if HF_AVAILABLE:
                logger.info(f"Loading models from Hugging Face {repo_id}...")
                api = HfApi()

                # 使用线程池执行器设置超时
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(list_repo_files, repo_id=repo_id, repo_type="model")
                    files = future.result(timeout=HF_TIMEOUT)
                    source = "Hugging Face"
                    logger.info(f"Successfully loaded models from Hugging Face {repo_id}")
        except:  # noqa E722
            # 如果 HF 失败，尝试从 ModelScope 获取
            if files is None and MS_AVAILABLE:
                logger.info(f"Loading models from ModelScope {repo_id}...")
                api = HubApi()
                # ModelScope API 获取文件列表
                model_files = api.get_model_files(model_id=repo_id, recursive=True)
                # 提取文件路径
                files = [file["Path"] for file in model_files if file.get("Type") == "blob"]
                source = "ModelScope"

        # 处理文件列表
        if files:
            model_names = process_files(files)
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


def check_model_exists(model_path, model_name):
    """检查模型是否已下载"""
    if not model_path or not os.path.exists(model_path):
        return False

    model_path_full = os.path.join(model_path, model_name)
    return os.path.exists(model_path_full)


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


def get_dit_choices(model_path, model_type="wan2.1", task_type=None, is_distill=None):
    """获取 Diffusion 模型可选项（从 Hugging Face 和本地）

    Args:
        model_path: 本地模型路径
        model_type: "wan2.1" 或 "wan2.2"
        task_type: "i2v" 或 "t2v"，None 表示不过滤任务类型
        is_distill: 是否为 distill 模型，None 表示同时获取 distill 和非 distill
    """
    excluded_keywords = ["vae", "tae", "clip", "t5", "high_noise", "low_noise"]
    fp8_supported = is_fp8_supported_gpu()

    # 根据模型类型和是否 distill 选择仓库
    if model_type == "wan2.1":
        if is_distill is True:
            repo_id = "lightx2v/Wan2.1-Distill-Models"
        elif is_distill is False:
            repo_id = "lightx2v/Wan2.1-Official-Models"
        else:
            # 同时获取两个仓库的模型
            repo_id_distill = "lightx2v/Wan2.1-Distill-Models"
            repo_id_official = "lightx2v/Wan2.1-Official-Models"
            hf_models_distill = get_hf_models(repo_id_distill, prefix_filter="wan2.1") if HF_AVAILABLE else []
            hf_models_official = get_hf_models(repo_id_official, prefix_filter="wan2.1") if HF_AVAILABLE else []
            hf_models = list(set(hf_models_distill + hf_models_official))
            repo_id = None  # 标记为已获取
    else:  # wan2.2
        if is_distill is True:
            repo_id = "lightx2v/Wan2.2-Distill-Models"
        elif is_distill is False:
            repo_id = "lightx2v/Wan2.2-Official-Models"
        else:
            # 同时获取两个仓库的模型
            repo_id_distill = "lightx2v/Wan2.2-Distill-Models"
            repo_id_official = "lightx2v/Wan2.2-Official-Models"
            hf_models_distill = get_hf_models(repo_id_distill, prefix_filter="wan2.2") if HF_AVAILABLE else []
            hf_models_official = get_hf_models(repo_id_official, prefix_filter="wan2.2") if HF_AVAILABLE else []
            hf_models = list(set(hf_models_distill + hf_models_official))
            repo_id = None  # 标记为已获取

    if repo_id:
        hf_models = get_hf_models(repo_id, prefix_filter=model_type) if HF_AVAILABLE else []

    # 筛选符合条件的模型
    def is_valid(name):
        name_lower = name.lower()
        # 过滤掉包含comfyui的文件
        if "comfyui" in name_lower:
            return False
        # 检查模型类型
        if model_type == "wan2.1":
            if "wan2.1" not in name_lower:
                return False
        else:
            if "wan2.2" not in name_lower:
                return False
        # 检查任务类型（如果指定）
        if task_type:
            if task_type.lower() not in name_lower:
                return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return not any(kw in name_lower for kw in excluded_keywords)

    # 筛选 HF 模型：只保留safetensors文件或_split目录
    valid_hf_models = []
    for m in hf_models:
        if not is_valid(m):
            continue
        # 如果是safetensors文件，或者包含_split的目录，则保留
        if m.endswith(".safetensors") or "_split" in m.lower():
            valid_hf_models.append(m)

    # 检查本地已存在的模型（只检索 safetensors 文件和目录，包括_split目录）
    contents = scan_model_path_contents(model_path)
    dir_choices = [d for d in contents["dirs"] if is_valid(d) and ("_split" in d.lower() or d in contents["safetensors_dirs"])]
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid(f)]
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if is_valid(d)]
    local_models = dir_choices + safetensors_choices + safetensors_dir_choices

    # 合并 HF 和本地模型，去重
    all_models = sorted(set(valid_hf_models + local_models))

    # 格式化选项，添加下载状态（✅ 已下载，❌ 未下载）
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_high_noise_choices(model_path, model_type="wan2.2", task_type=None, is_distill=None):
    """获取高噪模型可选项（从 Hugging Face 和本地，包含 high_noise 的文件/目录）

    Args:
        model_path: 本地模型路径
        model_type: "wan2.2"（高噪模型只用于wan2.2）
        task_type: "i2v" 或 "t2v"，None 表示不过滤任务类型
        is_distill: 是否为 distill 模型，None 表示同时获取 distill 和非 distill
    """
    fp8_supported = is_fp8_supported_gpu()

    # 根据是否 distill 选择仓库
    if is_distill is True:
        repo_id = "lightx2v/Wan2.2-Distill-Models"
    elif is_distill is False:
        repo_id = "lightx2v/Wan2.2-Official-Models"
    else:
        # 同时获取两个仓库的模型
        repo_id_distill = "lightx2v/Wan2.2-Distill-Models"
        repo_id_official = "lightx2v/Wan2.2-Official-Models"
        hf_models_distill = get_hf_models(repo_id_distill, keyword_filter="high_noise") if HF_AVAILABLE else []
        hf_models_official = get_hf_models(repo_id_official, keyword_filter="high_noise") if HF_AVAILABLE else []
        hf_models = list(set(hf_models_distill + hf_models_official))
        repo_id = None

    if repo_id:
        hf_models = get_hf_models(repo_id, keyword_filter="high_noise") if HF_AVAILABLE else []

    def is_valid(name):
        name_lower = name.lower()
        # 过滤掉包含comfyui的文件
        if "comfyui" in name_lower:
            return False
        # 检查模型类型
        if model_type.lower() not in name_lower:
            return False
        # 检查任务类型（如果指定）
        if task_type:
            if task_type.lower() not in name_lower:
                return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return "high_noise" in name_lower or "high-noise" in name_lower

    # 筛选 HF 模型：只保留safetensors文件或_split目录
    valid_hf_models = []
    for m in hf_models:
        if not is_valid(m):
            continue
        # 如果是safetensors文件，或者包含_split的目录，则保留
        if m.endswith(".safetensors") or "_split" in m.lower():
            valid_hf_models.append(m)

    # 检查本地已存在的模型（只检索 safetensors 文件和目录，包括_split目录）
    contents = scan_model_path_contents(model_path)
    dir_choices = [d for d in contents["dirs"] if is_valid(d) and ("_split" in d.lower() or d in contents["safetensors_dirs"])]
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid(f)]
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if is_valid(d)]
    local_models = dir_choices + safetensors_choices + safetensors_dir_choices

    # 合并 HF 和本地模型，去重
    all_models = sorted(set(valid_hf_models + local_models))

    # 格式化选项，添加下载状态（✅ 已下载，❌ 未下载）
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_low_noise_choices(model_path, model_type="wan2.2", task_type=None, is_distill=None):
    """获取低噪模型可选项（从 Hugging Face 和本地，包含 low_noise 的文件/目录）

    Args:
        model_path: 本地模型路径
        model_type: "wan2.2"（低噪模型只用于wan2.2）
        task_type: "i2v" 或 "t2v"，None 表示不过滤任务类型
        is_distill: 是否为 distill 模型，None 表示同时获取 distill 和非 distill
    """
    fp8_supported = is_fp8_supported_gpu()

    # 根据是否 distill 选择仓库
    if is_distill is True:
        repo_id = "lightx2v/Wan2.2-Distill-Models"
    elif is_distill is False:
        repo_id = "lightx2v/Wan2.2-Official-Models"
    else:
        # 同时获取两个仓库的模型
        repo_id_distill = "lightx2v/Wan2.2-Distill-Models"
        repo_id_official = "lightx2v/Wan2.2-Official-Models"
        hf_models_distill = get_hf_models(repo_id_distill, keyword_filter="low_noise") if HF_AVAILABLE else []
        hf_models_official = get_hf_models(repo_id_official, keyword_filter="low_noise") if HF_AVAILABLE else []
        hf_models = list(set(hf_models_distill + hf_models_official))
        repo_id = None

    if repo_id:
        hf_models = get_hf_models(repo_id, keyword_filter="low_noise") if HF_AVAILABLE else []

    def is_valid(name):
        name_lower = name.lower()
        # 过滤掉包含comfyui的文件
        if "comfyui" in name_lower:
            return False
        # 检查模型类型
        if model_type.lower() not in name_lower:
            return False
        # 检查任务类型（如果指定）
        if task_type:
            if task_type.lower() not in name_lower:
                return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return "low_noise" in name_lower or "low-noise" in name_lower

    # 筛选 HF 模型：只保留safetensors文件或_split目录
    valid_hf_models = []
    for m in hf_models:
        if not is_valid(m):
            continue
        # 如果是safetensors文件，或者包含_split的目录，则保留
        if m.endswith(".safetensors") or "_split" in m.lower():
            valid_hf_models.append(m)

    # 检查本地已存在的模型（只检索 safetensors 文件和目录，包括_split目录）
    contents = scan_model_path_contents(model_path)
    dir_choices = [d for d in contents["dirs"] if is_valid(d) and ("_split" in d.lower() or d in contents["safetensors_dirs"])]
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid(f)]
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if is_valid(d)]
    local_models = dir_choices + safetensors_choices + safetensors_dir_choices

    # 合并 HF 和本地模型，去重
    all_models = sorted(set(valid_hf_models + local_models))

    # 格式化选项，添加下载状态（✅ 已下载，❌ 未下载）
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_t5_model_choices(model_path):
    """获取 T5 模型可选项（从 Hugging Face Encoders 仓库和本地，包含 t5 关键字，只显示 safetensors，排除 google）"""
    fp8_supported = is_fp8_supported_gpu()

    # 从 Hugging Face Encoders 仓库获取
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # 筛选包含 t5 的文件，只显示 safetensors，排除 google
    def is_valid_hf(name):
        name_lower = name.lower()
        # 过滤掉包含comfyui的文件和 google 目录
        if "comfyui" in name_lower or name == "google":
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # 只显示 safetensors 文件
        return ("t5" in name_lower) and name.endswith(".safetensors")

    valid_hf_models = [m for m in hf_models if is_valid_hf(m)]

    # 检查本地已存在的模型
    contents = scan_model_path_contents(model_path)

    def is_valid_local(name):
        name_lower = name.lower()
        # 过滤掉包含comfyui的文件和 google 目录
        if "comfyui" in name_lower or name == "google":
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # 只显示 safetensors 文件
        return ("t5" in name_lower) and name.endswith(".safetensors")

    # 只从 .safetensors 文件中筛选
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid_local(f)]

    local_models = safetensors_choices

    # 合并 HF 和本地模型，去重
    all_models = sorted(set(valid_hf_models + local_models))

    # 格式化选项，添加下载状态（✅ 已下载，❌ 未下载）
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_t5_tokenizer_choices(model_path):
    """获取 T5 Tokenizer 可选项（google 目录）"""
    # 只返回 google 目录
    contents = scan_model_path_contents(model_path)
    dir_choices = ["google"] if "google" in contents["dirs"] else []

    # 从 HF 获取
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []
    hf_google = ["google"] if "google" in hf_models else []

    all_models = sorted(set(hf_google + dir_choices))
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_clip_model_choices(model_path):
    """获取 CLIP 模型可选项（从 Hugging Face Encoders 仓库和本地，包含 clip 关键字，只显示 safetensors，排除 xlm-roberta-large）"""
    fp8_supported = is_fp8_supported_gpu()

    # 从 Hugging Face Encoders 仓库获取
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # 筛选包含 clip 的文件，只显示 safetensors，排除 xlm-roberta-large
    def is_valid_hf(name):
        name_lower = name.lower()
        # 过滤掉包含comfyui的文件和 xlm-roberta-large 目录
        if "comfyui" in name_lower or name == "xlm-roberta-large":
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # 只显示 safetensors 文件
        return ("clip" in name_lower) and name.endswith(".safetensors")

    valid_hf_models = [m for m in hf_models if is_valid_hf(m)]

    # 检查本地已存在的模型
    contents = scan_model_path_contents(model_path)

    def is_valid_local(name):
        name_lower = name.lower()
        # 过滤掉包含comfyui的文件和 xlm-roberta-large 目录
        if "comfyui" in name_lower or name == "xlm-roberta-large":
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # 只显示 safetensors 文件
        return ("clip" in name_lower) and name.endswith(".safetensors")

    # 只从 .safetensors 文件中筛选
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid_local(f)]

    local_models = safetensors_choices

    # 合并 HF 和本地模型，去重
    all_models = sorted(set(valid_hf_models + local_models))

    # 格式化选项，添加下载状态（✅ 已下载，❌ 未下载）
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_clip_tokenizer_choices(model_path):
    """获取 CLIP Tokenizer 可选项（xlm-roberta-large 目录）"""
    # 只返回 xlm-roberta-large 目录
    contents = scan_model_path_contents(model_path)
    dir_choices = ["xlm-roberta-large"] if "xlm-roberta-large" in contents["dirs"] else []

    # 从 HF 获取
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []
    hf_xlm = ["xlm-roberta-large"] if "xlm-roberta-large" in hf_models else []

    all_models = sorted(set(hf_xlm + dir_choices))
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_vae_encoder_choices(model_path):
    """获取 VAE 编码器可选项，只返回 Wan2.1_VAE.safetensors"""
    encoder_name = "Wan2.1_VAE.safetensors"

    # 从 Hugging Face Autoencoders 仓库获取
    repo_id = "lightx2v/Autoencoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # 检查HF中是否有该文件
    hf_has = encoder_name in hf_models

    # 检查本地是否存在
    local_has = check_model_exists(model_path, encoder_name)

    # 如果HF或本地有，则返回
    if hf_has or local_has:
        return [format_model_choice(encoder_name, model_path)]
    else:
        return [format_model_choice(encoder_name, model_path)]


def get_vae_decoder_choices(model_path):
    """获取 VAE 解码器可选项（从 Hugging Face Autoencoders 仓库和本地，包含 vae/VAE/tae 关键字，只显示 safetensors）"""
    fp8_supported = is_fp8_supported_gpu()

    # 从 Hugging Face Autoencoders 仓库获取
    repo_id = "lightx2v/Autoencoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # 筛选包含 vae 或 tae 的文件，只显示 safetensors文件或_split目录
    def is_valid_hf(name):
        name_lower = name.lower()
        # 过滤掉包含comfyui的文件
        if "comfyui" in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # 只显示 safetensors 文件或_split目录，必须包含 vae 或 tae
        return any(kw in name_lower for kw in ["vae", "tae", "lightvae", "lighttae"]) and (name.endswith(".safetensors") or "_split" in name_lower)

    valid_hf_models = [m for m in hf_models if is_valid_hf(m)]

    # 检查本地已存在的模型
    contents = scan_model_path_contents(model_path)

    def is_valid_local(name):
        name_lower = name.lower()
        # 过滤掉包含comfyui的文件
        if "comfyui" in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # 只显示 safetensors 文件或_split目录，必须包含 vae 或 tae
        if not any(kw in name_lower for kw in ["vae", "tae", "lightvae", "lighttae"]):
            return False
        # 如果是文件，必须是safetensors
        if os.path.isfile(os.path.join(model_path, name)):
            return name.endswith(".safetensors")
        # 如果是目录，必须是包含safetensors的目录或_split目录
        return name in contents["safetensors_dirs"] or "_split" in name_lower

    # 从 .safetensors 文件中筛选
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid_local(f)]

    # 从包含 safetensors 的目录中筛选（包括_split目录）
    dir_choices = [d for d in contents["dirs"] if is_valid_local(d)]

    local_models = safetensors_choices + dir_choices

    # 合并 HF 和本地模型，去重
    all_models = sorted(set(valid_hf_models + local_models))

    # 对于VAE解码器，只显示包含"2_1"或"2.1"的选项
    all_models = [m for m in all_models if "2_1" in m or "2.1" in m]

    # 格式化选项，添加下载状态（✅ 已下载，❌ 未下载）
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


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


def download_model_from_hf(repo_id, model_name, model_path, progress=gr.Progress()):
    """从 Hugging Face 下载模型（支持文件和目录）"""
    if not HF_AVAILABLE:
        return f"❌ huggingface_hub 未安装，无法下载模型"

    progress(0, desc=f"开始从 Hugging Face 下载 {model_name}...")
    logger.info(f"开始从 Hugging Face {repo_id} 下载 {model_name} 到 {model_path}")

    target_path = os.path.join(model_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    import shutil

    # 判断是文件还是目录：如果名字不是 .safetensors 或 .pth 结尾，就是目录，否则就是单文件
    is_directory = not (model_name.endswith(".safetensors") or model_name.endswith(".pth"))

    if is_directory:
        # 下载目录
        progress(0.1, desc=f"下载目录 {model_name}...")
        logger.info(f"检测到 {model_name} 是目录，使用 snapshot_download")

        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hf_snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"{model_name}/**"],
                local_dir=model_path,
                local_dir_use_symlinks=False,
                repo_type="model",
            )

        # 移动文件到正确位置
        repo_name = repo_id.split("/")[-1]
        source_dir = os.path.join(model_path, repo_name, model_name)
        if os.path.exists(source_dir):
            shutil.move(source_dir, target_path)
            repo_dir = os.path.join(model_path, repo_name)
            if os.path.exists(repo_dir) and not os.listdir(repo_dir):
                os.rmdir(repo_dir)
        else:
            source_dir = os.path.join(model_path, model_name)
            if os.path.exists(source_dir) and source_dir != target_path:
                shutil.move(source_dir, target_path)

        logger.info(f"目录 {model_name} 下载完成，已移动到 {target_path}")
    else:
        # 下载文件
        progress(0.1, desc=f"下载文件 {model_name}...")
        logger.info(f"检测到 {model_name} 是文件，使用 hf_hub_download")

        if os.path.exists(target_path):
            os.remove(target_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                repo_type="model",
            )
        logger.info(f"文件 {model_name} 下载完成，保存到 {downloaded_path}")

    progress(1.0, desc=f"✅ {model_name} 下载完成")
    return f"✅ {model_name} 下载完成"


def download_model_from_ms(repo_id, model_name, model_path, progress=gr.Progress()):
    """从 ModelScope 下载模型（支持文件和目录）"""
    if not MS_AVAILABLE:
        return f"❌ modelscope 未安装，无法下载模型"

    progress(0, desc=f"开始从 ModelScope 下载 {model_name}...")
    logger.info(f"开始从 ModelScope {repo_id} 下载 {model_name} 到 {model_path}")

    target_path = os.path.join(model_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    import shutil

    # 判断是文件还是目录：如果名字不是 .safetensors 或 .pth 结尾，就是目录，否则就是单文件
    is_directory = not (model_name.endswith(".safetensors") or model_name.endswith(".pth"))
    is_file = not is_directory

    # 临时目录用于下载
    temp_dir = os.path.join(model_path, f".temp_{model_name}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # 处理目录下载
    if is_directory:
        progress(0.1, desc=f"下载目录 {model_name}...")
        logger.info(f"检测到 {model_name} 是目录，使用 snapshot_download")

        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        # 使用 snapshot_download 下载目录
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downloaded_path = ms_snapshot_download(
                model_id=repo_id,
                cache_dir=temp_dir,
                allow_patterns=[f"{model_name}/**"],
            )

        # 移动文件到目标位置
        source_dir = os.path.join(downloaded_path, model_name)
        if not os.path.exists(source_dir) and os.path.exists(downloaded_path):
            # 如果找不到，尝试从下载路径中查找
            for item in os.listdir(downloaded_path):
                item_path = os.path.join(downloaded_path, item)
                if model_name in item or os.path.isdir(item_path):
                    source_dir = item_path
                    break

        if os.path.exists(source_dir):
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.move(source_dir, target_path)

        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        logger.info(f"目录 {model_name} 下载完成，保存到 {target_path}")
    # 处理文件下载
    elif is_file:
        progress(0.1, desc=f"下载文件 {model_name}...")
        logger.info(f"检测到 {model_name} 是文件，使用 snapshot_download")

        if os.path.exists(target_path):
            os.remove(target_path)
        os.makedirs(os.path.dirname(target_path) if "/" in model_name else model_path, exist_ok=True)

        # 使用 snapshot_download 下载文件
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downloaded_path = ms_snapshot_download(
                model_id=repo_id,
                cache_dir=temp_dir,
                allow_patterns=[model_name],
            )

        # 查找并移动文件
        source_file = os.path.join(downloaded_path, model_name)
        if not os.path.exists(source_file):
            # 如果找不到，尝试从下载路径中查找
            for root, dirs, files_list in os.walk(downloaded_path):
                if model_name in files_list:
                    source_file = os.path.join(root, model_name)
                    break

        if os.path.exists(source_file):
            shutil.move(source_file, target_path)

        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        logger.info(f"文件 {model_name} 下载完成，保存到 {target_path}")
    else:
        return f"❌ 无法找到 {model_name}：既不是文件也不是目录"

    progress(1.0, desc=f"✅ {model_name} 下载完成")
    return f"✅ {model_name} 下载完成"


def download_model(repo_id, model_name, model_path, download_source="huggingface", progress=gr.Progress()):
    """统一的下载函数，根据下载源选择 Hugging Face 或 ModelScope"""
    if download_source == "modelscope":
        return download_model_from_ms(repo_id, model_name, model_path, progress)
    else:
        return download_model_from_hf(repo_id, model_name, model_path, progress)


def get_model_status(model_path, model_name, repo_id):
    """获取模型状态（已下载/未下载）"""
    exists = check_model_exists(model_path, model_name)
    if exists:
        return "✅ 已下载", gr.update(visible=False)
    else:
        return "❌ 未下载", gr.update(visible=True)


def update_model_path_options(model_path, model_type="wan2.1", task_type=None):
    """当 model_path 或 model_type 改变时，更新所有模型路径选择器"""
    dit_choices = get_dit_choices(model_path, model_type, task_type)
    high_noise_choices = get_high_noise_choices(model_path, model_type, task_type)
    low_noise_choices = get_low_noise_choices(model_path, model_type, task_type)
    t5_model_choices = get_t5_model_choices(model_path)
    t5_tokenizer_choices = get_t5_tokenizer_choices(model_path)
    clip_model_choices = get_clip_model_choices(model_path)
    clip_tokenizer_choices = get_clip_tokenizer_choices(model_path)
    vae_encoder_choices = get_vae_encoder_choices(model_path)
    vae_decoder_choices = get_vae_decoder_choices(model_path)

    return (
        gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else ""),
        gr.update(choices=high_noise_choices, value=high_noise_choices[0] if high_noise_choices else ""),
        gr.update(choices=low_noise_choices, value=low_noise_choices[0] if low_noise_choices else ""),
        gr.update(choices=t5_model_choices, value=t5_model_choices[0] if t5_model_choices else ""),
        gr.update(choices=t5_tokenizer_choices, value=t5_tokenizer_choices[0] if t5_tokenizer_choices else ""),
        gr.update(choices=clip_model_choices, value=clip_model_choices[0] if clip_model_choices else ""),
        gr.update(choices=clip_tokenizer_choices, value=clip_tokenizer_choices[0] if clip_tokenizer_choices else ""),
        gr.update(choices=vae_encoder_choices, value=vae_encoder_choices[0] if vae_encoder_choices else ""),
        gr.update(choices=vae_decoder_choices, value=vae_decoder_choices[0] if vae_decoder_choices else ""),
    )


def generate_random_seed():
    return random.randint(0, MAX_NUMPY_SEED)


def is_module_installed(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def get_available_quant_ops():
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
    with torch.cuda.device(gpu_idx):
        memory_info = torch.cuda.mem_get_info()
        total_memory = memory_info[1] / (1024**3)  # Convert bytes to GB
        return total_memory


def get_cpu_memory():
    available_bytes = psutil.virtual_memory().available
    return available_bytes / 1024**3


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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


def is_sm_greater_than_90():
    """检测计算能力是否大于 (9,0)"""
    if not torch.cuda.is_available():
        return False
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    return (major, minor) > (9, 0)


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


def is_distill_model_from_name(model_name):
    """根据模型名称判断是否是 distill 模型"""
    if not model_name:
        return None
    return "4step" in model_name.lower()


def get_repo_id_for_model(model_type, is_distill, model_category="dit"):
    """根据模型类型、是否 distill 和模型类别获取对应的 Hugging Face 仓库 ID"""
    if model_category == "dit":
        if model_type == "wan2.1":
            return "lightx2v/Wan2.1-Distill-Models" if is_distill else "lightx2v/Wan2.1-Official-Models"
        else:  # wan2.2
            return "lightx2v/Wan2.2-Distill-Models" if is_distill else "lightx2v/Wan2.2-Official-Models"
    elif model_category == "high_noise" or model_category == "low_noise":
        if is_distill:
            return "lightx2v/Wan2.2-Distill-Models"
        else:
            return "lightx2v/Wan2.2-Official-Models"
    elif model_category == "t5" or model_category == "clip":
        return "lightx2v/Encoders"
    elif model_category == "vae":
        return "lightx2v/Autoencoders"
    return None


global_runner = None
current_config = None
cur_dit_path = None
cur_t5_path = None
cur_clip_path = None

available_quant_ops = get_available_quant_ops()
quant_op_choices = []
for op_name, is_installed in available_quant_ops:
    status_text = "✅" if is_installed else "❌"
    display_text = f"{status_text}{op_name} "
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
        status_text = "✅"
        display_text = f"{status_text}{op_name}"
        attn_op_choices.append((op_name, display_text))

# 再添加未安装的（按优先级）
for op_name in attn_priority:
    if op_name in attn_op_dict and not attn_op_dict[op_name]:
        status_text = "❌"
        display_text = f"{status_text}{op_name}"
        attn_op_choices.append((op_name, display_text))

# 添加其他不在优先级列表中的算子（已安装的在前）
other_ops = [(op_name, is_installed) for op_name, is_installed in available_attn_ops if op_name not in attn_priority]
for op_name, is_installed in sorted(other_ops, key=lambda x: not x[1]):  # 已安装的在前
    status_text = "✅" if is_installed else "❌"
    display_text = f"{status_text}{op_name}"
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
    vae_encoder_path_input,
    vae_decoder_path_input,
    image_path=None,
):
    cleanup_memory()

    # 提取原始操作符名称（去掉状态标识 ✅/❌）
    def extract_op_name(op_str):
        """从格式化的操作符名称中提取原始名称"""
        if not op_str:
            return ""
        # 移除开头的 ✅ 或 ❌
        op_str = op_str.strip()
        if op_str.startswith("✅"):
            op_str = op_str[1:].strip()
        elif op_str.startswith("❌"):
            op_str = op_str[1:].strip()
        # 移除括号后的内容（如果有）
        if "(" in op_str:
            op_str = op_str.split("(")[0].strip()
        return op_str

    quant_op = extract_op_name(quant_op)
    attention_type = extract_op_name(attention_type)

    global global_runner, current_config, model_path, model_cls
    global cur_dit_path, cur_t5_path, cur_clip_path

    # 提取原始模型名称（去掉状态标识）
    dit_path_input = extract_model_name(dit_path_input) if dit_path_input else ""
    high_noise_path_input = extract_model_name(high_noise_path_input) if high_noise_path_input else ""
    low_noise_path_input = extract_model_name(low_noise_path_input) if low_noise_path_input else ""
    t5_path_input = extract_model_name(t5_path_input) if t5_path_input else ""
    # Tokenizer 固定名称
    t5_tokenizer_path_input = "google"
    clip_path_input = extract_model_name(clip_path_input) if clip_path_input else ""
    clip_tokenizer_path_input = "xlm-roberta-large"
    vae_encoder_path_input = extract_model_name(vae_encoder_path_input) if vae_encoder_path_input else ""
    vae_decoder_path_input = extract_model_name(vae_decoder_path_input) if vae_decoder_path_input else ""

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

    model_config = MODEL_CONFIG["Wan_14b"]

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

    # 处理 quant_op：如果是 torch，需要根据量化类型转换为 torchao
    def get_quant_scheme(quant_detected, quant_op_val):
        """根据量化类型和算子生成 quant_scheme"""
        if quant_op_val == "torch":
            # torch 选项需要转换为 torchao，格式为 fp8-torchao 或 int8-torchao
            return f"{quant_detected}-torchao"
        elif quant_op_val == "triton":
            # triton 选项格式为 fp8-triton 或 int8-triton
            return f"{quant_detected}-triton"
        else:
            return f"{quant_detected}-{quant_op_val}"

    if is_dit_quant:
        dit_quant_scheme = get_quant_scheme(dit_quant_detected, quant_op)
        if "wan2.1" in model_cls:
            dit_quantized_ckpt = os.path.join(model_path, dit_path_input)
        else:
            high_noise_quantized_ckpt = os.path.join(model_path, high_noise_path_input)
            low_noise_quantized_ckpt = os.path.join(model_path, low_noise_path_input)
    else:
        dit_quant_scheme = "Default"
        if "wan2.1" in model_cls:
            dit_original_ckpt = os.path.join(model_path, dit_path_input)
        else:
            high_noise_original_ckpt = os.path.join(model_path, high_noise_path_input)
            low_noise_original_ckpt = os.path.join(model_path, low_noise_path_input)

    # 使用前端选择的 T5 路径
    if is_t5_quant:
        t5_quantized_ckpt = os.path.join(model_path, t5_path_input)
        t5_quant_scheme = get_quant_scheme(t5_quant_detected, quant_op)
        t5_original_ckpt = None
    else:
        t5_quantized_ckpt = None
        t5_quant_scheme = None
        t5_original_ckpt = os.path.join(model_path, t5_path_input)

    # 使用前端选择的 CLIP 路径
    if is_clip_quant:
        clip_quantized_ckpt = os.path.join(model_path, clip_path_input)
        clip_quant_scheme = get_quant_scheme(clip_quant_detected, quant_op)
        clip_original_ckpt = None
    else:
        clip_quantized_ckpt = None
        clip_quant_scheme = None
        clip_original_ckpt = os.path.join(model_path, clip_path_input)

    if model_type_input == "wan2.1":
        current_dit_path = dit_path_input
    else:
        current_dit_path = f"{high_noise_path_input}|{low_noise_path_input}" if high_noise_path_input and low_noise_path_input else None

    current_t5_path = f"{t5_path_input}|{t5_tokenizer_path_input}" if t5_path_input and t5_tokenizer_path_input else t5_path_input
    # CLIP 路径：只有在 wan2.1 且 i2v 时才需要
    if model_type_input == "wan2.1" and task_type_input == "i2v":
        current_clip_path = f"{clip_path_input}|{clip_tokenizer_path_input}" if clip_path_input and clip_tokenizer_path_input else clip_path_input
    else:
        current_clip_path = None

    needs_reinit = lazy_load or unload_modules or global_runner is None or cur_dit_path != current_dit_path or cur_t5_path != current_t5_path or cur_clip_path != current_clip_path
    if cfg_scale == 1:
        enable_cfg = False
    else:
        enable_cfg = True

    # VAE 配置：根据解码器路径判断
    vae_encoder_path = vae_encoder_path_input if vae_encoder_path_input else "Wan2.1_VAE.safetensors"
    vae_decoder_path = vae_decoder_path_input if vae_decoder_path_input else None

    vae_decoder_name_lower = vae_decoder_path.lower() if vae_decoder_path else ""
    use_tae = "tae" in vae_decoder_name_lower or "lighttae" in vae_decoder_name_lower
    use_lightvae = "lightvae" in vae_decoder_name_lower
    need_scaled = "lighttae" in vae_decoder_name_lower

    # 根据 use_tae 设置 vae_path 和 tae_path
    if use_tae:
        # use_tae=True 时：tae_path 为解码器路径，vae_path 为编码器路径
        tae_path = os.path.join(model_path, vae_decoder_path) if vae_decoder_path else None
        vae_path = os.path.join(model_path, vae_encoder_path) if vae_encoder_path else None
    else:
        # 其他情况：vae_path 为解码器路径，tae_path 为 None
        vae_path = os.path.join(model_path, vae_decoder_path) if vae_decoder_path else None
        tae_path = None

    logger.info(
        f"VAE 配置 - use_tae: {use_tae}, use_lightvae: {use_lightvae}, need_scaled: {need_scaled} (VAE编码器: {vae_encoder_path}, VAE解码器: {vae_decoder_path}, vae_path: {vae_path}, tae_path: {tae_path})"
    )

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
        "vae_path": vae_path,
        "tae_path": tae_path,
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
        input_info = set_input_info(args)

    runner.run_pipeline(input_info)
    cleanup_memory()

    return save_result_path


def handle_lazy_load_change(lazy_load_enabled):
    """Handle lazy_load checkbox change to automatically enable unload_modules"""
    return gr.update(value=lazy_load_enabled)


def auto_configure(resolution, num_frames=81):
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

    # If num_frames > 81, set rope_chunk to True regardless of resolution
    if num_frames > 81:
        default_config["rope_chunk_val"] = True

    gpu_memory = round(get_gpu_memory())
    cpu_memory = round(get_cpu_memory())

    attn_priority = ["sage_attn2", "flash_attn3", "flash_attn2", "torch_sdpa"]

    # 如果 sm > (9,0)，且 sage_attn3 可用，将其放到 sage_attn2 后面
    if is_sm_greater_than_90():
        # 检查 sage_attn3 是否可用
        sage3_available = dict(available_attn_ops).get("sage_attn3", False)
        if sage3_available:
            # 找到 sage_attn2 的位置，在其后插入 sage_attn3
            if "sage_attn2" in attn_priority:
                sage2_index = attn_priority.index("sage_attn2")
                if "sage_attn3" not in attn_priority:
                    attn_priority.insert(sage2_index + 1, "sage_attn3")
                else:
                    # 如果已经在列表中，先移除再插入到正确位置
                    attn_priority.remove("sage_attn3")
                    attn_priority.insert(sage2_index + 1, "sage_attn3")
            else:
                # 如果没有 sage_attn2，就添加到最前
                if "sage_attn3" not in attn_priority:
                    attn_priority.insert(0, "sage_attn3")

    quant_op_priority = ["triton", "q8f", "vllm", "sgl", "torch"]

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
        (
            -1,
            {
                "t5_lazy_load": True,
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

    # 如果内存小于8GB，抛出异常
    if cpu_memory < 8:
        raise Exception(
            f"系统内存不足：当前可用内存为 {cpu_memory:.1f}GB，至少需要 8GB 内存才能正常运行。\n"
            f"建议解决方案：\n"
            f"1. 检查您的机器配置，确保有足够的内存\n"
            f"2. 使用量化模型（fp8/int8）以降低内存占用\n"
            f"3. 使用更小的模型进行推理"
        )

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


def main():
    # 在启动时加载 Hugging Face 模型列表缓存
    logger.info("正在加载 Hugging Face 模型列表缓存...")
    load_hf_models_cache()
    logger.info("模型列表缓存加载完成")

    with gr.Blocks(title="Lightx2v (轻量级视频推理和生成引擎)") as demo:
        gr.Markdown(f"# 🎬 LightX2V 视频生成器")
        gr.HTML(f"<style>{css}</style>")
        # 主布局：左右分栏
        with gr.Row():
            # 左侧：配置和输入区域
            with gr.Column(scale=5):
                # 模型配置区域
                with gr.Accordion("🗂️ 模型配置", open=True, elem_classes=["model-config"]):
                    gr.Markdown("💡 **提示**：请确保以下每个模型选项至少有一个已下载✅的模型可用，否则可能无法正常生成视频。")
                    # FP8 支持提示
                    if not is_fp8_supported_gpu():
                        gr.Markdown("⚠️ **您的设备不支持fp8推理**，已自动隐藏包含fp8的模型选项。")

                    # 隐藏的状态组件
                    model_path_input = gr.Textbox(value=model_path, visible=False)

                    # 模型类型 + 任务类型 + 下载源
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
                        download_source_input = gr.Radio(
                            label="📥 下载源",
                            choices=["huggingface", "modelscope"] if (HF_AVAILABLE and MS_AVAILABLE) else (["huggingface"] if HF_AVAILABLE else ["modelscope"] if MS_AVAILABLE else []),
                            value="huggingface" if HF_AVAILABLE else ("modelscope" if MS_AVAILABLE else None),
                            info="选择模型下载源",
                            visible=HF_AVAILABLE or MS_AVAILABLE,
                            elem_classes=["horizontal-radio"],
                        )

                    # wan2.1：Diffusion模型（美化布局）
                    with gr.Column(elem_classes=["diffusion-model-group"]) as wan21_row:
                        with gr.Row():
                            with gr.Column(scale=5):
                                dit_choices_init = get_dit_choices(model_path, "wan2.1", "i2v")
                                dit_path_input = gr.Dropdown(
                                    label="🎨 Diffusion模型",
                                    choices=dit_choices_init,
                                    value=dit_choices_init[0] if dit_choices_init else "",
                                    allow_custom_value=True,
                                    visible=True,
                                )
                            with gr.Column(scale=1, min_width=150):
                                dit_download_btn = gr.Button("📥 下载", visible=False, size="sm", variant="secondary")
                        dit_download_status = gr.Markdown("", visible=False)

                    # wan2.2 专用：高噪模型 + 低噪模型（默认隐藏）
                    with gr.Row(visible=False, elem_classes=["wan22-row"]) as wan22_row:
                        with gr.Column(scale=1):
                            high_noise_choices_init = get_high_noise_choices(model_path, "wan2.2", "i2v")
                            high_noise_path_input = gr.Dropdown(
                                label="🔊 高噪模型",
                                choices=high_noise_choices_init,
                                value=high_noise_choices_init[0] if high_noise_choices_init else "",
                                allow_custom_value=True,
                            )
                            high_noise_download_btn = gr.Button("📥 下载", visible=False, size="sm", variant="secondary")
                            high_noise_download_status = gr.Markdown("", visible=False)
                        with gr.Column(scale=1):
                            low_noise_choices_init = get_low_noise_choices(model_path, "wan2.2", "i2v")
                            low_noise_path_input = gr.Dropdown(
                                label="🔇 低噪模型",
                                choices=low_noise_choices_init,
                                value=low_noise_choices_init[0] if low_noise_choices_init else "",
                                allow_custom_value=True,
                            )
                            low_noise_download_btn = gr.Button("📥 下载", visible=False, size="sm", variant="secondary")
                            low_noise_download_status = gr.Markdown("", visible=False)

                    # 文本编码器（模型 + Tokenizer）
                    with gr.Row():
                        with gr.Column(scale=1):
                            t5_model_choices_init = get_t5_model_choices(model_path)
                            t5_path_input = gr.Dropdown(
                                label="📝 文本编码器",
                                choices=t5_model_choices_init,
                                value=t5_model_choices_init[0] if t5_model_choices_init else "",
                                allow_custom_value=True,
                            )
                            t5_download_btn = gr.Button("📥 下载", visible=False, size="sm", variant="secondary")
                            t5_download_status = gr.Markdown("", visible=False)
                        with gr.Column(scale=1):
                            t5_tokenizer_hint = gr.Dropdown(
                                label="📝 文本编码器 Tokenizer",
                                choices=["google ✅", "google ❌"],
                                value="google ❌",
                                interactive=False,
                            )
                            t5_tokenizer_download_btn = gr.Button("📥 下载", visible=False, size="sm", variant="secondary")
                            t5_tokenizer_download_status = gr.Markdown("", visible=False)

                    # 图像编码器（模型 + Tokenizer，条件显示）
                    with gr.Row(visible=True) as clip_row:
                        with gr.Column(scale=1):
                            clip_model_choices_init = get_clip_model_choices(model_path)
                            clip_path_input = gr.Dropdown(
                                label="🖼️ 图像编码器",
                                choices=clip_model_choices_init,
                                value=clip_model_choices_init[0] if clip_model_choices_init else "",
                                allow_custom_value=True,
                            )
                            clip_download_btn = gr.Button("📥 下载", visible=False, size="sm", variant="secondary")
                            clip_download_status = gr.Markdown("", visible=False)
                        with gr.Column(scale=1):
                            clip_tokenizer_hint = gr.Dropdown(
                                label="🖼️ 图像编码器 Tokenizer",
                                choices=["xlm-roberta-large ✅", "xlm-roberta-large ❌"],
                                value="xlm-roberta-large ❌",
                                interactive=False,
                            )
                            clip_tokenizer_download_btn = gr.Button("📥 下载", visible=False, size="sm", variant="secondary")
                            clip_tokenizer_download_status = gr.Markdown("", visible=False)

                    # VAE（编码器 + 解码器）
                    with gr.Row() as vae_row:
                        with gr.Column(scale=1) as vae_encoder_col:
                            vae_encoder_choices_init = get_vae_encoder_choices(model_path)
                            vae_encoder_path_input = gr.Dropdown(
                                label="VAE编码器",
                                choices=vae_encoder_choices_init,
                                value=vae_encoder_choices_init[0] if vae_encoder_choices_init else "",
                                allow_custom_value=True,
                                interactive=True,
                            )
                            vae_encoder_download_btn = gr.Button("📥 下载", visible=False, size="sm", variant="secondary")
                            vae_encoder_download_status = gr.Markdown("", visible=False)
                        with gr.Column(scale=1) as vae_decoder_col:
                            vae_decoder_choices_init = get_vae_decoder_choices(model_path)
                            vae_decoder_path_input = gr.Dropdown(
                                label="VAE解码器",
                                choices=vae_decoder_choices_init,
                                value=vae_decoder_choices_init[0] if vae_decoder_choices_init else "",
                                allow_custom_value=True,
                            )
                            vae_decoder_download_btn = gr.Button("📥 下载", visible=False, size="sm", variant="secondary")
                            vae_decoder_download_status = gr.Markdown("", visible=False)

                    # 注意力算子和量化矩阵乘法算子
                    with gr.Row():
                        attention_type = gr.Dropdown(
                            label="⚡ 注意力算子",
                            choices=[op[1] for op in attn_op_choices],
                            value=attn_op_choices[0][1] if attn_op_choices else "",
                            info="使用适当的注意力算子加速推理",
                        )
                        quant_op = gr.Dropdown(
                            label="⚡矩阵乘法算子",
                            choices=[op[1] for op in quant_op_choices],
                            value=quant_op_choices[0][1],
                            info="选择低精度矩阵乘法算子以加速推理",
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

                    # 任务类型切换事件
                    def on_task_type_change(model_type, task_type, model_path_val):
                        # 判断是否显示 CLIP（wan2.2 或 t2v 时不显示）
                        show_clip = model_type == "wan2.1" and task_type == "i2v"
                        # 判断是否显示 VAE编码器（t2v 时不显示）
                        show_vae_encoder = task_type == "i2v"
                        # VAE解码器始终显示
                        show_vae_decoder = True

                        # 根据任务类型更新模型选项
                        if model_type == "wan2.1":
                            dit_choices = get_dit_choices(model_path_val, "wan2.1", task_type)
                            t5_choices = get_t5_model_choices(model_path_val)
                            clip_choices = get_clip_model_choices(model_path_val) if show_clip else []
                            vae_encoder_choices = get_vae_encoder_choices(model_path_val) if show_vae_encoder else []
                            vae_decoder_choices = get_vae_decoder_choices(model_path_val)

                            return (
                                gr.update(visible=show_clip),  # clip_row
                                gr.update(visible=show_vae_encoder),  # vae_encoder_col
                                gr.update(visible=show_vae_decoder),  # vae_decoder_col
                                gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else ""),  # dit_path_input
                                gr.update(),  # high_noise_path_input (wan2.1不使用)
                                gr.update(),  # low_noise_path_input (wan2.1不使用)
                                gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),  # t5_path_input
                                gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else ""),  # clip_path_input
                                gr.update(choices=vae_encoder_choices, value=vae_encoder_choices[0] if vae_encoder_choices else ""),  # vae_encoder_path_input
                                gr.update(choices=vae_decoder_choices, value=vae_decoder_choices[0] if vae_decoder_choices else ""),  # vae_decoder_path_input
                            )
                        else:  # wan2.2
                            high_noise_choices = get_high_noise_choices(model_path_val, "wan2.2", task_type)
                            low_noise_choices = get_low_noise_choices(model_path_val, "wan2.2", task_type)
                            t5_choices = get_t5_model_choices(model_path_val)
                            vae_encoder_choices = get_vae_encoder_choices(model_path_val) if show_vae_encoder else []
                            vae_decoder_choices = get_vae_decoder_choices(model_path_val)

                            return (
                                gr.update(visible=show_clip),  # clip_row
                                gr.update(visible=show_vae_encoder),  # vae_encoder_col
                                gr.update(visible=show_vae_decoder),  # vae_decoder_col
                                gr.update(),  # dit_path_input (wan2.2不使用)
                                gr.update(choices=high_noise_choices, value=high_noise_choices[0] if high_noise_choices else ""),  # high_noise_path_input
                                gr.update(choices=low_noise_choices, value=low_noise_choices[0] if low_noise_choices else ""),  # low_noise_path_input
                                gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),  # t5_path_input
                                gr.update(choices=vae_encoder_choices, value=vae_encoder_choices[0] if vae_encoder_choices else ""),  # vae_encoder_path_input
                                gr.update(choices=vae_decoder_choices, value=vae_decoder_choices[0] if vae_decoder_choices else ""),  # vae_decoder_path_input
                            )

                    # 模型类型切换事件
                    def on_model_type_change(model_type, model_path_val, task_type):
                        # 判断是否显示 CLIP（wan2.2 或 t2v 时不显示）
                        show_clip = model_type == "wan2.1" and task_type == "i2v"
                        # 判断是否显示 VAE编码器（t2v 时不显示）
                        show_vae_encoder = task_type == "i2v"
                        # VAE解码器始终显示
                        show_vae_decoder = True

                        if model_type == "wan2.2":
                            # 更新 wan2.2 的高噪和低噪模型选项
                            high_noise_choices = get_high_noise_choices(model_path_val, "wan2.2", task_type)
                            low_noise_choices = get_low_noise_choices(model_path_val, "wan2.2", task_type)
                            t5_choices = get_t5_model_choices(model_path_val)
                            clip_choices = get_clip_model_choices(model_path_val) if show_clip else []
                            vae_encoder_choices = get_vae_encoder_choices(model_path_val) if show_vae_encoder else []
                            vae_decoder_choices = get_vae_decoder_choices(model_path_val)

                            return (
                                gr.update(visible=False),  # wan21_row
                                gr.update(visible=True),  # wan22_row
                                gr.update(visible=False),  # dit_path_input (wan2.2 时不使用)
                                gr.update(choices=high_noise_choices, value=high_noise_choices[0] if high_noise_choices else ""),  # high_noise_path_input
                                gr.update(choices=low_noise_choices, value=low_noise_choices[0] if low_noise_choices else ""),  # low_noise_path_input
                                gr.update(visible=show_clip),  # clip_row
                                gr.update(visible=show_vae_encoder),  # vae_encoder_col
                                gr.update(visible=show_vae_decoder),  # vae_decoder_col
                                gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),  # t5_path_input
                                gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else ""),  # clip_path_input
                                gr.update(choices=vae_encoder_choices, value=vae_encoder_choices[0] if vae_encoder_choices else ""),  # vae_encoder_path_input
                                gr.update(choices=vae_decoder_choices, value=vae_decoder_choices[0] if vae_decoder_choices else ""),  # vae_decoder_path_input
                                gr.update(visible=False),  # dit_download_btn
                                gr.update(visible=False),  # dit_download_status
                            )
                        else:
                            # 更新 wan2.1 的 Diffusion 模型选项
                            dit_choices = get_dit_choices(model_path_val, "wan2.1", task_type)
                            t5_choices = get_t5_model_choices(model_path_val)
                            clip_choices = get_clip_model_choices(model_path_val) if show_clip else []
                            vae_encoder_choices = get_vae_encoder_choices(model_path_val) if show_vae_encoder else []
                            vae_decoder_choices = get_vae_decoder_choices(model_path_val)

                            return (
                                gr.update(visible=True),  # wan21_row
                                gr.update(visible=False),  # wan22_row
                                gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else "", visible=True),  # dit_path_input
                                gr.update(),  # high_noise_path_input (wan2.2 时使用)
                                gr.update(),  # low_noise_path_input (wan2.2 时使用)
                                gr.update(visible=show_clip),  # clip_row
                                gr.update(visible=show_vae_encoder),  # vae_encoder_col
                                gr.update(visible=show_vae_decoder),  # vae_decoder_col
                                gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),  # t5_path_input
                                gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else ""),  # clip_path_input
                                gr.update(choices=vae_encoder_choices, value=vae_encoder_choices[0] if vae_encoder_choices else ""),  # vae_encoder_path_input
                                gr.update(choices=vae_decoder_choices, value=vae_decoder_choices[0] if vae_decoder_choices else ""),  # vae_decoder_path_input
                                gr.update(),  # dit_download_btn (可见性由 wan21_row 控制)
                                gr.update(),  # dit_download_status (可见性由 wan21_row 控制)
                            )

                    model_type_input.change(
                        fn=on_model_type_change,
                        inputs=[model_type_input, model_path_input, task_type_input],
                        outputs=[
                            wan21_row,
                            wan22_row,
                            dit_path_input,
                            high_noise_path_input,
                            low_noise_path_input,
                            clip_row,
                            vae_encoder_col,
                            vae_decoder_col,
                            t5_path_input,
                            clip_path_input,
                            vae_encoder_path_input,
                            vae_decoder_path_input,
                            dit_download_btn,
                            dit_download_status,
                        ],
                    )

                    task_type_input.change(
                        fn=on_task_type_change,
                        inputs=[model_type_input, task_type_input, model_path_input],
                        outputs=[
                            clip_row,
                            vae_encoder_col,
                            vae_decoder_col,
                            dit_path_input,
                            high_noise_path_input,
                            low_noise_path_input,
                            t5_path_input,
                            clip_path_input,
                            vae_encoder_path_input,
                            vae_decoder_path_input,
                        ],
                    )

                    # 更新模型下载状态的函数
                    def update_dit_status(model_path_val, model_name, model_type_val):
                        if not model_name:
                            return gr.update(visible=False)
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        is_distill = is_distill_model_from_name(actual_name)
                        repo_id = get_repo_id_for_model(model_type_val, is_distill, "dit")
                        exists = check_model_exists(model_path_val, actual_name)
                        if exists:
                            return gr.update(visible=False)
                        else:
                            return gr.update(visible=True)

                    def update_t5_model_status(model_path_val, model_name):
                        if not model_name:
                            return gr.update(visible=False)
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "t5")
                        exists = check_model_exists(model_path_val, actual_name)
                        if exists:
                            return gr.update(visible=False)
                        else:
                            return gr.update(visible=True)

                    def update_t5_tokenizer_status(model_path_val):
                        """更新 T5 Tokenizer (google) 状态"""
                        tokenizer_name = "google"
                        repo_id = get_repo_id_for_model(None, None, "t5")
                        exists = check_model_exists(model_path_val, tokenizer_name)
                        if exists:
                            status_text = f"{tokenizer_name} ✅"
                            return gr.update(value=status_text), gr.update(visible=False)
                        else:
                            status_text = f"{tokenizer_name} ❌"
                            return gr.update(value=status_text), gr.update(visible=True)

                    def update_clip_model_status(model_path_val, model_name):
                        if not model_name:
                            return gr.update(visible=False)
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "clip")
                        exists = check_model_exists(model_path_val, actual_name)
                        if exists:
                            return gr.update(visible=False)
                        else:
                            return gr.update(visible=True)

                    def update_clip_tokenizer_status(model_path_val):
                        """更新 CLIP Tokenizer (xlm-roberta-large) 状态"""
                        tokenizer_name = "xlm-roberta-large"
                        repo_id = get_repo_id_for_model(None, None, "clip")
                        exists = check_model_exists(model_path_val, tokenizer_name)
                        if exists:
                            status_text = f"{tokenizer_name} ✅"
                            return gr.update(value=status_text), gr.update(visible=False)
                        else:
                            status_text = f"{tokenizer_name} ❌"
                            return gr.update(value=status_text), gr.update(visible=True)

                    def update_vae_encoder_status(model_path_val, model_name):
                        if not model_name:
                            return gr.update(visible=False)
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "vae")
                        exists = check_model_exists(model_path_val, actual_name)
                        if exists:
                            return gr.update(visible=False)
                        else:
                            return gr.update(visible=True)

                    def update_vae_decoder_status(model_path_val, model_name):
                        if not model_name:
                            return gr.update(visible=False)
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "vae")
                        exists = check_model_exists(model_path_val, actual_name)
                        if exists:
                            return gr.update(visible=False)
                        else:
                            return gr.update(visible=True)

                    def update_high_noise_status(model_path_val, model_name):
                        if not model_name:
                            return gr.update(visible=False)
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        is_distill = is_distill_model_from_name(actual_name)
                        repo_id = get_repo_id_for_model("wan2.2", is_distill, "high_noise")
                        exists = check_model_exists(model_path_val, actual_name)
                        if exists:
                            return gr.update(visible=False)
                        else:
                            return gr.update(visible=True)

                    def update_low_noise_status(model_path_val, model_name):
                        if not model_name:
                            return gr.update(visible=False)
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        is_distill = is_distill_model_from_name(actual_name)
                        repo_id = get_repo_id_for_model("wan2.2", is_distill, "low_noise")
                        exists = check_model_exists(model_path_val, actual_name)
                        if exists:
                            return gr.update(visible=False)
                        else:
                            return gr.update(visible=True)

                    # 下载函数
                    def download_dit_model(model_path_val, model_name, model_type_val, task_type_val, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="请先选择模型"), gr.update(visible=False), gr.update()
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        is_distill = is_distill_model_from_name(actual_name)
                        repo_id = get_repo_id_for_model(model_type_val, is_distill, "dit")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        # 更新状态和选项
                        btn_visible = update_dit_status(model_path_val, format_model_choice(actual_name, model_path_val), model_type_val)
                        choices = get_dit_choices(model_path_val, model_type_val, task_type_val)
                        # 找到更新后的选项值
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_t5_model(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="请先选择模型"), gr.update(visible=False), gr.update()
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "t5")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_t5_model_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_t5_model_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_t5_tokenizer(model_path_val, download_source_val, progress=gr.Progress()):
                        """下载 T5 Tokenizer (google)"""
                        tokenizer_name = "google"
                        repo_id = get_repo_id_for_model(None, None, "t5")
                        result = download_model(repo_id, tokenizer_name, model_path_val, download_source_val, progress)
                        dropdown_update, btn_visible = update_t5_tokenizer_status(model_path_val)
                        return gr.update(value=result), dropdown_update, btn_visible

                    def download_clip_model(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="请先选择模型"), gr.update(visible=False), gr.update()
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "clip")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_clip_model_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_clip_model_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_clip_tokenizer(model_path_val, download_source_val, progress=gr.Progress()):
                        """下载 CLIP Tokenizer (xlm-roberta-large)"""
                        tokenizer_name = "xlm-roberta-large"
                        repo_id = get_repo_id_for_model(None, None, "clip")
                        result = download_model(repo_id, tokenizer_name, model_path_val, download_source_val, progress)
                        dropdown_update, btn_visible = update_clip_tokenizer_status(model_path_val)
                        return gr.update(value=result), dropdown_update, btn_visible

                    def download_vae_encoder(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="请先选择模型"), gr.update(visible=False), gr.update()
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "vae")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_vae_encoder_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_vae_encoder_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_vae_decoder(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="请先选择模型"), gr.update(visible=False), gr.update()
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "vae")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_vae_decoder_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_vae_decoder_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_high_noise_model(model_path_val, model_name, task_type_val, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="请先选择模型"), gr.update(visible=False), gr.update()
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        is_distill = is_distill_model_from_name(actual_name)
                        repo_id = get_repo_id_for_model("wan2.2", is_distill, "high_noise")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_high_noise_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_high_noise_choices(model_path_val, "wan2.2", task_type_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_low_noise_model(model_path_val, model_name, task_type_val, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="请先选择模型"), gr.update(visible=False), gr.update()
                        # 提取原始模型名称（去掉状态标识）
                        actual_name = extract_model_name(model_name)
                        is_distill = is_distill_model_from_name(actual_name)
                        repo_id = get_repo_id_for_model("wan2.2", is_distill, "low_noise")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_low_noise_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_low_noise_choices(model_path_val, "wan2.2", task_type_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    # 绑定事件：当模型选择改变时更新状态
                    dit_path_input.change(
                        fn=lambda mp, mn, mt: update_dit_status(mp, mn, mt),
                        inputs=[model_path_input, dit_path_input, model_type_input],
                        outputs=[dit_download_btn],
                    )

                    high_noise_path_input.change(
                        fn=update_high_noise_status,
                        inputs=[model_path_input, high_noise_path_input],
                        outputs=[high_noise_download_btn],
                    )

                    low_noise_path_input.change(
                        fn=update_low_noise_status,
                        inputs=[model_path_input, low_noise_path_input],
                        outputs=[low_noise_download_btn],
                    )

                    def update_t5_model_and_tokenizer_status(model_path_val, model_name):
                        """同时更新 T5 模型和 Tokenizer 状态"""
                        model_btn = update_t5_model_status(model_path_val, model_name)
                        tokenizer_dropdown, tokenizer_btn = update_t5_tokenizer_status(model_path_val)
                        return model_btn, tokenizer_dropdown, tokenizer_btn

                    t5_path_input.change(
                        fn=update_t5_model_and_tokenizer_status,
                        inputs=[model_path_input, t5_path_input],
                        outputs=[t5_download_btn, t5_tokenizer_hint, t5_tokenizer_download_btn],
                    )

                    def update_clip_model_and_tokenizer_status(model_path_val, model_name):
                        """同时更新 CLIP 模型和 Tokenizer 状态"""
                        model_btn = update_clip_model_status(model_path_val, model_name)
                        tokenizer_dropdown, tokenizer_btn = update_clip_tokenizer_status(model_path_val)
                        return model_btn, tokenizer_dropdown, tokenizer_btn

                    clip_path_input.change(
                        fn=update_clip_model_and_tokenizer_status,
                        inputs=[model_path_input, clip_path_input],
                        outputs=[clip_download_btn, clip_tokenizer_hint, clip_tokenizer_download_btn],
                    )

                    vae_encoder_path_input.change(
                        fn=update_vae_encoder_status,
                        inputs=[model_path_input, vae_encoder_path_input],
                        outputs=[vae_encoder_download_btn],
                    )

                    vae_decoder_path_input.change(
                        fn=update_vae_decoder_status,
                        inputs=[model_path_input, vae_decoder_path_input],
                        outputs=[vae_decoder_download_btn],
                    )

                    # 绑定下载按钮事件
                    dit_download_btn.click(
                        fn=download_dit_model,
                        inputs=[model_path_input, dit_path_input, model_type_input, task_type_input, download_source_input],
                        outputs=[dit_download_status, dit_download_btn, dit_path_input],
                    )

                    high_noise_download_btn.click(
                        fn=download_high_noise_model,
                        inputs=[model_path_input, high_noise_path_input, task_type_input, download_source_input],
                        outputs=[high_noise_download_status, high_noise_download_btn, high_noise_path_input],
                    )

                    low_noise_download_btn.click(
                        fn=download_low_noise_model,
                        inputs=[model_path_input, low_noise_path_input, task_type_input, download_source_input],
                        outputs=[low_noise_download_status, low_noise_download_btn, low_noise_path_input],
                    )

                    t5_download_btn.click(
                        fn=download_t5_model,
                        inputs=[model_path_input, t5_path_input, download_source_input],
                        outputs=[t5_download_status, t5_download_btn, t5_path_input],
                    )

                    t5_tokenizer_download_btn.click(
                        fn=download_t5_tokenizer,
                        inputs=[model_path_input, download_source_input],
                        outputs=[t5_tokenizer_download_status, t5_tokenizer_hint, t5_tokenizer_download_btn],
                    )

                    clip_download_btn.click(
                        fn=download_clip_model,
                        inputs=[model_path_input, clip_path_input, download_source_input],
                        outputs=[clip_download_status, clip_download_btn, clip_path_input],
                    )

                    clip_tokenizer_download_btn.click(
                        fn=download_clip_tokenizer,
                        inputs=[model_path_input, download_source_input],
                        outputs=[clip_tokenizer_download_status, clip_tokenizer_hint, clip_tokenizer_download_btn],
                    )

                    vae_encoder_download_btn.click(
                        fn=download_vae_encoder,
                        inputs=[model_path_input, vae_encoder_path_input, download_source_input],
                        outputs=[vae_encoder_download_status, vae_encoder_download_btn, vae_encoder_path_input],
                    )

                    vae_decoder_download_btn.click(
                        fn=download_vae_decoder,
                        inputs=[model_path_input, vae_decoder_path_input, download_source_input],
                        outputs=[vae_decoder_download_status, vae_decoder_download_btn, vae_decoder_path_input],
                    )

                    # 初始化所有模型的状态
                    def init_all_statuses(model_path_val, dit_name, high_noise_name, low_noise_name, t5_name, clip_name, vae_encoder_name, vae_decoder_name, model_type_val):
                        dit_btn_visible = update_dit_status(model_path_val, dit_name, model_type_val)
                        high_noise_btn_visible = update_high_noise_status(model_path_val, high_noise_name)
                        low_noise_btn_visible = update_low_noise_status(model_path_val, low_noise_name)
                        t5_btn_visible = update_t5_model_status(model_path_val, t5_name)
                        t5_tokenizer_dropdown_val, t5_tokenizer_btn_visible = update_t5_tokenizer_status(model_path_val)
                        clip_btn_visible = update_clip_model_status(model_path_val, clip_name)
                        clip_tokenizer_dropdown_val, clip_tokenizer_btn_visible = update_clip_tokenizer_status(model_path_val)
                        vae_encoder_btn_visible = update_vae_encoder_status(model_path_val, vae_encoder_name)
                        vae_decoder_btn_visible = update_vae_decoder_status(model_path_val, vae_decoder_name)
                        return (
                            dit_btn_visible,
                            high_noise_btn_visible,
                            low_noise_btn_visible,
                            t5_btn_visible,
                            t5_tokenizer_dropdown_val,
                            t5_tokenizer_btn_visible,
                            clip_btn_visible,
                            clip_tokenizer_dropdown_val,
                            clip_tokenizer_btn_visible,
                            vae_encoder_btn_visible,
                            vae_decoder_btn_visible,
                        )

                    demo.load(
                        fn=init_all_statuses,
                        inputs=[
                            model_path_input,
                            dit_path_input,
                            high_noise_path_input,
                            low_noise_path_input,
                            t5_path_input,
                            clip_path_input,
                            vae_encoder_path_input,
                            vae_decoder_path_input,
                            model_type_input,
                        ],
                        outputs=[
                            dit_download_btn,
                            high_noise_download_btn,
                            low_noise_download_btn,
                            t5_download_btn,
                            t5_tokenizer_hint,
                            t5_tokenizer_download_btn,
                            clip_download_btn,
                            clip_tokenizer_hint,
                            clip_tokenizer_download_btn,
                            vae_encoder_download_btn,
                            vae_decoder_download_btn,
                        ],
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
                            default_dit = get_dit_choices(model_path, "wan2.1", "i2v")[0] if get_dit_choices(model_path, "wan2.1", "i2v") else ""
                            default_high_noise = get_high_noise_choices(model_path, "wan2.2", "i2v")[0] if get_high_noise_choices(model_path, "wan2.2", "i2v") else ""
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
            inputs=[resolution, num_frames],
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

        num_frames.change(
            fn=auto_configure,
            inputs=[resolution, num_frames],
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
            fn=lambda res, nf: auto_configure(res, nf),
            inputs=[resolution, num_frames],
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
                vae_encoder_path_input,
                vae_decoder_path_input,
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
