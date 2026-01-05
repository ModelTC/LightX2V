import argparse
import gc
import glob
import importlib.util
import json
import logging
import os
import warnings

# Suppress network retry warnings during Hugging Face downloads (normal retry behavior)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.utils")
# Suppress reqwest retry warnings (JSON log outputs, not real errors)
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
    },
    "Qwen_Image_Edit_2511": {
        "_class_name": "QwenImageTransformer2DModel",
        "_diffusers_version": "0.36.0.dev0",
        "attention_head_dim": 128,
        "axes_dims_rope": [16, 56, 56],
        "guidance_embeds": False,
        "in_channels": 64,
        "joint_attention_dim": 3584,
        "num_attention_heads": 24,
        "num_layers": 60,
        "out_channels": 16,
        "patch_size": 2,
        "zero_cond_t": True,
    },
}
# Model list cache (avoid fetching from HF every time)
HF_MODELS_CACHE = {
    "lightx2v/wan2.1-Distill-Models": [],
    "lightx2v/wan2.1-Official-Models": [],
    "lightx2v/wan2.2-Distill-Models": [],
    "lightx2v/wan2.2-Official-Models": [],
    "lightx2v/Encoders": [],
    "lightx2v/Autoencoders": [],
    "lightx2v/Qwen-Image-Edit-2511-Lightning": [],
    "Qwen/Qwen-Image-Edit-2511": [],
}


def scan_model_path_contents(model_path):
    """Scan model_path directory, return available files and subdirectories"""
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
    """Load model list from Hugging Face and cache, if HF times out or fails, try ModelScope"""
    import concurrent.futures

    def process_files(files, repo_id=None):
        """Process file list, extract model names"""
        model_names = []
        seen_dirs = set()

        # For Qwen/Qwen-Image-Edit-2511 repository, keep vae and scheduler directories
        is_qwen_image_repo = repo_id == "Qwen/Qwen-Image-Edit-2511"

        for file in files:
            # Exclude files containing comfyui
            if "comfyui" in file.lower():
                continue

            # If it's a top-level file (no path separator)
            if "/" not in file:
                # Only keep safetensors files
                if file.endswith(".safetensors"):
                    model_names.append(file)
            else:
                # Extract top-level directory name (supports _split directories)
                top_dir = file.split("/")[0]
                if top_dir not in seen_dirs:
                    seen_dirs.add(top_dir)
                    # For Qwen repository, keep vae and scheduler directories
                    if is_qwen_image_repo and top_dir.lower() in ["vae", "scheduler"]:
                        model_names.append(top_dir)
                    # Support safetensors file directories and _split block storage directories
                    elif "_split" in top_dir or any(f.startswith(f"{top_dir}/") and f.endswith(".safetensors") for f in files):
                        model_names.append(top_dir)
        return sorted(set(model_names))

    # Timeout (seconds)
    HF_TIMEOUT = 30

    for repo_id in HF_MODELS_CACHE.keys():
        files = None
        source = None

        # First try to get from ModelScope
        try:
            if MS_AVAILABLE:
                logger.info(f"Loading models from ModelScope {repo_id}...")
                api = HubApi()
                # ModelScope API get file list
                model_files = api.get_model_files(model_id=repo_id, recursive=True)
                # Extract file paths
                files = [file["Path"] for file in model_files if file.get("Type") == "blob"]
                source = "ModelScope"
                logger.info(f"Successfully loaded models from ModelScope {repo_id}")
        except:  # noqa E722
            # If ModelScope fails, try to get from Hugging Face (with timeout)
            if files is None and HF_AVAILABLE:
                logger.info(f"Loading models from Hugging Face {repo_id}...")
                api = HfApi()

                # Use thread pool executor with timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(list_repo_files, repo_id=repo_id, repo_type="model")
                    files = future.result(timeout=HF_TIMEOUT)
                    source = "Hugging Face"

        # Process file list
        if files:
            model_names = process_files(files, repo_id)
            HF_MODELS_CACHE[repo_id] = model_names
            logger.info(f"Loaded {len(HF_MODELS_CACHE[repo_id])} models from {source} {repo_id}")
        else:
            logger.warning(f"No files retrieved from {repo_id}, setting empty cache")
            HF_MODELS_CACHE[repo_id] = []


def get_hf_models(repo_id, prefix_filter=None, keyword_filter=None):
    """Get models from cached model list (no longer fetching from HF in real-time)"""
    if repo_id not in HF_MODELS_CACHE:
        return []

    models = HF_MODELS_CACHE[repo_id]

    if prefix_filter:
        models = [m for m in models if m.lower().startswith(prefix_filter.lower())]

    if keyword_filter:
        models = [m for m in models if keyword_filter.lower() in m.lower()]

    return models


def check_model_exists(model_path, model_name):
    """Check if model is already downloaded"""
    if not model_path or not os.path.exists(model_path):
        return False

    model_path_full = os.path.join(model_path, model_name)
    # Check if exists (file or directory)
    if os.path.exists(model_path_full):
        return True

    # Additional check: if it's a safetensors file, also check for same-name directory (_split directory)
    if model_name.endswith(".safetensors"):
        # Check for same-name directory (may be stored in chunks)
        base_name = model_name.replace(".safetensors", "")
        split_dir = os.path.join(model_path, base_name + "_split")
        if os.path.exists(split_dir):
            return True

    return False


def format_model_choice(model_name, model_path, status_emoji=None):
    """Format model option, add download status indicator"""
    if not model_name:
        return ""

    # If status emoji is provided, use it directly
    if status_emoji is not None:
        return f"{status_emoji} {model_name}"

    # Otherwise check if it exists locally
    exists = check_model_exists(model_path, model_name)
    emoji = "✅" if exists else "❌"
    return f"{emoji} {model_name}"


def extract_model_name(formatted_name):
    """Extract original model name from formatted option name"""
    if not formatted_name:
        return ""
    # Remove leading emoji and space
    if formatted_name.startswith("✅ ") or formatted_name.startswith("❌ "):
        return formatted_name[2:].strip()
    return formatted_name.strip()


def sort_model_choices(models):
    """Sort model list based on device capability and model type

    Sort rules:
    - If device supports fp8: fp8+split > int8+split > fp8 > int8 > others
    - If device doesn't support fp8: int8+split > int8 > others
    """
    fp8_supported = is_fp8_supported_gpu()

    def get_priority(name):
        name_lower = name.lower()
        if fp8_supported:
            # fp8 device: fp8+split > int8+split > fp8 > int8 > others
            if "fp8" in name_lower and "_split" in name_lower:
                return 0  # Highest priority
            elif "int8" in name_lower and "_split" in name_lower:
                return 1
            elif "fp8" in name_lower:
                return 2
            elif "int8" in name_lower:
                return 3
            else:
                return 4  # Others
        else:
            # Non-fp8 device: int8+split > int8 > others
            if "int8" in name_lower and "_split" in name_lower:
                return 0  # Highest priority
            elif "int8" in name_lower:
                return 1
            else:
                return 2  # Others (fp8 already filtered out)

    return sorted(models, key=lambda x: (get_priority(x), x.lower()))


def get_dit_choices(model_path, model_type="wan2.1", task_type=None, is_distill=None):
    """Get Diffusion model options (from Hugging Face and local)

    Args:
        model_path: Local model path
        model_type: "wan2.1" or "wan2.2"
        task_type: "i2v" or "t2v", None means no task type filtering
        is_distill: Whether it's a distill model, None means get both distill and non-distill
    """
    excluded_keywords = ["vae", "tae", "clip", "t5", "high_noise", "low_noise"]
    fp8_supported = is_fp8_supported_gpu()

    # Select repository based on model type and whether distill
    if model_type == "wan2.1":
        if is_distill is True:
            repo_id = "lightx2v/wan2.1-Distill-Models"
        elif is_distill is False:
            repo_id = "lightx2v/wan2.1-Official-Models"
        else:
            # Get models from both repositories
            repo_id_distill = "lightx2v/wan2.1-Distill-Models"
            repo_id_official = "lightx2v/wan2.1-Official-Models"
            hf_models_distill = get_hf_models(repo_id_distill, prefix_filter="wan2.1") if HF_AVAILABLE else []
            hf_models_official = get_hf_models(repo_id_official, prefix_filter="wan2.1") if HF_AVAILABLE else []
            hf_models = list(set(hf_models_distill + hf_models_official))
            repo_id = None  # Mark as already fetched
    else:  # wan2.2
        if is_distill is True:
            repo_id = "lightx2v/wan2.2-Distill-Models"
        elif is_distill is False:
            repo_id = "lightx2v/wan2.2-Official-Models"
        else:
            # Get models from both repositories
            repo_id_distill = "lightx2v/wan2.2-Distill-Models"
            repo_id_official = "lightx2v/wan2.2-Official-Models"
            hf_models_distill = get_hf_models(repo_id_distill, prefix_filter="wan2.2") if HF_AVAILABLE else []
            hf_models_official = get_hf_models(repo_id_official, prefix_filter="wan2.2") if HF_AVAILABLE else []
            hf_models = list(set(hf_models_distill + hf_models_official))
            repo_id = None  # Mark as already fetched

    if repo_id:
        hf_models = get_hf_models(repo_id, prefix_filter=model_type) if HF_AVAILABLE else []

    # Filter models that meet criteria
    def is_valid(name):
        name_lower = name.lower()
        # Filter out files containing comfyui
        if "comfyui" in name_lower:
            return False
        # Check model type
        if model_type == "wan2.1":
            if "wan2.1" not in name_lower:
                return False
        else:
            if "wan2.2" not in name_lower:
                return False
        # Check task type (if specified)
        if task_type:
            if task_type.lower() not in name_lower:
                return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return not any(kw in name_lower for kw in excluded_keywords)

    # Filter HF models: only keep safetensors files or _split directories
    valid_hf_models = []
    for m in hf_models:
        if not is_valid(m):
            continue
        # Keep if it's a safetensors file or contains _split directory
        if m.endswith(".safetensors") or "_split" in m.lower():
            valid_hf_models.append(m)

    # Check locally existing models (only search safetensors files and directories, including _split directories)
    contents = scan_model_path_contents(model_path)
    dir_choices = [d for d in contents["dirs"] if is_valid(d) and ("_split" in d.lower() or d in contents["safetensors_dirs"])]
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid(f)]
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if is_valid(d)]
    local_models = dir_choices + safetensors_choices + safetensors_dir_choices

    # Merge HF and local models, deduplicate, and sort by priority
    all_models = sort_model_choices(list(set(valid_hf_models + local_models)))

    # Format options, add download status (✅ downloaded, ❌ not downloaded)
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_high_noise_choices(model_path, model_type="wan2.2", task_type=None, is_distill=None):
    """Get high noise model options (from Hugging Face and local, files/directories containing high_noise)

    Args:
        model_path: Local model path
        model_type: "wan2.2" (high noise model only for wan2.2)
        task_type: "i2v" or "t2v", None means no task type filtering
        is_distill: Whether it's a distill model, None means get both distill and non-distill
    """
    fp8_supported = is_fp8_supported_gpu()

    # Select repository based on whether distill
    if is_distill is True:
        repo_id = "lightx2v/wan2.2-Distill-Models"
    elif is_distill is False:
        repo_id = "lightx2v/wan2.2-Official-Models"
    else:
        # Get models from both repositories
        repo_id_distill = "lightx2v/wan2.2-Distill-Models"
        repo_id_official = "lightx2v/wan2.2-Official-Models"
        hf_models_distill = get_hf_models(repo_id_distill, keyword_filter="high_noise") if HF_AVAILABLE else []
        hf_models_official = get_hf_models(repo_id_official, keyword_filter="high_noise") if HF_AVAILABLE else []
        hf_models = list(set(hf_models_distill + hf_models_official))
        repo_id = None

    if repo_id:
        hf_models = get_hf_models(repo_id, keyword_filter="high_noise") if HF_AVAILABLE else []

    def is_valid(name):
        name_lower = name.lower()
        # Filter out files containing comfyui
        if "comfyui" in name_lower:
            return False
        # Check model type
        if model_type.lower() not in name_lower:
            return False
        # Check task type (if specified)
        if task_type:
            if task_type.lower() not in name_lower:
                return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return "high_noise" in name_lower or "high-noise" in name_lower

    # Filter HF models: only keep safetensors files or _split directories
    valid_hf_models = []
    for m in hf_models:
        if not is_valid(m):
            continue
        # Keep if it's a safetensors file or contains _split directory
        if m.endswith(".safetensors") or "_split" in m.lower():
            valid_hf_models.append(m)

    # Check locally existing models (only search safetensors files and directories, including _split directories)
    contents = scan_model_path_contents(model_path)
    dir_choices = [d for d in contents["dirs"] if is_valid(d) and ("_split" in d.lower() or d in contents["safetensors_dirs"])]
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid(f)]
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if is_valid(d)]
    local_models = dir_choices + safetensors_choices + safetensors_dir_choices

    # Merge HF and local models, deduplicate, and sort by priority
    all_models = sort_model_choices(list(set(valid_hf_models + local_models)))

    # Format options, add download status (✅ downloaded, ❌ not downloaded)
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_low_noise_choices(model_path, model_type="wan2.2", task_type=None, is_distill=None):
    """Get low noise model options (from Hugging Face and local, files/directories containing low_noise)

    Args:
        model_path: Local model path
        model_type: "wan2.2" (low noise model only for wan2.2)
        task_type: "i2v" or "t2v", None means no task type filtering
        is_distill: Whether it's a distill model, None means get both distill and non-distill
    """
    fp8_supported = is_fp8_supported_gpu()

    # Select repository based on whether distill
    if is_distill is True:
        repo_id = "lightx2v/wan2.2-Distill-Models"
    elif is_distill is False:
        repo_id = "lightx2v/wan2.2-Official-Models"
    else:
        # Get models from both repositories
        repo_id_distill = "lightx2v/wan2.2-Distill-Models"
        repo_id_official = "lightx2v/wan2.2-Official-Models"
        hf_models_distill = get_hf_models(repo_id_distill, keyword_filter="low_noise") if HF_AVAILABLE else []
        hf_models_official = get_hf_models(repo_id_official, keyword_filter="low_noise") if HF_AVAILABLE else []
        hf_models = list(set(hf_models_distill + hf_models_official))
        repo_id = None

    if repo_id:
        hf_models = get_hf_models(repo_id, keyword_filter="low_noise") if HF_AVAILABLE else []

    def is_valid(name):
        name_lower = name.lower()
        # Filter out files containing comfyui
        if "comfyui" in name_lower:
            return False
        # Check model type
        if model_type.lower() not in name_lower:
            return False
        # Check task type (if specified)
        if task_type:
            if task_type.lower() not in name_lower:
                return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return "low_noise" in name_lower or "low-noise" in name_lower

    # Filter HF models: only keep safetensors files or _split directories
    valid_hf_models = []
    for m in hf_models:
        if not is_valid(m):
            continue
        # Keep if it's a safetensors file or contains _split directory
        if m.endswith(".safetensors") or "_split" in m.lower():
            valid_hf_models.append(m)

    # Check locally existing models (only search safetensors files and directories, including _split directories)
    contents = scan_model_path_contents(model_path)
    dir_choices = [d for d in contents["dirs"] if is_valid(d) and ("_split" in d.lower() or d in contents["safetensors_dirs"])]
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid(f)]
    safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if is_valid(d)]
    local_models = dir_choices + safetensors_choices + safetensors_dir_choices

    # Merge HF and local models, deduplicate, and sort by priority
    all_models = sort_model_choices(list(set(valid_hf_models + local_models)))

    # Format options, add download status (✅ downloaded, ❌ not downloaded)
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_t5_model_choices(model_path):
    """Get T5 model options (from Hugging Face Encoders repository and local, containing t5 keyword, only safetensors, excluding google)"""
    fp8_supported = is_fp8_supported_gpu()

    # Get from Hugging Face Encoders repository
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # Filter files containing t5, only safetensors, excluding google
    def is_valid_hf(name):
        name_lower = name.lower()
        # Filter out files containing comfyui and google directory
        if "comfyui" in name_lower or name == "google":
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # Only show safetensors files
        return ("t5" in name_lower) and name.endswith(".safetensors")

    valid_hf_models = [m for m in hf_models if is_valid_hf(m)]

    # Check locally existing models
    contents = scan_model_path_contents(model_path)

    def is_valid_local(name):
        name_lower = name.lower()
        # Filter out files containing comfyui and google directory
        if "comfyui" in name_lower or name == "google":
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # Only show safetensors files
        return ("t5" in name_lower) and name.endswith(".safetensors")

    # Only filter from .safetensors files
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid_local(f)]

    local_models = safetensors_choices

    # Merge HF and local models, deduplicate, and sort by priority
    all_models = sort_model_choices(list(set(valid_hf_models + local_models)))

    # Format options, add download status (✅ downloaded, ❌ not downloaded)
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_t5_tokenizer_choices(model_path):
    """Get T5 Tokenizer options (google directory)"""
    # Only return google directory
    contents = scan_model_path_contents(model_path)
    dir_choices = ["google"] if "google" in contents["dirs"] else []

    # Get from HF
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []
    hf_google = ["google"] if "google" in hf_models else []

    all_models = sorted(set(hf_google + dir_choices))
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_clip_model_choices(model_path):
    """Get CLIP model options (from Hugging Face Encoders repository and local, containing clip keyword, only safetensors, excluding xlm-roberta-large)"""
    fp8_supported = is_fp8_supported_gpu()

    # Get from Hugging Face Encoders repository
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # Filter files containing clip, only safetensors, excluding xlm-roberta-large
    def is_valid_hf(name):
        name_lower = name.lower()
        # Filter out files containing comfyui and xlm-roberta-large directory
        if "comfyui" in name_lower or name == "xlm-roberta-large":
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # Only show safetensors files
        return ("clip" in name_lower) and name.endswith(".safetensors")

    valid_hf_models = [m for m in hf_models if is_valid_hf(m)]

    # Check locally existing models
    contents = scan_model_path_contents(model_path)

    def is_valid_local(name):
        name_lower = name.lower()
        # Filter out files containing comfyui and xlm-roberta-large directory
        if "comfyui" in name_lower or name == "xlm-roberta-large":
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # Only show safetensors files
        return ("clip" in name_lower) and name.endswith(".safetensors")

    # Only filter from .safetensors files
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid_local(f)]

    local_models = safetensors_choices

    # Merge HF and local models, deduplicate, and sort by priority
    all_models = sort_model_choices(list(set(valid_hf_models + local_models)))

    # Format options, add download status (✅ downloaded, ❌ not downloaded)
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_clip_tokenizer_choices(model_path):
    """Get CLIP Tokenizer options (xlm-roberta-large directory)"""
    # Only return xlm-roberta-large directory
    contents = scan_model_path_contents(model_path)
    dir_choices = ["xlm-roberta-large"] if "xlm-roberta-large" in contents["dirs"] else []

    # Get from HF
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []
    hf_xlm = ["xlm-roberta-large"] if "xlm-roberta-large" in hf_models else []

    all_models = sorted(set(hf_xlm + dir_choices))
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_vae_encoder_choices(model_path):
    """Get VAE encoder options, only return wan2.1_VAE.safetensors"""
    encoder_name = "wan2.1_VAE.safetensors"

    # Get from Hugging Face Autoencoders repository
    repo_id = "lightx2v/Autoencoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # Check if the file exists in HF
    hf_has = encoder_name in hf_models

    # Check if it exists locally
    local_has = check_model_exists(model_path, encoder_name)

    # If HF or local has it, return
    if hf_has or local_has:
        return [format_model_choice(encoder_name, model_path)]
    else:
        return [format_model_choice(encoder_name, model_path)]


def get_vae_decoder_choices(model_path):
    """Get VAE decoder options (from Hugging Face Autoencoders repository and local, containing vae/VAE/tae keyword, only safetensors)"""
    fp8_supported = is_fp8_supported_gpu()

    # Get from Hugging Face Autoencoders repository
    repo_id = "lightx2v/Autoencoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # Filter files containing vae or tae, only safetensors files or _split directories
    def is_valid_hf(name):
        name_lower = name.lower()
        # Filter out files containing comfyui
        if "comfyui" in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # Only show safetensors files or _split directories, must contain vae or tae
        return any(kw in name_lower for kw in ["vae", "tae", "lightvae", "lighttae"]) and (name.endswith(".safetensors") or "_split" in name_lower)

    valid_hf_models = [m for m in hf_models if is_valid_hf(m)]

    # Check locally existing models
    contents = scan_model_path_contents(model_path)

    def is_valid_local(name):
        name_lower = name.lower()
        # Filter out files containing comfyui
        if "comfyui" in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # Only show safetensors files or _split directories, must contain vae or tae
        if not any(kw in name_lower for kw in ["vae", "tae", "lightvae", "lighttae"]):
            return False
        # If it's a file, must be safetensors
        if os.path.isfile(os.path.join(model_path, name)):
            return name.endswith(".safetensors")
        # If it's a directory, must be a directory containing safetensors or _split directory
        return name in contents["safetensors_dirs"] or "_split" in name_lower

    # Filter from .safetensors files
    safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid_local(f)]

    # Filter from directories containing safetensors (including _split directories)
    dir_choices = [d for d in contents["dirs"] if is_valid_local(d)]

    local_models = safetensors_choices + dir_choices

    # Merge HF and local models, deduplicate
    all_models = list(set(valid_hf_models + local_models))

    # For VAE decoder, only show options containing "2_1" or "2.1"
    all_models = [m for m in all_models if "2_1" in m or "2.1" in m]

    # Sort by priority
    all_models = sort_model_choices(all_models)

    # Format options, add download status (✅ downloaded, ❌ not downloaded)
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_qwen_image_dit_choices(model_path):
    """Get Qwen Image Edit Diffusion model options
    From lightx2v/Qwen-Image-Edit-2511-Lightning repository
    Only list models containing qwen_image_edit_2511 and ending with lightning.safetensors or lightning_split
    """
    fp8_supported = is_fp8_supported_gpu()

    # Get from Hugging Face repository
    repo_id = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    def is_valid(name):
        name_lower = name.lower()
        # Filter out files containing comfyui
        if "comfyui" in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # Must contain qwen_image_edit_2511
        if "qwen_image_edit_2511" not in name_lower:
            return False
        # Only keep those ending with lightning.safetensors or lightning_split
        return name.endswith("lightning.safetensors") or name.endswith("_split") or "lightning_split" in name_lower

    # Filter HF models
    valid_hf_models = [m for m in hf_models if is_valid(m)]

    # Check locally existing models
    contents = scan_model_path_contents(model_path)
    local_models = []
    for item in contents["dirs"] + contents["files"]:
        if is_valid(item):
            local_models.append(item)

    # Merge HF and local models, deduplicate, and sort by priority
    all_models = sort_model_choices(list(set(valid_hf_models + local_models)))

    # Format options, add download status
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_qwen_image_vae_choices(model_path):
    """Get Qwen Image Edit VAE options
    Get vae directory from Qwen/Qwen-Image-Edit-2511 repository
    """
    # Get from Hugging Face repository
    repo_id = "Qwen/Qwen-Image-Edit-2511"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # Only keep vae directory
    valid_hf_models = [m for m in hf_models if m.lower() == "vae"]

    # Check locally existing models
    contents = scan_model_path_contents(model_path)
    local_models = [d for d in contents["dirs"] if d.lower() == "vae"]

    # Merge HF and local models, deduplicate
    all_models = sorted(set(valid_hf_models + local_models))

    # If not found, add default value "vae"
    if not all_models:
        all_models = ["vae"]

    # Format options, add download status
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices


def get_qwen_image_scheduler_choices(model_path):
    """Get Qwen Image Edit Scheduler options
    Get scheduler directory from Qwen/Qwen-Image-Edit-2511 repository
    """
    # Get from Hugging Face repository
    repo_id = "Qwen/Qwen-Image-Edit-2511"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # Only keep scheduler directory
    valid_hf_models = [m for m in hf_models if m.lower() == "scheduler"]

    # Check locally existing models
    contents = scan_model_path_contents(model_path)
    local_models = [d for d in contents["dirs"] if d.lower() == "scheduler"]

    # Merge HF and local models, deduplicate
    all_models = sorted(set(valid_hf_models + local_models))

    # If not found, add default value "scheduler"
    if not all_models:
        all_models = ["scheduler"]

    # Format options, add download status
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices


def get_qwen25vl_encoder_choices(model_path):
    """Get Qwen25-VL encoder options
    From lightx2v/Encoders repository, only need Qwen25-VL-4bit-GPTQ
    """
    # Get from Hugging Face Encoders repository
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # Only keep Qwen25-VL-4bit-GPTQ
    valid_hf_models = [m for m in hf_models if "qwen25-vl-4bit-gptq" in m.lower() or "qwen25_vl_4bit_gptq" in m.lower()]

    # Check locally existing models
    contents = scan_model_path_contents(model_path)
    local_models = [d for d in contents["dirs"] if "qwen25-vl-4bit-gptq" in d.lower() or "qwen25_vl_4bit_gptq" in d.lower()]

    # Merge HF and local models, deduplicate
    all_models = sorted(set(valid_hf_models + local_models))

    # Format options, add download status
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def detect_quant_scheme(model_name):
    """Automatically detect quantization precision based on model name
    - If model name contains "int8" → "int8"
    - If model name contains "fp8" and device supports it → "fp8"
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


def download_model_from_hf(repo_id, model_name, model_path, progress=gr.Progress()):
    """Download model from Hugging Face (supports files and directories)"""
    if not HF_AVAILABLE:
        return f"❌ huggingface_hub not installed, cannot download model"

    progress(0, desc=f"Starting to download {model_name} from Hugging Face...")
    logger.info(f"Starting to download {model_name} from Hugging Face {repo_id} to {model_path}")

    target_path = os.path.join(model_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    import shutil

    # Determine if it's a file or directory: if name doesn't end with .safetensors or .pth, it's a directory
    is_directory = not (model_name.endswith(".safetensors") or model_name.endswith(".pth"))

    if is_directory:
        # Download directory
        progress(0.1, desc=f"Downloading directory {model_name}...")
        logger.info(f"Detected {model_name} is a directory, using snapshot_download")

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

        # Move files to correct location
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

        logger.info(f"Directory {model_name} download complete, moved to {target_path}")
    else:
        # Download file
        progress(0.1, desc=f"Downloading file {model_name}...")
        logger.info(f"Detected {model_name} is a file, using hf_hub_download")

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
        logger.info(f"File {model_name} download complete, saved to {downloaded_path}")

    progress(1.0, desc=f"✅ {model_name} download complete")
    return f"✅ {model_name} download complete"


def download_model_from_ms(repo_id, model_name, model_path, progress=gr.Progress()):
    """Download model from ModelScope (supports files and directories)"""
    if not MS_AVAILABLE:
        return f"❌ modelscope not installed, cannot download model"

    progress(0, desc=f"Starting to download {model_name} from ModelScope...")
    logger.info(f"Starting to download {model_name} from ModelScope {repo_id} to {model_path}")

    target_path = os.path.join(model_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    import shutil

    # Determine if it's a file or directory: if name doesn't end with .safetensors or .pth, it's a directory
    is_directory = not (model_name.endswith(".safetensors") or model_name.endswith(".pth"))
    is_file = not is_directory

    # Temporary directory for download
    temp_dir = os.path.join(model_path, f".temp_{model_name}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Handle directory download
    if is_directory:
        progress(0.1, desc=f"Downloading directory {model_name}...")
        logger.info(f"Detected {model_name} is a directory, using snapshot_download")

        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        # Use snapshot_download to download directory
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downloaded_path = ms_snapshot_download(
                model_id=repo_id,
                cache_dir=temp_dir,
                allow_patterns=[f"{model_name}/**"],
            )

        # Move files to target location
        source_dir = os.path.join(downloaded_path, model_name)
        if not os.path.exists(source_dir) and os.path.exists(downloaded_path):
            # If not found, try to find from download path
            for item in os.listdir(downloaded_path):
                item_path = os.path.join(downloaded_path, item)
                if model_name in item or os.path.isdir(item_path):
                    source_dir = item_path
                    break

        if os.path.exists(source_dir):
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.move(source_dir, target_path)

        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        logger.info(f"Directory {model_name} download complete, saved to {target_path}")
    # Handle file download
    elif is_file:
        progress(0.1, desc=f"Downloading file {model_name}...")
        logger.info(f"Detected {model_name} is a file, using snapshot_download")

        if os.path.exists(target_path):
            os.remove(target_path)
        os.makedirs(os.path.dirname(target_path) if "/" in model_name else model_path, exist_ok=True)

        # Use snapshot_download to download file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downloaded_path = ms_snapshot_download(
                model_id=repo_id,
                cache_dir=temp_dir,
                allow_patterns=[model_name],
            )

        # Find and move file
        source_file = os.path.join(downloaded_path, model_name)
        if not os.path.exists(source_file):
            # If not found, try to find from download path
            for root, dirs, files_list in os.walk(downloaded_path):
                if model_name in files_list:
                    source_file = os.path.join(root, model_name)
                    break

        if os.path.exists(source_file):
            shutil.move(source_file, target_path)

        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        logger.info(f"File {model_name} download complete, saved to {target_path}")
    else:
        return f"❌ Cannot find {model_name}: neither a file nor a directory"

    progress(1.0, desc=f"✅ {model_name} download complete")
    return f"✅ {model_name} download complete"


def download_model(repo_id, model_name, model_path, download_source="huggingface", progress=gr.Progress()):
    """Unified download function, choose Hugging Face or ModelScope based on download source"""
    if download_source == "modelscope":
        return download_model_from_ms(repo_id, model_name, model_path, progress)
    else:
        return download_model_from_hf(repo_id, model_name, model_path, progress)


def get_model_status(model_path, model_name, repo_id):
    """Get model status (downloaded/not downloaded)"""
    exists = check_model_exists(model_path, model_name)
    if exists:
        return "✅ Downloaded", gr.update(visible=False)
    else:
        return "❌ Not Downloaded", gr.update(visible=True)


def update_model_path_options(model_path, model_type="wan2.1", task_type=None):
    """Update all model path selectors when model_path or model_type changes"""
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

    # Detect torch option: requires both hasattr(torch, "_scaled_mm") and torchao installed
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


def generate_unique_filename(output_dir, is_image=False):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = ".png" if is_image else ".mp4"
    return os.path.join(output_dir, f"{timestamp}{ext}")


def is_fp8_supported_gpu():
    if not torch.cuda.is_available():
        return False
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    return (major == 8 and minor == 9) or (major >= 9)


def is_sm_greater_than_90():
    """Detect if compute capability is greater than (9,0)"""
    if not torch.cuda.is_available():
        return False
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    return (major, minor) > (9, 0)


def get_gpu_generation():
    """Detect GPU series, return '40' for 40 series, '30' for 30 series, None for others"""
    if not torch.cuda.is_available():
        return None
    try:
        import re

        gpu_name = torch.cuda.get_device_name(0)
        gpu_name_lower = gpu_name.lower()

        # Detect 40 series GPUs (RTX 40xx, RTX 4060, RTX 4070, RTX 4080, RTX 4090, etc.)
        if any(keyword in gpu_name_lower for keyword in ["rtx 40", "rtx40", "geforce rtx 40"]):
            # Further check if it's 40xx series
            match = re.search(r"rtx\s*40\d+|40\d+", gpu_name_lower)
            if match:
                return "40"

        # Detect 30 series GPUs (RTX 30xx, RTX 3060, RTX 3070, RTX 3080, RTX 3090, etc.)
        if any(keyword in gpu_name_lower for keyword in ["rtx 30", "rtx30", "geforce rtx 30"]):
            # Further check if it's 30xx series
            match = re.search(r"rtx\s*30\d+|30\d+", gpu_name_lower)
            if match:
                return "30"

        return None
    except Exception as e:
        logger.warning(f"Cannot detect GPU series: {e}")
        return None


def get_quantization_options(model_path):
    """Dynamically get quantization options based on model_path"""
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

        # If no subdirectories but has original file, add original type
        if has_original_file:
            if not choices or "original" not in choices:
                choices.append(original_type)

        # If no options, use default
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
    """Determine model_cls based on model type and filename"""
    # Determine filename to check
    if model_type == "wan2.1":
        check_name = dit_name.lower() if dit_name else ""
        is_distill = "4step" in check_name
        return "wan2.1_distill" if is_distill else "wan2.1"
    elif model_type == "Qwen-Image-Edit-2511":
        # Qwen-Image-Edit-2511 uses qwen_image model class
        return "qwen_image"
    else:
        # wan2.2
        check_name = high_noise_name.lower() if high_noise_name else ""
        is_distill = "4step" in check_name
        return "wan2.2_moe_distill" if is_distill else "wan2.2_moe"


def is_distill_model_from_name(model_name):
    """Determine if it's a distill model based on model name"""
    if not model_name:
        return None
    return "4step" in model_name.lower()


def get_repo_id_for_model(model_type, is_distill, model_category="dit"):
    """Get corresponding Hugging Face repository ID based on model type, whether distill, and model category"""
    if model_category == "dit":
        if model_type == "wan2.1":
            return "lightx2v/wan2.1-Distill-Models" if is_distill else "lightx2v/wan2.1-Official-Models"
        elif model_type == "Qwen-Image-Edit-2511":
            return "lightx2v/Qwen-Image-Edit-2511-Lightning"
        else:  # wan2.2
            return "lightx2v/wan2.2-Distill-Models" if is_distill else "lightx2v/wan2.2-Official-Models"
    elif model_category == "high_noise" or model_category == "low_noise":
        if is_distill:
            return "lightx2v/wan2.2-Distill-Models"
        else:
            return "lightx2v/wan2.2-Official-Models"
    elif model_category == "t5" or model_category == "clip":
        return "lightx2v/Encoders"
    elif model_category == "qwen25vl":
        return "lightx2v/Encoders"
    elif model_category == "vae":
        return "lightx2v/Autoencoders"
    elif model_category == "qwen_image_vae" or model_category == "qwen_image_scheduler":
        return "Qwen/Qwen-Image-Edit-2511"
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
# Priority order
attn_priority = ["sage_attn2", "flash_attn3", "flash_attn2", "torch_sdpa"]
# Sort by priority, installed first, uninstalled last
attn_op_choices = []
attn_op_dict = dict(available_attn_ops)

# First add installed ones (by priority)
for op_name in attn_priority:
    if op_name in attn_op_dict and attn_op_dict[op_name]:
        status_text = "✅"
        display_text = f"{status_text}{op_name}"
        attn_op_choices.append((op_name, display_text))

# Then add uninstalled ones (by priority)
for op_name in attn_priority:
    if op_name in attn_op_dict and not attn_op_dict[op_name]:
        status_text = "❌"
        display_text = f"{status_text}{op_name}"
        attn_op_choices.append((op_name, display_text))

# Add other operators not in priority list (installed first)
other_ops = [(op_name, is_installed) for op_name, is_installed in available_attn_ops if op_name not in attn_priority]
for op_name, is_installed in sorted(other_ops, key=lambda x: not x[1]):  # Installed first
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
    # Qwen-Image-Edit-2511 related parameters
    qwen_image_dit_path_input=None,
    qwen_image_vae_path_input=None,
    qwen_image_scheduler_path_input=None,
    qwen25vl_encoder_path_input=None,
):
    cleanup_memory()

    # Extract original operator name (remove status indicator ✅/❌)
    def extract_op_name(op_str):
        """Extract original name from formatted operator name"""
        if not op_str:
            return ""
        # Remove leading ✅ or ❌
        op_str = op_str.strip()
        if op_str.startswith("✅"):
            op_str = op_str[1:].strip()
        elif op_str.startswith("❌"):
            op_str = op_str[1:].strip()
        # Remove content after parentheses (if any)
        if "(" in op_str:
            op_str = op_str.split("(")[0].strip()
        return op_str

    quant_op = extract_op_name(quant_op)
    attention_type = extract_op_name(attention_type)

    global global_runner, current_config, model_path, model_cls
    global cur_dit_path, cur_t5_path, cur_clip_path

    # Extract original model name (remove status indicator)
    dit_path_input = extract_model_name(dit_path_input) if dit_path_input else ""
    high_noise_path_input = extract_model_name(high_noise_path_input) if high_noise_path_input else ""
    low_noise_path_input = extract_model_name(low_noise_path_input) if low_noise_path_input else ""
    t5_path_input = extract_model_name(t5_path_input) if t5_path_input else ""
    # Tokenizer fixed name
    t5_tokenizer_path_input = "google"
    clip_path_input = extract_model_name(clip_path_input) if clip_path_input else ""
    clip_tokenizer_path_input = "xlm-roberta-large"
    vae_encoder_path_input = extract_model_name(vae_encoder_path_input) if vae_encoder_path_input else ""
    vae_decoder_path_input = extract_model_name(vae_decoder_path_input) if vae_decoder_path_input else ""

    task = task_type_input
    model_cls = determine_model_cls(model_type_input, dit_path_input, high_noise_path_input)
    logger.info(f"Auto-determined model_cls: {model_cls} (model type: {model_type_input})")

    # Qwen-Image-Edit-2511 related configuration
    qwen_image_dit_path_input = extract_model_name(qwen_image_dit_path_input) if qwen_image_dit_path_input else ""
    qwen_image_vae_path_input = extract_model_name(qwen_image_vae_path_input) if qwen_image_vae_path_input else ""
    qwen_image_scheduler_path_input = extract_model_name(qwen_image_scheduler_path_input) if qwen_image_scheduler_path_input else ""
    qwen25vl_encoder_path_input = extract_model_name(qwen25vl_encoder_path_input) if qwen25vl_encoder_path_input else ""

    if model_type_input == "wan2.1":
        dit_quant_detected = detect_quant_scheme(dit_path_input)
    else:
        dit_quant_detected = detect_quant_scheme(high_noise_path_input)
    t5_quant_detected = detect_quant_scheme(t5_path_input)
    clip_quant_detected = detect_quant_scheme(clip_path_input)
    logger.info(f"Auto-detected quantization precision - DIT: {dit_quant_detected}, T5: {t5_quant_detected}, CLIP: {clip_quant_detected}")

    if model_path_input and model_path_input.strip():
        model_path = model_path_input.strip()

    if model_type_input == "Qwen-Image-Edit-2511":
        model_config = MODEL_CONFIG["Qwen_Image_Edit_2511"]
    else:
        model_config = MODEL_CONFIG["Wan_14b"]

    # Generate output filename based on task type
    is_image_output = task == "i2i"
    save_result_path = generate_unique_filename(output_dir, is_image=is_image_output)

    is_dit_quant = dit_quant_detected in ["fp8", "int8"]
    is_t5_quant = t5_quant_detected in ["fp8", "int8"]
    is_clip_quant = clip_quant_detected in ["fp8", "int8"]

    dit_quantized_ckpt = None
    dit_original_ckpt = None
    high_noise_quantized_ckpt = None
    low_noise_quantized_ckpt = None
    high_noise_original_ckpt = None
    low_noise_original_ckpt = None

    # Handle quant_op: if torch, convert based on quantization type to torchao
    def get_quant_scheme(quant_detected, quant_op_val):
        """Generate quant_scheme based on quantization type and operator"""
        if quant_op_val == "torch":
            # torch option needs to be converted to torchao, format is fp8-torchao or int8-torchao
            return f"{quant_detected}-torchao"
        elif quant_op_val == "triton":
            # triton option format is fp8-triton or int8-triton
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

    # Use frontend-selected T5 path
    if is_t5_quant:
        t5_quantized_ckpt = os.path.join(model_path, t5_path_input)
        t5_quant_scheme = get_quant_scheme(t5_quant_detected, quant_op)
        t5_original_ckpt = None
    else:
        t5_quantized_ckpt = None
        t5_quant_scheme = None
        t5_original_ckpt = os.path.join(model_path, t5_path_input)

    # Use frontend-selected CLIP path
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
    # CLIP path: only needed for wan2.1 with i2v
    if model_type_input == "wan2.1" and task_type_input == "i2v":
        current_clip_path = f"{clip_path_input}|{clip_tokenizer_path_input}" if clip_path_input and clip_tokenizer_path_input else clip_path_input
    else:
        current_clip_path = None

    needs_reinit = lazy_load or unload_modules or global_runner is None or cur_dit_path != current_dit_path or cur_t5_path != current_t5_path or cur_clip_path != current_clip_path
    if cfg_scale == 1:
        enable_cfg = False
    else:
        enable_cfg = True

    # VAE configuration: determine based on decoder path
    vae_encoder_path = vae_encoder_path_input if vae_encoder_path_input else "wan2.1_VAE.safetensors"
    vae_decoder_path = vae_decoder_path_input if vae_decoder_path_input else None

    vae_decoder_name_lower = vae_decoder_path.lower() if vae_decoder_path else ""
    use_tae = "tae" in vae_decoder_name_lower or "lighttae" in vae_decoder_name_lower
    use_lightvae = "lightvae" in vae_decoder_name_lower
    need_scaled = "lighttae" in vae_decoder_name_lower

    # Set vae_path and tae_path based on use_tae
    if use_tae:
        # When use_tae=True: tae_path is decoder path, vae_path is encoder path
        tae_path = os.path.join(model_path, vae_decoder_path) if vae_decoder_path else None
        vae_path = os.path.join(model_path, vae_encoder_path) if vae_encoder_path else None
    else:
        # Otherwise: vae_path is decoder path, tae_path is None
        vae_path = os.path.join(model_path, vae_decoder_path) if vae_decoder_path else None
        tae_path = None

    logger.info(
        f"VAE config - use_tae: {use_tae}, use_lightvae: {use_lightvae}, need_scaled: {need_scaled} (VAE encoder: {vae_encoder_path}, VAE decoder: {vae_decoder_path}, vae_path: {vae_path}, tae_path: {tae_path})"
    )

    # Qwen-Image-Edit-2511 specific configuration
    is_qwen_image = model_cls == "qwen_image"
    qwen_image_config = {
        "attention_out_dim": 3072,
        "attention_dim_head": 128,
        "CONDITION_IMAGE_SIZE": 147456,
        "USE_IMAGE_ID_IN_PROMPT": True,
        "transformer_in_channels": 64,
        "prompt_template_encode": "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        "prompt_template_encode_start_idx": 64,
        "vae_scale_factor": 8,
    }

    if is_qwen_image:
        # Handle qwen_image dit model path
        qwen_dit_quant_detected = detect_quant_scheme(qwen_image_dit_path_input)
        is_qwen_dit_quant = qwen_dit_quant_detected in ["fp8", "int8"]
        if is_qwen_dit_quant:
            qwen_image_config["dit_quantized"] = True
            qwen_image_config["dit_quant_scheme"] = get_quant_scheme(qwen_dit_quant_detected, quant_op)
            qwen_image_config["dit_quantized_ckpt"] = os.path.join(model_path, qwen_image_dit_path_input) if qwen_image_dit_path_input else None
            qwen_image_config["dit_original_ckpt"] = None
        else:
            qwen_image_config["dit_quantized"] = False
            qwen_image_config["dit_quant_scheme"] = "Default"
            qwen_image_config["dit_original_ckpt"] = os.path.join(model_path, qwen_image_dit_path_input) if qwen_image_dit_path_input else None
            qwen_image_config["dit_quantized_ckpt"] = None

        # VAE and Scheduler
        qwen_image_config["vae_path"] = os.path.join(model_path, qwen_image_vae_path_input) if qwen_image_vae_path_input else None
        qwen_image_config["scheduler_path"] = os.path.join(model_path, qwen_image_scheduler_path_input) if qwen_image_scheduler_path_input else None

        # Qwen25-VL encoder (INT4 GPTQ)
        qwen_image_config["qwen25vl_quantized"] = True
        qwen_image_config["qwen25vl_quant_scheme"] = "int4"
        qwen_image_config["qwen25vl_quantized_ckpt"] = os.path.join(model_path, qwen25vl_encoder_path_input) if qwen25vl_encoder_path_input else None
        qwen_image_config["qwen25vl_tokenizer_path"] = os.path.join(model_path, qwen25vl_encoder_path_input) if qwen25vl_encoder_path_input else None
        qwen_image_config["qwen25vl_processor_path"] = os.path.join(model_path, qwen25vl_encoder_path_input) if qwen25vl_encoder_path_input else None

        # Set zero_cond_t to true (2511 feature)
        qwen_image_config["zero_cond_t"] = True
        qwen_image_config["attn_type"] = attention_type

    config_graio = {
        "infer_steps": infer_steps,
        "target_video_length": num_frames,
        "resolution": resolution,
        "resize_mode": "adaptive",
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
        "t5_lazy_load": lazy_load,
        "bucket_shape": {
            "0.667": [[480, 832], [544, 960], [720, 960]],
            "1.500": [[832, 480], [960, 544], [960, 720]],
            "1.000": [[480, 480], [576, 576], [720, 720]],
        },
    }

    # If it's a qwen_image model, override related configuration
    if is_qwen_image:
        config_graio.update(qwen_image_config)

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
    logger.info(f"Inference config:\n{json.dumps(config, indent=4, ensure_ascii=False)}")

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


def auto_configure(resolution, num_frames=81, task_type=None):
    """Automatically set inference options based on machine configuration and resolution"""
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
    if task_type == "i2i":
        attn_priority = ["torch_sdpa", "sage_attn2", "flash_attn3", "flash_attn2"]
    else:
        attn_priority = ["sage_attn2", "flash_attn3", "flash_attn2", "torch_sdpa"]
    # If sm > (9,0) and sage_attn3 is available, place it after sage_attn2
    if is_sm_greater_than_90():
        # Check if sage_attn3 is available
        sage3_available = dict(available_attn_ops).get("sage_attn3", False)
        if sage3_available:
            # Find sage_attn2 position and insert sage_attn3 after it
            if "sage_attn2" in attn_priority:
                sage2_index = attn_priority.index("sage_attn2")
                if "sage_attn3" not in attn_priority:
                    attn_priority.insert(sage2_index + 1, "sage_attn3")
                else:
                    # If already in list, remove and insert at correct position
                    attn_priority.remove("sage_attn3")
                    attn_priority.insert(sage2_index + 1, "sage_attn3")
            else:
                # If no sage_attn2, add to front
                if "sage_attn3" not in attn_priority:
                    attn_priority.insert(0, "sage_attn3")

    # Adjust quant_op priority based on GPU series
    gpu_gen = get_gpu_generation()
    if gpu_gen == "40":
        # 40 series GPU: q8f first
        quant_op_priority = ["q8f", "triton", "vllm", "sgl", "torch"]
    elif gpu_gen == "30":
        # 30 series GPU: vllm first
        quant_op_priority = ["vllm", "triton", "q8f", "sgl", "torch"]
    else:
        # Other cases: keep original order
        quant_op_priority = ["triton", "q8f", "vllm", "sgl", "torch"]

    for op in attn_priority:
        if dict(available_attn_ops).get(op):
            default_config["attention_type_val"] = dict(attn_op_choices)[op]
            break

    for op in quant_op_priority:
        if dict(available_quant_ops).get(op):
            default_config["quant_op_val"] = dict(quant_op_choices)[op]
            break

    if resolution in ["540p", "720p"]:
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

    # If memory is less than 8GB, raise exception
    if cpu_memory < 8:
        raise Exception(
            f"Insufficient system memory: Current available memory is {cpu_memory:.1f}GB, at least 8GB memory is required.\n"
            f"Suggested solutions:\n"
            f"1. Check your machine configuration to ensure sufficient memory\n"
            f"2. Use quantized models (fp8/int8) to reduce memory usage\n"
            f"3. Use smaller models for inference"
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

        /* Model status style */
        .model-status {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 12px !important;
            line-height: 1.2 !important;
            min-height: 20px !important;
        }

        /* Model configuration area style */
        .model-config {
            margin-bottom: 20px !important;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 15px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        /* Input parameters area style */
        .input-params {
            margin-bottom: 20px !important;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 15px;
            background: linear-gradient(135deg, #fff5f5 0%, #ffeef0 100%);
        }

        /* Output video area style */
        .output-video {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            min-height: 400px;
        }

        /* Generate button style */
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

        /* Accordion header style */
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

        /* Video player style */
        .output-video video {
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        /* Diffusion model container */
        .diffusion-model-group {
            margin-bottom: 20px !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /* Remove default border of Gradio components */
        .diffusion-model-group > div,
        .diffusion-model-group .gr-group,
        .diffusion-model-group .gr-box {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /* Encoder group container (text encoder, image encoder) */
        .encoder-group {
            margin-bottom: 20px !important;
        }

        /* VAE group container */
        .vae-group {
            margin-bottom: 20px !important;
        }

        /* Model group title style */
        .model-group-title {
            font-size: 16px !important;
            font-weight: 600 !important;
            margin-bottom: 12px !important;
            color: #24292f !important;
        }

        /* Download button style */
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

        /* Horizontal radio buttons */
        .horizontal-radio .form-radio {
            display: flex !important;
            flex-direction: row !important;
            gap: 20px !important;
        }
        .horizontal-radio .form-radio > label {
            margin-right: 20px !important;
        }

        /* wan2.2 row style - remove top border and divider */
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
    # Load Hugging Face model list cache at startup
    logger.info("Loading Hugging Face model list cache...")
    load_hf_models_cache()
    logger.info("Model list cache loaded")

    with gr.Blocks(title="Lightx2v (Lightweight Video Inference and Generation Engine)") as demo:
        gr.Markdown(f"# 🎬 LightX2V Image/Video Generator")
        gr.HTML(f"<style>{css}</style>")
        # Main layout: left-right columns
        with gr.Row():
            # Left side: configuration and input area
            with gr.Column(scale=5):
                # Model configuration area
                with gr.Accordion("🗂️ Model Configuration", open=True, elem_classes=["model-config"]):
                    gr.Markdown("💡 **Tip**: Please ensure that at least one downloaded ✅ model is available for each model option below, otherwise video generation may not work properly.")
                    # FP8 support notice
                    if not is_fp8_supported_gpu():
                        gr.Markdown("⚠️ **Your device does not support fp8 inference**, models containing fp8 have been automatically hidden.")

                    # Hidden state component
                    model_path_input = gr.Textbox(value=model_path, visible=False)

                    # Model type + Task type + Download source
                    with gr.Row():
                        model_type_input = gr.Radio(
                            label="Model Type",
                            choices=["wan2.1", "wan2.2", "Qwen-Image-Edit-2511"],
                            value="wan2.1",
                            info="wan2.2 requires separate high-noise and low-noise models; Qwen-Image-Edit-2511 is for image editing (i2i)",
                        )
                        task_type_input = gr.Radio(
                            label="Task Type",
                            choices=["i2v", "t2v"],
                            value="i2v",
                            info="i2v: Image-to-Video, t2v: Text-to-Video, i2i: Image Editing",
                        )
                        download_source_input = gr.Radio(
                            label="📥 Download Source",
                            choices=["huggingface", "modelscope"] if (HF_AVAILABLE and MS_AVAILABLE) else (["huggingface"] if HF_AVAILABLE else ["modelscope"] if MS_AVAILABLE else []),
                            value="modelscope" if MS_AVAILABLE else ("huggingface" if HF_AVAILABLE else None),
                            info="Select model download source",
                            visible=HF_AVAILABLE or MS_AVAILABLE,
                            elem_classes=["horizontal-radio"],
                        )

                    # wan2.1: Diffusion model (styled layout)
                    with gr.Column(elem_classes=["diffusion-model-group"]) as wan21_row:
                        with gr.Row():
                            with gr.Column(scale=5):
                                dit_choices_init = get_dit_choices(model_path, "wan2.1", "i2v")
                                dit_path_input = gr.Dropdown(
                                    label="🎨 Diffusion Model",
                                    choices=dit_choices_init,
                                    value=dit_choices_init[0] if dit_choices_init else "",
                                    allow_custom_value=True,
                                    visible=True,
                                )
                            with gr.Column(scale=1, min_width=150):
                                dit_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                        dit_download_status = gr.Markdown("", visible=False)

                    # wan2.2 specific: high-noise model + low-noise model (hidden by default)
                    with gr.Row(visible=False, elem_classes=["wan22-row"]) as wan22_row:
                        with gr.Column(scale=1):
                            high_noise_choices_init = get_high_noise_choices(model_path, "wan2.2", "i2v")
                            high_noise_path_input = gr.Dropdown(
                                label="🔊 High-Noise Model",
                                choices=high_noise_choices_init,
                                value=high_noise_choices_init[0] if high_noise_choices_init else "",
                                allow_custom_value=True,
                            )
                            high_noise_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                            high_noise_download_status = gr.Markdown("", visible=False)
                        with gr.Column(scale=1):
                            low_noise_choices_init = get_low_noise_choices(model_path, "wan2.2", "i2v")
                            low_noise_path_input = gr.Dropdown(
                                label="🔇 Low-Noise Model",
                                choices=low_noise_choices_init,
                                value=low_noise_choices_init[0] if low_noise_choices_init else "",
                                allow_custom_value=True,
                            )
                            low_noise_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                            low_noise_download_status = gr.Markdown("", visible=False)

                    # Qwen-Image-Edit-2511 specific (hidden by default)
                    with gr.Column(visible=False, elem_classes=["diffusion-model-group"]) as qwen_image_row:
                        # Diffusion model
                        with gr.Row():
                            with gr.Column(scale=5):
                                qwen_image_dit_choices_init = get_qwen_image_dit_choices(model_path)
                                qwen_image_dit_path_input = gr.Dropdown(
                                    label="🎨 Diffusion Model (Qwen Image Edit)",
                                    choices=qwen_image_dit_choices_init,
                                    value=qwen_image_dit_choices_init[0] if qwen_image_dit_choices_init else "",
                                    allow_custom_value=True,
                                )
                            with gr.Column(scale=1, min_width=150):
                                qwen_image_dit_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                        qwen_image_dit_download_status = gr.Markdown("", visible=False)

                        # VAE and Scheduler
                        with gr.Row():
                            with gr.Column(scale=1):
                                qwen_image_vae_choices_init = get_qwen_image_vae_choices(model_path)
                                qwen_image_vae_path_input = gr.Dropdown(
                                    label="🎞️ VAE (Image Encoder/Decoder)",
                                    choices=qwen_image_vae_choices_init,
                                    value=qwen_image_vae_choices_init[0] if qwen_image_vae_choices_init else "",
                                    allow_custom_value=True,
                                )
                                qwen_image_vae_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                                qwen_image_vae_download_status = gr.Markdown("", visible=False)
                            with gr.Column(scale=1):
                                qwen_image_scheduler_choices_init = get_qwen_image_scheduler_choices(model_path)
                                qwen_image_scheduler_path_input = gr.Dropdown(
                                    label="⏱️ Scheduler",
                                    choices=qwen_image_scheduler_choices_init,
                                    value=qwen_image_scheduler_choices_init[0] if qwen_image_scheduler_choices_init else "",
                                    allow_custom_value=True,
                                )
                                qwen_image_scheduler_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                                qwen_image_scheduler_download_status = gr.Markdown("", visible=False)

                        # Qwen25-VL encoder
                        with gr.Row():
                            with gr.Column(scale=1):
                                qwen25vl_encoder_choices_init = get_qwen25vl_encoder_choices(model_path)
                                qwen25vl_encoder_path_input = gr.Dropdown(
                                    label="📝 Qwen25-VL Encoder (INT4 GPTQ)",
                                    choices=qwen25vl_encoder_choices_init,
                                    value=qwen25vl_encoder_choices_init[0] if qwen25vl_encoder_choices_init else "",
                                    allow_custom_value=True,
                                )
                                qwen25vl_encoder_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                                qwen25vl_encoder_download_status = gr.Markdown("", visible=False)

                    # Text encoder (Model + Tokenizer)
                    with gr.Row() as t5_row:
                        with gr.Column(scale=1):
                            t5_model_choices_init = get_t5_model_choices(model_path)
                            t5_path_input = gr.Dropdown(
                                label="📝 Text Encoder",
                                choices=t5_model_choices_init,
                                value=t5_model_choices_init[0] if t5_model_choices_init else "",
                                allow_custom_value=True,
                            )
                            t5_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                            t5_download_status = gr.Markdown("", visible=False)
                        with gr.Column(scale=1):
                            t5_tokenizer_hint = gr.Dropdown(
                                label="📝 Text Encoder Tokenizer",
                                choices=["google ✅", "google ❌"],
                                value="google ❌",
                                interactive=False,
                            )
                            t5_tokenizer_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                            t5_tokenizer_download_status = gr.Markdown("", visible=False)

                    # Image encoder (Model + Tokenizer, conditionally shown)
                    with gr.Row(visible=True) as clip_row:
                        with gr.Column(scale=1):
                            clip_model_choices_init = get_clip_model_choices(model_path)
                            clip_path_input = gr.Dropdown(
                                label="🖼️ Image Encoder",
                                choices=clip_model_choices_init,
                                value=clip_model_choices_init[0] if clip_model_choices_init else "",
                                allow_custom_value=True,
                            )
                            clip_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                            clip_download_status = gr.Markdown("", visible=False)
                        with gr.Column(scale=1):
                            clip_tokenizer_hint = gr.Dropdown(
                                label="🖼️ Image Encoder Tokenizer",
                                choices=["xlm-roberta-large ✅", "xlm-roberta-large ❌"],
                                value="xlm-roberta-large ❌",
                                interactive=False,
                            )
                            clip_tokenizer_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                            clip_tokenizer_download_status = gr.Markdown("", visible=False)

                    # VAE (Encoder + Decoder)
                    with gr.Row() as vae_row:
                        with gr.Column(scale=1) as vae_encoder_col:
                            vae_encoder_choices_init = get_vae_encoder_choices(model_path)
                            vae_encoder_path_input = gr.Dropdown(
                                label="VAE Encoder",
                                choices=vae_encoder_choices_init,
                                value=vae_encoder_choices_init[0] if vae_encoder_choices_init else "",
                                allow_custom_value=True,
                                interactive=True,
                            )
                            vae_encoder_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                            vae_encoder_download_status = gr.Markdown("", visible=False)
                        with gr.Column(scale=1) as vae_decoder_col:
                            vae_decoder_choices_init = get_vae_decoder_choices(model_path)
                            vae_decoder_path_input = gr.Dropdown(
                                label="VAE Decoder",
                                choices=vae_decoder_choices_init,
                                value=vae_decoder_choices_init[0] if vae_decoder_choices_init else "",
                                allow_custom_value=True,
                            )
                            vae_decoder_download_btn = gr.Button("📥 Download", visible=False, size="sm", variant="secondary")
                            vae_decoder_download_status = gr.Markdown("", visible=False)

                    # Attention operators and quantized matrix multiplication operators
                    with gr.Row():
                        attention_type = gr.Dropdown(
                            label="⚡ Attention Operator",
                            choices=[op[1] for op in attn_op_choices],
                            value=attn_op_choices[0][1] if attn_op_choices else "",
                            info="Use appropriate attention operators to accelerate inference",
                        )
                        quant_op = gr.Dropdown(
                            label="⚡ MatMul Operator",
                            choices=[op[1] for op in quant_op_choices],
                            value=quant_op_choices[0][1],
                            info="Select low-precision matrix multiplication operators to accelerate inference",
                            interactive=True,
                        )

                    # Determine if model is a distill version
                    def is_distill_model(model_type, dit_path, high_noise_path):
                        """Determine if it's a distill version based on model type and path"""
                        if model_type == "wan2.1":
                            check_name = dit_path.lower() if dit_path else ""
                        else:
                            check_name = high_noise_path.lower() if high_noise_path else ""
                        return "4step" in check_name

                    # Task type change event
                    def on_task_type_change(model_type, task_type, model_path_val):
                        # Determine whether to show CLIP (hidden for wan2.2, t2v, or i2i)
                        show_clip = model_type == "wan2.1" and task_type == "i2v"
                        # Determine whether to show VAE encoder (hidden for t2v, qwen-image-edit has its own VAE)
                        show_vae_encoder = task_type == "i2v" and model_type != "Qwen-Image-Edit-2511"
                        # VAE decoder always shown (qwen-image-edit has its own VAE)
                        show_vae_decoder = model_type != "Qwen-Image-Edit-2511"
                        # Show image input (shown for i2v and i2i)
                        show_image_input = task_type in ["i2v", "i2i"]
                        # Output label
                        output_label = "Output Image Path" if task_type == "i2i" else "Output Video Path"

                        # Update model options based on task type
                        if model_type == "wan2.1":
                            dit_choices = get_dit_choices(model_path_val, "wan2.1", task_type if task_type != "i2i" else "i2v")
                            t5_choices = get_t5_model_choices(model_path_val)
                            clip_choices = get_clip_model_choices(model_path_val) if show_clip else []
                            vae_encoder_choices = get_vae_encoder_choices(model_path_val) if show_vae_encoder else []
                            vae_decoder_choices = get_vae_decoder_choices(model_path_val)

                            return (
                                gr.update(visible=show_clip),  # clip_row
                                gr.update(visible=show_vae_encoder),  # vae_encoder_col
                                gr.update(visible=show_vae_decoder),  # vae_decoder_col
                                gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else ""),  # dit_path_input
                                gr.update(),  # high_noise_path_input (not used for wan2.1)
                                gr.update(),  # low_noise_path_input (not used for wan2.1)
                                gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),  # t5_path_input
                                gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else ""),  # clip_path_input
                                gr.update(choices=vae_encoder_choices, value=vae_encoder_choices[0] if vae_encoder_choices else ""),  # vae_encoder_path_input
                                gr.update(choices=vae_decoder_choices, value=vae_decoder_choices[0] if vae_decoder_choices else ""),  # vae_decoder_path_input
                                gr.update(visible=show_image_input),  # image_input_row
                                gr.update(label=output_label),  # save_result_path
                            )
                        elif model_type == "Qwen-Image-Edit-2511":
                            # qwen-image-edit only supports i2i
                            qwen_image_dit_choices = get_qwen_image_dit_choices(model_path_val)
                            qwen_image_vae_choices = get_qwen_image_vae_choices(model_path_val)
                            qwen_image_scheduler_choices = get_qwen_image_scheduler_choices(model_path_val)
                            qwen25vl_encoder_choices = get_qwen25vl_encoder_choices(model_path_val)

                            return (
                                gr.update(visible=False),  # clip_row
                                gr.update(visible=False),  # vae_encoder_col
                                gr.update(visible=False),  # vae_decoder_col
                                gr.update(),  # dit_path_input
                                gr.update(),  # high_noise_path_input
                                gr.update(),  # low_noise_path_input
                                gr.update(),  # t5_path_input
                                gr.update(),  # clip_path_input
                                gr.update(),  # vae_encoder_path_input
                                gr.update(),  # vae_decoder_path_input
                                gr.update(visible=True),  # image_input_row (i2i needs image input)
                                gr.update(label="Output Image Path"),  # save_result_path
                            )
                        else:  # wan2.2
                            high_noise_choices = get_high_noise_choices(model_path_val, "wan2.2", task_type if task_type != "i2i" else "i2v")
                            low_noise_choices = get_low_noise_choices(model_path_val, "wan2.2", task_type if task_type != "i2i" else "i2v")
                            t5_choices = get_t5_model_choices(model_path_val)
                            vae_encoder_choices = get_vae_encoder_choices(model_path_val) if show_vae_encoder else []
                            vae_decoder_choices = get_vae_decoder_choices(model_path_val)

                            return (
                                gr.update(visible=show_clip),  # clip_row
                                gr.update(visible=show_vae_encoder),  # vae_encoder_col
                                gr.update(visible=show_vae_decoder),  # vae_decoder_col
                                gr.update(),  # dit_path_input (not used for wan2.2)
                                gr.update(choices=high_noise_choices, value=high_noise_choices[0] if high_noise_choices else ""),  # high_noise_path_input
                                gr.update(choices=low_noise_choices, value=low_noise_choices[0] if low_noise_choices else ""),  # low_noise_path_input
                                gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),  # t5_path_input
                                gr.update(),  # clip_path_input (not used for wan2.2)
                                gr.update(choices=vae_encoder_choices, value=vae_encoder_choices[0] if vae_encoder_choices else ""),  # vae_encoder_path_input
                                gr.update(choices=vae_decoder_choices, value=vae_decoder_choices[0] if vae_decoder_choices else ""),  # vae_decoder_path_input
                                gr.update(visible=show_image_input),  # image_input_row
                                gr.update(label=output_label),  # save_result_path
                            )

                    # Model type change event
                    def on_model_type_change(model_type, model_path_val, task_type):
                        # Determine whether to show CLIP (hidden for wan2.2, t2v, or qwen-image-edit)
                        show_clip = model_type == "wan2.1" and task_type == "i2v"
                        # Determine whether to show VAE encoder (hidden for t2v, qwen-image-edit has its own VAE)
                        show_vae_encoder = task_type == "i2v" and model_type != "Qwen-Image-Edit-2511"
                        # VAE decoder always shown (qwen-image-edit has its own VAE)
                        show_vae_decoder = model_type != "Qwen-Image-Edit-2511"
                        # Show text encoder row (qwen-image-edit has its own encoder)
                        show_t5_row = model_type != "Qwen-Image-Edit-2511"

                        if model_type == "Qwen-Image-Edit-2511":
                            # Update qwen-image-edit model options
                            qwen_image_dit_choices = get_qwen_image_dit_choices(model_path_val)
                            qwen_image_vae_choices = get_qwen_image_vae_choices(model_path_val)
                            qwen_image_scheduler_choices = get_qwen_image_scheduler_choices(model_path_val)
                            qwen25vl_encoder_choices = get_qwen25vl_encoder_choices(model_path_val)

                            return (
                                gr.update(visible=False),  # wan21_row
                                gr.update(visible=False),  # wan22_row
                                gr.update(visible=True),  # qwen_image_row
                                gr.update(visible=False),  # dit_path_input (not used for qwen-image-edit)
                                gr.update(),  # high_noise_path_input
                                gr.update(),  # low_noise_path_input
                                gr.update(visible=False),  # clip_row
                                gr.update(visible=False),  # vae_encoder_col
                                gr.update(visible=False),  # vae_decoder_col
                                gr.update(visible=False),  # t5_row
                                gr.update(),  # t5_path_input
                                gr.update(),  # clip_path_input
                                gr.update(),  # vae_encoder_path_input
                                gr.update(),  # vae_decoder_path_input
                                gr.update(visible=False),  # dit_download_btn
                                gr.update(visible=False),  # dit_download_status
                                gr.update(choices=qwen_image_dit_choices, value=qwen_image_dit_choices[0] if qwen_image_dit_choices else ""),  # qwen_image_dit_path_input
                                gr.update(choices=qwen_image_vae_choices, value=qwen_image_vae_choices[0] if qwen_image_vae_choices else ""),  # qwen_image_vae_path_input
                                gr.update(choices=qwen_image_scheduler_choices, value=qwen_image_scheduler_choices[0] if qwen_image_scheduler_choices else ""),  # qwen_image_scheduler_path_input
                                gr.update(choices=qwen25vl_encoder_choices, value=qwen25vl_encoder_choices[0] if qwen25vl_encoder_choices else ""),  # qwen25vl_encoder_path_input
                                gr.update(choices=["i2i"], value="i2i"),  # task_type_input only shows i2i
                            )
                        elif model_type == "wan2.2":
                            # Update wan2.2 high-noise and low-noise model options
                            high_noise_choices = get_high_noise_choices(model_path_val, "wan2.2", task_type)
                            low_noise_choices = get_low_noise_choices(model_path_val, "wan2.2", task_type)
                            t5_choices = get_t5_model_choices(model_path_val)
                            clip_choices = get_clip_model_choices(model_path_val) if show_clip else []
                            vae_encoder_choices = get_vae_encoder_choices(model_path_val) if show_vae_encoder else []
                            vae_decoder_choices = get_vae_decoder_choices(model_path_val)

                            return (
                                gr.update(visible=False),  # wan21_row
                                gr.update(visible=True),  # wan22_row
                                gr.update(visible=False),  # qwen_image_row
                                gr.update(visible=False),  # dit_path_input (not used for wan2.2)
                                gr.update(choices=high_noise_choices, value=high_noise_choices[0] if high_noise_choices else ""),  # high_noise_path_input
                                gr.update(choices=low_noise_choices, value=low_noise_choices[0] if low_noise_choices else ""),  # low_noise_path_input
                                gr.update(visible=show_clip),  # clip_row
                                gr.update(visible=show_vae_encoder),  # vae_encoder_col
                                gr.update(visible=show_vae_decoder),  # vae_decoder_col
                                gr.update(visible=True),  # t5_row
                                gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),  # t5_path_input
                                gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else ""),  # clip_path_input
                                gr.update(choices=vae_encoder_choices, value=vae_encoder_choices[0] if vae_encoder_choices else ""),  # vae_encoder_path_input
                                gr.update(choices=vae_decoder_choices, value=vae_decoder_choices[0] if vae_decoder_choices else ""),  # vae_decoder_path_input
                                gr.update(visible=False),  # dit_download_btn
                                gr.update(visible=False),  # dit_download_status
                                gr.update(),  # qwen_image_dit_path_input
                                gr.update(),  # qwen_image_vae_path_input
                                gr.update(),  # qwen_image_scheduler_path_input
                                gr.update(),  # qwen25vl_encoder_path_input
                                gr.update(choices=["i2v", "t2v"], value=task_type if task_type in ["i2v", "t2v"] else "i2v"),  # task_type_input only shows i2v and t2v
                            )
                        else:
                            # Update wan2.1 Diffusion model options
                            dit_choices = get_dit_choices(model_path_val, "wan2.1", task_type)
                            t5_choices = get_t5_model_choices(model_path_val)
                            clip_choices = get_clip_model_choices(model_path_val) if show_clip else []
                            vae_encoder_choices = get_vae_encoder_choices(model_path_val) if show_vae_encoder else []
                            vae_decoder_choices = get_vae_decoder_choices(model_path_val)

                            return (
                                gr.update(visible=True),  # wan21_row
                                gr.update(visible=False),  # wan22_row
                                gr.update(visible=False),  # qwen_image_row
                                gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else "", visible=True),  # dit_path_input
                                gr.update(),  # high_noise_path_input (used for wan2.2)
                                gr.update(),  # low_noise_path_input (used for wan2.2)
                                gr.update(visible=show_clip),  # clip_row
                                gr.update(visible=show_vae_encoder),  # vae_encoder_col
                                gr.update(visible=show_vae_decoder),  # vae_decoder_col
                                gr.update(visible=True),  # t5_row
                                gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),  # t5_path_input
                                gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else ""),  # clip_path_input
                                gr.update(choices=vae_encoder_choices, value=vae_encoder_choices[0] if vae_encoder_choices else ""),  # vae_encoder_path_input
                                gr.update(choices=vae_decoder_choices, value=vae_decoder_choices[0] if vae_decoder_choices else ""),  # vae_decoder_path_input
                                gr.update(),  # dit_download_btn (visibility controlled by wan21_row)
                                gr.update(),  # dit_download_status (visibility controlled by wan21_row)
                                gr.update(),  # qwen_image_dit_path_input
                                gr.update(),  # qwen_image_vae_path_input
                                gr.update(),  # qwen_image_scheduler_path_input
                                gr.update(),  # qwen25vl_encoder_path_input
                                gr.update(choices=["i2v", "t2v"], value=task_type if task_type in ["i2v", "t2v"] else "i2v"),  # task_type_input only shows i2v and t2v
                            )

                    model_type_input.change(
                        fn=on_model_type_change,
                        inputs=[model_type_input, model_path_input, task_type_input],
                        outputs=[
                            wan21_row,
                            wan22_row,
                            qwen_image_row,
                            dit_path_input,
                            high_noise_path_input,
                            low_noise_path_input,
                            clip_row,
                            vae_encoder_col,
                            vae_decoder_col,
                            t5_row,
                            t5_path_input,
                            clip_path_input,
                            vae_encoder_path_input,
                            vae_decoder_path_input,
                            dit_download_btn,
                            dit_download_status,
                            qwen_image_dit_path_input,
                            qwen_image_vae_path_input,
                            qwen_image_scheduler_path_input,
                            qwen25vl_encoder_path_input,
                            task_type_input,
                        ],
                    )

                    # Functions to update model download status
                    def update_dit_status(model_path_val, model_name, model_type_val):
                        if not model_name:
                            return gr.update(visible=False)
                        # Extract original model name (remove status indicator)
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
                        # Extract original model name (remove status indicator)
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "t5")
                        exists = check_model_exists(model_path_val, actual_name)
                        if exists:
                            return gr.update(visible=False)
                        else:
                            return gr.update(visible=True)

                    def update_t5_tokenizer_status(model_path_val):
                        """Update T5 Tokenizer (google) status"""
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
                        # Extract original model name (remove status indicator)
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "clip")
                        exists = check_model_exists(model_path_val, actual_name)
                        if exists:
                            return gr.update(visible=False)
                        else:
                            return gr.update(visible=True)

                    def update_clip_tokenizer_status(model_path_val):
                        """Update CLIP Tokenizer (xlm-roberta-large) status"""
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
                        # Extract original model name (remove status indicator)
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
                        # Extract original model name (remove status indicator)
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
                        # Extract original model name (remove status indicator)
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
                        # Extract original model name (remove status indicator)
                        actual_name = extract_model_name(model_name)
                        is_distill = is_distill_model_from_name(actual_name)
                        repo_id = get_repo_id_for_model("wan2.2", is_distill, "low_noise")
                        exists = check_model_exists(model_path_val, actual_name)
                        if exists:
                            return gr.update(visible=False)
                        else:
                            return gr.update(visible=True)

                    # qwen-image-edit related status update functions
                    def update_qwen_image_dit_status(model_path_val, model_name):
                        if not model_name:
                            return gr.update(visible=False)
                        actual_name = extract_model_name(model_name)
                        exists = check_model_exists(model_path_val, actual_name)
                        return gr.update(visible=not exists)

                    def update_qwen_image_vae_status(model_path_val, model_name):
                        if not model_name:
                            return gr.update(visible=False)
                        actual_name = extract_model_name(model_name)
                        exists = check_model_exists(model_path_val, actual_name)
                        return gr.update(visible=not exists)

                    def update_qwen_image_scheduler_status(model_path_val, model_name):
                        if not model_name:
                            return gr.update(visible=False)
                        actual_name = extract_model_name(model_name)
                        exists = check_model_exists(model_path_val, actual_name)
                        return gr.update(visible=not exists)

                    def update_qwen25vl_encoder_status(model_path_val, model_name):
                        if not model_name:
                            return gr.update(visible=False)
                        actual_name = extract_model_name(model_name)
                        exists = check_model_exists(model_path_val, actual_name)
                        return gr.update(visible=not exists)

                    # Download functions
                    def download_dit_model(model_path_val, model_name, model_type_val, task_type_val, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        # Extract original model name (remove status indicator)
                        actual_name = extract_model_name(model_name)
                        is_distill = is_distill_model_from_name(actual_name)
                        repo_id = get_repo_id_for_model(model_type_val, is_distill, "dit")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        # Update status and options
                        btn_visible = update_dit_status(model_path_val, format_model_choice(actual_name, model_path_val), model_type_val)
                        choices = get_dit_choices(model_path_val, model_type_val, task_type_val)
                        # Find the updated option value
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_t5_model(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        # Extract original model name (remove status indicator)
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "t5")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_t5_model_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_t5_model_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_t5_tokenizer(model_path_val, download_source_val, progress=gr.Progress()):
                        """Download T5 Tokenizer (google)"""
                        tokenizer_name = "google"
                        repo_id = get_repo_id_for_model(None, None, "t5")
                        result = download_model(repo_id, tokenizer_name, model_path_val, download_source_val, progress)
                        dropdown_update, btn_visible = update_t5_tokenizer_status(model_path_val)
                        return gr.update(value=result), dropdown_update, btn_visible

                    def download_clip_model(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        # Extract original model name (remove status indicator)
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "clip")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_clip_model_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_clip_model_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_clip_tokenizer(model_path_val, download_source_val, progress=gr.Progress()):
                        """Download CLIP Tokenizer (xlm-roberta-large)"""
                        tokenizer_name = "xlm-roberta-large"
                        repo_id = get_repo_id_for_model(None, None, "clip")
                        result = download_model(repo_id, tokenizer_name, model_path_val, download_source_val, progress)
                        dropdown_update, btn_visible = update_clip_tokenizer_status(model_path_val)
                        return gr.update(value=result), dropdown_update, btn_visible

                    def download_vae_encoder(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        # Extract original model name (remove status indicator)
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "vae")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_vae_encoder_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_vae_encoder_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_vae_decoder(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        # Extract original model name (remove status indicator)
                        actual_name = extract_model_name(model_name)
                        repo_id = get_repo_id_for_model(None, None, "vae")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_vae_decoder_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_vae_decoder_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_high_noise_model(model_path_val, model_name, task_type_val, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        # Extract original model name (remove status indicator)
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
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        # Extract original model name (remove status indicator)
                        actual_name = extract_model_name(model_name)
                        is_distill = is_distill_model_from_name(actual_name)
                        repo_id = get_repo_id_for_model("wan2.2", is_distill, "low_noise")
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_low_noise_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_low_noise_choices(model_path_val, "wan2.2", task_type_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    # qwen-image-edit download functions
                    def download_qwen_image_dit(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        actual_name = extract_model_name(model_name)
                        repo_id = "lightx2v/Qwen-Image-Edit-2511-Lightning"
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_qwen_image_dit_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_qwen_image_dit_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_qwen_image_vae(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        actual_name = extract_model_name(model_name)
                        repo_id = "Qwen/Qwen-Image-Edit-2511"
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_qwen_image_vae_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_qwen_image_vae_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_qwen_image_scheduler(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        actual_name = extract_model_name(model_name)
                        repo_id = "Qwen/Qwen-Image-Edit-2511"
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_qwen_image_scheduler_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_qwen_image_scheduler_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    def download_qwen25vl_encoder(model_path_val, model_name, download_source_val, progress=gr.Progress()):
                        if not model_name:
                            return gr.update(value="Please select a model first"), gr.update(visible=False), gr.update()
                        actual_name = extract_model_name(model_name)
                        repo_id = "lightx2v/Encoders"
                        result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress)
                        btn_visible = update_qwen25vl_encoder_status(model_path_val, format_model_choice(actual_name, model_path_val))
                        choices = get_qwen25vl_encoder_choices(model_path_val)
                        updated_value = format_model_choice(actual_name, model_path_val)
                        return gr.update(value=result), btn_visible, gr.update(choices=choices, value=updated_value)

                    # Bind events: update status when model selection changes
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
                        """Simultaneously update T5 model and Tokenizer status"""
                        model_btn = update_t5_model_status(model_path_val, model_name)
                        tokenizer_dropdown, tokenizer_btn = update_t5_tokenizer_status(model_path_val)
                        return model_btn, tokenizer_dropdown, tokenizer_btn

                    t5_path_input.change(
                        fn=update_t5_model_and_tokenizer_status,
                        inputs=[model_path_input, t5_path_input],
                        outputs=[t5_download_btn, t5_tokenizer_hint, t5_tokenizer_download_btn],
                    )

                    def update_clip_model_and_tokenizer_status(model_path_val, model_name):
                        """Simultaneously update CLIP model and Tokenizer status"""
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

                    # Bind download button events
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

                    # qwen-image-edit related event bindings
                    qwen_image_dit_path_input.change(
                        fn=update_qwen_image_dit_status,
                        inputs=[model_path_input, qwen_image_dit_path_input],
                        outputs=[qwen_image_dit_download_btn],
                    )

                    qwen_image_vae_path_input.change(
                        fn=update_qwen_image_vae_status,
                        inputs=[model_path_input, qwen_image_vae_path_input],
                        outputs=[qwen_image_vae_download_btn],
                    )

                    qwen_image_scheduler_path_input.change(
                        fn=update_qwen_image_scheduler_status,
                        inputs=[model_path_input, qwen_image_scheduler_path_input],
                        outputs=[qwen_image_scheduler_download_btn],
                    )

                    qwen25vl_encoder_path_input.change(
                        fn=update_qwen25vl_encoder_status,
                        inputs=[model_path_input, qwen25vl_encoder_path_input],
                        outputs=[qwen25vl_encoder_download_btn],
                    )

                    qwen_image_dit_download_btn.click(
                        fn=download_qwen_image_dit,
                        inputs=[model_path_input, qwen_image_dit_path_input, download_source_input],
                        outputs=[qwen_image_dit_download_status, qwen_image_dit_download_btn, qwen_image_dit_path_input],
                    )

                    qwen_image_vae_download_btn.click(
                        fn=download_qwen_image_vae,
                        inputs=[model_path_input, qwen_image_vae_path_input, download_source_input],
                        outputs=[qwen_image_vae_download_status, qwen_image_vae_download_btn, qwen_image_vae_path_input],
                    )

                    qwen_image_scheduler_download_btn.click(
                        fn=download_qwen_image_scheduler,
                        inputs=[model_path_input, qwen_image_scheduler_path_input, download_source_input],
                        outputs=[qwen_image_scheduler_download_status, qwen_image_scheduler_download_btn, qwen_image_scheduler_path_input],
                    )

                    qwen25vl_encoder_download_btn.click(
                        fn=download_qwen25vl_encoder,
                        inputs=[model_path_input, qwen25vl_encoder_path_input, download_source_input],
                        outputs=[qwen25vl_encoder_download_status, qwen25vl_encoder_download_btn, qwen25vl_encoder_path_input],
                    )

                    # Initialize all model statuses
                    def init_all_statuses(
                        model_path_val,
                        dit_name,
                        high_noise_name,
                        low_noise_name,
                        t5_name,
                        clip_name,
                        vae_encoder_name,
                        vae_decoder_name,
                        model_type_val,
                        qwen_image_dit_name,
                        qwen_image_vae_name,
                        qwen_image_scheduler_name,
                        qwen25vl_encoder_name,
                    ):
                        dit_btn_visible = update_dit_status(model_path_val, dit_name, model_type_val)
                        high_noise_btn_visible = update_high_noise_status(model_path_val, high_noise_name)
                        low_noise_btn_visible = update_low_noise_status(model_path_val, low_noise_name)
                        t5_btn_visible = update_t5_model_status(model_path_val, t5_name)
                        t5_tokenizer_dropdown_val, t5_tokenizer_btn_visible = update_t5_tokenizer_status(model_path_val)
                        clip_btn_visible = update_clip_model_status(model_path_val, clip_name)
                        clip_tokenizer_dropdown_val, clip_tokenizer_btn_visible = update_clip_tokenizer_status(model_path_val)
                        vae_encoder_btn_visible = update_vae_encoder_status(model_path_val, vae_encoder_name)
                        vae_decoder_btn_visible = update_vae_decoder_status(model_path_val, vae_decoder_name)
                        # qwen-image-edit related status
                        qwen_image_dit_btn_visible = update_qwen_image_dit_status(model_path_val, qwen_image_dit_name)
                        qwen_image_vae_btn_visible = update_qwen_image_vae_status(model_path_val, qwen_image_vae_name)
                        qwen_image_scheduler_btn_visible = update_qwen_image_scheduler_status(model_path_val, qwen_image_scheduler_name)
                        qwen25vl_encoder_btn_visible = update_qwen25vl_encoder_status(model_path_val, qwen25vl_encoder_name)
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
                            qwen_image_dit_btn_visible,
                            qwen_image_vae_btn_visible,
                            qwen_image_scheduler_btn_visible,
                            qwen25vl_encoder_btn_visible,
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
                            qwen_image_dit_path_input,
                            qwen_image_vae_path_input,
                            qwen_image_scheduler_path_input,
                            qwen25vl_encoder_path_input,
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
                            qwen_image_dit_download_btn,
                            qwen_image_vae_download_btn,
                            qwen_image_scheduler_download_btn,
                            qwen25vl_encoder_download_btn,
                        ],
                    )

                # Input parameters area
                with gr.Accordion("📥 Input Parameters", open=True, elem_classes=["input-params"]):
                    # Image input (shown for i2v)
                    with gr.Column(visible=True) as image_input_row:
                        image_files = gr.File(
                            label="Input Images (drag and drop multiple images)",
                            file_count="multiple",
                            file_types=["image"],
                            height=150,
                            interactive=True,
                        )
                        # Image preview Gallery
                        image_gallery = gr.Gallery(
                            label="Uploaded Images Preview",
                            columns=4,
                            rows=2,
                            height=200,
                            object_fit="contain",
                            show_label=True,
                        )
                        # Convert multiple file paths to comma-separated string
                        image_path = gr.Textbox(
                            label="Image Path",
                            visible=False,
                        )

                        def update_image_path_and_gallery(files):
                            if files is None or len(files) == 0:
                                return "", []
                            # Extract file paths
                            paths = [f.name if hasattr(f, "name") else f for f in files]
                            # Return comma-separated paths and image list for Gallery display
                            return ",".join(paths), paths

                        image_files.change(
                            fn=update_image_path_and_gallery,
                            inputs=[image_files],
                            outputs=[image_path, image_gallery],
                        )

                    # Task type change event (image input display logic)
                    def on_task_type_change_image(task_type):
                        # Show image input for i2v and i2i
                        return gr.update(visible=(task_type in ["i2v", "i2i"]))

                    task_type_input.change(
                        fn=on_task_type_change_image,
                        inputs=[task_type_input],
                        outputs=[image_input_row],
                    )

                    with gr.Row():
                        with gr.Column():
                            prompt = gr.Textbox(
                                label="Prompt",
                                lines=3,
                                placeholder="Describe video/image content...",
                                max_lines=5,
                            )
                        with gr.Column():
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                lines=3,
                                placeholder="Content you don't want in the video/image...",
                                max_lines=5,
                                value="camera shake, saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards",
                            )
                        with gr.Column(visible=True) as resolution_col:
                            resolution = gr.Dropdown(
                                choices=["480p", "540p", "720p"],
                                value="480p",
                                label="Max Resolution",
                                info="Lower the resolution if you run out of VRAM",
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
                            default_dit = get_dit_choices(model_path, "wan2.1", "i2v")[0] if get_dit_choices(model_path, "wan2.1", "i2v") else ""
                            default_high_noise = get_high_noise_choices(model_path, "wan2.2", "i2v")[0] if get_high_noise_choices(model_path, "wan2.2", "i2v") else ""
                            default_is_distill = is_distill_model("wan2.1", default_dit, default_high_noise)
                            if default_is_distill:
                                infer_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=4,
                                    info="Default 4 steps for distilled models.",
                                )
                            else:
                                infer_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=40,
                                    info="Inference steps for video generation. More steps may improve quality but reduce speed.",
                                )

                            # Dynamically update inference steps when model path changes
                            def update_infer_steps(model_type, dit_path, high_noise_path):
                                # Qwen-Image-Edit-2511 (i2i) defaults to 8 steps
                                if model_type == "Qwen-Image-Edit-2511":
                                    return gr.update(minimum=1, maximum=100, value=8, interactive=True)
                                else:
                                    is_distill = is_distill_model(model_type, dit_path, high_noise_path)
                                    if is_distill:
                                        return gr.update(minimum=1, maximum=100, value=4, interactive=True)
                                    else:
                                        return gr.update(minimum=1, maximum=100, value=40, interactive=True)

                            # Listen for model path changes
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

                    # Set default CFG based on model type
                    # CFG scale: defaults to 1 for distill, otherwise 5
                    default_cfg_scale = 1 if default_is_distill else 5
                    # enable_cfg not exposed to frontend, auto-set based on cfg_scale
                    # If cfg_scale == 1, enable_cfg = False, otherwise enable_cfg = True
                    default_enable_cfg = False if default_cfg_scale == 1 else True
                    enable_cfg = gr.Checkbox(
                        label="Enable Classifier-Free Guidance",
                        value=default_enable_cfg,
                        visible=False,  # Hidden from frontend
                    )

                    with gr.Row():
                        sample_shift = gr.Slider(
                            label="Sample Shift",
                            value=5,
                            minimum=0,
                            maximum=10,
                            step=1,
                            info="Controls the degree of sample distribution shift. Higher values mean more shift.",
                        )
                        cfg_scale = gr.Slider(
                            label="CFG Scale",
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=default_cfg_scale,
                            info="Controls prompt influence strength. Higher values increase prompt influence. CFG is automatically disabled when value is 1.",
                        )

                    # Update enable_cfg based on cfg_scale
                    def update_enable_cfg(cfg_scale_val):
                        """Auto-set enable_cfg based on cfg_scale value"""
                        if cfg_scale_val == 1:
                            return gr.update(value=False)
                        else:
                            return gr.update(value=True)

                    # Dynamically update CFG scale and enable_cfg when model path changes
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

                    with gr.Row(visible=True) as fps_frames_row:
                        fps = gr.Slider(
                            label="Frames Per Second (FPS)",
                            minimum=8,
                            maximum=30,
                            step=1,
                            value=16,
                            info="Video frames per second. Higher FPS produces smoother video.",
                        )
                        num_frames = gr.Slider(
                            label="Total Frames",
                            minimum=16,
                            maximum=120,
                            step=1,
                            value=81,
                            info="Total frames in the video. More frames produce longer video.",
                        )

                    save_result_path = gr.Textbox(
                        label="Output Video Path",
                        value=generate_unique_filename(output_dir),
                        info="Must include .mp4 extension. A unique filename will be auto-generated if left empty or using default.",
                        visible=False,  # Hidden, auto-generated
                    )

            with gr.Column(scale=4):
                with gr.Accordion("📤 Generated Result", open=True, elem_classes=["output-video"]) as output_accordion:
                    output_video = gr.Video(
                        label="",
                        height=600,
                        autoplay=True,
                        show_label=False,
                        visible=True,
                    )
                    output_image = gr.Image(
                        label="Output Image",
                        height=600,
                        show_label=False,
                        visible=False,
                    )

                    infer_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg", elem_classes=["generate-btn"])

            rope_chunk = gr.Checkbox(label="Chunked RoPE", value=False, visible=False)
            rope_chunk_size = gr.Slider(label="RoPE Chunk Size", value=100, minimum=100, maximum=10000, step=100, visible=False)
            unload_modules = gr.Checkbox(label="Unload Modules", value=False, visible=False)
            clean_cuda_cache = gr.Checkbox(label="Clean CUDA Cache", value=False, visible=False)
            cpu_offload = gr.Checkbox(label="CPU Offload", value=False, visible=False)
            lazy_load = gr.Checkbox(label="Enable Lazy Load", value=False, visible=False)
            offload_granularity = gr.Dropdown(label="DiT Offload Granularity", choices=["block", "phase"], value="phase", visible=False)
            t5_cpu_offload = gr.Checkbox(label="T5 CPU Offload", value=False, visible=False)
            clip_cpu_offload = gr.Checkbox(label="CLIP CPU Offload", value=False, visible=False)
            vae_cpu_offload = gr.Checkbox(label="VAE CPU Offload", value=False, visible=False)
            use_tiling_vae = gr.Checkbox(label="VAE Tiled Inference", value=False, visible=False)

        # Update output components and button when task type changes
        def on_task_type_change_output(task_type):
            is_i2i = task_type == "i2i"
            btn_text = "🖼️ Generate Image" if is_i2i else "🎬 Generate Video"
            # Default negative prompt to empty for i2i task
            default_negative = (
                ""
                if is_i2i
                else "camera shake, saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, many people in background, walking backwards"
            )
            return (
                gr.update(visible=not is_i2i),  # output_video
                gr.update(visible=is_i2i),  # output_image
                gr.update(value=btn_text),  # infer_btn
                gr.update(value=default_negative),  # negative_prompt
                gr.update(visible=not is_i2i),  # resolution_col (hide resolution for i2i)
                gr.update(visible=not is_i2i),  # fps_frames_row (hide frame settings for i2i)
            )

        task_type_input.change(
            fn=on_task_type_change_output,
            inputs=[task_type_input],
            outputs=[output_video, output_image, infer_btn, negative_prompt, resolution_col, fps_frames_row],
        )

        # Update model options when task type changes
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
                image_input_row,
                save_result_path,
            ],
        )

        task_type_input.change(
            fn=auto_configure,
            inputs=[resolution, num_frames, task_type_input],
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

        resolution.change(
            fn=auto_configure,
            inputs=[resolution, num_frames, task_type_input],
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
            inputs=[resolution, num_frames, task_type_input],
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

        # Wrap inference function to return to correct output component based on task type
        def run_inference_wrapper(*args):
            result = run_inference(*args)
            # Get task_type_input value (28th parameter, index 27)
            task_type = args[27]
            if task_type == "i2i":
                # i2i task returns image
                return gr.update(), gr.update(value=result)
            else:
                # Video task returns video
                return gr.update(value=result), gr.update()

        infer_btn.click(
            fn=run_inference_wrapper,
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
                # Qwen-Image-Edit-2511 related parameters
                qwen_image_dit_path_input,
                qwen_image_vae_path_input,
                qwen_image_scheduler_path_input,
                qwen25vl_encoder_path_input,
            ],
            outputs=[output_video, output_image],
        )

    # max_file_size: Set upload file size limit, "1gb" means maximum 1GB
    demo.launch(share=True, server_port=args.server_port, server_name=args.server_name, inbrowser=True, allowed_paths=[output_dir], max_file_size="1gb")


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
