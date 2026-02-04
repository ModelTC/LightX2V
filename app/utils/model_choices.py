"""模型选择相关函数模块"""

import os

from utils.model_utils import (
    HF_AVAILABLE,
    format_model_choice,
    get_hf_models,
    scan_model_path_contents,
    sort_model_choices,
)


def _get_models_from_repos(repo_ids, prefix_filter=None, keyword_filter=None):
    """从多个仓库获取模型并合并"""
    all_models = []
    for repo_id in repo_ids:
        models = get_hf_models(repo_id, prefix_filter=prefix_filter, keyword_filter=keyword_filter) if HF_AVAILABLE else []
        all_models.extend(models)
    return list(set(all_models))


def _filter_and_format_models(hf_models, model_path, is_valid_func, require_safetensors=True, additional_filters=None):
    """通用的模型过滤和格式化函数"""
    # 延迟导入避免循环依赖
    from utils.model_utils import is_fp8_supported_gpu

    fp8_supported = is_fp8_supported_gpu()

    # 筛选 HF 模型
    valid_hf_models = []
    for m in hf_models:
        if not is_valid_func(m, fp8_supported):
            continue
        if require_safetensors:
            if m.endswith(".safetensors") or "_split" in m.lower():
                valid_hf_models.append(m)
        else:
            valid_hf_models.append(m)

    # 检查本地已存在的模型
    contents = scan_model_path_contents(model_path)

    local_models = []
    if require_safetensors:
        dir_choices = [d for d in contents["dirs"] if is_valid_func(d, fp8_supported) and ("_split" in d.lower() or d in contents["safetensors_dirs"])]
        safetensors_choices = [f for f in contents["files"] if f.endswith(".safetensors") and is_valid_func(f, fp8_supported)]
        safetensors_dir_choices = [d for d in contents["safetensors_dirs"] if is_valid_func(d, fp8_supported)]
        local_models = dir_choices + safetensors_choices + safetensors_dir_choices
    else:
        for item in contents["dirs"] + contents["files"]:
            if is_valid_func(item, fp8_supported):
                local_models.append(item)

    # 应用额外过滤器
    if additional_filters:
        valid_hf_models = [m for m in valid_hf_models if additional_filters(m)]
        local_models = [m for m in local_models if additional_filters(m)]

    # 合并 HF 和本地模型，去重，并按优先级排序
    all_models = sort_model_choices(list(set(valid_hf_models + local_models)))

    # 格式化选项，添加下载状态
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_dit_choices(model_path, model_type="wan2.1", task_type=None, is_distill=None):
    """获取 Diffusion 模型可选项"""
    excluded_keywords = ["vae", "tae", "clip", "t5", "high_noise", "low_noise"]

    # 根据模型类型和是否 distill 选择仓库
    if model_type == "wan2.1":
        if is_distill is True:
            repo_ids = ["lightx2v/wan2.1-Distill-Models"]
        elif is_distill is False:
            repo_ids = ["lightx2v/wan2.1-Official-Models"]
        else:
            repo_ids = ["lightx2v/wan2.1-Distill-Models", "lightx2v/wan2.1-Official-Models"]
    else:  # wan2.2
        if is_distill is True:
            repo_ids = ["lightx2v/wan2.2-Distill-Models"]
        elif is_distill is False:
            repo_ids = ["lightx2v/wan2.2-Official-Models"]
        else:
            repo_ids = ["lightx2v/wan2.2-Distill-Models", "lightx2v/wan2.2-Official-Models"]

    hf_models = _get_models_from_repos(repo_ids, prefix_filter=model_type)

    def is_valid(name, fp8_supported):
        name_lower = name.lower()
        if "comfyui" in name_lower:
            return False
        if model_type == "wan2.1":
            if "wan2.1" not in name_lower:
                return False
        else:
            if "wan2.2" not in name_lower:
                return False
        if task_type and task_type.lower() not in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return not any(kw in name_lower for kw in excluded_keywords)

    return _filter_and_format_models(hf_models, model_path, is_valid)


def get_high_noise_choices(model_path, model_type="wan2.2", task_type=None, is_distill=None):
    """获取高噪模型可选项"""
    if is_distill is True:
        repo_ids = ["lightx2v/wan2.2-Distill-Models"]
    elif is_distill is False:
        repo_ids = ["lightx2v/wan2.2-Official-Models"]
    else:
        repo_ids = ["lightx2v/wan2.2-Distill-Models", "lightx2v/wan2.2-Official-Models"]

    hf_models = _get_models_from_repos(repo_ids, keyword_filter="high_noise")

    def is_valid(name, fp8_supported):
        name_lower = name.lower()
        if "comfyui" in name_lower:
            return False
        if model_type.lower() not in name_lower:
            return False
        if task_type and task_type.lower() not in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return "high_noise" in name_lower or "high-noise" in name_lower

    return _filter_and_format_models(hf_models, model_path, is_valid)


def get_low_noise_choices(model_path, model_type="wan2.2", task_type=None, is_distill=None):
    """获取低噪模型可选项"""
    if is_distill is True:
        repo_ids = ["lightx2v/wan2.2-Distill-Models"]
    elif is_distill is False:
        repo_ids = ["lightx2v/wan2.2-Official-Models"]
    else:
        repo_ids = ["lightx2v/wan2.2-Distill-Models", "lightx2v/wan2.2-Official-Models"]

    hf_models = _get_models_from_repos(repo_ids, keyword_filter="low_noise")

    def is_valid(name, fp8_supported):
        name_lower = name.lower()
        if "comfyui" in name_lower:
            return False
        if model_type.lower() not in name_lower:
            return False
        if task_type and task_type.lower() not in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return "low_noise" in name_lower or "low-noise" in name_lower

    return _filter_and_format_models(hf_models, model_path, is_valid)


def get_t5_model_choices(model_path):
    """获取 T5 模型可选项"""
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    def is_valid(name, fp8_supported):
        name_lower = name.lower()
        if "comfyui" in name_lower or name == "google":
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return ("t5" in name_lower) and name.endswith(".safetensors")

    return _filter_and_format_models(hf_models, model_path, is_valid)


def get_t5_tokenizer_choices(model_path):
    """获取 T5 Tokenizer 可选项"""
    contents = scan_model_path_contents(model_path)
    dir_choices = ["google"] if "google" in contents["dirs"] else []

    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []
    hf_google = ["google"] if "google" in hf_models else []

    all_models = sorted(set(hf_google + dir_choices))
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_clip_model_choices(model_path):
    """获取 CLIP 模型可选项"""
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    def is_valid(name, fp8_supported):
        name_lower = name.lower()
        if "comfyui" in name_lower or name == "xlm-roberta-large":
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        return ("clip" in name_lower) and name.endswith(".safetensors")

    return _filter_and_format_models(hf_models, model_path, is_valid)


def get_clip_tokenizer_choices(model_path):
    """获取 CLIP Tokenizer 可选项"""
    contents = scan_model_path_contents(model_path)
    dir_choices = ["xlm-roberta-large"] if "xlm-roberta-large" in contents["dirs"] else []

    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []
    hf_xlm = ["xlm-roberta-large"] if "xlm-roberta-large" in hf_models else []

    all_models = sorted(set(hf_xlm + dir_choices))
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_vae_choices(model_path):
    """获取 VAE 编码器可选项，只返回 Wan2.1_VAE.safetensors"""
    encoder_name = "Wan2.1_VAE.safetensors"
    return [format_model_choice(encoder_name, model_path)]


def get_qwen_image_dit_choices(model_path):
    """获取 Qwen Image Edit Diffusion 模型可选项
    注意：Diffusion 模型只从 model_path/Qwen-Image-Edit-2511 目录检索，如果不存在可以从 HF/MS 下载到子目录
    """
    repo_id = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    model_subdir = "Qwen-Image-Edit-2511"

    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    def is_valid(name, fp8_supported):
        name_lower = name.lower()
        if "comfyui" in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        if "qwen_image_edit_2511" not in name_lower:
            return False
        return name.endswith("lightning.safetensors") or name.endswith("_split") or "lightning_split" in name_lower

    # 筛选 HF 模型
    from utils.model_utils import is_fp8_supported_gpu

    fp8_supported = is_fp8_supported_gpu()
    valid_hf_models = [m for m in hf_models if is_valid(m, fp8_supported)]

    # 本地扫描：只从 model_path/Qwen-Image-Edit-2511 检索，不从 model_path 根目录检索
    subdir_path = os.path.join(model_path, model_subdir)
    local_models = []

    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
        subdir_contents = scan_model_path_contents(subdir_path)
        # 检查子目录中的文件和目录
        for item in subdir_contents["files"] + subdir_contents["dirs"]:
            if is_valid(item, fp8_supported):
                # 添加子目录路径前缀
                local_models.append(f"{model_subdir}/{item}")

    # 合并 HF 和本地模型，添加子目录路径前缀
    all_models = []
    for m in valid_hf_models:
        # HF 模型也需要添加子目录路径，确保下载到子目录
        all_models.append(f"{model_subdir}/{m}")

    all_models.extend(local_models)
    all_models = sorted(set(all_models))

    # 如果没有找到，返回空列表（让用户知道需要下载）
    if not all_models:
        # 至少返回一个选项，用于下载
        all_models = [f"{model_subdir}/"] if valid_hf_models else []

    formatted_choices = [format_model_choice(m, model_path) for m in all_models]
    return formatted_choices if formatted_choices else [""]


def get_qwen_image_vae_choices(model_path):
    """获取 Qwen Image Edit VAE 可选项
    注意：VAE 只从 model_path/Qwen-Image-Edit-2511/vae 目录检索，如果不存在可以从 HF/MS 下载到子目录
    """
    repo_id = "Qwen/Qwen-Image-Edit-2511"
    model_subdir = "Qwen-Image-Edit-2511"

    # 从 HF/MS 缓存获取模型列表（get_hf_models 实际上支持 MS，会先尝试 MS）
    remote_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # 从远程仓库获取的模型选项，需要添加子目录路径
    valid_remote_models = []
    for m in remote_models:
        if m.lower() == "vae":
            # 添加子目录路径，确保下载到子目录
            valid_remote_models.append(f"{model_subdir}/vae")

    # 本地扫描：只从 model_path/Qwen-Image-Edit-2511/vae 检索，不从 model_path 根目录检索
    subdir_path = os.path.join(model_path, model_subdir)
    local_models = []

    # 检查子目录是否存在
    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
        subdir_contents = scan_model_path_contents(subdir_path)
        if "vae" in [d.lower() for d in subdir_contents["dirs"]]:
            local_models.append(f"{model_subdir}/vae")

    all_models = sorted(set(valid_remote_models + local_models))
    # 如果没有找到，返回默认选项（包含子目录路径，用于下载）
    if not all_models:
        all_models = [f"{model_subdir}/vae"]

    formatted_choices = [format_model_choice(m, model_path) for m in all_models]
    return formatted_choices


def get_qwen_image_scheduler_choices(model_path):
    """获取 Qwen Image Edit Scheduler 可选项
    注意：Scheduler 只从 model_path/Qwen-Image-Edit-2511/scheduler 目录检索，如果不存在可以从 HF/MS 下载到子目录
    """
    repo_id = "Qwen/Qwen-Image-Edit-2511"
    model_subdir = "Qwen-Image-Edit-2511"

    # 从 HF/MS 缓存获取模型列表（get_hf_models 实际上支持 MS，会先尝试 MS）
    remote_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # 从远程仓库获取的模型选项，需要添加子目录路径
    valid_remote_models = []
    for m in remote_models:
        if m.lower() == "scheduler":
            # 添加子目录路径，确保下载到子目录
            valid_remote_models.append(f"{model_subdir}/scheduler")

    # 本地扫描：只从 model_path/Qwen-Image-Edit-2511/scheduler 检索，不从 model_path 根目录检索
    subdir_path = os.path.join(model_path, model_subdir)
    local_models = []

    # 检查子目录是否存在
    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
        subdir_contents = scan_model_path_contents(subdir_path)
        if "scheduler" in [d.lower() for d in subdir_contents["dirs"]]:
            local_models.append(f"{model_subdir}/scheduler")

    all_models = sorted(set(valid_remote_models + local_models))
    # 如果没有找到，返回默认选项（包含子目录路径，用于下载）
    if not all_models:
        all_models = [f"{model_subdir}/scheduler"]

    formatted_choices = [format_model_choice(m, model_path) for m in all_models]
    return formatted_choices


def get_qwen25vl_encoder_choices(model_path):
    """获取 Qwen25-VL 编码器可选项"""
    repo_id = "lightx2v/Encoders"
    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    valid_hf_models = [m for m in hf_models if "qwen25-vl-4bit-gptq" in m.lower() or "qwen25_vl_4bit_gptq" in m.lower()]

    contents = scan_model_path_contents(model_path)
    local_models = [d for d in contents["dirs"] if "qwen25-vl-4bit-gptq" in d.lower() or "qwen25_vl_4bit_gptq" in d.lower()]

    all_models = sorted(set(valid_hf_models + local_models))
    formatted_choices = [format_model_choice(m, model_path) for m in all_models]

    return formatted_choices if formatted_choices else [""]


def get_qwen_image_2512_dit_choices(model_path):
    """获取 Qwen Image 2512 Diffusion 模型可选项
    注意：Diffusion 模型只从 model_path/Qwen-Image-2512 目录检索，如果不存在可以从 HF/MS 下载到子目录
    只显示包含 "qwen_image_2512" 字段的模型
    """
    repo_id_lightning = "lightx2v/Qwen-Image-2512-Lightning"
    repo_id_official = "Qwen/Qwen-Image-2512"
    model_subdir = "Qwen-Image-2512"

    hf_models_lightning = get_hf_models(repo_id_lightning) if HF_AVAILABLE else []
    hf_models_official = get_hf_models(repo_id_official) if HF_AVAILABLE else []

    # 合并两个仓库的模型
    hf_models = hf_models_lightning + hf_models_official

    def is_valid(name, fp8_supported):
        name_lower = name.lower()
        if "comfyui" in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        # 只包含 "qwen_image_2512" 字段的模型
        if "qwen_image_2512" not in name_lower:
            return False
        return name.endswith("lightning.safetensors") or name.endswith("_split") or "lightning_split" in name_lower or name.endswith(".safetensors")

    # 筛选 HF 模型
    from utils.model_utils import is_fp8_supported_gpu

    fp8_supported = is_fp8_supported_gpu()
    valid_hf_models = [m for m in hf_models if is_valid(m, fp8_supported)]

    # 本地扫描：只从 model_path/Qwen-Image-2512 检索，不从 model_path 根目录检索
    subdir_path = os.path.join(model_path, model_subdir)
    local_models = []

    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
        subdir_contents = scan_model_path_contents(subdir_path)
        # 检查子目录中的文件和目录
        for item in subdir_contents["files"] + subdir_contents["dirs"]:
            if is_valid(item, fp8_supported):
                # 添加子目录路径前缀
                local_models.append(f"{model_subdir}/{item}")

    # 合并 HF 和本地模型，添加子目录路径前缀
    all_models = []
    for m in valid_hf_models:
        # HF 模型也需要添加子目录路径，确保下载到子目录
        all_models.append(f"{model_subdir}/{m}")

    all_models.extend(local_models)
    all_models = sorted(set(all_models))

    # 如果没有找到，返回空列表（让用户知道需要下载）
    if not all_models:
        # 至少返回一个选项，用于下载
        all_models = [f"{model_subdir}/"] if valid_hf_models else []

    formatted_choices = [format_model_choice(m, model_path) for m in all_models]
    return formatted_choices if formatted_choices else [""]


def get_qwen_image_2512_vae_choices(model_path):
    """获取 Qwen Image 2512 VAE 可选项
    注意：VAE 只从 model_path/Qwen-Image-2512/vae 目录检索，如果不存在可以从 HF/MS 下载到子目录
    """
    repo_id = "Qwen/Qwen-Image-2512"
    model_subdir = "Qwen-Image-2512"

    # 从 HF/MS 缓存获取模型列表（get_hf_models 实际上支持 MS，会先尝试 MS）
    remote_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # 从远程仓库获取的模型选项，需要添加子目录路径
    valid_remote_models = []
    for m in remote_models:
        if m.lower() == "vae":
            # 添加子目录路径，确保下载到子目录
            valid_remote_models.append(f"{model_subdir}/vae")

    # 本地扫描：只从 model_path/Qwen-Image-2512/vae 检索，不从 model_path 根目录检索
    subdir_path = os.path.join(model_path, model_subdir)
    local_models = []

    # 检查子目录是否存在
    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
        subdir_contents = scan_model_path_contents(subdir_path)
        if "vae" in [d.lower() for d in subdir_contents["dirs"]]:
            local_models.append(f"{model_subdir}/vae")

    all_models = sorted(set(valid_remote_models + local_models))
    # 如果没有找到，返回默认选项（包含子目录路径，用于下载）
    if not all_models:
        all_models = [f"{model_subdir}/vae"]

    formatted_choices = [format_model_choice(m, model_path) for m in all_models]
    return formatted_choices


def get_qwen_image_2512_scheduler_choices(model_path):
    """获取 Qwen Image 2512 Scheduler 可选项
    注意：Scheduler 只从 model_path/Qwen-Image-2512/scheduler 目录检索，如果不存在可以从 HF/MS 下载到子目录
    """
    repo_id = "Qwen/Qwen-Image-2512"
    model_subdir = "Qwen-Image-2512"

    # 从 HF/MS 缓存获取模型列表（get_hf_models 实际上支持 MS，会先尝试 MS）
    remote_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # 从远程仓库获取的模型选项，需要添加子目录路径
    valid_remote_models = []
    for m in remote_models:
        if m.lower() == "scheduler":
            # 添加子目录路径，确保下载到子目录
            valid_remote_models.append(f"{model_subdir}/scheduler")

    # 本地扫描：只从 model_path/Qwen-Image-2512/scheduler 检索，不从 model_path 根目录检索
    subdir_path = os.path.join(model_path, model_subdir)
    local_models = []

    # 检查子目录是否存在
    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
        subdir_contents = scan_model_path_contents(subdir_path)
        if "scheduler" in [d.lower() for d in subdir_contents["dirs"]]:
            local_models.append(f"{model_subdir}/scheduler")

    all_models = sorted(set(valid_remote_models + local_models))
    # 如果没有找到，返回默认选项（包含子目录路径，用于下载）
    if not all_models:
        all_models = [f"{model_subdir}/scheduler"]

    formatted_choices = [format_model_choice(m, model_path) for m in all_models]
    return formatted_choices


def get_z_image_turbo_dit_choices(model_path):
    """获取 Z-Image-Turbo Diffusion 模型可选项
    注意：Diffusion 模型只从 model_path/Z-Image-Turbo 目录检索，如果不存在可以从 HF/MS 下载到子目录
    """
    repo_id = "lightx2v/Z-Image-Turbo-Quantized"
    model_subdir = "Z-Image-Turbo"

    hf_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    def is_valid(name, fp8_supported):
        name_lower = name.lower()
        if "comfyui" in name_lower:
            return False
        if not fp8_supported and "fp8" in name_lower:
            return False
        if "z_image_turbo" not in name_lower and "z-image-turbo" not in name_lower:
            return False
        return name.endswith(".safetensors") or name.endswith("_split") or "_split" in name_lower

    # 筛选 HF 模型
    from utils.model_utils import is_fp8_supported_gpu

    fp8_supported = is_fp8_supported_gpu()
    valid_hf_models = [m for m in hf_models if is_valid(m, fp8_supported)]

    # 本地扫描：只从 model_path/Z-Image-Turbo 检索，不从 model_path 根目录检索
    subdir_path = os.path.join(model_path, model_subdir)
    local_models = []

    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
        subdir_contents = scan_model_path_contents(subdir_path)
        # 检查子目录中的文件和目录
        for item in subdir_contents["files"] + subdir_contents["dirs"]:
            if is_valid(item, fp8_supported):
                # 添加子目录路径前缀
                local_models.append(f"{model_subdir}/{item}")

    # 合并 HF 和本地模型，添加子目录路径前缀
    all_models = []
    for m in valid_hf_models:
        # HF 模型也需要添加子目录路径，确保下载到子目录
        all_models.append(f"{model_subdir}/{m}")

    all_models.extend(local_models)
    all_models = sorted(set(all_models))

    # 如果没有找到，返回空列表（让用户知道需要下载）
    if not all_models:
        # 至少返回一个选项，用于下载
        all_models = [f"{model_subdir}/"] if valid_hf_models else []

    formatted_choices = [format_model_choice(m, model_path) for m in all_models]
    return formatted_choices if formatted_choices else [""]


def get_z_image_turbo_vae_choices(model_path):
    """获取 Z-Image-Turbo VAE 可选项
    注意：VAE 只从 model_path/Z-Image-Turbo/vae 目录检索，如果不存在可以从 HF/MS 下载到子目录
    """
    repo_id = "Tongyi-MAI/Z-Image-Turbo"
    model_subdir = "Z-Image-Turbo"

    # 从 HF/MS 缓存获取模型列表（get_hf_models 实际上支持 MS，会先尝试 MS）
    remote_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # 从远程仓库获取的模型选项，需要添加子目录路径
    valid_remote_models = []
    for m in remote_models:
        if m.lower() == "vae":
            # 添加子目录路径，确保下载到子目录
            valid_remote_models.append(f"{model_subdir}/vae")

    # 本地扫描：只从 model_path/Z-Image-Turbo/vae 检索，不从 model_path 根目录检索
    subdir_path = os.path.join(model_path, model_subdir)
    local_models = []

    # 检查子目录是否存在
    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
        subdir_contents = scan_model_path_contents(subdir_path)
        if "vae" in [d.lower() for d in subdir_contents["dirs"]]:
            local_models.append(f"{model_subdir}/vae")

    all_models = sorted(set(valid_remote_models + local_models))
    # 如果没有找到，返回默认选项（包含子目录路径，用于下载）
    if not all_models:
        all_models = [f"{model_subdir}/vae"]

    formatted_choices = [format_model_choice(m, model_path) for m in all_models]
    return formatted_choices


def get_z_image_turbo_scheduler_choices(model_path):
    """获取 Z-Image-Turbo Scheduler 可选项
    注意：Scheduler 只从 model_path/Z-Image-Turbo/scheduler 目录检索，如果不存在可以从 HF/MS 下载到子目录
    """
    repo_id = "Tongyi-MAI/Z-Image-Turbo"
    model_subdir = "Z-Image-Turbo"

    # 从 HF/MS 缓存获取模型列表（get_hf_models 实际上支持 MS，会先尝试 MS）
    remote_models = get_hf_models(repo_id) if HF_AVAILABLE else []

    # 从远程仓库获取的模型选项，需要添加子目录路径
    valid_remote_models = []
    for m in remote_models:
        if m.lower() == "scheduler":
            # 添加子目录路径，确保下载到子目录
            valid_remote_models.append(f"{model_subdir}/scheduler")

    # 本地扫描：只从 model_path/Z-Image-Turbo/scheduler 检索，不从 model_path 根目录检索
    subdir_path = os.path.join(model_path, model_subdir)
    local_models = []

    # 检查子目录是否存在
    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
        subdir_contents = scan_model_path_contents(subdir_path)
        if "scheduler" in [d.lower() for d in subdir_contents["dirs"]]:
            local_models.append(f"{model_subdir}/scheduler")

    all_models = sorted(set(valid_remote_models + local_models))
    # 如果没有找到，返回默认选项（包含子目录路径，用于下载）
    if not all_models:
        all_models = [f"{model_subdir}/scheduler"]

    formatted_choices = [format_model_choice(m, model_path) for m in all_models]
    return formatted_choices


def get_qwen3_encoder_choices(model_path):
    """获取 Qwen3 编码器可选项
    注意：Qwen3 编码器只检测整个仓库目录（Qwen3-4B-GPTQ-Int4）是否存在，不存在可以下载
    """
    repo_id = "JunHowie/Qwen3-4B-GPTQ-Int4"
    repo_name = repo_id.split("/")[-1]  # "Qwen3-4B-GPTQ-Int4"

    # 只检测整个仓库目录是否存在
    repo_path = os.path.join(model_path, repo_name)
    if os.path.exists(repo_path) and os.path.isdir(repo_path):
        # 如果目录存在，返回目录名
        return [format_model_choice(repo_name, model_path)]
    else:
        # 如果目录不存在，也返回目录名（用于下载）
        return [format_model_choice(repo_name, model_path)]
