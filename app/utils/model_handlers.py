import os
import shutil
import warnings

import gradio as gr
from loguru import logger
from utils.model_utils import check_model_exists, extract_model_name, format_model_choice, is_distill_model_from_name

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub import snapshot_download as hf_snapshot_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

    MS_AVAILABLE = True
except ImportError:
    MS_AVAILABLE = False


def download_model_from_hf(repo_id, model_name, model_path, progress=gr.Progress(), download_entire_repo=False):
    """从 Hugging Face 下载模型（支持文件和目录）

    Args:
        repo_id: 仓库 ID
        model_name: 模型名称（文件或目录名）
        model_path: 模型保存路径
        progress: Gradio 进度条
        download_entire_repo: 是否下载整个仓库（用于 Qwen3 编码器等）
    """
    if not HF_AVAILABLE:
        return f"❌ huggingface_hub 未安装，无法下载模型"

    progress(0, desc=f"开始从 Hugging Face 下载 {model_name}...")
    logger.info(f"开始从 Hugging Face {repo_id} 下载 {model_name} 到 {model_path}")

    target_path = os.path.join(model_path, model_name)
    # 确保目标路径的父目录存在（如果 model_name 包含子目录路径，如 "Z-Image-Turbo/vae"）
    os.makedirs(os.path.dirname(target_path) if "/" in model_name else model_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # 如果指定下载整个仓库，直接下载整个仓库到目标目录
    if download_entire_repo:
        progress(0.1, desc=f"下载整个仓库 {repo_id}...")
        logger.info(f"下载整个仓库 {repo_id} 到 {target_path}")

        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hf_snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                repo_type="model",
            )

        # 移动文件到正确位置
        repo_name = repo_id.split("/")[-1]
        source_dir = os.path.join(model_path, repo_name)
        if os.path.exists(source_dir) and source_dir != target_path:
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.move(source_dir, target_path)

        logger.info(f"整个仓库 {repo_id} 下载完成，已移动到 {target_path}")
        progress(1.0, desc=f"✅ {model_name} 下载完成")
        return f"✅ {model_name} 下载完成"

    # 判断是文件还是目录
    is_directory = not (model_name.endswith(".safetensors") or model_name.endswith(".pth"))

    if is_directory:
        # 下载目录
        progress(0.1, desc=f"下载目录 {model_name}...")
        logger.info(f"检测到 {model_name} 是目录，使用 snapshot_download")

        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        # 从 model_name 中提取实际的目录名（如果包含子目录路径，如 "Z-Image-Turbo/scheduler" -> "scheduler"）
        actual_dir_name = model_name.split("/")[-1] if "/" in model_name else model_name

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hf_snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"{actual_dir_name}/**"],
                local_dir=model_path,
                local_dir_use_symlinks=False,
                repo_type="model",
            )

        # 移动文件到正确位置
        repo_name = repo_id.split("/")[-1]
        # 先尝试从 repo_name/actual_dir_name 查找
        source_dir = os.path.join(model_path, repo_name, actual_dir_name)
        if os.path.exists(source_dir):
            # 确保目标路径的父目录存在
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.move(source_dir, target_path)
            repo_dir = os.path.join(model_path, repo_name)
            if os.path.exists(repo_dir) and not os.listdir(repo_dir):
                os.rmdir(repo_dir)
        else:
            # 尝试从 model_path/actual_dir_name 查找
            source_dir = os.path.join(model_path, actual_dir_name)
            if os.path.exists(source_dir) and source_dir != target_path:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.move(source_dir, target_path)

        logger.info(f"目录 {model_name} 下载完成，已移动到 {target_path}")
    else:
        # 下载文件
        progress(0.1, desc=f"下载文件 {model_name}...")
        logger.info(f"检测到 {model_name} 是文件，使用 hf_hub_download")

        if os.path.exists(target_path):
            os.remove(target_path)

        # 如果 model_name 包含子目录路径，提取实际文件名
        actual_file_name = model_name.split("/")[-1] if "/" in model_name else model_name

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=actual_file_name,  # 使用实际文件名，不包含子目录路径
                local_dir=model_path,
                local_dir_use_symlinks=False,
                repo_type="model",
            )

        # 如果 model_name 包含子目录路径，需要移动文件到子目录
        if "/" in model_name:
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)
            if os.path.exists(downloaded_path) and downloaded_path != target_path:
                shutil.move(downloaded_path, target_path)
                logger.info(f"文件 {model_name} 下载完成，保存到 {target_path}")
            else:
                logger.info(f"文件 {model_name} 下载完成，保存到 {downloaded_path}")
        else:
            logger.info(f"文件 {model_name} 下载完成，保存到 {downloaded_path}")

    progress(1.0, desc=f"✅ {model_name} 下载完成")
    return f"✅ {model_name} 下载完成"


def download_model_from_ms(repo_id, model_name, model_path, progress=gr.Progress(), download_entire_repo=False):
    """从 ModelScope 下载模型（支持文件和目录）

    Args:
        repo_id: 仓库 ID
        model_name: 模型名称（文件或目录名）
        model_path: 模型保存路径
        progress: Gradio 进度条
        download_entire_repo: 是否下载整个仓库（用于 Qwen3 编码器等）
    """
    if not MS_AVAILABLE:
        return f"❌ modelscope 未安装，无法下载模型"

    progress(0, desc=f"开始从 ModelScope 下载 {model_name}...")
    logger.info(f"开始从 ModelScope {repo_id} 下载 {model_name} 到 {model_path}")

    target_path = os.path.join(model_path, model_name)
    # 确保目标路径的父目录存在（如果 model_name 包含子目录路径，如 "Z-Image-Turbo/vae"）
    os.makedirs(os.path.dirname(target_path) if "/" in model_name else model_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # 临时目录用于下载
    temp_dir = os.path.join(model_path, f".temp_{model_name}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # 如果指定下载整个仓库，直接下载整个仓库到目标目录
    if download_entire_repo:
        progress(0.1, desc=f"下载整个仓库 {repo_id}...")
        logger.info(f"下载整个仓库 {repo_id} 到 {target_path}")

        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downloaded_path = ms_snapshot_download(
                model_id=repo_id,
                cache_dir=temp_dir,
            )

        # 移动文件到目标位置
        if os.path.exists(downloaded_path):
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.move(downloaded_path, target_path)

        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        logger.info(f"整个仓库 {repo_id} 下载完成，已移动到 {target_path}")
        progress(1.0, desc=f"✅ {model_name} 下载完成")
        return f"✅ {model_name} 下载完成"

    # 判断是文件还是目录
    is_directory = not (model_name.endswith(".safetensors") or model_name.endswith(".pth"))
    is_file = not is_directory

    # 处理目录下载
    if is_directory:
        progress(0.1, desc=f"下载目录 {model_name}...")
        logger.info(f"检测到 {model_name} 是目录，使用 snapshot_download")

        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        # 从 model_name 中提取实际的目录名（如果包含子目录路径，如 "Z-Image-Turbo/scheduler" -> "scheduler"）
        actual_dir_name = model_name.split("/")[-1] if "/" in model_name else model_name

        # 使用 snapshot_download 下载目录（使用实际的目录名）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downloaded_path = ms_snapshot_download(
                model_id=repo_id,
                cache_dir=temp_dir,
                allow_patterns=[f"{actual_dir_name}/**"],
            )

        # 移动文件到目标位置
        # ModelScope 下载后，文件在 downloaded_path/actual_dir_name
        source_dir = os.path.join(downloaded_path, actual_dir_name)
        if not os.path.exists(source_dir) and os.path.exists(downloaded_path):
            # 如果找不到，尝试从下载路径中查找
            for item in os.listdir(downloaded_path):
                item_path = os.path.join(downloaded_path, item)
                if actual_dir_name.lower() in item.lower() or (os.path.isdir(item_path) and item.lower() == actual_dir_name.lower()):
                    source_dir = item_path
                    break

        if os.path.exists(source_dir):
            # 确保目标路径的父目录存在
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.move(source_dir, target_path)
            logger.info(f"目录 {model_name} 下载完成，已移动到 {target_path}")
        else:
            logger.error(f"无法找到下载的目录：{source_dir}，下载路径：{downloaded_path}")

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

        # 如果 model_name 包含子目录路径，提取实际文件名用于 allow_patterns
        actual_file_name = model_name.split("/")[-1] if "/" in model_name else model_name

        # 使用 snapshot_download 下载文件
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downloaded_path = ms_snapshot_download(
                model_id=repo_id,
                cache_dir=temp_dir,
                allow_patterns=[actual_file_name],  # 只使用文件名，不包含子目录路径
            )

        # 查找并移动文件
        # 如果 model_name 包含子目录路径（如 "Qwen-Image-2512/file.safetensors"），提取文件名
        actual_file_name = model_name.split("/")[-1] if "/" in model_name else model_name

        # 先尝试完整路径
        source_file = os.path.join(downloaded_path, model_name)
        if not os.path.exists(source_file):
            # 尝试直接使用文件名
            source_file = os.path.join(downloaded_path, actual_file_name)
            if not os.path.exists(source_file):
                # 如果找不到，从下载路径中递归查找
                for root, dirs, files_list in os.walk(downloaded_path):
                    if actual_file_name in files_list:
                        source_file = os.path.join(root, actual_file_name)
                        break

        if os.path.exists(source_file):
            # 确保目标目录存在
            target_dir = os.path.dirname(target_path)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            shutil.move(source_file, target_path)
            logger.info(f"文件 {model_name} 下载完成，保存到 {target_path}")
        else:
            logger.error(f"❌ 下载失败：无法找到文件 {actual_file_name} 在 {downloaded_path}")
            return f"❌ 下载失败：无法找到文件 {actual_file_name}"

        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    else:
        return f"❌ 无法找到 {model_name}：既不是文件也不是目录"

    progress(1.0, desc=f"✅ {model_name} 下载完成")
    return f"✅ {model_name} 下载完成"


def download_model(repo_id, model_name, model_path, download_source="huggingface", progress=gr.Progress(), download_entire_repo=False):
    """统一的下载函数，根据下载源选择 Hugging Face 或 ModelScope

    Args:
        repo_id: 仓库 ID
        model_name: 模型名称（文件或目录名）
        model_path: 模型保存路径
        download_source: 下载源 ("huggingface" 或 "modelscope")
        progress: Gradio 进度条
        download_entire_repo: 是否下载整个仓库（用于 Qwen3 编码器等）
    """
    if download_source == "modelscope":
        return download_model_from_ms(repo_id, model_name, model_path, progress, download_entire_repo)
    else:
        return download_model_from_hf(repo_id, model_name, model_path, progress, download_entire_repo)


def get_repo_id_for_model(model_type, is_distill, model_category="dit"):
    """根据模型类型、是否 distill 和模型类别获取对应的 Hugging Face 仓库 ID"""
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


MODEL_CONFIG_MAP = {
    "dit": {"get_repo_id": lambda model_type, is_distill, _: get_repo_id_for_model(model_type, is_distill, "dit")},
    "high_noise": {"get_repo_id": lambda model_type, is_distill, _: get_repo_id_for_model("wan2.2", is_distill, "high_noise")},
    "low_noise": {"get_repo_id": lambda model_type, is_distill, _: get_repo_id_for_model("wan2.2", is_distill, "low_noise")},
    "t5": {"get_repo_id": lambda _, __, ___: get_repo_id_for_model(None, None, "t5")},
    "clip": {"get_repo_id": lambda _, __, ___: get_repo_id_for_model(None, None, "clip")},
    "vae": {"get_repo_id": lambda _, __, ___: get_repo_id_for_model(None, None, "vae")},
    "vae_encoder": {"get_repo_id": lambda _, __, ___: get_repo_id_for_model(None, None, "vae")},  # 兼容旧代码
    "vae_decoder": {"get_repo_id": lambda _, __, ___: get_repo_id_for_model(None, None, "vae")},  # 兼容旧代码
    "qwen_image_dit": {"get_repo_id": lambda _, __, ___: "lightx2v/Qwen-Image-Edit-2511-Lightning"},
    "qwen_image_vae": {"get_repo_id": lambda _, __, ___: "Qwen/Qwen-Image-Edit-2511"},
    "qwen_image_scheduler": {"get_repo_id": lambda _, __, ___: "Qwen/Qwen-Image-Edit-2511"},
    "qwen_image_2512_dit": {"get_repo_id": lambda _, __, ___: "lightx2v/Qwen-Image-2512-Lightning"},
    "qwen_image_2512_vae": {"get_repo_id": lambda _, __, ___: "Qwen/Qwen-Image-2512"},
    "qwen_image_2512_scheduler": {"get_repo_id": lambda _, __, ___: "Qwen/Qwen-Image-2512"},
    "z_image_turbo_dit": {"get_repo_id": lambda _, __, ___: "lightx2v/Z-Image-Turbo-Quantized"},
    "z_image_turbo_vae": {"get_repo_id": lambda _, __, ___: "Tongyi-MAI/Z-Image-Turbo"},
    "z_image_turbo_scheduler": {"get_repo_id": lambda _, __, ___: "Tongyi-MAI/Z-Image-Turbo"},
    "qwen3_encoder": {"get_repo_id": lambda _, __, ___: "JunHowie/Qwen3-4B-GPTQ-Int4"},
    "qwen25vl_encoder": {"get_repo_id": lambda _, __, ___: "lightx2v/Encoders"},
}

# Tokenizer 配置
TOKENIZER_CONFIG = {
    "t5": {"name": "google", "repo_id": get_repo_id_for_model(None, None, "t5")},
    "clip": {"name": "xlm-roberta-large", "repo_id": get_repo_id_for_model(None, None, "clip")},
}


def update_model_status(model_path_val, model_name, model_category, model_type_val=None):
    """通用的模型状态更新函数

    Args:
        model_path_val: 模型路径
        model_name: 模型名称（可能包含状态标识）
        model_category: 模型类别 (dit, high_noise, low_noise, t5, clip, vae, qwen_image_dit, etc.)
        model_type_val: 模型类型 (wan2.1, wan2.2, Qwen-Image-Edit-2511)，某些类别需要此参数

    Returns:
        gr.update: 下载按钮的可见性更新
    """
    if not model_name:
        return gr.update(visible=False)

    actual_name = extract_model_name(model_name)

    # 对于 Qwen3 编码器，检查整个仓库目录是否存在
    if model_category == "qwen3_encoder":
        # Qwen3 编码器检查整个仓库目录 Qwen3-4B-GPTQ-Int4 是否存在
        repo_id = "JunHowie/Qwen3-4B-GPTQ-Int4"
        repo_name = repo_id.split("/")[-1]  # "Qwen3-4B-GPTQ-Int4"
        exists = check_model_exists(model_path_val, repo_name)
        return gr.update(visible=not exists)

    exists = check_model_exists(model_path_val, actual_name)
    return gr.update(visible=not exists)


def update_tokenizer_status(model_path_val, tokenizer_type):
    """更新 Tokenizer 状态

    Args:
        model_path_val: 模型路径
        tokenizer_type: tokenizer 类型 ("t5" 或 "clip")

    Returns:
        tuple: (dropdown_update, btn_visible_update)
    """
    config = TOKENIZER_CONFIG.get(tokenizer_type)
    if not config:
        return gr.update(), gr.update(visible=False)

    tokenizer_name = config["name"]
    exists = check_model_exists(model_path_val, tokenizer_name)

    if exists:
        status_text = f"{tokenizer_name} ✅"
        return gr.update(value=status_text), gr.update(visible=False)
    else:
        status_text = f"{tokenizer_name} ❌"
        return gr.update(value=status_text), gr.update(visible=True)


def download_model_handler(model_path_val, model_name, model_category, download_source_val, get_choices_func=None, model_type_val=None, task_type_val=None, progress=gr.Progress()):
    """通用的模型下载处理函数

    Args:
        model_path_val: 模型路径
        model_name: 模型名称（可能包含状态标识）
        model_category: 模型类别
        download_source_val: 下载源 ("huggingface" 或 "modelscope")
        get_choices_func: 获取模型选项列表的函数（可选）
        model_type_val: 模型类型（某些类别需要）
        task_type_val: 任务类型（某些类别需要）
        progress: Gradio 进度条

    Returns:
        tuple: (status_update, btn_visible_update, choices_update)
    """
    if not model_name:
        return gr.update(value="请先选择模型"), gr.update(visible=False), gr.update()

    actual_name = extract_model_name(model_name)

    # 获取 repo_id
    config = MODEL_CONFIG_MAP.get(model_category)
    if not config:
        return gr.update(value=f"未知的模型类别: {model_category}"), gr.update(visible=False), gr.update()

    is_distill = False
    if model_category in ["dit", "high_noise", "low_noise"]:
        is_distill = is_distill_model_from_name(actual_name)

    repo_id = config["get_repo_id"](model_type_val, is_distill, model_category)

    # 对于 Qwen3 编码器，始终下载整个仓库目录
    if model_category == "qwen3_encoder":
        # Qwen3 编码器下载整个仓库，使用仓库名作为目录名
        repo_name = repo_id.split("/")[-1]  # "Qwen3-4B-GPTQ-Int4"
        actual_name = repo_name

    # 对于 VAE、Scheduler 和 Diffusion 模型，需要下载到子目录
    elif model_category in [
        "qwen_image_vae",
        "qwen_image_scheduler",
        "qwen_image_dit",
        "qwen_image_2512_vae",
        "qwen_image_2512_scheduler",
        "qwen_image_2512_dit",
        "z_image_turbo_vae",
        "z_image_turbo_scheduler",
        "z_image_turbo_dit",
    ]:
        # 确定模型子目录名称
        if model_category in ["qwen_image_vae", "qwen_image_scheduler", "qwen_image_dit"]:
            model_subdir = "Qwen-Image-Edit-2511"
        elif model_category in ["qwen_image_2512_vae", "qwen_image_2512_scheduler", "qwen_image_2512_dit"]:
            model_subdir = "Qwen-Image-2512"
        elif model_category in ["z_image_turbo_vae", "z_image_turbo_scheduler", "z_image_turbo_dit"]:
            model_subdir = "Z-Image-Turbo"

        # 如果需要在子目录下载，修改 actual_name 为子目录路径
        # 如果 actual_name 已经包含子目录路径，则提取文件名
        if "/" in actual_name:
            # 提取文件名（去掉子目录前缀）
            parts = actual_name.split("/")
            if len(parts) > 1:
                actual_name = parts[-1]
        # 添加子目录路径前缀
        actual_name = f"{model_subdir}/{actual_name}"

    # 下载模型
    # 对于 Qwen3 编码器，下载整个仓库
    download_entire_repo = model_category == "qwen3_encoder"
    result = download_model(repo_id, actual_name, model_path_val, download_source_val, progress, download_entire_repo)

    # 下载完成后，直接标记为已存在（使用 ✅ 状态），避免文件系统同步延迟导致的状态检查失败
    formatted_name_with_status = format_model_choice(actual_name, model_path_val, status_emoji="✅")

    # 更新状态（下载完成后，模型应该存在，所以隐藏下载按钮）
    btn_visible = gr.update(visible=False)

    # 更新选项列表（如果提供了获取选项的函数）
    choices_update = gr.update()
    if get_choices_func:
        try:
            if model_category in ["dit"] and model_type_val and task_type_val:
                choices = get_choices_func(model_path_val, model_type_val, task_type_val)
            elif model_category in ["high_noise", "low_noise"] and task_type_val:
                choices = get_choices_func(model_path_val, "wan2.2", task_type_val)
            else:
                choices = get_choices_func(model_path_val)
            # 使用带状态标识的格式化名称
            choices_update = gr.update(choices=choices, value=formatted_name_with_status)
        except Exception as e:
            # 如果获取选项失败，只更新值
            choices_update = gr.update(value=formatted_name_with_status)

    return gr.update(value=result), btn_visible, choices_update


def download_tokenizer_handler(model_path_val, tokenizer_type, download_source_val, progress=gr.Progress()):
    """下载 Tokenizer 处理函数

    Args:
        model_path_val: 模型路径
        tokenizer_type: tokenizer 类型 ("t5" 或 "clip")
        download_source_val: 下载源
        progress: Gradio 进度条

    Returns:
        tuple: (status_update, dropdown_update, btn_visible_update)
    """
    config = TOKENIZER_CONFIG.get(tokenizer_type)
    if not config:
        return gr.update(value=f"未知的 tokenizer 类型: {tokenizer_type}"), gr.update(), gr.update(visible=False)

    tokenizer_name = config["name"]
    repo_id = config["repo_id"]

    result = download_model(repo_id, tokenizer_name, model_path_val, download_source_val, progress)
    dropdown_update, btn_visible = update_tokenizer_status(model_path_val, tokenizer_type)

    return gr.update(value=result), dropdown_update, btn_visible


def create_update_status_wrappers():
    """创建所有模型状态更新的包装函数

    Returns:
        dict: 包含所有 update 函数的字典
    """

    def update_dit_status(model_path_val, model_name, model_type_val):
        return update_model_status(model_path_val, model_name, "dit", model_type_val)

    def update_t5_model_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "t5")

    def update_t5_tokenizer_status(model_path_val):
        return update_tokenizer_status(model_path_val, "t5")

    def update_clip_model_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "clip")

    def update_clip_tokenizer_status(model_path_val):
        return update_tokenizer_status(model_path_val, "clip")

    def update_vae_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "vae")

    def update_vae_encoder_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "vae")  # 兼容旧代码，统一使用 vae

    def update_vae_decoder_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "vae")  # 兼容旧代码，统一使用 vae

    def update_high_noise_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "high_noise", "wan2.2")

    def update_low_noise_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "low_noise", "wan2.2")

    def update_qwen_image_dit_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "qwen_image_dit")

    def update_qwen_image_vae_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "qwen_image_vae")

    def update_qwen_image_scheduler_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "qwen_image_scheduler")

    def update_qwen25vl_encoder_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "qwen25vl_encoder")

    return {
        "update_dit_status": update_dit_status,
        "update_t5_model_status": update_t5_model_status,
        "update_t5_tokenizer_status": update_t5_tokenizer_status,
        "update_clip_model_status": update_clip_model_status,
        "update_clip_tokenizer_status": update_clip_tokenizer_status,
        "update_vae_status": update_vae_status,
        "update_vae_encoder_status": update_vae_encoder_status,  # 兼容旧代码
        "update_vae_decoder_status": update_vae_decoder_status,  # 兼容旧代码
        "update_high_noise_status": update_high_noise_status,
        "update_low_noise_status": update_low_noise_status,
        "update_qwen_image_dit_status": update_qwen_image_dit_status,
        "update_qwen_image_vae_status": update_qwen_image_vae_status,
        "update_qwen_image_scheduler_status": update_qwen_image_scheduler_status,
        "update_qwen25vl_encoder_status": update_qwen25vl_encoder_status,
    }


def create_download_wrappers(get_choices_funcs):
    """创建所有模型下载的包装函数

    Args:
        get_choices_funcs: 包含所有 get_choices 函数的字典

    Returns:
        dict: 包含所有 download 函数的字典
    """

    def download_dit_model(model_path_val, model_name, model_type_val, task_type_val, download_source_val, progress=gr.Progress()):
        return download_model_handler(
            model_path_val,
            model_name,
            "dit",
            download_source_val,
            get_choices_func=get_choices_funcs.get("get_dit_choices"),
            model_type_val=model_type_val,
            task_type_val=task_type_val,
            progress=progress,
        )

    def download_t5_model(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "t5", download_source_val, get_choices_func=get_choices_funcs.get("get_t5_model_choices"), progress=progress)

    def download_t5_tokenizer(model_path_val, download_source_val, progress=gr.Progress()):
        return download_tokenizer_handler(model_path_val, "t5", download_source_val, progress)

    def download_clip_model(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "clip", download_source_val, get_choices_func=get_choices_funcs.get("get_clip_model_choices"), progress=progress)

    def download_clip_tokenizer(model_path_val, download_source_val, progress=gr.Progress()):
        return download_tokenizer_handler(model_path_val, "clip", download_source_val, progress)

    def download_vae(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "vae", download_source_val, get_choices_func=get_choices_funcs.get("get_vae_choices"), progress=progress)

    def download_vae_encoder(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_vae(model_path_val, model_name, download_source_val, progress)  # 兼容旧代码

    def download_vae_decoder(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_vae(model_path_val, model_name, download_source_val, progress)  # 兼容旧代码

    def download_high_noise_model(model_path_val, model_name, task_type_val, download_source_val, progress=gr.Progress()):
        return download_model_handler(
            model_path_val,
            model_name,
            "high_noise",
            download_source_val,
            get_choices_func=get_choices_funcs.get("get_high_noise_choices"),
            model_type_val="wan2.2",
            task_type_val=task_type_val,
            progress=progress,
        )

    def download_low_noise_model(model_path_val, model_name, task_type_val, download_source_val, progress=gr.Progress()):
        return download_model_handler(
            model_path_val,
            model_name,
            "low_noise",
            download_source_val,
            get_choices_func=get_choices_funcs.get("get_low_noise_choices"),
            model_type_val="wan2.2",
            task_type_val=task_type_val,
            progress=progress,
        )

    def download_qwen_image_dit(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "qwen_image_dit", download_source_val, get_choices_func=get_choices_funcs.get("get_qwen_image_dit_choices"), progress=progress)

    def download_qwen_image_vae(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "qwen_image_vae", download_source_val, get_choices_func=get_choices_funcs.get("get_qwen_image_vae_choices"), progress=progress)

    def download_qwen_image_scheduler(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(
            model_path_val, model_name, "qwen_image_scheduler", download_source_val, get_choices_func=get_choices_funcs.get("get_qwen_image_scheduler_choices"), progress=progress
        )

    def download_qwen25vl_encoder(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "qwen25vl_encoder", download_source_val, get_choices_func=get_choices_funcs.get("get_qwen25vl_encoder_choices"), progress=progress)

    return {
        "download_dit_model": download_dit_model,
        "download_t5_model": download_t5_model,
        "download_t5_tokenizer": download_t5_tokenizer,
        "download_clip_model": download_clip_model,
        "download_clip_tokenizer": download_clip_tokenizer,
        "download_vae": download_vae,
        "download_vae_encoder": download_vae_encoder,  # 兼容旧代码
        "download_vae_decoder": download_vae_decoder,  # 兼容旧代码
        "download_high_noise_model": download_high_noise_model,
        "download_low_noise_model": download_low_noise_model,
        "download_qwen_image_dit": download_qwen_image_dit,
        "download_qwen_image_vae": download_qwen_image_vae,
        "download_qwen_image_scheduler": download_qwen_image_scheduler,
        "download_qwen25vl_encoder": download_qwen25vl_encoder,
    }
