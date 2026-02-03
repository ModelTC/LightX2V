import os
import random
import subprocess
from typing import Optional

import imageio
import imageio_ffmpeg as ffmpeg
import numpy as np
import safetensors
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
from loguru import logger
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_device_module.manual_seed(seed)
    torch_device_module.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def cache_video(
    tensor,
    save_file: str,
    fps=30,
    suffix=".mp4",
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
    retry=5,
):
    save_dir = os.path.dirname(save_file)
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory: {save_dir}, error: {e}")
        return None

    cache_file = save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))  # type: ignore
            tensor = torch.stack(
                [torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range) for u in tensor.unbind(2)],
                dim=1,
            ).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(cache_file, fps=fps, codec="libx264", quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            del tensor
            torch.cuda.empty_cache()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        logger.info(f"cache_video failed, error: {error}", flush=True)
        return None


def vae_to_comfyui_image(vae_output: torch.Tensor) -> torch.Tensor:
    """
    Convert VAE decoder output to ComfyUI Image format

    Args:
        vae_output: VAE decoder output tensor, typically in range [-1, 1]
                    Shape: [B, C, T, H, W] or [B, C, H, W]

    Returns:
        ComfyUI Image tensor in range [0, 1]
        Shape: [B, H, W, C] for single frame or [B*T, H, W, C] for video
    """
    # Handle video tensor (5D) vs image tensor (4D)
    if vae_output.dim() == 5:
        # Video tensor: [B, C, T, H, W]
        B, C, T, H, W = vae_output.shape
        # Reshape to [B*T, C, H, W] for processing
        vae_output = vae_output.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    # Normalize from [-1, 1] to [0, 1]
    images = (vae_output + 1) / 2

    # Clamp values to [0, 1]
    images = torch.clamp(images, 0, 1)

    # Convert from [B, C, H, W] to [B, H, W, C]
    images = images.permute(0, 2, 3, 1).cpu()

    return images


def vae_to_comfyui_image_inplace(vae_output: torch.Tensor) -> torch.Tensor:
    """
    Convert VAE decoder output to ComfyUI Image format (inplace operation)

    Args:
        vae_output: VAE decoder output tensor, typically in range [-1, 1]
                    Shape: [B, C, T, H, W] or [B, C, H, W]
                    WARNING: This tensor will be modified in-place!

    Returns:
        ComfyUI Image tensor in range [0, 1]
        Shape: [B, H, W, C] for single frame or [B*T, H, W, C] for video
        Note: The returned tensor is the same object as input (modified in-place)
    """
    # Handle video tensor (5D) vs image tensor (4D)
    if vae_output.dim() == 5:
        # Video tensor: [B, C, T, H, W]
        B, C, T, H, W = vae_output.shape
        # Reshape to [B*T, C, H, W] for processing (inplace view)
        vae_output = vae_output.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

    # Normalize from [-1, 1] to [0, 1] (inplace)
    vae_output.add_(1).div_(2)

    # Clamp values to [0, 1] (inplace)
    vae_output.clamp_(0, 1)

    # Convert from [B, C, H, W] to [B, H, W, C] and move to CPU
    vae_output = vae_output.permute(0, 2, 3, 1).cpu()

    return vae_output


def wan_vae_to_comfy(vae_output: torch.Tensor) -> torch.Tensor:
    """
    Convert VAE decoder output to ComfyUI Image format (inplace operation)

    Args:
        vae_output: VAE decoder output tensor, typically in range [-1, 1]
                    Shape: [B, C, T, H, W] or [B, C, H, W]
                    WARNING: This tensor will be modified in-place!

    Returns:
        ComfyUI Image tensor in range [0, 1]
        Shape: [B, H, W, C] for single frame or [B*T, H, W, C] for video
        Note: The returned tensor is the same object as input (modified in-place)
    """

    vae_output.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)

    if vae_output.ndim == 5:
        # Video: [B, C, T, H, W] -> [B, T, H, W, C]
        vae_output = vae_output.permute(0, 2, 3, 4, 1)
        # -> [B*T, H, W, C]
        return vae_output.cpu().flatten(0, 1)
    else:
        # Image: [B, C, H, W] -> [B, H, W, C]
        return vae_output.permute(0, 2, 3, 1).cpu()


def diffusers_vae_to_comfy(vae_output: torch.Tensor) -> torch.Tensor:
    """
    Convert Diffusers VAE decoder output to ComfyUI Image format
    Image processor for VAE, return tensor in range [0, 1] when do_denormalize is True.

    ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/image_processor.py#L744

    """
    return vae_output.permute(0, 2, 3, 1).cpu()


def save_to_video(
    images: torch.Tensor,
    output_path: str,
    fps: float = 24.0,
    method: str = "imageio",
    lossless: bool = False,
    output_pix_fmt: Optional[str] = "yuv420p",
) -> None:
    """
    Save ComfyUI Image tensor to video file

    Args:
        images: ComfyUI Image tensor [N, H, W, C] in range [0, 1]
        output_path: Path to save the video
        fps: Frames per second
        method: Save method - "imageio" or "ffmpeg"
        lossless: Whether to use lossless encoding (ffmpeg method only)
        output_pix_fmt: Pixel format for output (ffmpeg method only)
    """
    assert images.dim() == 4 and images.shape[-1] == 3, "Input must be [N, H, W, C] with C=3"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if method == "imageio":
        # Convert to uint8
        # frames = (images * 255).cpu().numpy().astype(np.uint8)
        frames = (images * 255).to(torch.uint8).cpu().numpy()
        imageio.mimsave(output_path, frames, fps=fps)  # type: ignore

    elif method == "ffmpeg":
        # Convert to numpy and scale to [0, 255]
        # frames = (images * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        frames = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        # Convert RGB to BGR for OpenCV/FFmpeg
        frames = frames[..., ::-1].copy()

        N, height, width, _ = frames.shape

        # Ensure even dimensions for x264
        width += width % 2
        height += height % 2

        # Get ffmpeg executable from imageio_ffmpeg
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

        if lossless:
            command = [
                ffmpeg_exe,
                "-y",  # Overwrite output file if it exists
                "-f",
                "rawvideo",
                "-s",
                f"{int(width)}x{int(height)}",
                "-pix_fmt",
                "bgr24",
                "-r",
                f"{fps}",
                "-loglevel",
                "error",
                "-threads",
                "4",
                "-i",
                "-",  # Input from pipe
                "-vcodec",
                "libx264rgb",
                "-crf",
                "0",
                "-an",  # No audio
                output_path,
            ]
        else:
            command = [
                ffmpeg_exe,
                "-y",  # Overwrite output file if it exists
                "-f",
                "rawvideo",
                "-s",
                f"{int(width)}x{int(height)}",
                "-pix_fmt",
                "bgr24",
                "-r",
                f"{fps}",
                "-loglevel",
                "error",
                "-threads",
                "4",
                "-i",
                "-",  # Input from pipe
                "-vcodec",
                "libx264",
                "-pix_fmt",
                output_pix_fmt,
                "-an",  # No audio
                output_path,
            ]

        # Run FFmpeg
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if process.stdin is None:
            raise BrokenPipeError("No stdin buffer received.")

        # Write frames to FFmpeg
        for frame in frames:
            # Pad frame if needed
            if frame.shape[0] < height or frame.shape[1] < width:
                padded = np.zeros((height, width, 3), dtype=np.uint8)
                padded[: frame.shape[0], : frame.shape[1]] = frame
                frame = padded
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            error_output = process.stderr.read().decode() if process.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed with error: {error_output}")

    else:
        raise ValueError(f"Unknown save method: {method}")


def remove_substrings_from_keys(original_dict, substr):
    new_dict = {}
    for key, value in original_dict.items():
        new_dict[key.replace(substr, "")] = value
    return new_dict


def find_torch_model_path(config, ckpt_config_key=None, filename=None, subdir=["original", "fp8", "int8", "distill_models", "distill_fp8", "distill_int8"]):
    if ckpt_config_key and config.get(ckpt_config_key, None) is not None:
        return config.get(ckpt_config_key)

    paths_to_check = [
        os.path.join(config["model_path"], filename),
    ]
    if isinstance(subdir, list):
        for sub in subdir:
            paths_to_check.insert(0, os.path.join(config["model_path"], sub, filename))
    else:
        paths_to_check.insert(0, os.path.join(config["model_path"], subdir, filename))

    for path in paths_to_check:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"PyTorch model file '{filename}' not found.\nPlease download the model from https://huggingface.co/lightx2v/ or specify the model path in the configuration file.")


def load_safetensors(in_path, remove_key=None, include_keys=None):
    """加载safetensors文件或目录，支持按key包含筛选或排除"""
    include_keys = include_keys or []
    if os.path.isdir(in_path):
        return load_safetensors_from_dir(in_path, remove_key, include_keys)
    elif os.path.isfile(in_path):
        return load_safetensors_from_path(in_path, remove_key, include_keys)
    else:
        raise ValueError(f"{in_path} does not exist")


def load_safetensors_from_path(in_path, remove_key=None, include_keys=None):
    include_keys = include_keys or []
    tensors = {}
    with safetensors.safe_open(in_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if include_keys:
                if any(inc_key in key for inc_key in include_keys):
                    tensors[key] = f.get_tensor(key)
            else:
                if not (remove_key and remove_key in key):
                    tensors[key] = f.get_tensor(key)
    return tensors


def load_safetensors_from_dir(in_dir, remove_key=None, include_keys=None):
    """从目录加载所有safetensors文件，支持按key筛选"""
    include_keys = include_keys or []
    tensors = {}
    safetensors_files = os.listdir(in_dir)
    safetensors_files = [f for f in safetensors_files if f.endswith(".safetensors")]
    for f in safetensors_files:
        tensors.update(load_safetensors_from_path(os.path.join(in_dir, f), remove_key, include_keys))
    return tensors


def load_pt_safetensors(in_path, remove_key=None, include_keys=None):
    """加载pt/pth或safetensors权重，支持按key筛选"""
    include_keys = include_keys or []
    ext = os.path.splitext(in_path)[-1]
    if ext in (".pt", ".pth", ".tar"):
        state_dict = torch.load(in_path, map_location="cpu", weights_only=True)
        # 处理筛选逻辑
        keys_to_keep = []
        for key in state_dict.keys():
            if include_keys:
                if any(inc_key in key for inc_key in include_keys):
                    keys_to_keep.append(key)
            else:
                if not (remove_key and remove_key in key):
                    keys_to_keep.append(key)
        # 只保留符合条件的key
        state_dict = {k: state_dict[k] for k in keys_to_keep}
    else:
        state_dict = load_safetensors(in_path, remove_key, include_keys)
    return state_dict


def load_weights(checkpoint_path, cpu_offload=False, remove_key=None, load_from_rank0=False, include_keys=None):
    if not dist.is_initialized() or not load_from_rank0:
        # Single GPU mode
        logger.info(f"Loading weights from {checkpoint_path}")
        cpu_weight_dict = load_pt_safetensors(checkpoint_path, remove_key, include_keys)
        return cpu_weight_dict

    # Multi-GPU mode
    is_weight_loader = False
    current_rank = dist.get_rank()
    if current_rank == 0:
        is_weight_loader = True

    cpu_weight_dict = {}
    if is_weight_loader:
        logger.info(f"Loading weights from {checkpoint_path}")
        cpu_weight_dict = load_pt_safetensors(checkpoint_path, remove_key)

    meta_dict = {}
    if is_weight_loader:
        for key, tensor in cpu_weight_dict.items():
            meta_dict[key] = {"shape": tensor.shape, "dtype": tensor.dtype}

    obj_list = [meta_dict] if is_weight_loader else [None]

    src_global_rank = 0
    dist.broadcast_object_list(obj_list, src=src_global_rank)
    synced_meta_dict = obj_list[0]

    if cpu_offload:
        target_device = "cpu"
        distributed_weight_dict = {key: torch.empty(meta["shape"], dtype=meta["dtype"], device=target_device) for key, meta in synced_meta_dict.items()}
        dist.barrier()
    else:
        target_device = torch.device(f"cuda:{current_rank}")
        distributed_weight_dict = {key: torch.empty(meta["shape"], dtype=meta["dtype"], device=target_device) for key, meta in synced_meta_dict.items()}
        dist.barrier(device_ids=[torch.cuda.current_device()])

    for key in sorted(synced_meta_dict.keys()):
        tensor_to_broadcast = distributed_weight_dict[key]
        if is_weight_loader:
            tensor_to_broadcast.copy_(cpu_weight_dict[key], non_blocking=True)

        if cpu_offload:
            if is_weight_loader:
                gpu_tensor = tensor_to_broadcast.cuda()
                dist.broadcast(gpu_tensor, src=src_global_rank)
                tensor_to_broadcast.copy_(gpu_tensor.cpu(), non_blocking=True)
                del gpu_tensor
                torch.cuda.empty_cache()
            else:
                gpu_tensor = torch.empty_like(tensor_to_broadcast, device="cuda")
                dist.broadcast(gpu_tensor, src=src_global_rank)
                tensor_to_broadcast.copy_(gpu_tensor.cpu(), non_blocking=True)
                del gpu_tensor
                torch.cuda.empty_cache()
        else:
            dist.broadcast(tensor_to_broadcast, src=src_global_rank)

    if is_weight_loader:
        del cpu_weight_dict

    if cpu_offload:
        torch.cuda.empty_cache()

    logger.info(f"Weights distributed across {dist.get_world_size()} devices on {target_device}")
    return distributed_weight_dict


def masks_like(tensor, zero=False, generator=None, p=0.2, prev_len=1):
    assert isinstance(tensor, torch.Tensor)
    out = torch.ones_like(tensor)
    if zero:
        if generator is not None:
            random_num = torch.rand(1, generator=generator, device=generator.device).item()
            if random_num < p:
                out[:, :prev_len] = torch.zeros_like(out[:, :prev_len])
        else:
            out[:, :prev_len] = torch.zeros_like(out[:, :prev_len])
    return out


def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio) ** 0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2


def get_optimal_patched_size_with_sp(patched_h, patched_w, sp_size):
    assert sp_size > 0 and (sp_size & (sp_size - 1)) == 0, "sp_size must be a power of 2"

    h_ratio, w_ratio = 1, 1
    while sp_size != 1:
        sp_size //= 2
        if patched_h % 2 == 0:
            patched_h //= 2
            h_ratio *= 2
        elif patched_w % 2 == 0:
            patched_w //= 2
            w_ratio *= 2
        else:
            if patched_h > patched_w:
                patched_h //= 2
                h_ratio *= 2
            else:
                patched_w //= 2
                w_ratio *= 2
    return patched_h * h_ratio, patched_w * w_ratio


def get_crop_bbox(ori_h, ori_w, tgt_h, tgt_w):
    tgt_ar = tgt_h / tgt_w
    ori_ar = ori_h / ori_w
    if abs(ori_ar - tgt_ar) < 0.01:
        return 0, ori_h, 0, ori_w
    if ori_ar > tgt_ar:
        crop_h = int(tgt_ar * ori_w)
        y0 = (ori_h - crop_h) // 2
        y1 = y0 + crop_h
        return y0, y1, 0, ori_w
    else:
        crop_w = int(ori_h / tgt_ar)
        x0 = (ori_w - crop_w) // 2
        x1 = x0 + crop_w
        return 0, ori_h, x0, x1


def isotropic_crop_resize(frames: torch.Tensor, size: tuple):
    """
    frames: (C, H, W) or (T, C, H, W) or (N, C, H, W)
    size: (H, W)
    """
    original_shape = frames.shape

    if len(frames.shape) == 3:
        frames = frames.unsqueeze(0)
    elif len(frames.shape) == 4 and frames.shape[0] > 1:
        pass

    ori_h, ori_w = frames.shape[2:]
    h, w = size
    y0, y1, x0, x1 = get_crop_bbox(ori_h, ori_w, h, w)
    cropped_frames = frames[:, :, y0:y1, x0:x1]
    resized_frames = resize(cropped_frames, [h, w], InterpolationMode.BICUBIC, antialias=True)

    if len(original_shape) == 3:
        resized_frames = resized_frames.squeeze(0)

    return resized_frames


def fixed_shape_resize(img, target_height, target_width):
    orig_height, orig_width = img.shape[-2:]

    target_ratio = target_height / target_width
    orig_ratio = orig_height / orig_width

    if orig_ratio > target_ratio:
        crop_width = orig_width
        crop_height = int(crop_width * target_ratio)
    else:
        crop_height = orig_height
        crop_width = int(crop_height / target_ratio)

    cropped_img = TF.center_crop(img, [crop_height, crop_width])

    resized_img = TF.resize(cropped_img, [target_height, target_width], antialias=True)

    h, w = resized_img.shape[-2:]
    return resized_img, h, w


def check_path_exists(path: str) -> None:
    """
    Check if a file path exists and raise an error if it doesn't.

    Args:
        path: The file path to check

    Raises:
        FileNotFoundError: If the path is not empty and the file does not exist
    """
    if path and not path.startswith(("http://", "https://", "data:")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")


def validate_task_arguments(args: "argparse.Namespace") -> None:
    """
    Validate arguments based on the task type.

    Args:
        args: Parsed arguments from argparse

    Raises:
        AssertionError: If required arguments are missing or invalid for the task
    """
    # Check model_path exists
    model_path = getattr(args, "model_path", None)
    if model_path:
        check_path_exists(model_path)

    task = args.task

    # Define required file paths for each task
    task_requirements = {
        "i2i": {"required_paths": ["image_path"], "description": "Image-to-Image task requires --image_path"},
        "i2v": {"required_paths": ["image_path"], "description": "Image-to-Video task requires --image_path"},
        "flf2v": {"required_paths": ["image_path", "last_frame_path"], "description": "First-Last-Frame-to-Video task requires --image_path and --last_frame_path"},
        "s2v": {"required_paths": ["image_path", "audio_path"], "description": "Speech-to-Video task requires --image_path and --audio_path"},
        "rs2v": {"required_paths": ["image_path", "audio_path"], "description": "Ref-speech-to-Video task requires --image_path and --audio_path"},
        "vace": {"required_paths": ["src_ref_images"], "description": "Video Appearance Change Editing task requires --src_ref_images"},
        "animate": {"required_paths": ["image_path"], "description": "Animate task requires --image_path"},
        "t2v": {"required_paths": [], "description": "Text-to-Video task"},
        "t2i": {"required_paths": [], "description": "Text-to-Image task"},
        "i2av": {"required_paths": ["image_path"], "description": "Image-to-Audio-Video task requires --image_path"},
    }

    if task not in task_requirements:
        logger.warning(f"Unknown task type: {task}, skipping validation")
        return

    requirements = task_requirements[task]

    # Check required paths
    for path_arg in requirements["required_paths"]:
        path_value = getattr(args, path_arg, "")

        # Check if path is provided
        if not path_value:
            raise ValueError(f"{requirements['description']}: --{path_arg} cannot be empty")

        # For comma-separated paths (like i2i with multiple images)
        if "," in path_value:
            paths = [p.strip() for p in path_value.split(",")]
            for path in paths:
                check_path_exists(path)
        else:
            check_path_exists(path_value)

    logger.info(f"✓ Task '{task}' arguments validated successfully")


def validate_config_paths(config: dict) -> None:
    """
    Validate checkpoint paths in config dictionary.

    Args:
        config: Configuration dictionary

    Raises:
        FileNotFoundError: If any checkpoint path in config does not exist
    """
    # Check dit_quantized_ckpt or dit_original_ckpt
    if "dit_quantized_ckpt" in config and config["dit_quantized_ckpt"] is not None:
        check_path_exists(config["dit_quantized_ckpt"])
        logger.debug(f"✓ Verified dit_quantized_ckpt: {config['dit_quantized_ckpt']}")

    if "dit_original_ckpt" in config and config["dit_original_ckpt"] is not None:
        check_path_exists(config["dit_original_ckpt"])
        logger.debug(f"✓ Verified dit_original_ckpt: {config['dit_original_ckpt']}")

    # For wan2.2, check high and low noise checkpoints
    model_cls = config.get("model_cls", "")
    if model_cls and "wan2.2" in model_cls:
        # Check high noise checkpoints
        if "high_noise_original_ckpt" in config and config["high_noise_original_ckpt"] is not None:
            check_path_exists(config["high_noise_original_ckpt"])
            logger.debug(f"✓ Verified high_noise_original_ckpt: {config['high_noise_original_ckpt']}")

        if "high_noise_quantized_ckpt" in config and config["high_noise_quantized_ckpt"] is not None:
            check_path_exists(config["high_noise_quantized_ckpt"])
            logger.debug(f"✓ Verified high_noise_quantized_ckpt: {config['high_noise_quantized_ckpt']}")

        # Check low noise checkpoints
        if "low_noise_original_ckpt" in config and config["low_noise_original_ckpt"] is not None:
            check_path_exists(config["low_noise_original_ckpt"])
            logger.debug(f"✓ Verified low_noise_original_ckpt: {config['low_noise_original_ckpt']}")

        if "low_noise_quantized_ckpt" in config and config["low_noise_quantized_ckpt"] is not None:
            check_path_exists(config["low_noise_quantized_ckpt"])
            logger.debug(f"✓ Verified low_noise_quantized_ckpt: {config['low_noise_quantized_ckpt']}")

    logger.info("✓ Config checkpoint paths validated successfully")
