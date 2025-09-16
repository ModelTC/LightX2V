import glob
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
from einops import rearrange
from loguru import logger


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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


def find_torch_model_path(config, ckpt_config_key=None, filename=None, subdir=["original", "fp8", "int8", "distill_models", "distill_fp8", "distill_int8"]):
    if ckpt_config_key and config.get(ckpt_config_key, None) is not None:
        return config.get(ckpt_config_key)

    paths_to_check = [
        os.path.join(config.model_path, filename),
    ]
    if isinstance(subdir, list):
        for sub in subdir:
            paths_to_check.insert(0, os.path.join(config.model_path, sub, filename))
    else:
        paths_to_check.insert(0, os.path.join(config.model_path, subdir, filename))

    for path in paths_to_check:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"PyTorch model file '{filename}' not found.\nPlease download the model from https://huggingface.co/lightx2v/ or specify the model path in the configuration file.")


def find_hf_model_path(config, model_path, ckpt_config_key=None, subdir=["original", "fp8", "int8", "distill_models", "distill_fp8", "distill_int8"]):
    if ckpt_config_key and config.get(ckpt_config_key, None) is not None:
        return config.get(ckpt_config_key)

    paths_to_check = [model_path]
    if isinstance(subdir, list):
        for sub in subdir:
            paths_to_check.insert(0, os.path.join(model_path, sub))
    else:
        paths_to_check.insert(0, os.path.join(model_path, subdir))
    for path in paths_to_check:
        safetensors_pattern = os.path.join(path, "*.safetensors")
        safetensors_files = glob.glob(safetensors_pattern)
        if safetensors_files:
            return path
    raise FileNotFoundError(f"No Hugging Face model files (.safetensors) found.\nPlease download the model from: https://huggingface.co/lightx2v/ or specify the model path in the configuration file.")


def find_gguf_model_path(config, ckpt_config_key=None, subdir=None):
    gguf_path = config.get(ckpt_config_key, None)
    if gguf_path is None:
        raise ValueError(f"GGUF path not found in config with key '{ckpt_config_key}'")
    if not isinstance(gguf_path, str) or not gguf_path.endswith(".gguf"):
        raise ValueError(f"GGUF path must be a string ending with '.gguf', got: {gguf_path}")
    if os.sep in gguf_path or (os.altsep and os.altsep in gguf_path):
        if os.path.exists(gguf_path):
            logger.info(f"Found GGUF model file in: {gguf_path}")
            return os.path.abspath(gguf_path)
        else:
            raise FileNotFoundError(f"GGUF file not found at path: {gguf_path}")
    else:
        # It's just a filename, search in predefined paths
        paths_to_check = [config.model_path]
        if subdir:
            paths_to_check.append(os.path.join(config.model_path, subdir))

        for path in paths_to_check:
            gguf_file_path = os.path.join(path, gguf_path)
            gguf_file = glob.glob(gguf_file_path)
            if gguf_file:
                logger.info(f"Found GGUF model file in: {gguf_file_path}")
                return gguf_file_path

    raise FileNotFoundError(f"No GGUF model files (.gguf) found.\nPlease download the model from: https://huggingface.co/lightx2v/ or specify the model path in the configuration file.")


def load_safetensors(in_path, remove_key):
    if os.path.isdir(in_path):
        return load_safetensors_from_dir(in_path, remove_key)
    elif os.path.isfile(in_path):
        return load_safetensors_from_path(in_path, remove_key)
    else:
        raise ValueError(f"{in_path} does not exist")


def load_safetensors_from_path(in_path, remove_key):
    tensors = {}
    with safetensors.safe_open(in_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if remove_key not in key:
                tensors[key] = f.get_tensor(key)
    return tensors


def load_safetensors_from_dir(in_dir, remove_key):
    tensors = {}
    safetensors = os.listdir(in_dir)
    safetensors = [f for f in safetensors if f.endswith(".safetensors")]
    for f in safetensors:
        tensors.update(load_safetensors_from_path(os.path.join(in_dir, f), remove_key))
    return tensors


def load_pt_safetensors(in_path, remove_key):
    ext = os.path.splitext(in_path)[-1]
    if ext in (".pt", ".pth", ".tar"):
        state_dict = torch.load(in_path, map_location="cpu", weights_only=True)
        for key in list(state_dict.keys()):
            if remove_key and remove_key in key:
                state_dict.pop(key)
    else:
        state_dict = load_safetensors(in_path, remove_key)
    return state_dict


def load_weights(checkpoint_path, cpu_offload=False, remove_key=None):
    if not dist.is_initialized():
        # Single GPU mode
        logger.info(f"Loading weights from {checkpoint_path}")
        cpu_weight_dict = load_pt_safetensors(checkpoint_path, remove_key)
        return cpu_weight_dict

    # Multi-GPU mode
    is_weight_loader = False
    current_rank = dist.get_rank()
    if current_rank == 0:
        is_weight_loader = True

    cpu_weight_dict = {}
    if is_weight_loader:
        logger.info(f"Loading weights from {checkpoint_path}")
        cpu_weight_dict = load_pt_safetensors(checkpoint_path)
        for key in list(cpu_weight_dict.keys()):
            if remove_key and remove_key in key:
                cpu_weight_dict.pop(key)

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
