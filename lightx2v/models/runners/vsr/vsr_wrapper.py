import os
from typing import Optional

import torch
from torch.nn import functional as F

from lightx2v.utils.profiler import *

try:
    from diffsynth import FlashVSRTinyPipeline, ModelManager
except ImportError:
    ModelManager = None
    FlashVSRTinyPipeline = None


from .utils.TCDecoder import build_tcdecoder
from .utils.utils import Buffer_LQ4x_Proj


def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. Increase scale (got {scale}).")

    return sW, sH, tW, tH


def prepare_input_tensor(input_tensor, scale: float = 2.0, dtype=torch.bfloat16, device="cuda"):
    """
    视频预处理: [T,H,W,3] -> [1,C,F,H,W]
    1. GPU 上完成插值 + 中心裁剪
    2. 自动 pad 帧数到 8n-3
    """

    input_tensor = input_tensor.to(device=device, dtype=torch.float32)  # [T,H,W,3]
    total, h0, w0, _ = input_tensor.shape

    # 计算缩放与目标分辨率
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    print(f"Scaled (x{scale:.2f}): {sW}x{sH} -> Target: {tW}x{tH}")

    # Pad 帧数到 8n-3
    idx = list(range(total)) + [total - 1] * 4
    F_target = largest_8n1_leq(len(idx))
    if F_target == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {len(idx)}.")
    idx = idx[:F_target]
    print(f"Target Frames (8n-3): {F_target - 4}")

    # 取帧并转为 tensor 格式 [B,C,H,W]
    frames = input_tensor[idx]  # [F,H,W,3]
    frames = frames.permute(0, 3, 1, 2) * 2.0 - 1.0  # [F,3,H,W] -> [-1,1]

    # 上采样 (Bilinear)
    frames = F.interpolate(frames, scale_factor=scale, mode="bicubic", align_corners=False)
    _, _, sH, sW = frames.shape

    # 中心裁剪
    left = (sW - tW) // 2
    top = (sH - tH) // 2
    frames = frames[:, :, top : top + tH, left : left + tW]

    # 输出 [1, C, F, H, W]
    vid = frames.permute(1, 0, 2, 3).unsqueeze(0).to(dtype)
    return vid, tH, tW, F_target


def init_pipeline(model_path):
    # print(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models(
        [
            model_path + "/diffusion_pytorch_model_streaming_dmd.safetensors",
        ]
    )
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    LQ_proj_in_path = model_path + "/LQ_proj_in.ckpt"
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to("cuda")

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16 + 768)
    mis = pipe.TCDecoder.load_state_dict(torch.load(model_path + "/TCDecoder.ckpt"), strict=False)
    # print(mis)

    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


class VSRWrapper:
    def __init__(self, model_path, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup torch for optimal performance
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # Load model
        self.dtype, self.device = torch.bfloat16, "cuda"
        self.sparse_ratio = 2.0  # Recommended: 1.5 or 2.0. 1.5 → faster; 2.0 → more stable.
        with ProfilingContext4DebugL2("Load VSR model"):
            self.pipe = init_pipeline(model_path)

    @ProfilingContext4DebugL2("VSR video")
    def super_resolve_frames(
        self,
        video: torch.Tensor,  # [T,H,W,C]
        seed: float = 0.0,
        scale: float = 2.0,
    ) -> torch.Tensor:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        LQ, th, tw, F = prepare_input_tensor(video, scale=scale, dtype=self.dtype, device=self.device)

        video = self.pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=seed,
            LQ_video=LQ,
            num_frames=F,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=self.sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=3.0,
            local_range=11,  # Recommended: 9 or 11. local_range=9 → sharper details; 11 → more stable results.
            color_fix=True,
        )
        video = (video + 1.0) / 2.0  # 将 [-1,1] 映射到 [0,1]
        video = video.permute(1, 2, 3, 0).clamp(0.0, 1.0)  # [C,T,H,W] -> [T,H,W,C]
        return video
