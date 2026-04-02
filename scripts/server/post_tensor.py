import argparse
import base64
from io import BytesIO
from pathlib import Path

import requests
import torch
from loguru import logger


def tensor_to_base64(t: torch.Tensor) -> str:
    buffer = BytesIO()
    torch.save(t, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_tensor(s: str) -> torch.Tensor:
    raw = base64.b64decode(s)
    return torch.load(BytesIO(raw), map_location="cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare tensor_infer output with saved Self-Forcing real_score dump.")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/tasks/video/tensor_infer")
    parser.add_argument(
        "--dump_path",
        type=str,
        default="/data/nvme4/gushiqiao/Self-Forcing/debug_real_score/real_score_rank0.pt",
        help="Path to saved torch.dump from Self-Forcing/model/dmd.py",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    dump_path = Path(args.dump_path)
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump file not found: {dump_path}")

    saved = torch.load(str(dump_path), map_location="cpu")
    noisy = saved["noisy_image_or_video"]  # [B,F,C,H,W]
    conditional_dict = saved["conditional_dict"]
    unconditional_dict = saved["unconditional_dict"]
    timestep = saved["timestep"]  # [B,F]
    target_pred_real_image = saved["pred_real_image"]  # [B,F,C,H,W]

    context = conditional_dict["prompt_embeds"]
    context_null = unconditional_dict["prompt_embeds"]
    timestep_scalar = timestep.flatten()[0].to(torch.int64)

    message = {
        "prompt": "",
        "negative_prompt": "",
        "seed": 0,
        "noisy_tensor": tensor_to_base64(noisy),
        "context_tensor": tensor_to_base64(context),
        "context_null_tensor": tensor_to_base64(context_null),
        "timestep_tensor": tensor_to_base64(timestep_scalar.unsqueeze(0)),
        "return_pred_x0": True,
    }

    logger.info(f"POST {args.url}")
    response = requests.post(args.url, json=message, timeout=120)
    logger.info(f"status_code={response.status_code}")
    response.raise_for_status()

    data = response.json()
    logger.info(f"response keys={list(data.keys())}")
    if data.get("status") != "success":
        logger.error(data)
        raise SystemExit(1)

    noise_pred = base64_to_tensor(data["noise_pred_tensor"])
    logger.info(f"noise_pred.shape={tuple(noise_pred.shape)}, dtype={noise_pred.dtype}")
    pred_x0 = base64_to_tensor(data["pred_x0_tensor"])
    logger.info(f"pred_x0.shape={tuple(pred_x0.shape)}, dtype={pred_x0.dtype}")

    # Align target shape [B,F,C,H,W] -> [C,F,H,W]
    target = target_pred_real_image
    if target.ndim == 5:
        if target.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got {tuple(target.shape)}")
        target = target.squeeze(0).permute(1, 0, 2, 3).contiguous()
    elif target.ndim != 4:
        raise ValueError(f"Unexpected target ndim: {target.ndim}")

    pred_x0 = pred_x0.to(torch.float32)
    target = target.to(torch.float32)

    # print(pred_x0)
    # print(pred_x0.shape)
    # print(target)
    # print(target.shape)
    # exit()

    abs_diff = (pred_x0 - target).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        pred_x0.reshape(1, -1),
        target.reshape(1, -1),
        dim=1,
    ).item()
    is_close = torch.allclose(pred_x0, target, atol=args.atol, rtol=args.rtol)

    logger.info(f"allclose={is_close}, atol={args.atol}, rtol={args.rtol}")
    logger.info(f"abs diff: max={max_diff:.6e}, mean={mean_diff:.6e}")
    logger.info(f"cosine similarity={cos_sim:.8f}")

    if not is_close:
        raise SystemExit(2)
    logger.info("Verification passed: service output matches saved pred_real_image.")
