# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Depth model utilities: MoGe monocular depth and Depth Anything 3 loaders."""

import os
import re
import sys
from typing import Tuple

import cv2
import torch

from loguru import logger


# ---------------------------------------------------------------------------
# MoGe
# ---------------------------------------------------------------------------

def load_moge_model(device: torch.device):
    # Disable xformers for MoGe's DINOv2 backbone so it doesn't dispatch to
    # flash_attn CUDA kernels that segfault on aarch64 / Grace Hopper.
    os.environ["XFORMERS_DISABLED"] = "1"
    try:
        from moge.model.v1 import MoGeModel  # type: ignore
    except Exception as e:
        raise ImportError("MoGe is required for --input_image_path flow. Please install `moge`. Error: " + str(e))

    for _name, _mod in sys.modules.items():
        if "moge" in _name and hasattr(_mod, "XFORMERS_ENABLED"):
            _mod.XFORMERS_ENABLED = False

    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    model.eval()
    return model


def moge_infer_depth_intrinsics(
    moge_model,
    img_rgb_uint8: torch.Tensor,
    depth_pred_hw: Tuple[int, int] = (720, 1280),
    target_hw: Tuple[int, int] = (704, 1280),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        img_rgb_uint8: [H,W,3] uint8 on CPU
    Returns:
        image_chw_norm: [1,3,Ht,Wt] float in [0,1]
        depth_hw: [Ht,Wt] float (large for invalid)
        intrinsics_33: [3,3] pixel units
        mask_hw: [Ht,Wt] bool mask
    """
    device = next(moge_model.parameters()).device
    Ht, Wt = target_hw

    img_resized = cv2.resize(img_rgb_uint8.numpy(), (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    img_chw_0_1 = torch.tensor(img_resized / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

    with torch.no_grad():
        out = moge_model.infer(img_chw_0_1)

    depth_hw = out["depth"].to(device)
    mask_hw = out["mask"].to(device)
    K_norm = out["intrinsics"].to(device)  # 3x3 normalized

    depth_hw = torch.nan_to_num(depth_hw, nan=1e4).clamp(min=0, max=1e4)
    depth_hw = torch.where(mask_hw == 0, torch.tensor(1000.0, device=device, dtype=depth_hw.dtype), depth_hw)

    # Scale intrinsics to pixel units for (Wt, Ht)
    K = K_norm.clone()
    K[0, 0] *= Wt
    K[1, 1] *= Ht
    K[0, 2] *= Wt
    K[1, 2] *= Ht

    return img_chw_0_1.unsqueeze(0), depth_hw, K, mask_hw


# ---------------------------------------------------------------------------
# Depth Anything 3
# ---------------------------------------------------------------------------

def _import_da3_api():
    """Lazy-import DepthAnything3 from the vendored depth_anything_3 submodule."""
    da3_src_root = os.path.join(
        os.path.dirname(__file__),
        "depth_anything_3",
        "src",
    )
    if da3_src_root not in sys.path:
        sys.path.insert(0, da3_src_root)
    stale_keys = [k for k in sys.modules if k == "depth_anything_3" or k.startswith("depth_anything_3.")]
    for k in stale_keys:
        del sys.modules[k]
    try:
        from depth_anything_3.api import DepthAnything3  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import DepthAnything3 from vendored depth_anything_3. "
            "Make sure the depth_anything_3 submodule is present and its "
            "dependencies are available."
        ) from e
    return DepthAnything3


def _resolve_da3_local_model_name(model_name: str) -> str:
    """Map a Hub-style model id to a local DA3 config name when possible."""
    da3_src_root = os.path.join(
        os.path.dirname(__file__),
        "depth_anything_3",
        "src",
    )
    if da3_src_root not in sys.path:
        sys.path.insert(0, da3_src_root)
    from depth_anything_3.registry import MODEL_REGISTRY  # type: ignore

    candidates = [str(model_name).strip()]
    tail = candidates[0].split("/")[-1]
    candidates.extend(
        [
            tail,
            tail.lower(),
            tail.lower().replace("_", "-"),
            re.sub(r"[-_]\d+(?:\.\d+)*$", "", tail.lower().replace("_", "-")),
        ]
    )
    for candidate in candidates:
        if candidate in MODEL_REGISTRY:
            return candidate
    raise KeyError(
        "Unable to map DA3 model name to a local config: "
        f"{model_name}. Available local configs: {', '.join(MODEL_REGISTRY.keys())}"
    )


def load_da3_from_custom_checkpoint(
    ckpt_path: str,
    pretrained_path: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    device: str = "cuda",
    strict: bool = True,
):
    """Load DepthAnything3 from a custom finetuned checkpoint."""
    DepthAnything3 = _import_da3_api()
    local_model_name = _resolve_da3_local_model_name(pretrained_path)
    logger.info(
        f"Initializing DA3 architecture from local config: {local_model_name} "
        f"(requested={pretrained_path})"
    )
    model = DepthAnything3(model_name=local_model_name)

    logger.info(f"Loading custom checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "module" in checkpoint:
        state_dict = checkpoint["module"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Strip "model." prefix
    prefix = "model."
    converted = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    missing, unexpected = model.model.load_state_dict(converted, strict=strict)
    if missing and not strict:
        logger.info(
            f"Custom checkpoint is missing {len(missing)} tensors; "
            f"falling back to pretrained base weights from {pretrained_path}."
        )
        model = DepthAnything3.from_pretrained(pretrained_path)
        missing, unexpected = model.model.load_state_dict(converted, strict=strict)
    if missing:
        logger.info(f"Missing keys when loading custom checkpoint: {len(missing)}")
    if unexpected:
        logger.info(f"Unexpected keys when loading custom checkpoint: {len(unexpected)}")
    logger.info(f"Loaded {len(converted)} parameters from custom checkpoint")

    model = model.to(device=device)
    model.eval()
    return model


def load_da3_model(
    da3_model_name: str,
    da3_model_path_custom: str = None,
    device: str = "cuda",
):
    """Load a DepthAnything3 model, optionally from a custom checkpoint."""
    if da3_model_path_custom is not None:
        logger.info(f"Loading DA3 model with custom checkpoint: {da3_model_path_custom}")
        model = load_da3_from_custom_checkpoint(
            ckpt_path=da3_model_path_custom,
            pretrained_path=da3_model_name,
            device=device,
            strict=False,
        )
    else:
        logger.info(f"Loading DA3 model from pretrained: {da3_model_name}")
        DepthAnything3 = _import_da3_api()
        model = DepthAnything3.from_pretrained(da3_model_name).to(device)

    model.eval()
    return model
