#!/usr/bin/env python3
"""Run single-GPU FP32 Wan training with memory-efficient SDPA enforced."""

import runpy
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")
