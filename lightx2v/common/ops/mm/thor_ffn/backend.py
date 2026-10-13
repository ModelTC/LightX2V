import os
from functools import lru_cache
from pathlib import Path
from threading import local

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_plan_cache = local()


@lru_cache(None)
def native_down_extension():
    cutlass_root = Path(os.environ.get("CUTLASS_PATH") or "/opt/cutlass").expanduser().resolve()
    return load(
        name="thor_ffn_dram_opt_down_v1",
        sources=[str(Path(__file__).with_name("down_dram_opt.cu"))],
        extra_include_paths=[str(cutlass_root / "include"), str(cutlass_root / "tools/util/include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_110a", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-diag-suppress=20012,177"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def down_bias(mm_weight, x):
    """DreamZero BF16 down projection with current weights and fused bias."""
    from lightx2v.common.ops.mm.mm_weight import MMWeight

    if type(mm_weight) is not MMWeight or mm_weight._get_actual_bias() is None:
        return mm_weight.apply(x)
    weight = mm_weight._get_actual_weight().t().contiguous()
    bias = mm_weight._get_actual_bias().contiguous()
    if (
        not mm_weight.has_lora_branch
        and x.is_cuda
        and x.dtype == weight.dtype == bias.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape == (1785, 13824)
        and weight.shape == (5120, 13824)
        and bias.shape == (5120,)
        and torch.cuda.get_device_capability(x.device) == (11, 0)
    ):
        y = torch.empty((1785, 5120), device=x.device, dtype=x.dtype)
        key = (x.device, torch.cuda.current_stream(x.device).cuda_stream)
        cached = getattr(_plan_cache, "down_entry", None)
        if cached is None or cached[0] != key:
            plan = native_down_extension().DownPlan(x)
            _plan_cache.down_entry = (key, plan)
        else:
            plan = cached[1]
        plan.run_with(x, weight, bias, y)
    else:
        return mm_weight.apply(x)
    return y


@lru_cache(None)
def extension():
    root = Path(__file__).resolve().parent
    # CUTLASS_PATH points to the checkout root, not its include directory.
    cutlass_root = Path(os.environ.get("CUTLASS_PATH") or "/opt/cutlass").expanduser().resolve()
    return load(
        name="thor_ffn_cutlass_v1",
        sources=[str(root / "extension.cu")],
        extra_include_paths=[str(cutlass_root / "include"), str(cutlass_root / "tools/util/include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-arch=sm_110a",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-diag-suppress=20012",
        ],
        verbose=False,
    )


def up_gelu(mm_weight, x):
    """DreamZero up projection plus bias and tanh GELU, selected by thor."""
    if mm_weight.has_lora_branch:
        # apply() adds the adapter contribution before the activation.
        return F.gelu(mm_weight.apply(x), approximate="tanh")

    weight = mm_weight._get_actual_weight().t().contiguous()
    bias = mm_weight._get_actual_bias()
    x = x.contiguous()
    y = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    # One execution plan per calling thread, shared across model layers. The
    # native plan owns only workspace, never model weights or activations.
    # A shape/device/stream change replaces the entry instead of growing it.
    key = (x.device, x.dtype, x.shape[0], torch.cuda.current_stream(x.device).cuda_stream)
    cached = getattr(_plan_cache, "entry", None)
    if cached is None or cached[0] != key:
        plan = extension().prepare(x, weight, bias, y)
        _plan_cache.entry = (key, plan)
    else:
        plan = cached[1]
    plan.run_with(x, weight, bias, y)
    return y
