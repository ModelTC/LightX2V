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
        name="thor_ffn_dram_opt_down_v3",
        sources=[str(Path(__file__).with_name("down_dram_opt.cu"))],
        extra_include_paths=[str(cutlass_root / "include"), str(cutlass_root / "tools/util/include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_110a", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-diag-suppress=20012,177", "-DNDEBUG"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def down_bias(mm_weight, x):
    """DreamZero BF16 down projection with current weights and fused bias.

    1785 rows (the denoising forward) run the hand-written kernel with its
    tuned K-reversing tile order.  Other row counts previously ran cuBLAS; they
    now use kernels that accumulate K in ascending order like cuBLAS and give
    bitwise-identical results: the same hand-written kernel with every tile
    forward-K for 1536 < rows <= 1792 (the 1760-row cache-update forward), and
    the CUTLASS linear otherwise.
    """
    from lightx2v.common.ops.mm.mm_weight import MMWeight

    if type(mm_weight) is not MMWeight or mm_weight._get_actual_bias() is None:
        return mm_weight.apply(x)
    weight = mm_weight._get_actual_weight().t().contiguous()
    bias = mm_weight._get_actual_bias().contiguous()
    if not (
        not mm_weight.has_lora_branch
        and x.is_cuda
        and x.dtype == weight.dtype == bias.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.dim() == 2
        and x.shape[1] == 13824
        and weight.shape == (5120, 13824)
        and bias.shape == (5120,)
        and torch.cuda.get_device_capability(x.device) == (11, 0)
    ):
        return mm_weight.apply(x)
    rows = x.shape[0]
    if not 1536 < rows <= 1792:
        from lightx2v.common.ops.mm.thor_qkv import linear

        return linear(mm_weight, x)
    forward_k = rows != 1785
    y = torch.empty((rows, 5120), device=x.device, dtype=x.dtype)
    key = (x.device, torch.cuda.current_stream(x.device).cuda_stream, forward_k)
    cache = getattr(_plan_cache, "down_plans", None)
    if cache is None:
        cache = _plan_cache.down_plans = {}
    plan = cache.get(key)
    if plan is None:
        plan = cache[key] = native_down_extension().DownPlan(x, forward_k)
    plan.run_with(x, weight, bias, y)
    return y


@lru_cache(None)
def extension():
    root = Path(__file__).resolve().parent
    # CUTLASS_PATH points to the checkout root, not its include directory.
    cutlass_root = Path(os.environ.get("CUTLASS_PATH") or "/opt/cutlass").expanduser().resolve()
    return load(
        name="thor_ffn_cutlass_v2",
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
            "-DNDEBUG",
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
