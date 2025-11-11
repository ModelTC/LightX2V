from __future__ import annotations

from typing import Iterable, Sequence

import torch


def svdq_gemm_w4a4_cuda(
    act: torch.Tensor | None,
    wgt: torch.Tensor | None,
    out: torch.Tensor | None = None,
    qout: torch.Tensor | None = None,
    ascales: torch.Tensor | None = None,
    wscales: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    poolout: torch.Tensor | None = None,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    norm_q: torch.Tensor | None = None,
    norm_k: torch.Tensor | None = None,
    rotary_emb: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    smooth_factor: torch.Tensor | None = None,
    out_vk: torch.Tensor | None = None,
    out_linearattn: torch.Tensor | None = None,
    act_unsigned: bool = False,
    lora_scales: Sequence[float] | None = None,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float = 1.0,
    wcscales: torch.Tensor | None = None,
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    attn_tokens: int = 0,
) -> None:
    """
    請參考 nunchaku 的同名函數說明。
    """
    if act is None or wgt is None:
        raise ValueError("act與wgt必須提供。")

    if lora_scales is None:
        lora_scales = []

    torch.ops.lightx2v_kernel.svdq_gemm_w4a4_cuda(
        act,
        wgt,
        out,
        qout,
        ascales,
        wscales,
        oscales,
        poolout,
        lora_act_in,
        lora_up,
        lora_down,
        lora_act_out,
        norm_q,
        norm_k,
        rotary_emb,
        bias,
        smooth_factor,
        out_vk,
        out_linearattn,
        act_unsigned,
        list(lora_scales),
        fuse_silu,
        fp4,
        alpha,
        wcscales,
        out_q,
        out_k,
        out_v,
        attn_tokens,
    )


def _ceil_divide(val: int, div: int) -> int:
    return (val + div - 1) // div


def svdq_quantize_w4a4_act_fuse_lora_cuda(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    量化輸入並選擇性輸出LoRA結果，介面與 nunchaku 保持一致。
    """
    if input is None:
        raise ValueError("input 不可為 None。")

    batch_size, channels = input.shape
    rank = lora_down.shape[1] if lora_down is not None else 0
    batch_size_pad = _ceil_divide(batch_size, pad_size) * pad_size

    if output is None:
        output = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
    if oscales is None:
        if fp4:
            if channels % 16 != 0:
                raise ValueError("NVFP4 模式要求 channels 可被 16 整除。")
            oscales = torch.empty(channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=input.device)
        else:
            if channels % 64 != 0:
                raise ValueError("INT4 模式要求 channels 可被 64 整除。")
            oscales = torch.empty(channels // 64, batch_size_pad, dtype=input.dtype, device=input.device)
    call_lora_act_out = lora_act_out
    if lora_down is not None and call_lora_act_out is None:
        call_lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)
    if lora_down is None:
        call_lora_act_out = None

    torch.ops.lightx2v_kernel.svdq_quantize_w4a4_act_fuse_lora_cuda(
        input, output, oscales, lora_down, call_lora_act_out, smooth, fuse_glu, fp4
    )

    if lora_down is None:
        call_lora_act_out = torch.empty(0, device=input.device)

    return output, oscales, call_lora_act_out
