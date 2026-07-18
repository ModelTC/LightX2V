import torch


def cute_dsl_vit_fmha_sm110(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    try:
        op = torch.ops.lightx2v_kernel.cute_dsl_vit_fmha_sm110.default
    except AttributeError as error:
        raise RuntimeError(
            "cute_dsl_vit_fmha_sm110 is unavailable. Rebuild lightx2v-kernel with "
            "LIGHTX2V_THOR_NVFP4_ONLY=ON and LIGHTX2V_ENABLE_CUTEDSL_VIT_FMHA=ON."
        ) from error
    return op(q, k, v, cu_seqlens, max_seqlen)
