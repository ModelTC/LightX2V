import torch

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

try:
    from lightx2v_kernel.fmha import cute_dsl_vit_fmha_sm110
except ImportError:
    cute_dsl_vit_fmha_sm110 = None


@ATTN_WEIGHT_REGISTER("cute_dsl_vit_fmha_sm110")
class CuteDslViTFMHASm110Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        if cute_dsl_vit_fmha_sm110 is None:
            raise RuntimeError(
                "cute_dsl_vit_fmha_sm110 is unavailable. Install a Thor lightx2v-kernel wheel "
                "built with LIGHTX2V_ENABLE_CUTEDSL_VIT_FMHA=ON."
            )
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("CuTe DSL ViT FMHA expects Q, K, and V with shape [total_S, H, D].")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("CuTe DSL ViT FMHA requires self-attention Q, K, and V to have identical shapes.")
        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise ValueError("CuTe DSL ViT FMHA requires CUDA Q, K, and V tensors.")
        if q.device != k.device or q.device != v.device:
            raise ValueError("CuTe DSL ViT FMHA requires Q, K, and V on the same device.")
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"CuTe DSL ViT FMHA only accepts BF16 or FP16 inputs, got {q.dtype}.")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("CuTe DSL ViT FMHA requires Q, K, and V to use the same dtype.")
        if cu_seqlens_q is None or cu_seqlens_kv is None:
            raise ValueError("CuTe DSL ViT FMHA requires cu_seqlens_q and cu_seqlens_kv.")
        if cu_seqlens_q is not cu_seqlens_kv and not torch.equal(cu_seqlens_q, cu_seqlens_kv):
            raise ValueError("CuTe DSL ViT FMHA requires identical Q and KV sequence boundaries.")
        if max_seqlen_q is None or max_seqlen_kv is None:
            raise ValueError("CuTe DSL ViT FMHA requires max_seqlen_q and max_seqlen_kv.")
        if max_seqlen_q != max_seqlen_kv:
            raise ValueError("CuTe DSL ViT FMHA requires equal Q and KV maximum sequence lengths.")

        q_fp16 = q.to(dtype=torch.float16).contiguous()
        k_fp16 = k.to(dtype=torch.float16).contiguous()
        v_fp16 = v.to(dtype=torch.float16).contiguous()
        cu_seqlens = cu_seqlens_q.to(device=q.device, dtype=torch.int32, non_blocking=True).contiguous()
        output = cute_dsl_vit_fmha_sm110(
            q_fp16,
            k_fp16,
            v_fp16,
            cu_seqlens,
            int(max_seqlen_q),
        )
        return output.reshape(q.shape[0], -1)
