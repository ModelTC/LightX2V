import torch

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

try:
    import torch_br
except ImportError:
    torch_br = None


@PLATFORM_ATTN_WEIGHT_REGISTER("biren_fmha_attn")
class BirenFMHAAttnWeight(AttnWeightTemplate):
    def __init__(self):
        if torch_br is None:
            raise ImportError("torch_br is not installed.")
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
        """
        Apply SUPA FMHA attention.

        Input shapes:
            q: [total_seq_len, num_heads, head_dim]
            k: [total_seq_len_kv, num_heads, head_dim]
            v: [total_seq_len_kv, num_heads, head_dim]

        SUPA kernel expects:
            q/k/v: [batch * num_heads, seq_len, head_dim]

        Returns:
            [total_seqlen, num_heads * head_dim]
        """
        half_dtypes = (torch.float16, torch.bfloat16)

        def to_half(x):
            return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

        if len(q.shape) == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])

        seq_len_q = q.shape[0]
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        seq_len_kv = k.shape[0]

        # Transpose: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        # This is equivalent to [batch*num_heads, seq_len, head_dim] with batch=1
        q_t = to_half(q).permute(1, 0, 2).contiguous()
        k_t = to_half(k).permute(1, 0, 2).contiguous()
        v_t = to_half(v).permute(1, 0, 2).contiguous()

        # Prepare cu_seqlens for SUPA
        cu_seqlens_q_t = torch.tensor([0, seq_len_q], dtype=torch.int32, device=q.device)
        max_seqlen_q_t = torch.tensor([seq_len_q], dtype=torch.int32, device=q.device)

        # BR166 FMHA API only accepts cu_seqlens_q/max_seqlen_q.
        # Cross-attention (different Q/KV lengths) is handled by the kernel
        # based on the actual tensor shapes.
        output, _ = torch_br.sueager_scaled_dot_product_attention_fwd(
            q_t, k_t, v_t,
            mask=None,
            dropout_prob=0,
            is_causal=False,
            scale=None,
            algorithm='FMHA',
            cu_seqlens_q=cu_seqlens_q_t,
            max_seqlen_q=max_seqlen_q_t,
        )

        # output: [num_heads, seq_len_q, head_dim] with Split_0 across 2 dies
        # Need to convert to [seq_len_q, num_heads * head_dim]
        # The FMHA kernel produces output split on dim 0 (heads) across dies.
        # Naive transpose(0,1).contiguous() fails because Split_0 becomes Split_1
        # after transpose, and D2D copy doesn't support split on dim != 0.
        # Solution: copy Split_0 to a standard BB tensor (copy from split dim=0
        # IS supported), then do normal transpose+reshape on the BB tensor.
        output_bb = torch.empty(
            num_heads, seq_len_q, head_dim, dtype=output.dtype, device=output.device
        )
        output_bb.copy_(output)
        output = output_bb.transpose(0, 1).contiguous().view(seq_len_q, num_heads * head_dim)

        return output
