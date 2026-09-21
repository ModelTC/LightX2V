import torch
import torch.nn.functional as F

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .torch_sdpa import TorchSDPAWeight


@ATTN_WEIGHT_REGISTER("torch_sdpa_mps")
class TorchSDPAMPSWeight(TorchSDPAWeight):
    def apply(self, q, k, v, drop_rate=0, attn_mask=None, causal=False, **kwargs):
        chunk_size = kwargs.get("mps_sdpa_query_chunk_size", 0)
        if not isinstance(chunk_size, int) or chunk_size <= 0 or q.shape[-3] <= chunk_size or attn_mask is not None or causal or drop_rate != 0:
            return super().apply(q, k, v, drop_rate=drop_rate, attn_mask=attn_mask, causal=causal, **kwargs)

        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # Each query chunk attends to all keys/values, preserving the softmax domain.
        x = torch.cat(
            [
                F.scaled_dot_product_attention(q[:, :, start : start + chunk_size, :], k, v, attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=q.shape[1] != k.shape[1])
                for start in range(0, q.shape[2], chunk_size)
            ],
            dim=2,
        )
        x = x.transpose(1, 2)
        return x.reshape(x.shape[0], x.shape[1], -1).squeeze(0)
