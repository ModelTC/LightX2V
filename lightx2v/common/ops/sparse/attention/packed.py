from torch import nn

from lightx2v.common.ops.inference_module import InferenceAttention

from ..basic import VarLenTensor


class PackedAttention(nn.Module):
    """Dense or ragged attention without padding across sample boundaries."""

    def __init__(self, backend="torch_sdpa"):
        super().__init__()
        self.attention = InferenceAttention(backend)

    def forward(self, q, k=None, v=None):
        if k is None:
            q, k, v = q.unbind(dim=-3)
        elif v is None:
            k, v = k.unbind(dim=-3)
        if isinstance(q, VarLenTensor):
            output = []
            for batch, section in enumerate(q.layout):
                query = q.feats[section].unsqueeze(0)
                key = k.feats[k.layout[batch]].unsqueeze(0) if isinstance(k, VarLenTensor) else k[batch : batch + 1]
                value = v.feats[v.layout[batch]].unsqueeze(0) if isinstance(v, VarLenTensor) else v[batch : batch + 1]
                output.append(self._dense(query, key, value).squeeze(0))
            import torch

            return q.replace(torch.cat(output, dim=0))
        return self._dense(q, k, v)

    def _dense(self, q, k, v):
        return self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
