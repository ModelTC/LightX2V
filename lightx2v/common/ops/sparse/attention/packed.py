from torch import nn

from lightx2v.common.ops.inference_module import InferenceAttention

from ..basic import VarLenTensor


class PackedAttention(nn.Module):
    """Dense or ragged attention without padding across sample boundaries."""

    def __init__(self, backend="torch_sdpa"):
        super().__init__()
        self.attention = InferenceAttention(backend)

    def forward(self, q, k=None, v=None):
        # Match TRELLIS.2's variable-length FlashAttention entry points, even
        # for a single sample: the dense entry point can produce other values.
        if isinstance(q, VarLenTensor) and self.attention.op.__class__.__name__ == "FlashAttn2Weight":
            import torch
            import flash_attn

            lengths = [part.stop - part.start for part in q.layout]
            cu_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(lengths), dim=0)]).int().to(q.device)
            if k is None:
                return q.replace(flash_attn.flash_attn_varlen_qkvpacked_func(q.feats, cu_q, max(lengths)))
            if isinstance(k, VarLenTensor):
                kv_lengths = [part.stop - part.start for part in k.layout]
                k_values = k.feats
            else:
                kv_lengths = [k.shape[1]] * k.shape[0]
                k_values = k.reshape(-1, *k.shape[2:])
            cu_k = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_lengths), dim=0)]).int().to(q.device)
            if v is None:
                output = flash_attn.flash_attn_varlen_kvpacked_func(q.feats, k_values, cu_q, cu_k, max(lengths), max(kv_lengths))
            else:
                v_values = v.feats if isinstance(v, VarLenTensor) else v.reshape(-1, *v.shape[2:])
                output = flash_attn.flash_attn_varlen_func(q.feats, k_values, v_values, cu_q, cu_k, max(lengths), max(kv_lengths))
            return q.replace(output)
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
