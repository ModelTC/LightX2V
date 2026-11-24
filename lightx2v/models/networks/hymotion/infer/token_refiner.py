# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.hymotion.weights.token_refiner import Optional, Tensor, apply_gate, attention, rearrange, torch


def IndividualTokenRefinerBlock_forward(self, x: Tensor, c: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
    gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=-1)
    norm_x = self.norm1(x)
    qkv = self.self_attn_qkv(norm_x)
    q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
    q = self.self_attn_q_norm(q).to(v)
    k = self.self_attn_k_norm(k).to(v)
    attn = attention(q, k, v, mode="torch", attn_mask=attn_mask)
    x = x + apply_gate(self.self_attn_proj(attn), gate_msa)
    x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)
    return x


def IndividualTokenRefiner_forward(self, x: Tensor, c: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    self_attn_mask = None
    if mask is not None:
        batch_size = mask.shape[0]
        seq_len = mask.shape[1]
        mask = mask.to(x.device)
        self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
        self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
        self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
        L = self_attn_mask.size(-1)
        diag = torch.eye(L, dtype=torch.bool, device=self_attn_mask.device).view(1, 1, L, L)
        all_false = ~self_attn_mask.any(dim=-1, keepdim=False)
        all_false = all_false.unsqueeze(-1).expand(-1, -1, -1, L)
        self_attn_mask = torch.where(all_false, diag.expand_as(self_attn_mask), self_attn_mask)
    if self_attn_mask is not None:
        self_attn_mask = torch.where(self_attn_mask, torch.zeros_like(self_attn_mask, dtype=torch.float), torch.full_like(self_attn_mask, float("-inf"), dtype=torch.float))
    for block in self.blocks:
        x = block(x, c, self_attn_mask)
    return x


def SingleTokenRefiner_forward(self, x: Tensor, t: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    timestep_aware_representations = self.timestep_encoder(t)
    if mask is None:
        context_aware_representations = x.mean(dim=1)
    else:
        mask_float = mask.float().unsqueeze(-1)
        denom = mask_float.sum(dim=1).clamp_min(1e-06)
        context_aware_representations = (x * mask_float).sum(dim=1) / denom
    context_aware_representations = self.context_encoder(context_aware_representations).unsqueeze(1)
    c = timestep_aware_representations + context_aware_representations
    x = self.input_embedder(x)
    x = self.individual_token_refiner(x, c, mask)
    return x
