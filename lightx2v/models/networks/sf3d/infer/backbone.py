# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.sf3d.weights.backbone import F, _registered_sdpa, nn, torch


def GEGLU_gelu(self, gate: torch.Tensor) -> torch.Tensor:
    if gate.device.type != "mps":
        return F.gelu(gate)
    return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)


def GEGLU_forward(self, hidden_states, scale: float = 1.0):
    args = ()
    hidden_states, gate = self.proj(hidden_states, *args).chunk(2, dim=-1)
    return hidden_states * self.gelu(gate)


def CrossAttention_forward(self, x_q, x_kv):
    B, N_q, C = x_q.shape
    B, N_kv, _ = x_kv.shape
    q = self.wq(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads)
    k = self.wk(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads)
    v = self.wv(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads)
    x = _registered_sdpa(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), attn_mask=None, dropout_p=self.attn_drop, scale=self.scale).permute(0, 2, 1, 3)
    x = x.reshape(B, N_q, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def FeedForward_forward(self, x: torch.Tensor) -> torch.Tensor:
    for module in self.net:
        x = module(x)
    return x


def BasicBlock_forward(self, z, x):
    z_norm = self.norm1(z)
    z = z + self.attn1(z_norm, z_norm)
    z_norm = self.norm2(z)
    z = z + self.attn2(z_norm, x if x is not None else z_norm)
    z_norm = self.norm3(z)
    z = z + self.ff(z_norm)
    return z


def SingleStreamTransformer_forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
    residual = hidden_states
    hidden_states = self.norm(hidden_states)
    hidden_states = hidden_states.permute(0, 2, 1)
    hidden_states = self.proj_in(hidden_states)
    for block in self.transformer_blocks:
        hidden_states = block(hidden_states, encoder_hidden_states)
    hidden_states = self.proj_out(hidden_states).permute(0, 2, 1).contiguous()
    hidden_states = hidden_states + residual
    return hidden_states


def FuseBlock_forward(self, z, x):
    z = z + self.attn(self.norm_z1(z), self.norm_x(x) if self.norm_x_input else x)
    z = z + self.ff(self.norm_z2(z))
    return z


def TriplaneAttention_forward(self, x):
    B, N, C = x.shape
    q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads)
    k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads)
    v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads)
    assert N == self.resolution**2 * 3
    attn_bias = self.attn_mask.to(q).unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1) if not self.full_attention else None
    x = _registered_sdpa(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), attn_mask=attn_bias, dropout_p=self.attn_drop, scale=self.scale).permute(0, 2, 1, 3)
    x = x.reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def TwoStreamBlock_forward(self, latent, input, cross_input):
    latent = self.fuse_block_in(latent, input)
    for block in self.transformer_block:
        latent = block(latent, cross_input)
    input = self.fuse_block_out(input, latent)
    return (latent, input)


def TwoStreamInterleaveTransformer_forward(self, hidden_states, encoder_hidden_states, **kwargs):
    if isinstance(self.norm_triplane, nn.GroupNorm):
        triplane_tokens = self.norm_triplane(hidden_states)
        triplane_tokens = triplane_tokens.permute(0, 2, 1)
    elif isinstance(self.norm_triplane, nn.LayerNorm):
        triplane_tokens = self.norm_triplane(hidden_states.permute(0, 2, 1))
    else:
        raise TypeError("Unknown normalization layer")
    triplane_tokens = self.proj_triplane(triplane_tokens)
    if self.mix_latent:
        image_tokens = self.norm_image(encoder_hidden_states)
        image_tokens = self.proj_image(image_tokens)
    init_latents = self.latent_init.expand(hidden_states.shape[0], -1, -1)
    init_latents = self.norm_latent(init_latents)
    init_latents = self.proj_latent(init_latents)
    if self.mix_latent:
        latent_tokens = torch.cat([image_tokens, init_latents], dim=1)
    else:
        latent_tokens = init_latents
    for block in self.main_blocks:
        latent_tokens, triplane_tokens = block(latent_tokens, triplane_tokens, encoder_hidden_states)
    triplane_tokens = self.proj_out(triplane_tokens).permute(0, 2, 1).contiguous()
    triplane_tokens = triplane_tokens + hidden_states
    return triplane_tokens
