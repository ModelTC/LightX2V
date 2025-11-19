from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer

from .module_io import HunyuanVideo15ImgBranchOutput, HunyuanVideo15TxtBranchOutput
from .triton_ops import fuse_scale_shift_kernel, norm_infer


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x, gate=None, tanh=False):
    """AI is creating summary for apply_gate

    Args:
        x (torch.Tensor): input tensor.
        gate (torch.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        torch.Tensor: the output tensor after apply gate.
    """
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    else:
        return x * gate.unsqueeze(1)


def apply_hunyuan_rope_with_flashinfer(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, H, D = xq.shape

    query = xq.reshape(B * L, H * D).contiguous()
    key = xk.reshape(B * L, H * D).contiguous()

    positions = torch.arange(B * L, device=xq.device, dtype=torch.long)

    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=query,
        key=key,
        head_size=D,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
    )

    xq_out = query.view(B, L, H, D)
    xk_out = key.view(B, L, H, D)
    return xq_out, xk_out


def apply_hunyuan_rope_with_torch(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, H, D = xq.shape

    cos = cos_sin_cache[:, : D // 2]
    sin = cos_sin_cache[:, D // 2 :]

    def _apply_rope(x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(B * L, H, D)
        x1 = x_flat[..., ::2]
        x2 = x_flat[..., 1::2]

        cos_ = cos.unsqueeze(1)
        sin_ = sin.unsqueeze(1)

        o1 = x1.float() * cos_ - x2.float() * sin_
        o2 = x2.float() * cos_ + x1.float() * sin_

        out = torch.empty_like(x_flat)
        out[..., ::2] = o1
        out[..., 1::2] = o2
        return out.view(B, L, H, D)

    xq_out = _apply_rope(xq)
    xk_out = _apply_rope(xk)
    return xq_out, xk_out


class HunyuanVideo15TransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.double_blocks_num = config["mm_double_blocks_depth"]
        self.heads_num = config["heads_num"]
        self.infer_func = self.infer_without_offload

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.scheduler.transformer_infer = self

    @torch.no_grad()
    def infer(self, weights, infer_module_out):
        self.infer_func(weights, infer_module_out)
        x = self.infer_final_layer(weights, infer_module_out)
        return x

    @torch.no_grad()
    def infer_without_offload(self, weights, infer_module_out):
        for i in range(self.double_blocks_num):
            infer_module_out.img, infer_module_out.txt = self.infer_double_block(weights.double_blocks[i], infer_module_out)

    def infer_final_layer(self, weights, infer_module_out):
        x = torch.cat((infer_module_out.img, infer_module_out.txt), 1)
        img = x[:, : infer_module_out.img.shape[1], ...]
        shift, scale = weights.final_layer.adaLN_modulation.apply(infer_module_out.vec).chunk(2, dim=1)
        if self.config.get("modulate_type", "triton") == "triton":
            img = fuse_scale_shift_kernel(norm_infer(img.squeeze(0), None, None, 1e-6).unsqueeze(0), scale, shift).squeeze(0)
        else:
            img = modulate(norm_infer(img.squeeze(0), None, None, 1e-6), scale=scale, shift=shift).squeeze(0)
        img = weights.final_layer.linear.apply(img)
        return img.unsqueeze(0)

    @torch.no_grad()
    def infer_double_block(self, weights, infer_module_out):
        img_q, img_k, img_v, img_branch_out = self._infer_img_branch_before_attn(weights, infer_module_out)
        txt_q, txt_k, txt_v, txt_branch_out = self._infer_txt_branch_before_attn(weights, infer_module_out)
        img_attn, txt_attn = self._infer_attn(weights, img_q, img_k, img_v, txt_q, txt_k, txt_v)
        img = self._infer_img_branch_after_attn(weights, img_attn, infer_module_out.img, img_branch_out)
        txt = self._infer_txt_branch_after_attn(weights, txt_attn, infer_module_out.txt, txt_branch_out)
        return img, txt

    @torch.no_grad()
    def _infer_img_branch_before_attn(self, weights, infer_module_out):
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = weights.img_branch.img_mod.apply(infer_module_out.vec).chunk(6, dim=-1)
        img_modulated = norm_infer(infer_module_out.img.squeeze(0), None, None, 1e-6)
        if self.config.get("modulate_type", "triton") == "triton":
            img_modulated = fuse_scale_shift_kernel(img_modulated.unsqueeze(0), img_mod1_scale, img_mod1_shift).squeeze(0)
        else:
            img_modulated = modulate(img_modulated, scale=img_mod1_scale, shift=img_mod1_shift).squeeze(0)
        img_q = weights.img_branch.img_attn_q.apply(img_modulated)
        img_k = weights.img_branch.img_attn_k.apply(img_modulated)
        img_v = weights.img_branch.img_attn_v.apply(img_modulated)
        img_q = rearrange(img_q, "L (H D) -> L H D", H=self.heads_num)
        img_k = rearrange(img_k, "L (H D) -> L H D", H=self.heads_num)
        img_v = rearrange(img_v, "L (H D) -> L H D", H=self.heads_num)
        img_q = weights.img_branch.img_attn_q_norm.apply(img_q)
        img_k = weights.img_branch.img_attn_k_norm.apply(img_k)
        if self.config.get("rope_type", "flashinfer") == "flashinfer":
            img_q, img_k = apply_hunyuan_rope_with_flashinfer(img_q.unsqueeze(0), img_k.unsqueeze(0), cos_sin_cache=self.scheduler.cos_sin)
        else:
            img_q, img_k = apply_hunyuan_rope_with_torch(img_q.unsqueeze(0), img_k.unsqueeze(0), cos_sin_cache=self.scheduler.cos_sin)
        return (
            img_q,
            img_k,
            img_v.unsqueeze(0),
            HunyuanVideo15ImgBranchOutput(
                img_mod1_gate=img_mod1_gate,
                img_mod2_shift=img_mod2_shift,
                img_mod2_scale=img_mod2_scale,
                img_mod2_gate=img_mod2_gate,
            ),
        )

    @torch.no_grad()
    def _infer_txt_branch_before_attn(self, weights, infer_module_out):
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = weights.txt_branch.txt_mod.apply(infer_module_out.vec).chunk(6, dim=-1)
        txt_modulated = norm_infer(infer_module_out.txt.squeeze(0), None, None, 1e-6)
        if self.config.get("modulate_type", "triton") == "triton":
            txt_modulated = fuse_scale_shift_kernel(txt_modulated.unsqueeze(0), txt_mod1_scale, txt_mod1_shift).squeeze(0)
        else:
            txt_modulated = modulate(txt_modulated, scale=txt_mod1_scale, shift=txt_mod1_shift).squeeze(0)
        txt_q = weights.txt_branch.txt_attn_q.apply(txt_modulated)
        txt_k = weights.txt_branch.txt_attn_k.apply(txt_modulated)
        txt_v = weights.txt_branch.txt_attn_v.apply(txt_modulated)
        txt_q = rearrange(txt_q, "L (H D) -> L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "L (H D) -> L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "L (H D) -> L H D", H=self.heads_num)
        txt_q = weights.txt_branch.txt_attn_q_norm.apply(txt_q).to(txt_v)
        txt_k = weights.txt_branch.txt_attn_k_norm.apply(txt_k).to(txt_v)
        return (
            txt_q.unsqueeze(0),
            txt_k.unsqueeze(0),
            txt_v.unsqueeze(0),
            HunyuanVideo15TxtBranchOutput(
                txt_mod1_gate=txt_mod1_gate,
                txt_mod2_shift=txt_mod2_shift,
                txt_mod2_scale=txt_mod2_scale,
                txt_mod2_gate=txt_mod2_gate,
            ),
        )

    @torch.no_grad()
    def _infer_attn(self, weights, img_q, img_k, img_v, txt_q, txt_k, txt_v):
        img_seqlen = img_q.shape[1]
        query = torch.cat([img_q, txt_q], dim=1)
        key = torch.cat([img_k, txt_k], dim=1)
        value = torch.cat([img_v, txt_v], dim=1)
        seqlen = query.shape[1]
        cu_seqlens_qkv = torch.tensor([0, seqlen], dtype=torch.int32, device="cpu").to("cuda", non_blocking=True)

        attn_out = weights.self_attention.apply(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_qkv,
            cu_seqlens_kv=cu_seqlens_qkv,
            max_seqlen_q=seqlen,
            max_seqlen_kv=seqlen,
        )
        img_attn, txt_attn = attn_out[:img_seqlen], attn_out[img_seqlen:]
        return img_attn, txt_attn

    @torch.no_grad()
    def _infer_img_branch_after_attn(self, weights, img_attn, img, img_branch_out):
        img = img + apply_gate(weights.img_branch.img_attn_proj.apply(img_attn).unsqueeze(0), gate=img_branch_out.img_mod1_gate)
        if self.config.get("modulate_type", "triton") == "triton":
            out = weights.img_branch.img_mlp_fc1.apply(
                fuse_scale_shift_kernel(norm_infer(img.squeeze(0), None, None, 1e-6).unsqueeze(0), img_branch_out.img_mod2_scale, img_branch_out.img_mod2_shift).squeeze(0)
            )
        else:
            out = weights.img_branch.img_mlp_fc1.apply(modulate(norm_infer(img.squeeze(0), None, None, 1e-6), scale=img_branch_out.img_mod2_scale, shift=img_branch_out.img_mod2_shift).squeeze(0))
        out = weights.img_branch.img_mlp_fc2.apply(F.gelu(out, approximate="tanh"))
        img = img + apply_gate(out.unsqueeze(0), gate=img_branch_out.img_mod2_gate)
        return img

    @torch.no_grad()
    def _infer_txt_branch_after_attn(self, weights, txt_attn, txt, txt_branch_out):
        txt = txt + apply_gate(weights.txt_branch.txt_attn_proj.apply(txt_attn).unsqueeze(0), gate=txt_branch_out.txt_mod1_gate)
        if self.config.get("modulate_type", "triton") == "triton":
            out = weights.txt_branch.txt_mlp_fc1.apply(
                fuse_scale_shift_kernel(norm_infer(txt.squeeze(0), None, None, 1e-6).unsqueeze(0), txt_branch_out.txt_mod2_scale, txt_branch_out.txt_mod2_shift).squeeze(0)
            )
        else:
            out = weights.txt_branch.txt_mlp_fc1.apply(modulate(norm_infer(txt.squeeze(0), None, None, 1e-6), scale=txt_branch_out.txt_mod2_scale, shift=txt_branch_out.txt_mod2_shift).squeeze(0))
        out = weights.txt_branch.txt_mlp_fc2.apply(F.gelu(out, approximate="tanh"))
        txt = txt + apply_gate(out.unsqueeze(0), gate=txt_branch_out.txt_mod2_gate)
        return txt
