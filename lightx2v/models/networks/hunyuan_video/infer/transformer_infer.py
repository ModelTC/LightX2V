import torch
import torch.nn.functional as F
from einops import rearrange

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer

from .flash_attn_no_pad import flash_attn_no_pad_v3
from .module_io import HunyuanVideo15ImgBranchOutput, HunyuanVideo15TxtBranchOutput
from .posemb_layers import apply_rotary_emb


def modulate(x, shift=None, scale=None):
    """modulate by shift and scale

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
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


class HunyuanVideo15TransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.double_blocks_num = config["mm_double_blocks_depth"]
        self.heads_num = config["heads_num"]

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, weights, infer_module_out):
        for i in range(self.double_blocks_num):
            infer_module_out.img, infer_module_out.txt = self.infer_double_block(weights.double_blocks[i], infer_module_out)
        x = self.infer_final_layer(weights, infer_module_out)
        return x

    def infer_final_layer(self, weights, infer_module_out):
        x = torch.cat((infer_module_out.img, infer_module_out.txt), 1)
        img = x[:, : infer_module_out.img.shape[1], ...]
        shift, scale = weights.final_layer.adaLN_modulation.apply(infer_module_out.vec).chunk(2, dim=1)
        img = modulate(weights.final_layer.norm_final.apply(img), shift=shift, scale=scale).squeeze(0)
        img = weights.final_layer.linear.apply(img)
        return img.unsqueeze(0)

    @torch.no_grad()
    def infer_double_block(self, weights, infer_module_out):
        img_q, img_k, img_v, img_branch_out = self._infer_img_branch_before_attn(weights, infer_module_out)
        txt_q, txt_k, txt_v, txt_branch_out = self._infer_txt_branch_before_attn(weights, infer_module_out)
        img_attn, txt_attn = self._infer_attn(img_q, img_k, img_v, txt_q, txt_k, txt_v, infer_module_out.text_mask)
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
        img_modulated = weights.img_branch.img_norm1.apply(infer_module_out.img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        img_modulated = img_modulated.squeeze(0)
        img_q = weights.img_branch.img_attn_q.apply(img_modulated)
        img_k = weights.img_branch.img_attn_k.apply(img_modulated)
        img_v = weights.img_branch.img_attn_v.apply(img_modulated)
        img_q = rearrange(img_q, "L (H D) -> L H D", H=self.heads_num)
        img_k = rearrange(img_k, "L (H D) -> L H D", H=self.heads_num)
        img_v = rearrange(img_v, "L (H D) -> L H D", H=self.heads_num)
        img_q = weights.img_branch.img_attn_q_norm.apply(img_q).to(img_v)
        img_k = weights.img_branch.img_attn_k_norm.apply(img_k).to(img_v)

        img_q, img_k = apply_rotary_emb(img_q.unsqueeze(0), img_k.unsqueeze(0), (infer_module_out.freqs_cos, infer_module_out.freqs_sin), head_first=False)
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
        txt_modulated = weights.txt_branch.txt_norm1.apply(infer_module_out.txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_modulated = txt_modulated.squeeze(0)
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
    def _infer_attn(self, img_q, img_k, img_v, txt_q, txt_k, txt_v, text_mask):
        # Attention
        sequence_length = img_q.size(1)
        query = torch.cat([img_q, txt_q], dim=1)
        key = torch.cat([img_k, txt_k], dim=1)
        value = torch.cat([img_v, txt_v], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad_v3(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)
        b, s, a, d = hidden_states.shape
        hidden_states = hidden_states.reshape(b, s, -1)
        img_attn, txt_attn = hidden_states[:, : img_q.shape[1]].contiguous(), hidden_states[:, img_q.shape[1] :].contiguous()
        return img_attn, txt_attn

    @torch.no_grad()
    def _infer_img_branch_after_attn(self, weights, img_attn, img, img_branch_out):
        img = img + apply_gate(weights.img_branch.img_attn_proj.apply(img_attn.squeeze(0)).unsqueeze(0), gate=img_branch_out.img_mod1_gate)
        out = weights.img_branch.img_mlp_fc1.apply(modulate(weights.img_branch.img_norm2.apply(img), shift=img_branch_out.img_mod2_shift, scale=img_branch_out.img_mod2_scale).squeeze(0))
        out = weights.img_branch.img_mlp_fc2.apply(F.gelu(out, approximate="tanh"))
        img = img + apply_gate(out.unsqueeze(0), gate=img_branch_out.img_mod2_gate)
        return img

    @torch.no_grad()
    def _infer_txt_branch_after_attn(self, weights, txt_attn, txt, txt_branch_out):
        txt = txt + apply_gate(weights.txt_branch.txt_attn_proj.apply(txt_attn.squeeze(0)).unsqueeze(0), gate=txt_branch_out.txt_mod1_gate)
        out = weights.txt_branch.txt_mlp_fc1.apply(modulate(weights.txt_branch.txt_norm2.apply(txt), shift=txt_branch_out.txt_mod2_shift, scale=txt_branch_out.txt_mod2_scale).squeeze(0))
        out = weights.txt_branch.txt_mlp_fc2.apply(F.gelu(out, approximate="tanh"))
        txt = txt + apply_gate(out.unsqueeze(0), gate=txt_branch_out.txt_mod2_gate)
        return txt
