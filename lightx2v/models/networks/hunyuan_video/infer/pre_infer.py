import math
from typing import Optional

import torch
from einops import rearrange

from lightx2v.utils.envs import *

from .flash_attn_no_pad import flash_attn_no_pad
from .module_io import HunyuanVideo15InferModuleOutput
from .posemb_layers import get_nd_rotary_pos_embed


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


@torch.compiler.disable
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    drop_rate: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Compute attention using flash_attn_no_pad.

    Args:
        q: Query tensor of shape [B, L, H, D]
        k: Key tensor of shape [B, L, H, D]
        v: Value tensor of shape [B, L, H, D]
        drop_rate: Dropout rate for attention weights.
        attn_mask: Optional attention mask of shape [B, L].
        causal: Whether to apply causal masking.

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """
    qkv = torch.stack([q, k, v], dim=2)
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()
    x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


class HunyuanVideo15PreInfer:
    def __init__(self, config):
        self.config = config
        self.patch_size = config["patch_size"]
        self.heads_num = config["heads_num"]
        self.frequency_embedding_size = 256
        self.max_period = 10000

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, weights, inputs):
        latents = self.scheduler.latents
        grid_sizes_t, grid_sizes_h, grid_sizes_w = latents.shape[2:]

        cond_latents_concat = self.scheduler.cond_latents_concat
        mask_concat = self.scheduler.mask_concat
        timesteps = self.scheduler.timesteps
        t = timesteps[self.scheduler.step_index]
        txt, text_mask = inputs["text_encoder_output"]["context"][0], inputs["text_encoder_output"]["context"][1]

        if self.scheduler.infer_condition:
            txt, text_mask = inputs["text_encoder_output"]["context"][0], inputs["text_encoder_output"]["context"][1]
        else:
            txt, text_mask = inputs["text_encoder_output"]["context_null"][0], inputs["text_encoder_output"]["context_null"][1]

        byt5_txt, byt5_text_mask = inputs["text_encoder_output"]["byt5_features"], inputs["text_encoder_output"]["byt5_masks"]
        image_encoder_output, image_encoder_mask = inputs["image_encoder_output"], inputs["image_encoder_mask"]
        txt = txt.to(torch.bfloat16)

        img = x = latent_model_input = torch.concat([latents, cond_latents_concat, mask_concat], dim=1)
        img = img.to(torch.bfloat16)

        t_expand = t.repeat(latent_model_input.shape[0])
        guidance_expand = None

        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
        img = weights.img_in.apply(img)
        img = img.flatten(2).transpose(1, 2)

        t_freq = self.timestep_embedding(t_expand, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
        vec = weights.time_in_0.apply(t_freq)
        vec = torch.nn.functional.silu(vec)
        vec = weights.time_in_2.apply(vec)

        t_freq = self.timestep_embedding(t_expand, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
        timestep_aware_representations = weights.txt_in_t_embedder_0.apply(t_freq)
        timestep_aware_representations = torch.nn.functional.silu(timestep_aware_representations)
        timestep_aware_representations = weights.txt_in_t_embedder_2.apply(timestep_aware_representations)

        mask_float = text_mask.float().unsqueeze(-1)
        context_aware_representations = (txt * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        context_aware_representations = context_aware_representations.to(torch.bfloat16)
        context_aware_representations = weights.txt_in_c_embedder_0.apply(context_aware_representations)
        context_aware_representations = torch.nn.functional.silu(context_aware_representations)
        context_aware_representations = weights.txt_in_c_embedder_2.apply(context_aware_representations)

        c = timestep_aware_representations + context_aware_representations
        out = weights.txt_in_input_embedder.apply(txt[0].to(torch.bfloat16))
        txt = self.run_individual_token_refiner(weights, out, text_mask, c)

        # TODO: 可以删除这段计算
        txt = txt.unsqueeze(0)
        txt = txt + weights.cond_type_embedding.apply(torch.zeros_like(txt[:, :, 0], device=txt.device, dtype=torch.long))
        byt5_txt = byt5_txt + weights.cond_type_embedding.apply(torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long))
        txt, text_mask = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True)

        image_encoder_output = image_encoder_output + weights.cond_type_embedding.apply(2 * torch.ones_like(image_encoder_output[:, :, 0], dtype=torch.long, device=torch.device("cuda")))
        txt, text_mask = self.reorder_txt_token(image_encoder_output, txt, image_encoder_mask, text_mask)

        return HunyuanVideo15InferModuleOutput(
            img=img,
            txt=txt,
            vec=torch.nn.functional.silu(vec),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            text_mask=text_mask,
            grid_sizes=(grid_sizes_t, grid_sizes_h, grid_sizes_w),
        )

    def run_individual_token_refiner(self, weights, out, mask, c):
        mask = mask.clone().bool()
        mask[:, 0] = True  # Prevent attention weights from becoming NaN
        for block in weights.individual_token_refiner:  # block num = 2
            gate_msa, gate_mlp = self.adaLN_modulation(block, c)
            norm_x = block.norm1.apply(out.unsqueeze(0)).squeeze(0)
            qkv = block.self_attn_qkv.apply(norm_x).unsqueeze(0)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
            attn = attention(q, k, v, attn_mask=mask).squeeze(0)
            out = out + apply_gate(block.self_attn_proj.apply(attn).unsqueeze(0), gate_msa).squeeze(0)
            tmp = block.mlp_fc1.apply(block.norm2.apply(out))
            tmp = torch.nn.functional.silu(tmp)
            tmp = block.mlp_fc2.apply(tmp)
            out = out + apply_gate(tmp.unsqueeze(0), gate_mlp).squeeze(0)
        return out

    def adaLN_modulation(self, weights, c):
        c = torch.nn.functional.silu(c)
        gate_msa, gate_mlp = weights.adaLN_modulation.apply(c).chunk(2, dim=1)
        return gate_msa, gate_mlp

    def get_rotary_pos_embed(self, rope_sizes):
        target_ndim = 3
        head_dim = self.config["hidden_size"] // self.config["heads_num"]
        rope_dim_list = self.config["rope_dim_list"]
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.config["rope_theta"],
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim (int): the dimension of the output.
            max_period (int): controls the minimum frequency of the embeddings.

        Returns:
            embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

        .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()

                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    # When using block mask with approximate computation, set pad to zero to reduce error
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat([byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], pad_byt5, pad_text], dim=0)
                else:
                    reorder_txt_i = torch.cat([byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], byt5_txt_i[~byt5_text_mask_i], txt_i[~text_mask_i]], dim=0)
                reorder_mask_i = torch.cat([byt5_text_mask_i[byt5_text_mask_i], text_mask_i[text_mask_i], byt5_text_mask_i[~byt5_text_mask_i], text_mask_i[~text_mask_i]], dim=0)

                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)

            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int64)

        return reorder_txt, reorder_mask
