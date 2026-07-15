import math

import torch
import torch.distributed as dist
from einops import rearrange
from torch.nn import functional as F

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE, PLATFORM

from .module_io import HunyuanVideo15InferModuleOutput
from .posemb_layers import get_nd_rotary_pos_embed

try:
    from sgl_kernel.elementwise import timestep_embedding as timestep_embedding_cuda

    TIMESTEP_EMBEDDING_CUDA_AVAILABLE = PLATFORM == "cuda"
except ImportError:
    TIMESTEP_EMBEDDING_CUDA_AVAILABLE = False


def apply_gate(x, gate=None, tanh=False):
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    return x * gate.unsqueeze(1)


class HunyuanVideo15PreInfer:
    def __init__(self, config):
        self.config = config
        self.patch_size = config["patch_size"]
        self.heads_num = config["heads_num"]
        self.frequency_embedding_size = 256
        self.max_period = 10000
        self.cos_sin = None
        self.grid_sizes = (0, 0, 0)  # (t, h, w)
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def prepare_cos_sin(self, rope_sizes):
        target_ndim = 3
        head_dim = self.config["hidden_size"] // self.config["heads_num"]
        rope_dim_list = self.config["rope_dim_list"]
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(rope_dim_list, rope_sizes, theta=self.config["rope_theta"], use_real=True, theta_rescale_factor=1, device=AI_DEVICE)
        cos_half = freqs_cos[:, ::2].contiguous()
        sin_half = freqs_sin[:, ::2].contiguous()
        cos_sin = torch.cat([cos_half, sin_half], dim=-1)
        if self.seq_p_group is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            cur_rank = dist.get_rank(self.seq_p_group)
            seqlen = cos_sin.shape[0]
            padding_size = (world_size - (seqlen % world_size)) % world_size
            if padding_size > 0:
                cos_sin = F.pad(cos_sin, (0, 0, 0, padding_size))
            cos_sin = torch.chunk(cos_sin, world_size, dim=0)[cur_rank]
        return cos_sin

    @torch.no_grad()
    def infer(self, weights, inputs):
        latents = self.scheduler.latents
        grid_sizes_t, grid_sizes_h, grid_sizes_w = latents.shape[2:]

        timesteps = self.scheduler.timesteps
        t = timesteps[self.scheduler.step_index]

        if self.scheduler.infer_condition:
            txt, text_mask = inputs["text_encoder_output"]["context"][0], inputs["text_encoder_output"]["context"][1]
        else:
            txt, text_mask = inputs["text_encoder_output"]["context_null"][0], inputs["text_encoder_output"]["context_null"][1]

        byt5_txt, byt5_text_mask = inputs["text_encoder_output"]["byt5_features"], inputs["text_encoder_output"]["byt5_masks"]
        siglip_output, siglip_mask = inputs["image_encoder_output"]["siglip_output"], inputs["image_encoder_output"]["siglip_mask"]
        txt = txt.to(torch.bfloat16)

        if self.config["is_sr_running"]:
            if t < 1000 * self.scheduler.noise_scale:
                condition = self.scheduler.zero_condition
            else:
                condition = self.scheduler.condition

            img = x = latent_model_input = torch.concat([latents, condition], dim=1)
        else:
            cond_latents_concat = self.scheduler.cond_latents_concat
            mask_concat = self.scheduler.mask_concat
            img = x = latent_model_input = torch.concat([latents, cond_latents_concat, mask_concat], dim=1)

        img = img.to(torch.bfloat16)

        t_expand = t.repeat(latent_model_input.shape[0])
        guidance_expand = None

        img = weights.img_in.apply(img)
        img = img.flatten(2).transpose(1, 2)

        t_freq = self.timestep_embedding(t_expand, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
        vec = weights.time_in_0.apply(t_freq)
        vec = torch.nn.functional.silu(vec)
        vec = weights.time_in_2.apply(vec)

        if self.config["is_sr_running"]:
            use_meanflow = self.config.get("video_super_resolution", {}).get("use_meanflow", False)
            if use_meanflow:
                if self.scheduler.step_index == len(timesteps) - 1:
                    timesteps_r = torch.tensor([0.0], device=latent_model_input.device)
                else:
                    timesteps_r = timesteps[self.scheduler.step_index + 1]
                timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
            else:
                timesteps_r = None

            if timesteps_r is not None:
                t_freq = self.timestep_embedding(timesteps_r, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
                vec_res = weights.time_r_in_0.apply(t_freq)
                vec_res = torch.nn.functional.silu(vec_res)
                vec_res = weights.time_r_in_2.apply(vec_res)
                vec = vec + vec_res

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

        siglip_output = siglip_output + weights.cond_type_embedding.apply(2 * torch.ones_like(siglip_output[:, :, 0], dtype=torch.long, device=AI_DEVICE))
        txt, text_mask = self.reorder_txt_token(siglip_output, txt, siglip_mask, text_mask)
        txt = txt[:, : text_mask.sum(), :]

        grid_sizes = (grid_sizes_t, grid_sizes_h, grid_sizes_w)

        if self.cos_sin is None or self.grid_sizes != grid_sizes:
            self.grid_sizes = grid_sizes
            self.cos_sin = self.prepare_cos_sin(grid_sizes)

        return HunyuanVideo15InferModuleOutput(
            img=img.contiguous(),
            txt=txt.contiguous(),
            vec=torch.nn.functional.silu(vec),
            grid_sizes=grid_sizes,
            cos_sin=self.cos_sin,
        )

    def _infer_token_refiner_attn(self, attention_module, q, k, v, mask):
        if mask is None:
            mask = torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
        else:
            mask = mask.to(device=q.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and q.shape[0] > 1:
                mask = mask.expand(q.shape[0], -1)

        batch, seqlen, heads, head_dim = q.shape
        out = torch.zeros(batch, seqlen, heads * head_dim, dtype=q.dtype, device=q.device)
        for batch_idx in range(batch):
            valid_mask = mask[batch_idx]
            valid_len = int(valid_mask.sum().item())
            if valid_len == 0:
                continue
            cu_seqlens = torch.tensor([0, valid_len], dtype=torch.int32, device=q.device)
            attn = attention_module.apply(
                q=q[batch_idx, valid_mask].contiguous(),
                k=k[batch_idx, valid_mask].contiguous(),
                v=v[batch_idx, valid_mask].contiguous(),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=valid_len,
                max_seqlen_kv=valid_len,
            )
            out[batch_idx, valid_mask] = attn.reshape(valid_len, -1)
        return out

    def run_individual_token_refiner(self, weights, out, mask, c):
        mask = mask.clone().bool()
        mask[:, 0] = True  # Prevent attention weights from becoming NaN
        for block in weights.individual_token_refiner:  # block num = 2
            gate_msa, gate_mlp = self.adaLN_modulation(block, c)
            norm_x = block.norm1.apply(out.unsqueeze(0)).squeeze(0)
            qkv = block.self_attn_qkv.apply(norm_x).unsqueeze(0)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
            attn = self._infer_token_refiner_attn(block.self_attention, q, k, v, mask).squeeze(0)
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
        if TIMESTEP_EMBEDDING_CUDA_AVAILABLE:
            return timestep_embedding_cuda(
                t,
                dim,
                flip_sin_to_cos=True,
                max_period=max_period,
            )

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
