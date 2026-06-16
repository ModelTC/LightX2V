import os
import math
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import time
import mindiesd
from mindiesd.layers.flash_attn.attention_forward import attention_forward


class Rainfusion_blockwise(nn.Module):
    def __init__(
            self,
            grid_size: list,
            pool_size: int = 128,
            sparsity: float = 0.9,
            skip_timesteps: int = 0,
            txt_len: int = 0,
            txt_first: bool = False,
    ) -> None:
        """
        参数:
            grid_size (list): latents的THW网格大小。
            sparsity (float, optional): 稀疏度, 取值范围[0, 1]，默认为 0.50。
        """
        super().__init__()

        # Rainfusion_param
        self.grid_size = grid_size
        self.frame_num = self.grid_size[0]
        self.num_tokens_per_frame = self.grid_size[1] * self.grid_size[2]
        self.first_frame_len = self.num_tokens_per_frame

        self.pool_size = pool_size
        self.sparsity = sparsity
        self.skip_timesteps = skip_timesteps
        self.text_len = txt_len
        self.txt_first = txt_first

    @staticmethod
    def get_grid_size(latent_size, patch_size):
        t, h, w = latent_size[-3:]
        return [t // patch_size[0], h // patch_size[1], w // patch_size[2]]

    def avgpool(self, input_tensor, pool_size=128):  # BSND in,  BSND out
        batch, seqlen, headnum, dim = input_tensor.shape

        num_full_blocks = seqlen // pool_size
        tail_size = seqlen % pool_size

        if num_full_blocks > 0:
            full_blocks = input_tensor[:, :num_full_blocks * pool_size, :, :]
            full_blocks_reshaped = full_blocks.view(batch, num_full_blocks, pool_size, headnum, dim)
            full_pooled = full_blocks_reshaped.mean(dim=2)
        else:
            full_pooled = torch.empty(0, device=input_tensor.device)
        if tail_size > 0:
            tail_block = input_tensor[:, num_full_blocks * pool_size:, :, :]
            tail_reshaped = tail_block.view(batch, 1, tail_size, headnum, dim)
            tail_pooled = tail_reshaped.mean(dim=2)
        else:
            tail_pooled = torch.empty(0, device=input_tensor.device)

        if num_full_blocks > 0 and tail_size > 0:
            output_tensor = torch.cat([full_pooled, tail_pooled], dim=1)
        elif num_full_blocks > 0:
            output_tensor = full_pooled
        else:
            output_tensor = tail_pooled

        return output_tensor

    def get_mask_index(self, mask):
        B, N, S, _ = mask.shape
        device = mask.device

        # 1. 重塑维度 → (B*N)×S×S
        mask_reshaped = mask.reshape(-1, S, S)
        batch_size = mask_reshaped.shape[0]

        # 2. 生成行索引  标记False位置为S（大于所有有效索引）
        row_indices = torch.arange(S, device=device).expand(batch_size, S, -1)  # (B*N, S, S)
        sorted_vals = torch.where(mask_reshaped, row_indices, S)
        sorted_vals, _ = torch.sort(sorted_vals, dim=-1)
        valid_count = mask_reshaped.sum(dim=-1, keepdim=True)
        keep_mask = row_indices < valid_count
        result = torch.where(keep_mask, sorted_vals, -1)

        pos_matrix = result.reshape(B, N, S, S)
        return pos_matrix

    def get_blockwise_mask(self, score_matrix, sparsity):
        batch_size, num_heads, rows, cols = score_matrix.shape

        keep_len = math.ceil(cols * (1 - sparsity))
        topk_values, _ = torch.topk(score_matrix, k=keep_len, dim=-1)
        thresholds = topk_values[..., -1:]
        mask = score_matrix >= thresholds

        protect_len = (self.first_frame_len + self.text_len + self.pool_size - 1) // self.pool_size

        if protect_len > 0:
            mask[:, :, -protect_len:, :] = True
            mask[:, :, :, -protect_len:] = True

        selectIdx = self.get_mask_index(mask)
        selectIdx = selectIdx[0].transpose(0, 1)
        selectNumIdx = mask[0].transpose(0, 1).sum(dim=-1)
        return selectIdx, selectNumIdx

    def rearrange_with_remaining(self, tensor):  # BSND in ,  BSND out
        b, s, n, d = tensor.shape
        h = self.grid_size[1]
        w = self.grid_size[2]
        h_res_len, w_res_len = 0, 0
        first_frame_num = self.first_frame_len // h // w

        tensor_hwt = rearrange(tensor, 'b (f h w) n d -> (b n) f h w d', f=self.frame_num - first_frame_num, h=h, w=w)
        if h % 8 != 0:
            tensor_hwt, tensor_h_r = torch.split(tensor_hwt, h - (h % 8), dim=2)
            tensor_h_r = tensor_h_r.reshape(b * n, -1, d)
            h_res_len = tensor_h_r.shape[1]
        if w % 8 != 0:
            tensor_hwt, tensor_w_r = torch.split(tensor_hwt, w - (w % 8), dim=3)
            tensor_w_r = tensor_w_r.reshape(b * n, -1, d)
            w_res_len = tensor_w_r.shape[1]
        tensor_hwt = rearrange(tensor_hwt, 'b (fn fb) (hn hb) (wn wb) d -> b (fn hn wn fb hb wb) d',
                               fn=(self.frame_num - first_frame_num) // 2, fb=2, hb=8, wb=8, hn=h // 8, wn=w // 8)
        if h % 8 != 0:
            tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=1)
        if w % 8 != 0:
            tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=1)
        tensor_hwt = rearrange(tensor_hwt, '(b n) s d -> b s n d', b=b, n=n)
        return tensor_hwt, h_res_len, w_res_len

    def inv_rearrange_with_remaining(self, tensor, h_res_len, w_res_len):  # BSND in ,  BSND out
        b, s, n, d = tensor.shape
        h = self.grid_size[1]
        w = self.grid_size[2]
        h_sr, w_sr = h % 8, w % 8

        tensor = rearrange(tensor, 'b s n d->(b n) s d', b=b, n=n)
        tensor_hwt, tensor_h, tensor_w = torch.split(tensor, [s - h_res_len - w_res_len, h_res_len, w_res_len], dim=1)
        tensor_hwt = rearrange(tensor_hwt, 'b (fn hn wn fb hb wb) d -> b (fn fb) (hn hb) (wn wb) d',
                               fn=(self.frame_num - 1) // 2, fb=2, hb=8, wb=8, hn=h // 8, wn=w // 8)
        if w_res_len != 0:
            tensor_w = tensor_w.reshape(b * n, self.frame_num - 1, h - h_sr, w_sr, d)
            tensor_hwt = torch.cat((tensor_hwt, tensor_w), dim=3)
        if h_sr != 0:
            tensor_h = tensor_h.reshape(b * n, self.frame_num - 1, h_sr, w, d)
            tensor_hwt = torch.cat((tensor_hwt, tensor_h), dim=2)
        tensor_hwt = tensor_hwt.reshape(b * n, -1, d)
        tensor_hwt = rearrange(tensor_hwt, '(b n) s d -> b s n d', b=b, n=n)
        return tensor_hwt

    def do_tensor_rearrange_pooling(self, tensor):  # BSND in ,  BSND out
        b, s, n, d = tensor.shape
        if self.txt_first:
            tensor_t, tensor_f, tensor_i = torch.split(tensor, [self.text_len, self.first_frame_len,
                                                                s - self.text_len - self.first_frame_len], dim=1)
        else:
            tensor_f, tensor_i, tensor_t = torch.split(tensor,
                                                       [self.first_frame_len, s - self.text_len - self.first_frame_len,
                                                        self.text_len], dim=1)
        tensor_i_2, h_res_len, w_res_len = self.rearrange_with_remaining(tensor_i)
        tensor = torch.concat((tensor_i_2, tensor_f, tensor_t), dim=1)
        tensor_pool = self.avgpool(tensor, pool_size=128)
        return tensor, tensor_pool, h_res_len, w_res_len

    def do_tensor_inv_rearrange(self, tensor, h_res_len, w_res_len):
        b, s, n, d = tensor.shape
        tensor_i, tensor_f, tensor_t = torch.split(tensor,
                                                   [s - self.text_len - self.first_frame_len, self.first_frame_len,
                                                    self.text_len], dim=1)
        tensor_i = self.inv_rearrange_with_remaining(tensor_i, h_res_len, w_res_len)

        if self.txt_first:
            tensor = torch.concat((tensor_t, tensor_f, tensor_i), dim=1)
        else:
            tensor = torch.concat((tensor_f, tensor_i, tensor_t), dim=1)
        return tensor

    def do_tensor_pooling(self, tensor):
        tensor_t = tensor[:, :self.text_len, :, :]
        tensor_i = tensor[:, self.text_len:, :, :]

        tensor_i_pool = self.avgpool(tensor_i, pool_size=128)
        tensor_t_pool = self.avgpool(tensor_t, pool_size=128)

        tensor_pool = torch.concat((tensor_t_pool, tensor_i_pool), dim=1)
        return tensor_pool

    def forward(
            self,
            q,  # BSND
            k,
            v,
            t_b_idx,
            base_blockmask,
    ):
        t_idx = t_b_idx[0]

        if t_idx < self.skip_timesteps:
            base_blockmask = None
            x = attention_forward(q, k, v,
                                  opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
        else:
            batch, qSeqlen, numHeads, headDim = q.shape
            assert batch == 1, "Rainfusion_blockwise currently only supports batch size 1."
            _, kvSeqlen, _, _ = k.shape
            blockShapeX, blockShapeY = self.pool_size, self.pool_size
            scale = headDim ** -0.5

            blockShape = [blockShapeX, blockShapeY]
            actualSeqLengthsHost = [qSeqlen for _ in range(batch)]
            actualSeqLengthsKvHost = [kvSeqlen for _ in range(batch)]

            sparsity = self.sparsity

            h_res_len, w_res_len = 0, 0
            if base_blockmask is None:
                qkv = torch.cat((q, k, v), dim=0)
                qkv, qkv_pool, h_res_len, w_res_len = self.do_tensor_rearrange_pooling(qkv)
                q, k, v = torch.chunk(qkv, 3, dim=0)
                # qkv_pool = self.do_tensor_pooling(qkv)
                query_pool, key_pool, value_pool = torch.chunk(qkv_pool, 3, dim=0)

                attn_scores_head = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
                attn_scores_fake = torch.nn.functional.softmax(attn_scores_head, dim=-1)

                selectIdx, selectNumIdx = self.get_blockwise_mask(attn_scores_fake, sparsity)
                base_blockmask = [selectIdx, selectNumIdx]
            else:
                selectIdx = base_blockmask[0]
                selectNumIdx = base_blockmask[1]
                qkv = torch.cat((q, k, v), dim=0)
                qkv, qkv_pool, h_res_len, w_res_len = self.do_tensor_rearrange_pooling(qkv)
                q, k, v = torch.chunk(qkv, 3, dim=0)

            q_bnsd = q.transpose(1, 2)
            k_bnsd = k.transpose(1, 2)
            v_bnsd = v.transpose(1, 2)
            x = mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2.rain_fusion_attention(
                q_bnsd, k_bnsd, v_bnsd,
                scale=scale,
                head_num=numHeads,
                input_layout="BNSD",
                select_idx=selectIdx,
                select_num_idx=selectNumIdx,
                blockshape=blockShape,
                actual_seq_lengths=actualSeqLengthsHost,
                actual_seq_lengths_kv=actualSeqLengthsKvHost
            )

            x = x.transpose(1, 2).view(batch, qSeqlen, numHeads, headDim)

            x = self.do_tensor_inv_rearrange(x, h_res_len, w_res_len)
            base_blockmask = None

        return x, base_blockmask