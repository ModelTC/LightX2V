import math

import torch
from einops import rearrange

try:
    import mindiesd
    from mindiesd.layers.flash_attn.attention_forward import attention_forward
    _HAS_MINDIESD = True
except ImportError:
    mindiesd = None
    attention_forward = None
    _HAS_MINDIESD = False

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

try:
    import torch_npu
except ImportError:
    torch_npu = None

"""
RainFusion is a sparse attention acceleration algorithm.
RainFusion requires MindIE-SD to be installed. Installation guide: https://gitcode.com/Ascend/MindIE-SD/blob/master/docs/zh/installation.md
"""
@PLATFORM_ATTN_WEIGHT_REGISTER("npu_rainfusion_attn")
class NpuRainfusionAttnWeight(AttnWeightTemplate):
    def __init__(self, grid_size=None, pool_size=128, sparsity=0.8, skip_timesteps=-1, txt_len=0, txt_first=False):
        """
        参数:
            grid_size (list): latents的THW网格大小。
            sparsity (float, optional): 稀疏度, 取值范围[0, 1]，默认为 0.50。
        """
        self.config = {}
        assert torch_npu is not None, "torch_npu is not installed."
        assert _HAS_MINDIESD, "mindiesd is not installed. RainFusion requires MindIE-SD."

        self.pool_size = pool_size
        self.sparsity = sparsity
        self.skip_timesteps = skip_timesteps
        self.text_len = txt_len
        self.txt_first = txt_first

        if grid_size is not None:
            self.update_grid_size(grid_size)

    def update_grid_size(self, grid_size):
        self.grid_size = grid_size
        self.frame_num = self.grid_size[0]
        self.num_tokens_per_frame = self.grid_size[1] * self.grid_size[2]
        self.first_frame_len = self.num_tokens_per_frame

    @staticmethod
    def get_grid_size(latent_size, patch_size):
        t, h, w = latent_size[-3:]
        return [t // patch_size[0], h // patch_size[1], w // patch_size[2]]

    def avgpool(self, input_tensor, pool_size=128):
        batch, seqlen, headnum, dim = input_tensor.shape

        num_full_blocks = seqlen // pool_size
        tail_size = seqlen % pool_size

        pooled_tensors = []
        if num_full_blocks > 0:
            full_blocks = input_tensor[:, :num_full_blocks * pool_size, :, :]
            full_blocks_reshaped = full_blocks.reshape(batch, num_full_blocks, pool_size, headnum, dim)
            pooled_tensors.append(full_blocks_reshaped.mean(dim=2))
        if tail_size > 0:
            tail_block = input_tensor[:, num_full_blocks * pool_size:, :, :]
            tail_reshaped = tail_block.reshape(batch, 1, tail_size, headnum, dim)
            pooled_tensors.append(tail_reshaped.mean(dim=2))

        return torch.cat(pooled_tensors, dim=1)

    def get_mask_index(self, mask):
        B, N, S, _ = mask.shape
        device = mask.device

        mask_reshaped = mask.reshape(-1, S, S)
        batch_size = mask_reshaped.shape[0]

        row_indices = torch.arange(S, device=device).view(1, 1, S).expand(batch_size, S, S)
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
        keep_len = max(1, min(keep_len, cols))
        topk_values, _ = torch.topk(score_matrix, k=keep_len, dim=-1)
        thresholds = topk_values[..., -1:]
        mask = score_matrix >= thresholds

        total_len = self.frame_num * self.num_tokens_per_frame + self.text_len
        protected_start_idx = total_len - self.first_frame_len - self.text_len
        protected_start_block = protected_start_idx // self.pool_size
        protect_len = cols - protected_start_block

        if protect_len > 0:
            mask[:, :, -protect_len:, :] = True
            mask[:, :, :, -protect_len:] = True

        selectIdx = self.get_mask_index(mask)
        selectIdx = selectIdx[0].transpose(0, 1)
        selectNumIdx = mask[0].transpose(0, 1).sum(dim=-1)
        return selectIdx, selectNumIdx

    def rearrange_with_remaining(self, tensor):
        b, s, n, d = tensor.shape
        h = self.grid_size[1]
        w = self.grid_size[2]
        h_res_len, w_res_len = 0, 0
        first_frame_num = self.first_frame_len // h // w

        tensor_hwt = rearrange(tensor, "b (f h w) n d -> (b n) f h w d", f=self.frame_num - first_frame_num, h=h, w=w)
        if h % 8 != 0:
            tensor_hwt, tensor_h_r = torch.split(tensor_hwt, [h - (h % 8), h % 8], dim=2)
            tensor_h_r = tensor_h_r.reshape(b * n, -1, d)
            h_res_len = tensor_h_r.shape[1]
        if w % 8 != 0:
            tensor_hwt, tensor_w_r = torch.split(tensor_hwt, [w - (w % 8), w % 8], dim=3)
            tensor_w_r = tensor_w_r.reshape(b * n, -1, d)
            w_res_len = tensor_w_r.shape[1]
        remaining_frames = self.frame_num - first_frame_num
        if remaining_frames % 2 != 0:
            raise ValueError(f"The number of remaining frames ({remaining_frames}) must be even to be rearranged with block size 2.")
        tensor_hwt = rearrange(tensor_hwt, "bn (fn fb) (hn hb) (wn wb) d -> bn (fn hn wn fb hb wb) d", fn=remaining_frames // 2, fb=2, hb=8, wb=8, hn=h // 8, wn=w // 8)
        if h % 8 != 0:
            tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=1)
        if w % 8 != 0:
            tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=1)
        tensor_hwt = rearrange(tensor_hwt, "(b n) s d -> b s n d", b=b, n=n)
        return tensor_hwt, h_res_len, w_res_len

    def inv_rearrange_with_remaining(self, tensor, h_res_len, w_res_len):
        b, s, n, d = tensor.shape
        h = self.grid_size[1]
        w = self.grid_size[2]
        h_sr, w_sr = h % 8, w % 8
        first_frame_num = self.first_frame_len // (h * w)
        remaining_frames = self.frame_num - first_frame_num

        tensor = rearrange(tensor, "b s n d->(b n) s d", b=b, n=n)
        tensor_hwt, tensor_h, tensor_w = torch.split(tensor, [s - h_res_len - w_res_len, h_res_len, w_res_len], dim=1)
        tensor_hwt = rearrange(tensor_hwt, "bn (fn hn wn fb hb wb) d -> bn (fn fb) (hn hb) (wn wb) d", fn=remaining_frames // 2, fb=2, hb=8, wb=8, hn=h // 8, wn=w // 8)
        if w_res_len != 0:
            tensor_w = tensor_w.reshape(b * n, remaining_frames, h - h_sr, w_sr, d)
            tensor_hwt = torch.cat((tensor_hwt, tensor_w), dim=3)
        if h_sr != 0:
            tensor_h = tensor_h.reshape(b * n, remaining_frames, h_sr, w, d)
            tensor_hwt = torch.cat((tensor_hwt, tensor_h), dim=2)
        tensor_hwt = tensor_hwt.reshape(b * n, -1, d)
        tensor_hwt = rearrange(tensor_hwt, "(b n) s d -> b s n d", b=b, n=n)
        return tensor_hwt

    def do_tensor_rearrange_pooling(self, tensor):
        b, s, n, d = tensor.shape
        if s < self.text_len + self.first_frame_len:
            raise ValueError(f"Sequence length {s} is too small for text_len {self.text_len} and first_frame_len {self.first_frame_len}")
        if self.txt_first:
            tensor_t, tensor_f, tensor_i = torch.split(tensor, [self.text_len, self.first_frame_len, s - self.text_len - self.first_frame_len], dim=1)
        else:
            tensor_f, tensor_i, tensor_t = torch.split(tensor, [self.first_frame_len, s - self.text_len - self.first_frame_len, self.text_len], dim=1)
        tensor_i_2, h_res_len, w_res_len = self.rearrange_with_remaining(tensor_i)
        tensor = torch.cat((tensor_i_2, tensor_f, tensor_t), dim=1)
        tensor_pool = self.avgpool(tensor, pool_size=128)
        return tensor, tensor_pool, h_res_len, w_res_len

    def do_tensor_inv_rearrange(self, tensor, h_res_len, w_res_len):
        b, s, n, d = tensor.shape
        tensor_i, tensor_f, tensor_t = torch.split(tensor, [s - self.text_len - self.first_frame_len, self.first_frame_len, self.text_len], dim=1)
        tensor_i = self.inv_rearrange_with_remaining(tensor_i, h_res_len, w_res_len)

        if self.txt_first:
            tensor = torch.cat((tensor_t, tensor_f, tensor_i), dim=1)
        else:
            tensor = torch.cat((tensor_f, tensor_i, tensor_t), dim=1)
        return tensor

    def apply(self, q, k, v, grid_size=None, t_b_idx=None, base_blockmask=None, **kwds):
        if grid_size is not None:
            self.update_grid_size(grid_size)
        t_idx = t_b_idx[0]

        if t_idx < self.skip_timesteps:
            base_blockmask = None
            x = attention_forward(q, k, v, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
        else:
            batch, qSeqlen, numHeads, headDim = q.shape
            if batch != 1:
                raise ValueError("NpuRainfusionAttnWeight currently only supports batch size 1.")
            _, kvSeqlen, _, _ = k.shape
            blockShapeX, blockShapeY = self.pool_size, self.pool_size
            scale = headDim**-0.5

            blockShape = [blockShapeX, blockShapeY]
            actualSeqLengthsHost = [qSeqlen for _ in range(batch)]
            actualSeqLengthsKvHost = [kvSeqlen for _ in range(batch)]

            sparsity = self.sparsity

            h_res_len, w_res_len = 0, 0
            qkv = torch.cat((q, k, v), dim=0)
            qkv, qkv_pool, h_res_len, w_res_len = self.do_tensor_rearrange_pooling(qkv)
            q, k, v = torch.chunk(qkv, 3, dim=0)

            if base_blockmask is None:
                query_pool, key_pool, value_pool = torch.chunk(qkv_pool, 3, dim=0)

                attn_scores_head = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
                attn_scores_fake = torch.nn.functional.softmax(attn_scores_head, dim=-1)

                selectIdx, selectNumIdx = self.get_blockwise_mask(attn_scores_fake, sparsity)
                base_blockmask = [selectIdx, selectNumIdx]
            else:
                selectIdx = base_blockmask[0]
                selectNumIdx = base_blockmask[1]

            q_bnsd = q.transpose(1, 2)
            k_bnsd = k.transpose(1, 2)
            v_bnsd = v.transpose(1, 2)
            x = mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2.rain_fusion_attention(
                q_bnsd,
                k_bnsd,
                v_bnsd,
                scale=scale,
                head_num=numHeads,
                input_layout="BNSD",
                select_idx=selectIdx,
                select_num_idx=selectNumIdx,
                blockshape=blockShape,
                actual_seq_lengths=actualSeqLengthsHost,
                actual_seq_lengths_kv=actualSeqLengthsKvHost,
            )

            x = x.transpose(1, 2).reshape(batch, qSeqlen, numHeads, headDim)

            x = self.do_tensor_inv_rearrange(x, h_res_len, w_res_len)

        return x, base_blockmask


class RainfusionAdapter:
    def __init__(self, rf_attn, manager):
        self.rf = rf_attn
        self.manager = manager

    def _pad_to_expected(self, tensor, actual_seqlen, expected_seqlen):
        if actual_seqlen > expected_seqlen:
            return tensor[:expected_seqlen], True
        elif actual_seqlen < expected_seqlen:
            diff = expected_seqlen - actual_seqlen
            pad = tensor.new_zeros((diff,) + tensor.shape[1:])
            return torch.cat([tensor, pad], dim=0), False
        else:
            return tensor, None

    def apply(self, q, k, v, **kwargs):
        grid_sizes = self.manager.grid_sizes
        expected_seqlen = grid_sizes[0] * grid_sizes[1] * grid_sizes[2]
        actual_seqlen = q.shape[0]

        q_valid, trunc_input = self._pad_to_expected(q, actual_seqlen, expected_seqlen)
        k_valid, _ = self._pad_to_expected(k, actual_seqlen, expected_seqlen)
        v_valid, _ = self._pad_to_expected(v, actual_seqlen, expected_seqlen)

        attn_out, self.manager._base_blockmask = self.rf.apply(
            q=q_valid.unsqueeze(0),
            k=k_valid.unsqueeze(0),
            v=v_valid.unsqueeze(0),
            grid_size=list(grid_sizes),
            t_b_idx=(self.manager.step_index, 0),
            base_blockmask=self.manager._base_blockmask,
        )
        attn_out = attn_out.squeeze(0).reshape(attn_out.shape[1], -1)

        if trunc_input is True:
            pad_out = attn_out.new_zeros((actual_seqlen - expected_seqlen, attn_out.shape[1]))
            attn_out = torch.cat([attn_out, pad_out], dim=0)
        elif trunc_input is False:
            attn_out = attn_out[:actual_seqlen]

        return attn_out


class RainfusionManager:
    def __init__(self, config):
        self.config = config
        self._rf_cfg = config.get("rainfusion", {})
        self._rf = None
        self._base_blockmask = None
        self.grid_sizes = None
        self.step_index = 0

    def update_grid_sizes(self, grid_sizes):
        self.grid_sizes = grid_sizes
        self._init_rf()

    def update_step_index(self, step_index):
        self.step_index = step_index

    def reset(self):
        self._base_blockmask = None

    def _init_rf(self):
        if self._rf is not None and list(self._rf.grid_size) == list(self.grid_sizes):
            return
        self._base_blockmask = None
        self._rf = NpuRainfusionAttnWeight(
            grid_size=list(self.grid_sizes),
            pool_size=128,
            sparsity=self._rf_cfg.get("sparsity", 0.8),
            skip_timesteps=self._rf_cfg.get("skip_timesteps", -1),
            txt_len=0,
            txt_first=False,
        )

    def apply_non_sp(self, q, k, v, grid_sizes):
        attn_out, self._base_blockmask = self._rf.apply(
            q=q, k=k, v=v,
            grid_size=list(grid_sizes),
            t_b_idx=(self.step_index, 0),
            base_blockmask=self._base_blockmask,
        )
        return attn_out

    def get_sp_adapter(self):
        return RainfusionAdapter(self._rf, self)

    @property
    def is_active(self):
        return self._rf is not None
