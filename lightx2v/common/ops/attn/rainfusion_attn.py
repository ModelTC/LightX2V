import math

import torch
from einops import rearrange

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

from .template import AttnWeightTemplate


@ATTN_WEIGHT_REGISTER("rainfusion_attn")
class RainfusionAttnWeight(AttnWeightTemplate):
    pool_size = 128
    sparsity = 0.8
    skip_timesteps = -1
    text_len = 0
    txt_first = False

    _operator = None
    _base_blockmask = None
    _grid_size = None
    _step_index = None

    def __init__(self):
        self.config = {}

    @classmethod
    def configure(cls, setting):
        cls.pool_size = setting.get("pool_size", cls.pool_size)
        cls.sparsity = setting.get("sparsity", cls.sparsity)
        cls.skip_timesteps = setting.get("skip_timesteps", cls.skip_timesteps)
        cls.text_len = setting.get("text_len", setting.get("txt_len", cls.text_len))
        cls.txt_first = setting.get("txt_first", cls.txt_first)

    @classmethod
    def reset_state(cls):
        cls._base_blockmask = None
        cls._step_index = None

    @classmethod
    def _get_operator(cls):
        if cls._operator is not None:
            return cls._operator
        if "npu" not in AI_DEVICE:
            raise RuntimeError("rainfusion_attn currently requires ascend_npu platform.")
        from lightx2v_platform.ops.attn.ascend_npu.npu_rainfusion_attn import NpuRainfusionOperator

        cls._operator = NpuRainfusionOperator()
        return cls._operator

    @classmethod
    def update_grid_size(cls, grid_size):
        grid_size = tuple(grid_size)
        if cls._grid_size != grid_size:
            cls._grid_size = grid_size
            cls._base_blockmask = None

    @classmethod
    def _get_grid_shape(cls):
        if cls._grid_size is None:
            raise ValueError("rainfusion_attn requires grid_sizes in apply kwargs.")
        frame_num, h, w = cls._grid_size
        return frame_num, h, w, h * w

    @classmethod
    def _get_expected_seqlen(cls):
        frame_num, _, _, num_tokens_per_frame = cls._get_grid_shape()
        return frame_num * num_tokens_per_frame

    @classmethod
    def avgpool(cls, input_tensor):
        batch, seqlen, headnum, dim = input_tensor.shape
        num_full_blocks = seqlen // cls.pool_size
        tail_size = seqlen % cls.pool_size

        pooled_tensors = []
        if num_full_blocks > 0:
            full_blocks = input_tensor[:, : num_full_blocks * cls.pool_size, :, :]
            full_blocks_reshaped = full_blocks.reshape(batch, num_full_blocks, cls.pool_size, headnum, dim)
            pooled_tensors.append(full_blocks_reshaped.mean(dim=2))
        if tail_size > 0:
            tail_block = input_tensor[:, num_full_blocks * cls.pool_size :, :, :]
            tail_reshaped = tail_block.reshape(batch, 1, tail_size, headnum, dim)
            pooled_tensors.append(tail_reshaped.mean(dim=2))

        return torch.cat(pooled_tensors, dim=1)

    @classmethod
    def get_mask_index(cls, mask):
        b, n, s, _ = mask.shape
        device = mask.device

        mask_reshaped = mask.reshape(-1, s, s)
        batch_size = mask_reshaped.shape[0]

        row_indices = torch.arange(s, device=device).view(1, 1, s).expand(batch_size, s, s)
        sorted_vals = torch.where(mask_reshaped, row_indices, s)
        sorted_vals, _ = torch.sort(sorted_vals, dim=-1)
        valid_count = mask_reshaped.sum(dim=-1, keepdim=True)
        keep_mask = row_indices < valid_count
        result = torch.where(keep_mask, sorted_vals, -1)

        pos_matrix = result.reshape(b, n, s, s)
        return pos_matrix

    @classmethod
    def get_blockwise_mask(cls, score_matrix):
        batch_size, num_heads, rows, cols = score_matrix.shape

        keep_len = math.ceil(cols * (1 - cls.sparsity))
        keep_len = max(1, min(keep_len, cols))
        topk_values, _ = torch.topk(score_matrix, k=keep_len, dim=-1)
        thresholds = topk_values[..., -1:]
        mask = score_matrix >= thresholds

        frame_num, _, _, num_tokens_per_frame = cls._get_grid_shape()
        first_frame_len = num_tokens_per_frame
        total_len = frame_num * num_tokens_per_frame + cls.text_len
        protected_start_idx = total_len - first_frame_len - cls.text_len
        protected_start_block = protected_start_idx // cls.pool_size
        protect_len = cols - protected_start_block

        if protect_len > 0:
            mask[:, :, -protect_len:, :] = True
            mask[:, :, :, -protect_len:] = True

        select_idx = cls.get_mask_index(mask)
        select_idx = select_idx[0].transpose(0, 1)
        select_num_idx = mask[0].transpose(0, 1).sum(dim=-1)
        return select_idx, select_num_idx

    @classmethod
    def rearrange_with_remaining(cls, tensor):
        b, s, n, d = tensor.shape
        frame_num, h, w, first_frame_len = cls._get_grid_shape()
        h_res_len, w_res_len = 0, 0
        first_frame_num = first_frame_len // h // w

        tensor_hwt = rearrange(tensor, "b (f h w) n d -> (b n) f h w d", f=frame_num - first_frame_num, h=h, w=w)
        if h % 8 != 0:
            tensor_hwt, tensor_h_r = torch.split(tensor_hwt, [h - (h % 8), h % 8], dim=2)
            tensor_h_r = tensor_h_r.reshape(b * n, -1, d)
            h_res_len = tensor_h_r.shape[1]
        if w % 8 != 0:
            tensor_hwt, tensor_w_r = torch.split(tensor_hwt, [w - (w % 8), w % 8], dim=3)
            tensor_w_r = tensor_w_r.reshape(b * n, -1, d)
            w_res_len = tensor_w_r.shape[1]
        remaining_frames = frame_num - first_frame_num
        if remaining_frames % 2 != 0:
            raise ValueError(f"The number of remaining frames ({remaining_frames}) must be even to be rearranged with block size 2.")
        tensor_hwt = rearrange(tensor_hwt, "bn (fn fb) (hn hb) (wn wb) d -> bn (fn hn wn fb hb wb) d", fn=remaining_frames // 2, fb=2, hb=8, wb=8, hn=h // 8, wn=w // 8)
        if h % 8 != 0:
            tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=1)
        if w % 8 != 0:
            tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=1)
        tensor_hwt = rearrange(tensor_hwt, "(b n) s d -> b s n d", b=b, n=n)
        return tensor_hwt, h_res_len, w_res_len

    @classmethod
    def inv_rearrange_with_remaining(cls, tensor, h_res_len, w_res_len):
        b, s, n, d = tensor.shape
        frame_num, h, w, first_frame_len = cls._get_grid_shape()
        h_sr, w_sr = h % 8, w % 8
        first_frame_num = first_frame_len // (h * w)
        remaining_frames = frame_num - first_frame_num

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

    @classmethod
    def do_tensor_rearrange_pooling(cls, tensor):
        b, s, n, d = tensor.shape
        _, _, _, first_frame_len = cls._get_grid_shape()
        if s < cls.text_len + first_frame_len:
            raise ValueError(f"Sequence length {s} is too small for text_len {cls.text_len} and first_frame_len {first_frame_len}")
        if cls.txt_first:
            tensor_t, tensor_f, tensor_i = torch.split(tensor, [cls.text_len, first_frame_len, s - cls.text_len - first_frame_len], dim=1)
        else:
            tensor_f, tensor_i, tensor_t = torch.split(tensor, [first_frame_len, s - cls.text_len - first_frame_len, cls.text_len], dim=1)
        tensor_i_2, h_res_len, w_res_len = cls.rearrange_with_remaining(tensor_i)
        tensor = torch.cat((tensor_i_2, tensor_f, tensor_t), dim=1)
        tensor_pool = cls.avgpool(tensor)
        return tensor, tensor_pool, h_res_len, w_res_len

    @classmethod
    def do_tensor_inv_rearrange(cls, tensor, h_res_len, w_res_len):
        b, s, n, d = tensor.shape
        _, _, _, first_frame_len = cls._get_grid_shape()
        tensor_i, tensor_f, tensor_t = torch.split(tensor, [s - cls.text_len - first_frame_len, first_frame_len, cls.text_len], dim=1)
        tensor_i = cls.inv_rearrange_with_remaining(tensor_i, h_res_len, w_res_len)

        if cls.txt_first:
            tensor = torch.cat((tensor_t, tensor_f, tensor_i), dim=1)
        else:
            tensor = torch.cat((tensor_f, tensor_i, tensor_t), dim=1)
        return tensor

    @staticmethod
    def _pad_to_expected(tensor, expected_seqlen):
        actual_seqlen = tensor.shape[0]
        if actual_seqlen > expected_seqlen:
            return tensor[:expected_seqlen], True
        if actual_seqlen < expected_seqlen:
            diff = expected_seqlen - actual_seqlen
            pad = tensor.new_zeros((diff,) + tensor.shape[1:])
            return torch.cat([tensor, pad], dim=0), False
        return tensor, None

    @classmethod
    def _get_step_index(cls, scheduler):
        if scheduler is None:
            return 0
        return getattr(scheduler, "step_index", 0)

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        grid_sizes = kwargs.get("grid_sizes", None)
        if grid_sizes is None:
            raise ValueError("rainfusion_attn requires grid_sizes in apply kwargs.")
        cls = type(self)
        cls.update_grid_size(grid_sizes)

        scheduler = kwargs.get("scheduler", None)
        step_index = cls._get_step_index(scheduler)
        if cls._step_index != step_index:
            cls._base_blockmask = None
            cls._step_index = step_index

        if len(q.shape) == 4:
            bs = q.shape[0]
            if bs != 1:
                raise ValueError("rainfusion_attn currently only supports batch size 1.")
            q_4d, k_4d, v_4d = q, k, v
            actual_seqlen = q.shape[1]
            trunc_input = None
        else:
            expected_seqlen = self._get_expected_seqlen()
            q_valid, trunc_input = self._pad_to_expected(q, expected_seqlen)
            k_valid, _ = self._pad_to_expected(k, expected_seqlen)
            v_valid, _ = self._pad_to_expected(v, expected_seqlen)
            actual_seqlen = q.shape[0]
            q_4d = q_valid.unsqueeze(0)
            k_4d = k_valid.unsqueeze(0)
            v_4d = v_valid.unsqueeze(0)

        if step_index < self.skip_timesteps:
            out = cls._get_operator().dense_attention(q_4d, k_4d, v_4d)
        else:
            batch, q_seqlen, num_heads, head_dim = q_4d.shape
            if batch != 1:
                raise ValueError("rainfusion_attn currently only supports batch size 1.")
            kv_seqlen = k_4d.shape[1]
            scale = head_dim**-0.5

            h_res_len, w_res_len = 0, 0
            qkv = torch.cat((q_4d, k_4d, v_4d), dim=0)
            qkv, qkv_pool, h_res_len, w_res_len = self.do_tensor_rearrange_pooling(qkv)
            q_4d, k_4d, v_4d = torch.chunk(qkv, 3, dim=0)

            if cls._base_blockmask is None:
                query_pool, key_pool, _ = torch.chunk(qkv_pool, 3, dim=0)
                attn_scores_head = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
                attn_scores_fake = torch.nn.functional.softmax(attn_scores_head, dim=-1)
                cls._base_blockmask = cls.get_blockwise_mask(attn_scores_fake)

            select_idx, select_num_idx = cls._base_blockmask
            out = cls._get_operator().sparse_attention(
                q_4d,
                k_4d,
                v_4d,
                scale=scale,
                head_num=num_heads,
                select_idx=select_idx,
                select_num_idx=select_num_idx,
                block_shape=[self.pool_size, self.pool_size],
                actual_seq_lengths=[q_seqlen for _ in range(batch)],
                actual_seq_lengths_kv=[kv_seqlen for _ in range(batch)],
            )
            out = self.do_tensor_inv_rearrange(out, h_res_len, w_res_len)

        out = out.squeeze(0).reshape(out.shape[1], -1)
        if trunc_input is True:
            pad_out = out.new_zeros((actual_seqlen - out.shape[0], out.shape[1]))
            out = torch.cat([out, pad_out], dim=0)
        elif trunc_input is False:
            out = out[:actual_seqlen]
        return out
