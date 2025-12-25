import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate
from .utils.ring_comm import RingComm

try:
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")
    flash_attn_varlen_func = None


@torch.jit.script
def _update_out_and_lse(
    out,
    lse,
    block_out,
    block_lse,
):
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


@ATTN_WEIGHT_REGISTER("ring")
class RingAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, img_qkv_len, cu_seqlens_qkv, attention_module=None, seq_p_group=None, model_cls=None, use_fp8_comm=False, enable_head_parallel=False):
        """
        执行 Ring 注意力机制，结合图像和文本的查询、键和值。

        参数:
            q (torch.Tensor): 查询张量，形状为 [shard_seqlen, heads, hidden_dims]
            k (torch.Tensor): 键张量，形状为 [shard_seqlen, heads, hidden_dims]
            v (torch.Tensor): 值张量，形状为 [shard_seqlen, heads, hidden_dims]
            img_qkv_len (int): 图像查询、键和值的长度
            cu_seqlens_qkv (torch.Tensor): 累积序列长度，包含文本和图像的长度信息
            attention_type (str): 注意力类型，默认为 "flash_attn2"

        返回:
            torch.Tensor: 计算得到的注意力结果
        """
        assert not use_fp8_comm, "RingAttn can't support fp8 comm now."
        assert not enable_head_parallel, "RingAttn can't support head parallel mode."

        # 获取当前进程的排名和全局进程数
        cur_rank = dist.get_rank(seq_p_group)
        world_size = dist.get_world_size(seq_p_group)

        if len(cu_seqlens_qkv) == 3:
            txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
            txt_mask_len = cu_seqlens_qkv[2] - img_qkv_len  # 文本掩码长度
        elif len(cu_seqlens_qkv) == 2:
            txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
            txt_mask_len = 0

        # if RING_COMM is None:
        #     init_ring_comm()

        RING_COMM = RingComm(seq_p_group)

        # if len(cu_seqlens_qkv) == 3:
        #     txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
        #     txt_mask_len = cu_seqlens_qkv[2] - img_qkv_len  # 文本掩码长度
        # elif len(cu_seqlens_qkv) == 2:
        #     txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
        #     txt_mask_len = None
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        img_q, img_k, img_v = q[:, :img_qkv_len, :, :].contiguous(), k[:, :img_qkv_len, :, :].contiguous(), v[:, :img_qkv_len, :, :].contiguous()
        txt_q, txt_k, txt_v = (
            q[:, img_qkv_len : img_qkv_len + txt_qkv_len, :, :].contiguous(),
            k[:, img_qkv_len : img_qkv_len + txt_qkv_len, :, :].contiguous(),
            v[:, img_qkv_len : img_qkv_len + txt_qkv_len, :, :].contiguous(),
        )

        out, lse, next_k, next_v = None, None, None, None

        if len(cu_seqlens_qkv) == 3:
            q = torch.cat((img_q, txt_q), dim=1)
        k = img_k
        v = img_v

        for step in range(world_size):
            if step + 1 != world_size:
                next_k = RING_COMM.send_recv(k)
                next_v = RING_COMM.send_recv(v)
                RING_COMM.commit()

            if step + 1 == world_size:
                k = torch.cat((k, txt_k), dim=1)
                v = torch.cat((v, txt_v), dim=1)

            block_out, block_lse = self.ring_attn_sub(q, k, v)

            out, lse = self.update_out_and_lse(out, lse, block_out, block_lse)

            if step + 1 != world_size:
                RING_COMM.wait()
                k = next_k
                v = next_v

        attn1 = out.to(GET_DTYPE()).squeeze(0).reshape(img_qkv_len + txt_qkv_len, -1)

        if txt_mask_len > 0:
            attn2, *_ = flash_attn.flash_attn_interface._flash_attn_forward(
                q[:, -(txt_mask_len - txt_qkv_len) :, :, :].contiguous(),
                k[:, -(txt_mask_len - txt_qkv_len) :, :, :].contiguous(),
                v[:, -(txt_mask_len - txt_qkv_len) :, :, :].contiguous(),
                dropout_p=0.0,
                softmax_scale=q.shape[-1] ** (-0.5),
                causal=False,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )

            attn2 = attn2.to(GET_DTYPE()).squeeze(0).reshape((txt_mask_len - txt_qkv_len), -1)
            attn1 = torch.cat([attn1, attn2], dim=0)

        return attn1

    def ring_attn_sub(self, q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_softmax=False):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        block_out, block_lse, _, _ = flash_attn.flash_attn_interface._flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
        return block_out, block_lse

    def update_out_and_lse(
        self,
        out,
        lse,
        block_out,
        block_lse,
        slice_=None,
    ):
        if out is None:
            if slice_ is not None:
                raise RuntimeError("first update_out_and_lse should not pass slice_ args")
            out = block_out.to(torch.float32)
            lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        elif slice_ is not None:
            slice_out, slice_lse = out[slice_], lse[slice_]
            slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
            out[slice_], lse[slice_] = slice_out, slice_lse
        else:
            out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
        return out, lse
