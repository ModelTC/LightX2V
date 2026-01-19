import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.utils.envs import *
from lightx2v.utils.quant_utils import dequant_fp8_vllm, quant_fp8_vllm
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
        self.helper = RingAttnHelper()

    def apply(
        self,
        q,
        k,
        v,
        slice_qkv_len,
        cu_seqlens_qkv,
        attention_module=None,
        attention_type="flash_attn2",
        seq_p_group=None,
        use_fp8_comm=False,
        use_kv_fusion=False,
        enable_head_parallel=False,
        **kwargs,
    ):
        """
        执行 Ring 注意力机制，结合图像和文本的查询、键和值。

        参数:
            q (torch.Tensor): 查询张量，形状为 [shard_seqlen, heads, hidden_dims]
            k (torch.Tensor): 键张量，形状为 [shard_seqlen, heads, hidden_dims]
            v (torch.Tensor): 值张量，形状为 [shard_seqlen, heads, hidden_dims]
            slice_qkv_len (int): 图像查询、键和值的长度
            cu_seqlens_qkv (torch.Tensor): 累积序列长度，包含文本和图像的长度信息
            attention_type (str): 注意力类型，默认为 "flash_attn2"

        返回:
            torch.Tensor: 计算得到的注意力结果
        """
        assert not enable_head_parallel, "RingAttn can't support head parallel mode."

        # 获取当前进程的排名和全局进程数
        cur_rank = dist.get_rank(seq_p_group)
        world_size = dist.get_world_size(seq_p_group)

        img_qkv_len = slice_qkv_len
        txt_qkv_len, txt_mask_len = self.helper._get_text_lengths(cu_seqlens_qkv, img_qkv_len)

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

        heads, hidden_dims = k.shape[-2], k.shape[-1]
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

        if use_kv_fusion:
            txt_kv = torch.stack([txt_k, txt_v], dim=0).reshape(2, txt_qkv_len, heads, hidden_dims).contiguous()
            kv, original_dtype, original_shape = self.helper._prepare_kv_tensors(k, v, use_kv_fusion)
        else:
            original_dtype = k.dtype
            original_shape = k.shape

        for step in range(world_size):
            if step + 1 != world_size:
                if use_fp8_comm:
                    if use_kv_fusion:
                        next_kv_fp8, next_kv_scale = self.helper._send_recv_tensor(kv, hidden_dims, RING_COMM, use_fp8_comm, original_shape)
                    else:
                        next_k_fp8, next_k_scale = self.helper._send_recv_tensor(k, hidden_dims, RING_COMM, use_fp8_comm, original_shape)
                        next_v_fp8, next_v_scale = self.helper._send_recv_tensor(v, hidden_dims, RING_COMM, use_fp8_comm, original_shape)
                else:
                    if use_kv_fusion:
                        next_kv = self.helper._send_recv_tensor(kv, hidden_dims, RING_COMM, use_fp8_comm, original_shape)[0]
                    else:
                        next_k = self.helper._send_recv_tensor(k, hidden_dims, RING_COMM, use_fp8_comm, original_shape)[0]
                        next_v = self.helper._send_recv_tensor(v, hidden_dims, RING_COMM, use_fp8_comm, original_shape)[0]
                RING_COMM.commit()

            if step + 1 == world_size:
                if use_kv_fusion:
                    kv = torch.cat((kv, txt_kv), dim=1)
                else:
                    k = torch.cat((k, txt_k), dim=1)
                    v = torch.cat((v, txt_v), dim=1)

            if use_kv_fusion:
                block_out, block_lse = self.ring_attn_sub_kv_fusion(q, kv)
            else:
                block_out, block_lse = self.ring_attn_sub(q, k, v)

            out, lse = self.update_out_and_lse(out, lse, block_out, block_lse)

            if step + 1 != world_size:
                RING_COMM.wait()

                if use_fp8_comm:
                    if use_kv_fusion:
                        kv = self.helper._dequantize_received(next_kv_fp8, next_kv_scale, original_dtype, original_shape, use_kv_fusion=True, is_kv_fusion=True)
                    else:
                        k, v = self.helper._dequantize_received(
                            next_k_fp8, next_k_scale, original_dtype, original_shape, use_kv_fusion=False, is_kv_fusion=False, v_fp8=next_v_fp8, v_scale=next_v_scale
                        )
                else:
                    if use_kv_fusion:
                        kv = next_kv
                    else:
                        k, v = next_k, next_v

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

    def ring_attn_sub_kv_fusion(self, q, kv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_softmax=False):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        block_out, block_lse, _, _ = flash_attn.flash_attn_interface._flash_attn_forward(
            q,
            kv[:1, :, :, :],
            kv[1:, :, :, :],
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


class RingAttnHelper:
    """辅助函数类，处理 Ring Attention 中的量化、通信和反量化逻辑"""

    @staticmethod
    def _quant_and_send(tensor, hidden_dims, comm, original_shape=None):
        """
        对张量进行 FP8 量化并通过通信器发送/接收

        参数:
            tensor: 要量化和发送的张量
            hidden_dims: 隐藏维度大小
            comm: 通信器对象
            original_shape: 原始形状（用于 reshape 回原始形状）

        返回:
            tuple: (量化后的张量, scale 张量)
        """
        if original_shape is None:
            original_shape = tensor.shape

        # 量化为 FP8
        tensor_fp8, tensor_scale = quant_fp8_vllm(tensor.reshape(-1, hidden_dims))

        # reshape 回原始形状
        tensor_fp8 = tensor_fp8.reshape(original_shape)
        tensor_scale = tensor_scale.reshape(original_shape[0], original_shape[1], original_shape[2], 1)

        # 发送/接收量化后的张量
        next_tensor_fp8 = comm.send_recv(tensor_fp8)
        next_tensor_scale = comm.send_recv(tensor_scale)

        return next_tensor_fp8, next_tensor_scale

    @staticmethod
    def _prepare_kv_tensors(k, v, use_kv_fusion):
        """
        准备 K 和 V 张量，根据是否使用 KV 融合返回适当的张量

        参数:
            k: 键张量
            v: 值张量
            use_kv_fusion: 是否使用 KV 融合

        返回:
            tuple: (主张量, 原始数据类型, 原始形状)
        """
        original_dtype = k.dtype
        original_shape = k.shape

        if use_kv_fusion:
            # 融合 K 和 V
            kv = torch.stack([k, v], dim=0).reshape(2, k.shape[1], k.shape[2], k.shape[3]).contiguous()
            return kv, original_dtype, kv.shape
        else:
            return k, original_dtype, original_shape

    @staticmethod
    def _dequantize_received(next_tensor_fp8, next_tensor_scale, original_dtype, original_shape, use_kv_fusion=False, is_kv_fusion=False, v_fp8=None, v_scale=None):
        """
        反量化接收到的 FP8 张量

        参数:
            next_tensor_fp8: 接收到的量化张量
            next_tensor_scale: 接收到的 scale 张量
            original_dtype: 原始数据类型
            original_shape: 原始形状
            use_kv_fusion: 是否使用 KV 融合模式
            is_kv_fusion: 当前张量是否为 KV 融合张量
            v_fp8, v_scale: 分离模式下的 V 张量和 scale

        返回:
            tuple: 反量化后的张量 (k, v) 或 kv
        """
        if use_kv_fusion and is_kv_fusion:
            # KV 融合模式
            return dequant_fp8_vllm(next_tensor_fp8, next_tensor_scale, original_dtype)
        elif not use_kv_fusion:
            # 分离模式
            k = dequant_fp8_vllm(next_tensor_fp8, next_tensor_scale, original_dtype)
            v = dequant_fp8_vllm(v_fp8, v_scale, original_dtype)
            return k, v
        else:
            # 默认返回单个张量
            return dequant_fp8_vllm(next_tensor_fp8, next_tensor_scale, original_dtype)

    @staticmethod
    def _send_recv_tensor(tensor, hidden_dims, comm, use_fp8_comm, original_shape=None):
        """
        发送/接收张量，根据是否使用 FP8 选择通信方式

        参数:
            tensor: 要发送的张量
            hidden_dims: 隐藏维度大小
            comm: 通信器对象
            use_fp8_comm: 是否使用 FP8 通信
            original_shape: 原始形状

        返回:
            tuple: 接收到的张量（和可能的 scale）
        """
        if use_fp8_comm:
            return RingAttnHelper._quant_and_send(tensor, hidden_dims, comm, original_shape)
        else:
            next_tensor = comm.send_recv(tensor)
            return next_tensor, None

    @staticmethod
    def _get_text_lengths(cu_seqlens_qkv, img_qkv_len):
        """
        从累积序列长度中获取文本长度

        参数:
            cu_seqlens_qkv: 累积序列长度
            img_qkv_len: 图像序列长度

        返回:
            tuple: (文本QKV长度, 文本掩码长度)
        """
        if len(cu_seqlens_qkv) == 3:
            txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len
            txt_mask_len = cu_seqlens_qkv[2] - img_qkv_len
        elif len(cu_seqlens_qkv) == 2:
            txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len
            txt_mask_len = 0
        else:
            raise ValueError(f"Invalid cu_seqlens_qkv length: {len(cu_seqlens_qkv)}")

        return txt_qkv_len, txt_mask_len
