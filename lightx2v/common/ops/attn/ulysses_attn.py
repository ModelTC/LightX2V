import torch
import torch.distributed as dist

from lightx2v.utils.quant_utils import dequant_fp8_vllm, quant_fp8_vllm
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

from .template import AttnWeightTemplate
from .utils.all2all import all2all_head2seq, all2all_seq2head


@ATTN_WEIGHT_REGISTER("ulysses")
class UlyssesAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, img_qkv_len, cu_seqlens_qkv, attention_module=None, seq_p_group=None, model_cls=None, use_fp8_comm=False):
        """
        执行 Ulysses 注意力机制，结合图像和文本的查询、键和值。

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
        if len(q.shape) == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])

        # 获取当前进程的排名和全局进程数
        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)

        # 获取序列长度和文本相关的长度
        seq_len = q.shape[0]
        if len(cu_seqlens_qkv) == 3:
            txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
            txt_mask_len = cu_seqlens_qkv[2] - img_qkv_len  # 文本掩码长度
        elif len(cu_seqlens_qkv) == 2:
            txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
            txt_mask_len = None

        # 获取查询张量的头数和隐藏维度
        _, heads, hidden_dims = q.shape
        shard_heads = heads // world_size  # 每个进程处理的头数
        shard_seqlen = img_qkv_len  # 每个进程处理的序列长度

        # 分割图像和文本的查询、键和值
        img_q, img_k, img_v = q[:img_qkv_len, :, :].contiguous(), k[:img_qkv_len, :, :].contiguous(), v[:img_qkv_len, :, :].contiguous()
        txt_q, txt_k, txt_v = q[img_qkv_len:, :, :].contiguous(), k[img_qkv_len:, :, :].contiguous(), v[img_qkv_len:, :, :].contiguous()

        # 将图像的查询、键和值转换为头的格式
        if use_fp8_comm:
            original_dtype = img_q.dtype
            original_shape = img_q.shape
            img_q_fp8, q_scale = quant_fp8_vllm(img_q.reshape(-1, original_shape[-1]))
            img_k_fp8, k_scale = quant_fp8_vllm(img_k.reshape(-1, original_shape[-1]))
            img_v_fp8, v_scale = quant_fp8_vllm(img_v.reshape(-1, original_shape[-1]))
            img_q_fp8 = all2all_seq2head(img_q_fp8.reshape(original_shape), group=seq_p_group)
            img_k_fp8 = all2all_seq2head(img_k_fp8.reshape(original_shape), group=seq_p_group)
            img_v_fp8 = all2all_seq2head(img_v_fp8.reshape(original_shape), group=seq_p_group)
            q_scale = all2all_seq2head(q_scale.reshape(original_shape[0], original_shape[1], 1), group=seq_p_group)
            k_scale = all2all_seq2head(k_scale.reshape(original_shape[0], original_shape[1], 1), group=seq_p_group)
            v_scale = all2all_seq2head(v_scale.reshape(original_shape[0], original_shape[1], 1), group=seq_p_group)
            img_q = dequant_fp8_vllm(img_q_fp8, q_scale, original_dtype)
            img_k = dequant_fp8_vllm(img_k_fp8, k_scale, original_dtype)
            img_v = dequant_fp8_vllm(img_v_fp8, v_scale, original_dtype)
        else:
            img_q = all2all_seq2head(img_q, group=seq_p_group)
            img_k = all2all_seq2head(img_k, group=seq_p_group)
            img_v = all2all_seq2head(img_v, group=seq_p_group)
        self.device_synchronize()  # 确保CUDA操作完成

        # 处理文本的查询、键和值，选择当前进程的头
        txt_q = txt_q[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
        txt_k = txt_k[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
        txt_v = txt_v[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]

        # 合并图像和文本的查询、键和值
        q = torch.cat((img_q, txt_q), dim=0)
        k = torch.cat((img_k, txt_k), dim=0)
        v = torch.cat((img_v, txt_v), dim=0)

        # 初始化累积序列长度张量
        cu_seqlens_qkv = torch.zeros([2], dtype=torch.int32, device=AI_DEVICE)
        s = txt_qkv_len + img_q.shape[0]  # 计算文本和图像的总长度
        s1 = s  # 当前样本的结束位置
        cu_seqlens_qkv[1] = s1  # 设置累积序列长度
        if txt_mask_len:
            s2 = txt_mask_len + img_q.shape[0]  # 文本掩码的结束位置
            cu_seqlens_qkv = torch.cat(cu_seqlens_qkv, s2)
        max_seqlen_qkv = img_q.shape[0] + txt_q.shape[0]  # 最大序列长度

        # 调用注意力函数计算注意力结果
        # attn = attention(attention_type=attention_type, q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv)
        attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, model_cls=model_cls)

        # 分割图像和文本的注意力结果
        img_attn, txt_attn = attn[: img_q.shape[0], :], attn[img_q.shape[0] :,]

        # 收集所有进程的文本注意力结果
        gathered_txt_attn = [torch.empty_like(txt_attn) for _ in range(world_size)]
        dist.all_gather(gathered_txt_attn, txt_attn, group=seq_p_group)

        img_attn = self._reshape_img_attn(img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group, use_fp8_comm)

        txt_attn = torch.cat(gathered_txt_attn, dim=1)  # 合并所有进程的文本注意力结果

        # 合并图像和文本的注意力结果
        attn = torch.cat([img_attn, txt_attn], dim=0)

        return attn  # 返回最终的注意力结果

    @torch.compiler.disable
    def _reshape_img_attn(self, img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group, use_fp8_comm):
        img_attn = img_attn.reshape(world_size * shard_seqlen, shard_heads, hidden_dims)  # 重塑图像注意力结果

        # 将头的格式转换回序列格式
        if use_fp8_comm:
            original_dtype = img_attn.dtype
            original_shape = img_attn.shape
            img_attn_fp8, attn_scale = quant_fp8_vllm(img_attn.reshape(-1, original_shape[-1]))
            img_attn_fp8 = all2all_head2seq(img_attn_fp8.reshape(original_shape), group=seq_p_group)
            attn_scale = all2all_head2seq(attn_scale.reshape(original_shape[0], original_shape[1], 1), group=seq_p_group)
            img_attn = dequant_fp8_vllm(img_attn_fp8, attn_scale, original_dtype)
        else:
            img_attn = all2all_head2seq(img_attn, group=seq_p_group)

        img_attn = img_attn.reshape(shard_seqlen, -1)  # 重塑为 [shard_seqlen, -1] 形状
        return img_attn


@ATTN_WEIGHT_REGISTER("ulysses-4090")
class Ulysses4090AttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, img_qkv_len, cu_seqlens_qkv, attention_module=None, seq_p_group=None, model_cls=None, use_fp8_comm=False):
        """
        执行 Ulysses 注意力机制，结合图像和文本的查询、键和值。

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
        if len(q.shape) == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
        # 获取当前进程的排名和全局进程数
        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)
        global_world_size = dist.get_world_size()
        global_rank = dist.get_global_rank(seq_p_group, cur_rank)
        cfg_p_group_index = global_rank // world_size

        # 获取序列长度和文本相关的长度
        seq_len = q.shape[0]
        if len(cu_seqlens_qkv) == 3:
            txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
            txt_mask_len = cu_seqlens_qkv[2] - img_qkv_len  # 文本掩码长度
        elif len(cu_seqlens_qkv) == 2:
            txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
            txt_mask_len = None

        # 获取查询张量的头数和隐藏维度
        _, heads, hidden_dims = q.shape
        shard_heads = heads // world_size  # 每个进程处理的头数
        shard_seqlen = img_qkv_len  # 每个进程处理的序列长度

        # 分割图像和文本的查询、键和值
        img_q, img_k, img_v = q[:img_qkv_len, :, :].contiguous(), k[:img_qkv_len, :, :].contiguous(), v[:img_qkv_len, :, :].contiguous()
        txt_q, txt_k, txt_v = q[img_qkv_len:, :, :].contiguous(), k[img_qkv_len:, :, :].contiguous(), v[img_qkv_len:, :, :].contiguous()

        # 计算每个进程应该持有的头数分片
        num_heads = img_q.shape[1]
        shard_heads = num_heads // world_size

        # 将 QKV 按头维度切分成 N 份,每份大小为 D/N
        q_shards = [img_q[:, i * shard_heads : (i + 1) * shard_heads, :].contiguous() for i in range(world_size)]
        k_shards = [img_k[:, i * shard_heads : (i + 1) * shard_heads, :].contiguous() for i in range(world_size)]
        v_shards = [img_v[:, i * shard_heads : (i + 1) * shard_heads, :].contiguous() for i in range(world_size)]

        if use_fp8_comm:
            original_dtype = img_q.dtype
            original_shape = img_q.shape

            # 量化所有分片
            q_shards_fp8 = []
            q_scales = []
            k_shards_fp8 = []
            k_scales = []
            v_shards_fp8 = []
            v_scales = []
            for i in range(world_size):
                q_fp8, q_scale = quant_fp8_vllm(q_shards[i].reshape(-1, original_shape[-1]))
                q_shards_fp8.append(q_fp8)
                q_scales.append(q_scale)
                k_fp8, k_scale = quant_fp8_vllm(k_shards[i].reshape(-1, original_shape[-1]))
                k_shards_fp8.append(k_fp8)
                k_scales.append(k_scale)
                v_fp8, v_scale = quant_fp8_vllm(v_shards[i].reshape(-1, original_shape[-1]))
                v_shards_fp8.append(v_fp8)
                v_scales.append(v_scale)

            # 准备接收缓冲区 (FP8 + scale)
            gathered_q_fp8 = [None] * world_size
            gathered_q_scales = [None] * world_size
            gathered_k_fp8 = [None] * world_size
            gathered_k_scales = [None] * world_size
            gathered_v_fp8 = [None] * world_size
            gathered_v_scales = [None] * world_size
            for target_rank in range(world_size):
                if target_rank != cur_rank:
                    gathered_q_fp8[target_rank] = torch.empty_like(q_shards_fp8[target_rank])
                    gathered_q_scales[target_rank] = torch.empty_like(q_scales[target_rank])
                    gathered_k_fp8[target_rank] = torch.empty_like(k_shards_fp8[target_rank])
                    gathered_k_scales[target_rank] = torch.empty_like(k_scales[target_rank])
                    gathered_v_fp8[target_rank] = torch.empty_like(v_shards_fp8[target_rank])
                    gathered_v_scales[target_rank] = torch.empty_like(v_scales[target_rank])
                else:
                    gathered_q_fp8[cur_rank] = q_shards_fp8[cur_rank]
                    gathered_q_scales[cur_rank] = q_scales[cur_rank]
                    gathered_k_fp8[cur_rank] = k_shards_fp8[cur_rank]
                    gathered_k_scales[cur_rank] = k_scales[cur_rank]
                    gathered_v_fp8[cur_rank] = v_shards_fp8[cur_rank]
                    gathered_v_scales[cur_rank] = v_scales[cur_rank]

            # 通信 FP8 数据和 scales
            for target_rank in range(world_size):
                target_global_rank = cfg_p_group_index * world_size + target_rank
                if target_rank != cur_rank:
                    if cur_rank < target_rank:
                        sendq_fp8_req = dist.isend(q_shards_fp8[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendk_fp8_req = dist.isend(k_shards_fp8[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendv_fp8_req = dist.isend(v_shards_fp8[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendq_scale_req = dist.isend(q_scales[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendk_scale_req = dist.isend(k_scales[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendv_scale_req = dist.isend(v_scales[target_rank], dst=target_global_rank, group=seq_p_group)
                        recvq_fp8_req = dist.irecv(gathered_q_fp8[target_rank], src=target_global_rank, group=seq_p_group)
                        recvk_fp8_req = dist.irecv(gathered_k_fp8[target_rank], src=target_global_rank, group=seq_p_group)
                        recvv_fp8_req = dist.irecv(gathered_v_fp8[target_rank], src=target_global_rank, group=seq_p_group)
                        recvq_scale_req = dist.irecv(gathered_q_scales[target_rank], src=target_global_rank, group=seq_p_group)
                        recvk_scale_req = dist.irecv(gathered_k_scales[target_rank], src=target_global_rank, group=seq_p_group)
                        recvv_scale_req = dist.irecv(gathered_v_scales[target_rank], src=target_global_rank, group=seq_p_group)
                    else:
                        recvq_fp8_req = dist.irecv(gathered_q_fp8[target_rank], src=target_global_rank, group=seq_p_group)
                        recvk_fp8_req = dist.irecv(gathered_k_fp8[target_rank], src=target_global_rank, group=seq_p_group)
                        recvv_fp8_req = dist.irecv(gathered_v_fp8[target_rank], src=target_global_rank, group=seq_p_group)
                        recvq_scale_req = dist.irecv(gathered_q_scales[target_rank], src=target_global_rank, group=seq_p_group)
                        recvk_scale_req = dist.irecv(gathered_k_scales[target_rank], src=target_global_rank, group=seq_p_group)
                        recvv_scale_req = dist.irecv(gathered_v_scales[target_rank], src=target_global_rank, group=seq_p_group)
                        sendq_fp8_req = dist.isend(q_shards_fp8[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendk_fp8_req = dist.isend(k_shards_fp8[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendv_fp8_req = dist.isend(v_shards_fp8[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendq_scale_req = dist.isend(q_scales[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendk_scale_req = dist.isend(k_scales[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendv_scale_req = dist.isend(v_scales[target_rank], dst=target_global_rank, group=seq_p_group)
                    sendq_fp8_req.wait()
                    sendk_fp8_req.wait()
                    sendv_fp8_req.wait()
                    sendq_scale_req.wait()
                    sendk_scale_req.wait()
                    sendv_scale_req.wait()
                    recvq_fp8_req.wait()
                    recvk_fp8_req.wait()
                    recvv_fp8_req.wait()
                    recvq_scale_req.wait()
                    recvk_scale_req.wait()
                    recvv_scale_req.wait()

            # 反量化
            gathered_q_shards = []
            gathered_k_shards = []
            gathered_v_shards = []
            for i in range(world_size):
                q_shards_new = dequant_fp8_vllm(gathered_q_fp8[i], gathered_q_scales[i], original_dtype).reshape(-1, shard_heads, hidden_dims)
                k_shards_new = dequant_fp8_vllm(gathered_k_fp8[i], gathered_k_scales[i], original_dtype).reshape(-1, shard_heads, hidden_dims)
                v_shards_new = dequant_fp8_vllm(gathered_v_fp8[i], gathered_v_scales[i], original_dtype).reshape(-1, shard_heads, hidden_dims)
                gathered_q_shards.append(q_shards_new)
                gathered_k_shards.append(k_shards_new)
                gathered_v_shards.append(v_shards_new)

        else:
            # 准备接收缓冲区
            gathered_q_shards = [None] * world_size
            gathered_k_shards = [None] * world_size
            gathered_v_shards = [None] * world_size
            for target_rank in range(world_size):
                if target_rank != cur_rank:
                    gathered_q_shards[target_rank] = torch.empty_like(q_shards[target_rank])
                    gathered_k_shards[target_rank] = torch.empty_like(k_shards[target_rank])
                    gathered_v_shards[target_rank] = torch.empty_like(v_shards[target_rank])
                else:
                    gathered_q_shards[cur_rank] = q_shards[cur_rank]
                    gathered_k_shards[cur_rank] = k_shards[cur_rank]
                    gathered_v_shards[cur_rank] = v_shards[cur_rank]

            # 通信
            for target_rank in range(world_size):
                target_global_rank = cfg_p_group_index * world_size + target_rank
                if target_rank != cur_rank:
                    if cur_rank < target_rank:
                        sendq_req = dist.isend(q_shards[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendk_req = dist.isend(k_shards[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendv_req = dist.isend(v_shards[target_rank], dst=target_global_rank, group=seq_p_group)
                        recvq_req = dist.irecv(gathered_q_shards[target_rank], src=target_global_rank, group=seq_p_group)
                        recvk_req = dist.irecv(gathered_k_shards[target_rank], src=target_global_rank, group=seq_p_group)
                        recvv_req = dist.irecv(gathered_v_shards[target_rank], src=target_global_rank, group=seq_p_group)
                    else:
                        recvq_req = dist.irecv(gathered_q_shards[target_rank], src=target_global_rank, group=seq_p_group)
                        recvk_req = dist.irecv(gathered_k_shards[target_rank], src=target_global_rank, group=seq_p_group)
                        recvv_req = dist.irecv(gathered_v_shards[target_rank], src=target_global_rank, group=seq_p_group)
                        sendq_req = dist.isend(q_shards[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendk_req = dist.isend(k_shards[target_rank], dst=target_global_rank, group=seq_p_group)
                        sendv_req = dist.isend(v_shards[target_rank], dst=target_global_rank, group=seq_p_group)
                    sendq_req.wait()
                    sendk_req.wait()
                    sendv_req.wait()
                    recvq_req.wait()
                    recvk_req.wait()
                    recvv_req.wait()

        # 拼接所有分片 (在序列维度上)
        # 每个 gathered_*_shards[i] 的形状是 (seq_len/N, num_heads/N, head_dim)
        # 拼接后形状是 (seq_len, num_heads/N, head_dim)
        img_q = torch.cat(gathered_q_shards, dim=0)
        img_k = torch.cat(gathered_k_shards, dim=0)
        img_v = torch.cat(gathered_v_shards, dim=0)

        # 处理文本的查询、键和值，选择当前进程的头
        txt_q = txt_q[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
        txt_k = txt_k[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
        txt_v = txt_v[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]

        # 合并图像和文本的查询、键和值
        q = torch.cat((img_q, txt_q), dim=0)
        k = torch.cat((img_k, txt_k), dim=0)
        v = torch.cat((img_v, txt_v), dim=0)

        # 初始化累积序列长度张量
        cu_seqlens_qkv = torch.zeros([2], dtype=torch.int32, device="cuda")
        s = txt_qkv_len + img_q.shape[0]  # 计算文本和图像的总长度
        s1 = s  # 当前样本的结束位置
        cu_seqlens_qkv[1] = s1  # 设置累积序列长度
        if txt_mask_len:
            s2 = txt_mask_len + img_q.shape[0]  # 文本掩码的结束位置
            cu_seqlens_qkv = torch.cat(cu_seqlens_qkv, s2)
        max_seqlen_qkv = img_q.shape[0] + txt_q.shape[0]  # 最大序列长度

        # 调用注意力函数计算注意力结果
        # attn = attention(attention_type=attention_type, q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv)
        attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, model_cls=model_cls)

        # 分割图像和文本的注意力结果
        img_attn, txt_attn = attn[: img_q.shape[0], :], attn[img_q.shape[0] :,]

        # 收集所有进程的文本注意力结果
        gathered_txt_attn = [torch.empty_like(txt_attn) for _ in range(world_size)]
        dist.all_gather(gathered_txt_attn, txt_attn, group=seq_p_group)

        img_attn = self._reshape_img_attn(img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group, use_fp8_comm)

        txt_attn = torch.cat(gathered_txt_attn, dim=1)  # 合并所有进程的文本注意力结果

        # 合并图像和文本的注意力结果
        attn = torch.cat([img_attn, txt_attn], dim=0)

        return attn  # 返回最终的注意力结果

    @torch.compiler.disable
    def _reshape_img_attn(self, img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group, use_fp8_comm):
        cur_rank = dist.get_rank(seq_p_group)
        global_world_size = dist.get_world_size()
        global_rank = dist.get_global_rank(seq_p_group, cur_rank)
        cfg_p_group_index = global_rank // world_size

        img_attn = img_attn.reshape(world_size * shard_seqlen, shard_heads, hidden_dims)  # 重塑图像注意力结果

        # 按序列维度切分成 N 份
        attn_shards = [img_attn[i * shard_seqlen : (i + 1) * shard_seqlen, :, :].contiguous() for i in range(world_size)]

        if use_fp8_comm:
            original_dtype = img_attn.dtype
            original_shape = img_attn.shape

            # 量化所有分片
            attn_shards_fp8 = []
            attn_scales = []
            for i in range(world_size):
                attn_fp8, attn_scale = quant_fp8_vllm(attn_shards[i].reshape(-1, original_shape[-1]))
                attn_shards_fp8.append(attn_fp8)
                attn_scales.append(attn_scale)

            # 准备接收缓冲区 (FP8 + scale)
            gathered_attn_fp8 = [None] * world_size
            gathered_attn_scales = [None] * world_size
            for target_rank in range(world_size):
                if target_rank != cur_rank:
                    gathered_attn_fp8[target_rank] = torch.empty_like(attn_shards_fp8[target_rank])
                    gathered_attn_scales[target_rank] = torch.empty_like(attn_scales[target_rank])
                else:
                    gathered_attn_fp8[cur_rank] = attn_shards_fp8[cur_rank]
                    gathered_attn_scales[cur_rank] = attn_scales[cur_rank]

            # 通信 FP8 数据和 scales
            for target_rank in range(world_size):
                target_global_rank = cfg_p_group_index * world_size + target_rank
                if target_rank != cur_rank:
                    if cur_rank < target_rank:
                        send_attn_fp8_req = dist.isend(attn_shards_fp8[target_rank], dst=target_global_rank, group=seq_p_group)
                        send_attn_scale_req = dist.isend(attn_scales[target_rank], dst=target_global_rank, group=seq_p_group)
                        recv_attn_fp8_req = dist.irecv(gathered_attn_fp8[target_rank], src=target_global_rank, group=seq_p_group)
                        recv_attn_scale_req = dist.irecv(gathered_attn_scales[target_rank], src=target_global_rank, group=seq_p_group)
                    else:
                        recv_attn_fp8_req = dist.irecv(gathered_attn_fp8[target_rank], src=target_global_rank, group=seq_p_group)
                        recv_attn_scale_req = dist.irecv(gathered_attn_scales[target_rank], src=target_global_rank, group=seq_p_group)
                        send_attn_fp8_req = dist.isend(attn_shards_fp8[target_rank], dst=target_global_rank, group=seq_p_group)
                        send_attn_scale_req = dist.isend(attn_scales[target_rank], dst=target_global_rank, group=seq_p_group)
                    send_attn_fp8_req.wait()
                    send_attn_scale_req.wait()
                    recv_attn_fp8_req.wait()
                    recv_attn_scale_req.wait()

            # 反量化
            gathered_attn_shards = []
            for i in range(world_size):
                attn_shards_new = dequant_fp8_vllm(gathered_attn_fp8[i], gathered_attn_scales[i], original_dtype).reshape(-1, shard_heads, hidden_dims)
                gathered_attn_shards.append(attn_shards_new)
        else:
            # 准备接收缓冲区
            gathered_attn_shards = [None] * world_size
            for target_rank in range(world_size):
                if target_rank != cur_rank:
                    gathered_attn_shards[target_rank] = torch.empty_like(attn_shards[target_rank])
                else:
                    gathered_attn_shards[cur_rank] = attn_shards[cur_rank]

            # 通信
            for target_rank in range(world_size):
                target_global_rank = cfg_p_group_index * world_size + target_rank
                if target_rank != cur_rank:
                    if cur_rank < target_rank:
                        send_req = dist.isend(attn_shards[target_rank], dst=target_global_rank, group=seq_p_group)
                        recv_req = dist.irecv(gathered_attn_shards[target_rank], src=target_global_rank, group=seq_p_group)
                    else:
                        recv_req = dist.irecv(gathered_attn_shards[target_rank], src=target_global_rank, group=seq_p_group)
                        send_req = dist.isend(attn_shards[target_rank], dst=target_global_rank, group=seq_p_group)
                    send_req.wait()
                    recv_req.wait()

        # 拼接所有分片 (在头维度上)
        img_attn = torch.cat(gathered_attn_shards, dim=1)
        img_attn = img_attn.reshape(shard_seqlen, -1)  # 重塑为 [shard_seqlen, -1] 形状

        return img_attn
