import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.quant_utils import dequant_fp8_vllm, quant_fp8_vllm
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate
from .utils.all2all import all2all_head2seq


@ATTN_WEIGHT_REGISTER("ulysses")
class UlyssesAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, slice_qkv_len, cu_seqlens_qkv, attention_module=None, seq_p_group=None, model_cls=None, use_fp8_comm=False, enable_head_parallel=False, img_first=True):
        """
        执行 Ulysses 注意力机制，结合图像和文本的查询、键和值。

        参数:
            q (torch.Tensor): 查询张量，形状为 [shard_seqlen, heads, hidden_dims]
            k (torch.Tensor): 键张量，形状为 [shard_seqlen, heads, hidden_dims]
            v (torch.Tensor): 值张量，形状为 [shard_seqlen, heads, hidden_dims]
            slice_qkv_len (int): 图像或者文本查询、键和值的长度，根据img_first确定谁在前半部分
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
        if img_first:
            img_qkv_len = slice_qkv_len
            if len(cu_seqlens_qkv) == 3:
                txt_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len  # 文本查询、键和值的长度
                txt_mask_len = cu_seqlens_qkv[2] - slice_qkv_len  # 文本掩码长度
            elif len(cu_seqlens_qkv) == 2:
                txt_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len  # 文本查询、键和值的长度
                txt_mask_len = None
        else:
            # assert len(cu_seqlens_qkv) == 2
            txt_qkv_len = slice_qkv_len
            img_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len
            txt_mask_len = None

        # 获取查询张量的头数和隐藏维度
        _, heads, hidden_dims = q.shape
        shard_heads = heads // world_size  # 每个进程处理的头数
        shard_seqlen = img_qkv_len  # 每个进程处理的序列长度
        global_img_seqlen = shard_seqlen * world_size  # 全局序列长度

        # 初始化累积序列长度张量
        cu_seqlens_qkv = torch.zeros([2], dtype=torch.int32)
        s = txt_qkv_len + global_img_seqlen  # 计算文本和图像的总长度
        s1 = s  # 当前样本的结束位置
        cu_seqlens_qkv[1] = s1  # 设置累积序列长度
        if txt_mask_len:
            s2 = txt_mask_len + global_img_seqlen  # 文本掩码的结束位置
            cu_seqlens_qkv = torch.cat((cu_seqlens_qkv, torch.tensor([s2], dtype=torch.int32)))
        max_seqlen_qkv = global_img_seqlen + txt_qkv_len  # 最大序列长度

        # 分割图像和文本的查询、键和值
        if img_first:
            img_q, img_k, img_v = q[:img_qkv_len, :, :].contiguous(), k[:img_qkv_len, :, :].contiguous(), v[:img_qkv_len, :, :].contiguous()
            txt_q, txt_k, txt_v = q[img_qkv_len:, :, :].contiguous(), k[img_qkv_len:, :, :].contiguous(), v[img_qkv_len:, :, :].contiguous()
        else:
            txt_q, txt_k, txt_v = q[:txt_qkv_len, :, :].contiguous(), k[:txt_qkv_len, :, :].contiguous(), v[:txt_qkv_len, :, :].contiguous()
            img_q, img_k, img_v = q[txt_qkv_len:, :, :].contiguous(), k[txt_qkv_len:, :, :].contiguous(), v[txt_qkv_len:, :, :].contiguous()

        img_qkv = torch.stack([img_q, img_k, img_v], dim=0).reshape(3, img_qkv_len, world_size, shard_heads, hidden_dims)
        original_dtype = img_qkv.dtype

        if enable_head_parallel:
            img_qkv = img_qkv.permute(3, 2, 1, 0, 4).contiguous()  # (shard_heads, world_size, img_qkv_len, 3, hidden_dims)
            output_qkv = torch.empty_like(img_qkv)

            # 通信图像的查询、键和值
            if use_fp8_comm:
                img_qkv_fp8, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                img_qkv_fp8 = img_qkv_fp8.reshape(shard_heads, world_size, img_qkv_len, 3, hidden_dims)
                img_qkv_scale = img_qkv_scale.reshape(shard_heads, world_size, img_qkv_len, 3, 1)
                output_qkv_fp8 = torch.empty_like(img_qkv_fp8)
                output_qkv_scale = torch.empty_like(img_qkv_scale)
                comm_fp8_works = []
                comm_scale_works = []
                for h in range(shard_heads):
                    work_fp8 = dist.all_to_all_single(output_qkv_fp8[h], img_qkv_fp8[h], group=seq_p_group, async_op=True)
                    work_scale = dist.all_to_all_single(output_qkv_scale[h], img_qkv_scale[h], group=seq_p_group, async_op=True)
                    comm_fp8_works.append(work_fp8)
                    comm_scale_works.append(work_scale)
            else:
                comm_works = []
                for h in range(shard_heads):
                    work = dist.all_to_all_single(output_qkv[h], img_qkv[h], group=seq_p_group, async_op=True)
                    comm_works.append(work)

            # 逐个head完成Attention计算
            single_head = 1
            head_attns = []
            for h in range(shard_heads):
                if use_fp8_comm:
                    comm_fp8_works[h].wait()
                    comm_scale_works[h].wait()
                    output_qkv[h] = dequant_fp8_vllm(output_qkv_fp8[h], output_qkv_scale[h], original_dtype)
                else:
                    comm_works[h].wait()

                qkv = output_qkv[h].reshape(global_img_seqlen, 3, single_head, hidden_dims).transpose(0, 1)
                shard_img_q = qkv[0]  # (global_img_seqlen, single_head, hidden_dims)
                shard_img_k = qkv[1]
                shard_img_v = qkv[2]

                # 处理文本的查询、键和值，选择当前进程的当前头
                shard_txt_q = txt_q[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                shard_txt_k = txt_k[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                shard_txt_v = txt_v[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]

                # 合并图像和文本的查询、键和值
                q = torch.cat((shard_img_q, shard_txt_q), dim=0)
                k = torch.cat((shard_img_k, shard_txt_k), dim=0)
                v = torch.cat((shard_img_v, shard_txt_v), dim=0)

                # 调用注意力函数计算注意力结果
                head_attn = attention_module.apply(
                    q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, model_cls=model_cls
                )
                head_attns.append(head_attn)

            # 合并当前进程的所有head的attn
            attn = torch.cat(head_attns, dim=1)

        else:
            img_qkv = img_qkv.permute(2, 1, 0, 3, 4).contiguous()  # (world_size, img_qkv_len, 3, shard_heads, hidden_dims)

            # 通信图像的查询、键和值
            if use_fp8_comm:
                img_qkv_fp8, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                img_qkv_fp8 = img_qkv_fp8.reshape(world_size, img_qkv_len, shard_heads, 3, hidden_dims)
                img_qkv_scale = img_qkv_scale.reshape(world_size, img_qkv_len, shard_heads, 3, 1)
                output_qkv_fp8 = torch.empty_like(img_qkv_fp8)
                output_qkv_scale = torch.empty_like(img_qkv_scale)
                dist.all_to_all_single(output_qkv_fp8, img_qkv_fp8, group=seq_p_group)
                dist.all_to_all_single(output_qkv_scale, img_qkv_scale, group=seq_p_group)
                output_qkv = dequant_fp8_vllm(output_qkv_fp8, output_qkv_scale, original_dtype)
            else:
                output_qkv = torch.empty_like(img_qkv)
                dist.all_to_all_single(output_qkv, img_qkv, group=seq_p_group)

            # 完成Attention计算
            qkv = output_qkv.reshape(global_img_seqlen, 3, shard_heads, hidden_dims).transpose(0, 1)
            shard_img_q = qkv[0]  # (global_img_seqlen, shard_head, hidden_dims)
            shard_img_k = qkv[1]
            shard_img_v = qkv[2]

            # 处理文本的查询、键和值，选择当前进程的当前头
            shard_txt_q = txt_q[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
            shard_txt_k = txt_k[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
            shard_txt_v = txt_v[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]

            # 合并图像和文本的查询、键和值
            q = torch.cat((shard_img_q, shard_txt_q), dim=0)
            k = torch.cat((shard_img_k, shard_txt_k), dim=0)
            v = torch.cat((shard_img_v, shard_txt_v), dim=0)

            # 调用注意力函数计算注意力结果
            attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, model_cls=model_cls)

        # 分割图像和文本的注意力结果
        if img_first:
            img_attn, txt_attn = attn[:global_img_seqlen, :], attn[global_img_seqlen:]
        else:
            txt_attn, img_attn = attn[:txt_qkv_len, :], attn[txt_qkv_len:]

        # 通信所有进程的图像注意力结果
        img_attn = self._reshape_img_attn(img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group, use_fp8_comm)

        # 收集所有进程的文本注意力结果
        gathered_txt_attn = [torch.empty_like(txt_attn) for _ in range(world_size)]
        dist.all_gather(gathered_txt_attn, txt_attn, group=seq_p_group)
        txt_attn = torch.cat(gathered_txt_attn, dim=1)  # 合并所有进程的文本注意力结果

        # 合并图像和文本的注意力结果
        if img_first:
            attn = torch.cat([img_attn, txt_attn], dim=0)
        else:
            attn = torch.cat([txt_attn, img_attn], dim=0)

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
        self.rounds = []

    def generate_round_robin_pairs(self, seq_p_group=None):
        """
        生成循环赛配对表，并确保每个配对中的第一个元素小于第二个
        这样我们可以用简单的规则确定通信顺序
        """
        cur_rank = dist.get_rank(seq_p_group)
        world_size = dist.get_world_size(seq_p_group)
        if world_size % 2 != 0:
            raise ValueError("world_size必须是偶数，奇数情况需要特殊处理")

        teams = list(range(world_size))
        for _ in range(world_size - 1):
            round_schedule = {}
            for i in range(world_size // 2):
                team1, team2 = teams[i], teams[world_size - 1 - i]
                smaller, larger = min(team1, team2), max(team1, team2)
                round_schedule[smaller] = (larger, True)
                round_schedule[larger] = (smaller, False)
            self.rounds.append(round_schedule)
            # 旋转列表（固定第一个元素）
            teams = [teams[0]] + [teams[-1]] + teams[1:-1]

        # if cur_rank == 0:
        #    self.print_pairing_schedule(seq_p_group)

    def print_pairing_schedule(self, seq_p_group):
        """打印通信调度表"""
        world_size = dist.get_world_size(seq_p_group)
        logger.info("循环赛通信调度表:")
        logger.info("=" * 50)
        for i, round_schedule in enumerate(self.rounds):
            logger.info(f"第 {i + 1} 轮:")
            for cur_rank in range(world_size):
                partner, is_smaller_in_pair = round_schedule[cur_rank]
                logger.info(f"  进程 {cur_rank} ←→ 进程 {partner}")
        logger.info("=" * 50)

    def load_balanced_all_to_all(self, shards, seq_p_group=None):
        """
        负载均衡all-to-all通信实现
        """
        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)
        global_rank = dist.get_global_rank(seq_p_group, cur_rank)
        cfg_p_group_index = global_rank // world_size

        # 准备接收缓冲区
        gathered_shards = [None] * world_size
        for target_rank in range(world_size):
            if target_rank != cur_rank:
                gathered_shards[target_rank] = torch.empty_like(shards[target_rank])
            else:
                gathered_shards[cur_rank] = shards[cur_rank]

        for i, round_schedule in enumerate(self.rounds):
            # 查找当前进程在本轮的配对
            partner = None
            is_smaller_in_pair = False
            if cur_rank in round_schedule:
                partner, is_smaller_in_pair = round_schedule[cur_rank]

            # 如果没有找到配对，说明本轮当前进程空闲
            if partner is None:
                continue

            # 计算全局rank
            partner_global_rank = cfg_p_group_index * world_size + partner

            if is_smaller_in_pair:
                # 当前进程是配对中的较小者，先发送后接收
                send_req = dist.isend(shards[partner], dst=partner_global_rank, group=seq_p_group)
                recv_req = dist.irecv(gathered_shards[partner], src=partner_global_rank, group=seq_p_group)
                send_req.wait()
                recv_req.wait()
            else:
                # 当前进程是配对中的较大者，先接收后发送
                recv_req = dist.irecv(gathered_shards[partner], src=partner_global_rank, group=seq_p_group)
                send_req = dist.isend(shards[partner], dst=partner_global_rank, group=seq_p_group)
                recv_req.wait()
                send_req.wait()

        return gathered_shards

    def apply(self, q, k, v, slice_qkv_len, cu_seqlens_qkv, attention_module=None, seq_p_group=None, model_cls=None, use_fp8_comm=False, enable_head_parallel=False, img_first=True):
        """
        执行 Ulysses 注意力机制，结合图像和文本的查询、键和值。

        参数:
            q (torch.Tensor): 查询张量，形状为 [shard_seqlen, heads, hidden_dims]
            k (torch.Tensor): 键张量，形状为 [shard_seqlen, heads, hidden_dims]
            v (torch.Tensor): 值张量，形状为 [shard_seqlen, heads, hidden_dims]
            slice_qkv_len (int): 图像或者文本查询、键和值的长度，根据img_first确定谁在前半部分
            cu_seqlens_qkv (torch.Tensor): 累积序列长度，包含文本和图像的长度信息
            attention_type (str): 注意力类型，默认为 "flash_attn2"

        返回:
            torch.Tensor: 计算得到的注意力结果
        """
        assert not enable_head_parallel, "Ulysses-4090 can't support head parallel mode."

        if len(self.rounds) == 0:
            self.generate_round_robin_pairs(seq_p_group)

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
        if img_first:
            img_qkv_len = slice_qkv_len
            if len(cu_seqlens_qkv) == 3:
                txt_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len  # 文本查询、键和值的长度
                txt_mask_len = cu_seqlens_qkv[2] - slice_qkv_len  # 文本掩码长度
            elif len(cu_seqlens_qkv) == 2:
                txt_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len  # 文本查询、键和值的长度
                txt_mask_len = None
        else:
            # assert len(cu_seqlens_qkv) == 2
            txt_qkv_len = slice_qkv_len
            img_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len
            txt_mask_len = None

        # 获取查询张量的头数和隐藏维度
        _, heads, hidden_dims = q.shape
        shard_heads = heads // world_size  # 每个进程处理的头数
        shard_seqlen = img_qkv_len  # 每个进程处理的序列长度

        # 分割图像和文本的查询、键和值
        if img_first:
            img_q, img_k, img_v = q[:img_qkv_len, :, :].contiguous(), k[:img_qkv_len, :, :].contiguous(), v[:img_qkv_len, :, :].contiguous()
            txt_q, txt_k, txt_v = q[img_qkv_len:, :, :].contiguous(), k[img_qkv_len:, :, :].contiguous(), v[img_qkv_len:, :, :].contiguous()
        else:
            txt_q, txt_k, txt_v = q[:txt_qkv_len, :, :].contiguous(), k[:txt_qkv_len, :, :].contiguous(), v[:txt_qkv_len, :, :].contiguous()
            img_q, img_k, img_v = q[txt_qkv_len:, :, :].contiguous(), k[txt_qkv_len:, :, :].contiguous(), v[txt_qkv_len:, :, :].contiguous()

        # 计算每个进程应该持有的头数分片
        num_heads = img_q.shape[1]
        shard_heads = num_heads // world_size

        # 将 image QKV 拼接后，按头维度切分成 N 份,每份大小为 D/N
        img_qkv = torch.stack([img_q, img_k, img_v], dim=0)
        qkv_shards = [img_qkv[:, :, i * shard_heads : (i + 1) * shard_heads, :].contiguous() for i in range(world_size)]
        qkv_dtype = img_qkv.dtype

        if use_fp8_comm:
            qkv_fp8_byte_tensors = []
            qkv_fp8_bytes = 0
            qkv_fp8_dtype = None
            qkv_scale_dtype = None
            for i in range(world_size):
                qkv_fp8, qkv_scale = quant_fp8_vllm(qkv_shards[i].reshape(-1, hidden_dims))
                if i == 0:
                    qkv_fp8_bytes = qkv_fp8.numel() * qkv_fp8.element_size()
                    qkv_fp8_dtype = qkv_fp8.dtype
                    qkv_scale_dtype = qkv_scale.dtype
                qkv_fp8_byte_tensors.append(torch.cat([qkv_fp8.contiguous().reshape(-1).view(torch.uint8), qkv_scale.contiguous().reshape(-1).view(torch.uint8)], dim=0))

            gathered_qkv_fp8_byte_tensors = self.load_balanced_all_to_all(qkv_fp8_byte_tensors, seq_p_group)

            gathered_q_shards = []
            gathered_k_shards = []
            gathered_v_shards = []
            for i in range(world_size):
                qkv_fp8_byte_tensor = gathered_qkv_fp8_byte_tensors[i]
                qkv_fp8 = qkv_fp8_byte_tensor[:qkv_fp8_bytes].view(qkv_fp8_dtype).reshape(3, -1, hidden_dims)
                qkv_scale = qkv_fp8_byte_tensor[qkv_fp8_bytes:].view(qkv_scale_dtype).reshape(3, -1, 1)
                q_shards_new = dequant_fp8_vllm(qkv_fp8[0], qkv_scale[0], qkv_dtype).reshape(-1, shard_heads, hidden_dims)
                k_shards_new = dequant_fp8_vllm(qkv_fp8[1], qkv_scale[1], qkv_dtype).reshape(-1, shard_heads, hidden_dims)
                v_shards_new = dequant_fp8_vllm(qkv_fp8[2], qkv_scale[2], qkv_dtype).reshape(-1, shard_heads, hidden_dims)
                gathered_q_shards.append(q_shards_new)
                gathered_k_shards.append(k_shards_new)
                gathered_v_shards.append(v_shards_new)
        else:
            gathered_qkv_byte_tensors = self.load_balanced_all_to_all(qkv_shards, seq_p_group)

            gathered_q_shards = []
            gathered_k_shards = []
            gathered_v_shards = []
            for i in range(world_size):
                qkv_tensor = gathered_qkv_byte_tensors[i].view(qkv_dtype).reshape(3, -1, shard_heads, hidden_dims)
                gathered_q_shards.append(qkv_tensor[0])
                gathered_k_shards.append(qkv_tensor[1])
                gathered_v_shards.append(qkv_tensor[2])

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
        if img_first:
            q = torch.cat((img_q, txt_q), dim=0)
            k = torch.cat((img_k, txt_k), dim=0)
            v = torch.cat((img_v, txt_v), dim=0)
        else:
            q = torch.cat((txt_q, img_q), dim=0)
            k = torch.cat((txt_k, img_k), dim=0)
            v = torch.cat((txt_v, img_v), dim=0)

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
        if img_first:
            img_attn, txt_attn = attn[: img_q.shape[0], :], attn[img_q.shape[0] :,]
        else:
            txt_attn, img_attn = attn[: txt_q.shape[0], :], attn[txt_q.shape[0] :,]

        # 收集所有进程的文本注意力结果
        gathered_txt_attn = [torch.empty_like(txt_attn) for _ in range(world_size)]
        dist.all_gather(gathered_txt_attn, txt_attn, group=seq_p_group)

        img_attn = self._reshape_img_attn(img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group, use_fp8_comm)

        txt_attn = torch.cat(gathered_txt_attn, dim=1)  # 合并所有进程的文本注意力结果

        # 合并图像和文本的注意力结果
        if img_first:
            attn = torch.cat([img_attn, txt_attn], dim=0)
        else:
            attn = torch.cat([txt_attn, img_attn], dim=0)

        return attn  # 返回最终的注意力结果

    @torch.compiler.disable
    def _reshape_img_attn(self, img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group, use_fp8_comm):
        cur_rank = dist.get_rank(seq_p_group)
        global_world_size = dist.get_world_size()
        global_rank = dist.get_global_rank(seq_p_group, cur_rank)
        cfg_p_group_index = global_rank // world_size

        img_attn = img_attn.reshape(world_size * shard_seqlen, shard_heads, hidden_dims)  # 重塑图像注意力结果
        attn_dtype = img_attn.dtype

        # 按序列维度切分成 N 份
        attn_shards = [img_attn[i * shard_seqlen : (i + 1) * shard_seqlen, :, :].contiguous() for i in range(world_size)]

        if use_fp8_comm:
            attn_fp8_byte_tensors = []
            attn_fp8_bytes = 0
            attn_fp8_dtype = None
            attn_scale_dtype = None
            for i in range(world_size):
                attn_fp8, attn_scale = quant_fp8_vllm(attn_shards[i].reshape(-1, hidden_dims))
                if i == 0:
                    attn_fp8_bytes = attn_fp8.numel() * attn_fp8.element_size()
                    attn_fp8_dtype = attn_fp8.dtype
                    attn_scale_dtype = attn_scale.dtype
                attn_fp8_byte_tensors.append(torch.cat([attn_fp8.contiguous().reshape(-1).view(torch.uint8), attn_scale.contiguous().reshape(-1).view(torch.uint8)], dim=0))

            gathered_attn_fp8_byte_tensors = self.load_balanced_all_to_all(attn_fp8_byte_tensors, seq_p_group)

            gathered_attn_shards = []
            for i in range(world_size):
                attn_fp8_byte_tensor = gathered_attn_fp8_byte_tensors[i]
                attn_fp8 = attn_fp8_byte_tensor[:attn_fp8_bytes].view(attn_fp8_dtype).reshape(-1, hidden_dims)
                attn_scale = attn_fp8_byte_tensor[attn_fp8_bytes:].view(attn_scale_dtype).reshape(-1, 1)
                attn_shards_new = dequant_fp8_vllm(attn_fp8, attn_scale, attn_dtype).reshape(-1, shard_heads, hidden_dims)
                gathered_attn_shards.append(attn_shards_new)

        else:
            gathered_attn_shards = self.load_balanced_all_to_all(attn_shards, seq_p_group)

        # 拼接所有分片 (在头维度上)
        img_attn = torch.cat(gathered_attn_shards, dim=1)
        img_attn = img_attn.reshape(shard_seqlen, -1)  # 重塑为 [shard_seqlen, -1] 形状

        return img_attn
