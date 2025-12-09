import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.quant_utils import dequant_fp8_vllm, quant_fp8_vllm
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

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
        cu_seqlens_qkv = torch.zeros([2], dtype=torch.int32, device=self.config.get("run_device", "cuda"))
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
        self.device_synchronize()  # 确保CUDA操作完成
        return img_attn

    def device_synchronize(
        self,
    ):
        if torch.cuda.is_available():
            # no need to sync between comm and comp
            # torch.cuda.synchronize()
            self.config["run_device"] = "cuda"
        elif hasattr(torch, "mlu") and torch.mlu.is_available():
            torch.mlu.synchronize()
            self.config["run_device"] = "mlu"


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
        for round_num in range(world_size - 1):
            pairs = []

            for i in range(world_size // 2):
                team1 = teams[i]
                team2 = teams[world_size - 1 - i]
                # 确保配对中较小的进程在前
                if team1 < team2:
                    pairs.append((team1, team2))
                else:
                    pairs.append((team2, team1))

            self.rounds.append(pairs)

            # 旋转列表（固定第一个元素）
            teams = [teams[0]] + [teams[-1]] + teams[1:-1]

        # if cur_rank == 0:
        #    self.print_pairing_schedule()

    def print_pairing_schedule(self):
        """打印通信调度表"""
        logger.info("循环赛通信调度表:")
        logger.info("=" * 50)
        for i, round_pairs in enumerate(self.rounds):
            logger.info(f"第 {i + 1} 轮:")
            for src, dst in round_pairs:
                logger.info(f"  进程 {src} ←→ 进程 {dst}")
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

        for round_num, round_pairs in enumerate(self.rounds):
            # 查找当前进程在本轮的配对
            partner = None
            is_smaller_in_pair = False

            for smaller, larger in round_pairs:
                if cur_rank == smaller:
                    partner = larger
                    is_smaller_in_pair = True
                    break
                elif cur_rank == larger:
                    partner = smaller
                    is_smaller_in_pair = False
                    break

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
