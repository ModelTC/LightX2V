import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.quant_utils import dequant_fp8_vllm, quant_fp8_vllm
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

from .template import AttnWeightTemplate
from .utils.all2all import all2all_head2seq

try:
    from sageattn3_sparse import dequant_fp4 as dequant_fp4_sage3
    from sageattn3_sparse import quant_fp4 as quant_fp4_sage3
except ImportError:
    logger.info("sageattn3_sparse not found, to use quant_fp4 and dequant_fp4, please install sageattention sparse first")
    quant_fp4_sage3 = None
    dequant_fp4_sage3 = None


@ATTN_WEIGHT_REGISTER("ulysses")
class UlyssesAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        slice_qkv_len,
        cu_seqlens_qkv,
        attention_module=None,
        seq_p_group=None,
        use_fp8_comm=False,
        use_fp4_comm=False,
        use_tensor_fusion=False,
        enable_head_parallel=False,
        img_first=True,
        q_only_img=False,
        **kwargs,
    ):
        """
        执行 Ulysses 注意力机制，结合图像和文本的查询、键和值。

        参数:
            q (torch.Tensor): 查询张量，形状为 [shard_seqlen, q_heads, hidden_dims]。
                              若 q_only_img=True，则 q 只含图像 token，形状为 [img_shard_seqlen, q_heads, hidden_dims]
            k (torch.Tensor): 键张量，形状为 [shard_seqlen, kv_heads, hidden_dims]
            v (torch.Tensor): 值张量，形状为 [shard_seqlen, kv_heads, hidden_dims]
            slice_qkv_len (int): 图像或者文本查询、键和值的长度，根据 img_first 确定谁在前半部分
            cu_seqlens_qkv (torch.Tensor): 累积序列长度，包含文本和图像的长度信息
            q_only_img (bool): 若为 True，q 只含图像 token，k/v 同时含图像和文本 token。
                               此时只对 k/v 做 img/txt 分割，q 整体参与图像侧 all-to-all。
                               支持 cross-attention 等 q 不含文本 token 的场景。
            注意: q_heads != kv_heads（GQA）时会自动检测并分别处理，无需额外参数。
                 GQA 模式与 enable_head_parallel 不兼容。

        返回:
            torch.Tensor: 计算得到的注意力结果
        """
        assert not (use_fp8_comm and use_fp4_comm), "use_fp8_comm and use_fp4_comm can't be enabled at the same time."

        use_qkv_fusion = use_tensor_fusion

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
            txt_qkv_len = slice_qkv_len
            img_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len
            txt_mask_len = None

        # 分别获取 q 和 kv 的头数，支持 GQA（k/v 头数可能少于 q）
        _, q_heads, hidden_dims = q.shape
        _, kv_heads, _ = k.shape
        is_gqa = q_heads != kv_heads
        q_shard_heads = q_heads // world_size  # q 每个进程处理的头数
        kv_shard_heads = kv_heads // world_size  # k/v 每个进程处理的头数
        shard_heads = q_shard_heads  # 输出侧以 q 的头数为准

        # GQA 或 q_only_img 时 q 与 kv 形状不一致，无法做 tensor fusion
        if is_gqa or q_only_img:
            use_qkv_fusion = False

        shard_seqlen = img_qkv_len  # 每个进程处理的图像序列长度
        global_img_seqlen = shard_seqlen * world_size  # 全局图像序列长度

        # 重建 kv 侧累积序列长度（img + txt）
        cu_seqlens_kv = torch.zeros([2], dtype=torch.int32)
        cu_seqlens_kv[1] = txt_qkv_len + global_img_seqlen
        if txt_mask_len:
            cu_seqlens_kv = torch.cat((cu_seqlens_kv, torch.tensor([txt_mask_len + global_img_seqlen], dtype=torch.int32)))
        max_seqlen_kv = global_img_seqlen + txt_qkv_len

        # q_only_img 时 q 只含图像 token，cu_seqlens_q 与 kv 侧不同
        if q_only_img:
            cu_seqlens_q = torch.zeros([2], dtype=torch.int32)
            cu_seqlens_q[1] = global_img_seqlen
            max_seqlen_q = global_img_seqlen
        else:
            cu_seqlens_q = cu_seqlens_kv
            max_seqlen_q = max_seqlen_kv

        # 分割图像和文本的查询、键和值
        if q_only_img:
            # q 只含图像 token，无需分割；仅 k/v 需要拆出图像和文本部分
            img_q = q.contiguous()
            txt_q = None
            if img_first:
                img_k = k[:img_qkv_len, :, :].contiguous()
                img_v = v[:img_qkv_len, :, :].contiguous()
                txt_k = k[img_qkv_len:, :, :].contiguous()
                txt_v = v[img_qkv_len:, :, :].contiguous()
            else:
                txt_k = k[:txt_qkv_len, :, :].contiguous()
                txt_v = v[:txt_qkv_len, :, :].contiguous()
                img_k = k[txt_qkv_len:, :, :].contiguous()
                img_v = v[txt_qkv_len:, :, :].contiguous()
        else:
            if img_first:
                img_q = q[:img_qkv_len, :, :].contiguous()
                img_k = k[:img_qkv_len, :, :].contiguous()
                img_v = v[:img_qkv_len, :, :].contiguous()
                txt_q = q[img_qkv_len:, :, :].contiguous()
                txt_k = k[img_qkv_len:, :, :].contiguous()
                txt_v = v[img_qkv_len:, :, :].contiguous()
            else:
                txt_q = q[:txt_qkv_len, :, :].contiguous()
                txt_k = k[:txt_qkv_len, :, :].contiguous()
                txt_v = v[:txt_qkv_len, :, :].contiguous()
                img_q = q[txt_qkv_len:, :, :].contiguous()
                img_k = k[txt_qkv_len:, :, :].contiguous()
                img_v = v[txt_qkv_len:, :, :].contiguous()

        if use_qkv_fusion:
            # fusion 路径：q_shard_heads == kv_shard_heads（非 GQA、非 q_only_img 时才走此分支）
            img_qkv = torch.stack([img_q, img_k, img_v], dim=0).reshape(3, img_qkv_len, world_size, shard_heads, hidden_dims)
            original_dtype = img_qkv.dtype
        else:
            # 非 fusion：q 和 kv 分别 reshape，支持 GQA 下头数不同
            img_q = img_q.reshape(img_qkv_len, world_size, q_shard_heads, hidden_dims)
            img_k = img_k.reshape(img_qkv_len, world_size, kv_shard_heads, hidden_dims)
            img_v = img_v.reshape(img_qkv_len, world_size, kv_shard_heads, hidden_dims)
            original_dtype = img_q.dtype

        if enable_head_parallel:
            assert not is_gqa, "GQA（q_heads != kv_heads）暂不支持 enable_head_parallel 模式"
            # head_parallel 路径下 q_shard_heads == kv_shard_heads
            if use_qkv_fusion:
                img_qkv = img_qkv.permute(3, 2, 1, 0, 4).contiguous()  # (shard_heads, world_size, img_qkv_len, 3, hidden_dims)
                output_qkv = torch.empty_like(img_qkv)
                if use_fp8_comm or use_fp4_comm:
                    if use_fp8_comm:
                        img_qkv_quant, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                        img_qkv_quant = img_qkv_quant.reshape(shard_heads, world_size, img_qkv_len, 3, hidden_dims)
                        img_qkv_scale = img_qkv_scale.reshape(shard_heads, world_size, img_qkv_len, 3, 1)
                    else:
                        img_qkv_quant, img_qkv_scale = quant_fp4_sage3(img_qkv.reshape(1, 1, -1, hidden_dims))
                        img_qkv_quant = img_qkv_quant.reshape(shard_heads, world_size, img_qkv_len, 3, hidden_dims // 2)
                        img_qkv_scale = img_qkv_scale.reshape(shard_heads, world_size, img_qkv_len, 3, hidden_dims // 16)
                    output_qkv_quant = torch.empty_like(img_qkv_quant)
                    output_qkv_scale = torch.empty_like(img_qkv_scale)
                    comm_quant_works = []
                    comm_scale_works = []
                    for h in range(shard_heads):
                        work_quant = dist.all_to_all_single(output_qkv_quant[h], img_qkv_quant[h], group=seq_p_group, async_op=True)
                        work_scale = dist.all_to_all_single(output_qkv_scale[h], img_qkv_scale[h], group=seq_p_group, async_op=True)
                        comm_quant_works.append(work_quant)
                        comm_scale_works.append(work_scale)
                else:
                    comm_works = []
                    for h in range(shard_heads):
                        work = dist.all_to_all_single(output_qkv[h], img_qkv[h], group=seq_p_group, async_op=True)
                        comm_works.append(work)
            else:
                img_q = img_q.permute(2, 1, 0, 3).contiguous()  # (shard_heads, world_size, img_qkv_len, hidden_dims)
                img_k = img_k.permute(2, 1, 0, 3).contiguous()
                img_v = img_v.permute(2, 1, 0, 3).contiguous()
                output_q = torch.empty_like(img_q)
                output_k = torch.empty_like(img_k)
                output_v = torch.empty_like(img_v)

                if use_fp8_comm or use_fp4_comm:
                    if use_fp8_comm:
                        img_q_quant, img_q_scale = quant_fp8_vllm(img_q.reshape(-1, hidden_dims))
                        img_k_quant, img_k_scale = quant_fp8_vllm(img_k.reshape(-1, hidden_dims))
                        img_v_quant, img_v_scale = quant_fp8_vllm(img_v.reshape(-1, hidden_dims))
                        img_q_quant = img_q_quant.reshape(shard_heads, world_size, img_qkv_len, hidden_dims)
                        img_k_quant = img_k_quant.reshape(shard_heads, world_size, img_qkv_len, hidden_dims)
                        img_v_quant = img_v_quant.reshape(shard_heads, world_size, img_qkv_len, hidden_dims)
                        img_q_scale = img_q_scale.reshape(shard_heads, world_size, img_qkv_len, 1)
                        img_k_scale = img_k_scale.reshape(shard_heads, world_size, img_qkv_len, 1)
                        img_v_scale = img_v_scale.reshape(shard_heads, world_size, img_qkv_len, 1)
                    else:
                        img_q_quant, img_q_scale = quant_fp4_sage3(img_q.reshape(1, 1, -1, hidden_dims))
                        img_k_quant, img_k_scale = quant_fp4_sage3(img_k.reshape(1, 1, -1, hidden_dims))
                        img_v_quant, img_v_scale = quant_fp4_sage3(img_v.reshape(1, 1, -1, hidden_dims))
                        img_q_quant = img_q_quant.reshape(shard_heads, world_size, img_qkv_len, hidden_dims // 2)
                        img_k_quant = img_k_quant.reshape(shard_heads, world_size, img_qkv_len, hidden_dims // 2)
                        img_v_quant = img_v_quant.reshape(shard_heads, world_size, img_qkv_len, hidden_dims // 2)
                        img_q_scale = img_q_scale.reshape(shard_heads, world_size, img_qkv_len, hidden_dims // 16)
                        img_k_scale = img_k_scale.reshape(shard_heads, world_size, img_qkv_len, hidden_dims // 16)
                        img_v_scale = img_v_scale.reshape(shard_heads, world_size, img_qkv_len, hidden_dims // 16)
                    output_q_quant = torch.empty_like(img_q_quant)
                    output_k_quant = torch.empty_like(img_k_quant)
                    output_v_quant = torch.empty_like(img_v_quant)
                    output_q_scale = torch.empty_like(img_q_scale)
                    output_k_scale = torch.empty_like(img_k_scale)
                    output_v_scale = torch.empty_like(img_v_scale)
                    comm_quant_works = []
                    comm_scale_works = []
                    for h in range(shard_heads):
                        work_q_quant = dist.all_to_all_single(output_q_quant[h], img_q_quant[h], group=seq_p_group, async_op=True)
                        work_k_quant = dist.all_to_all_single(output_k_quant[h], img_k_quant[h], group=seq_p_group, async_op=True)
                        work_v_quant = dist.all_to_all_single(output_v_quant[h], img_v_quant[h], group=seq_p_group, async_op=True)
                        work_q_scale = dist.all_to_all_single(output_q_scale[h], img_q_scale[h], group=seq_p_group, async_op=True)
                        work_k_scale = dist.all_to_all_single(output_k_scale[h], img_k_scale[h], group=seq_p_group, async_op=True)
                        work_v_scale = dist.all_to_all_single(output_v_scale[h], img_v_scale[h], group=seq_p_group, async_op=True)
                        comm_quant_works.append(work_q_quant)
                        comm_quant_works.append(work_k_quant)
                        comm_quant_works.append(work_v_quant)
                        comm_scale_works.append(work_q_scale)
                        comm_scale_works.append(work_k_scale)
                        comm_scale_works.append(work_v_scale)
                else:
                    comm_works = []
                    for h in range(shard_heads):
                        work_q = dist.all_to_all_single(output_q[h], img_q[h], group=seq_p_group, async_op=True)
                        work_k = dist.all_to_all_single(output_k[h], img_k[h], group=seq_p_group, async_op=True)
                        work_v = dist.all_to_all_single(output_v[h], img_v[h], group=seq_p_group, async_op=True)
                        comm_works.append(work_q)
                        comm_works.append(work_k)
                        comm_works.append(work_v)

            # 逐个 head 完成 Attention 计算
            single_head = 1
            head_attns = []
            for h in range(shard_heads):
                if use_qkv_fusion:
                    if use_fp8_comm or use_fp4_comm:
                        comm_quant_works[h].wait()
                        comm_scale_works[h].wait()
                        if use_fp8_comm:
                            output_qkv[h] = dequant_fp8_vllm(output_qkv_quant[h], output_qkv_scale[h], original_dtype)
                        else:
                            output_qkv[h] = dequant_fp4_sage3(output_qkv_quant[h].reshape(1, 1, -1, hidden_dims // 2), output_qkv_scale[h].reshape(1, 1, -1, hidden_dims // 16)).reshape(
                                world_size, img_qkv_len, 3, hidden_dims
                            )
                    else:
                        comm_works[h].wait()

                    qkv = output_qkv[h].reshape(global_img_seqlen, 3, single_head, hidden_dims).transpose(0, 1)
                    shard_img_q = qkv[0]  # (global_img_seqlen, single_head, hidden_dims)
                    shard_img_k = qkv[1]
                    shard_img_v = qkv[2]
                else:
                    if use_fp8_comm or use_fp4_comm:
                        comm_quant_works[3 * h].wait()
                        comm_quant_works[3 * h + 1].wait()
                        comm_quant_works[3 * h + 2].wait()
                        comm_scale_works[3 * h].wait()
                        comm_scale_works[3 * h + 1].wait()
                        comm_scale_works[3 * h + 2].wait()
                        if use_fp8_comm:
                            output_q[h] = dequant_fp8_vllm(output_q_quant[h], output_q_scale[h], original_dtype)
                            output_k[h] = dequant_fp8_vllm(output_k_quant[h], output_k_scale[h], original_dtype)
                            output_v[h] = dequant_fp8_vllm(output_v_quant[h], output_v_scale[h], original_dtype)
                        else:
                            output_q[h] = dequant_fp4_sage3(output_q_quant[h].reshape(1, 1, -1, hidden_dims // 2), output_q_scale[h].reshape(1, 1, -1, hidden_dims // 16)).reshape(
                                world_size, img_qkv_len, hidden_dims
                            )
                            output_k[h] = dequant_fp4_sage3(output_k_quant[h].reshape(1, 1, -1, hidden_dims // 2), output_k_scale[h].reshape(1, 1, -1, hidden_dims // 16)).reshape(
                                world_size, img_qkv_len, hidden_dims
                            )
                            output_v[h] = dequant_fp4_sage3(output_v_quant[h].reshape(1, 1, -1, hidden_dims // 2), output_v_scale[h].reshape(1, 1, -1, hidden_dims // 16)).reshape(
                                world_size, img_qkv_len, hidden_dims
                            )
                    else:
                        comm_works[3 * h].wait()
                        comm_works[3 * h + 1].wait()
                        comm_works[3 * h + 2].wait()

                    shard_img_q = output_q[h].reshape(global_img_seqlen, single_head, hidden_dims)
                    shard_img_k = output_k[h].reshape(global_img_seqlen, single_head, hidden_dims)
                    shard_img_v = output_v[h].reshape(global_img_seqlen, single_head, hidden_dims)

                if q_only_img:
                    # q 只含图像 token，无 txt_q；k/v 需拼接图像和文本部分
                    shard_txt_k = txt_k[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                    shard_txt_v = txt_v[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                    q_h = shard_img_q
                    if img_first:
                        k_h = torch.cat((shard_img_k, shard_txt_k), dim=0)
                        v_h = torch.cat((shard_img_v, shard_txt_v), dim=0)
                    else:
                        k_h = torch.cat((shard_txt_k, shard_img_k), dim=0)
                        v_h = torch.cat((shard_txt_v, shard_img_v), dim=0)
                else:
                    # 处理文本的查询、键和值，选择当前进程的当前头
                    shard_txt_q = txt_q[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                    shard_txt_k = txt_k[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                    shard_txt_v = txt_v[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                    if img_first:
                        q_h = torch.cat((shard_img_q, shard_txt_q), dim=0)
                        k_h = torch.cat((shard_img_k, shard_txt_k), dim=0)
                        v_h = torch.cat((shard_img_v, shard_txt_v), dim=0)
                    else:
                        q_h = torch.cat((shard_txt_q, shard_img_q), dim=0)
                        k_h = torch.cat((shard_txt_k, shard_img_k), dim=0)
                        v_h = torch.cat((shard_txt_v, shard_img_v), dim=0)

                # 调用注意力函数计算注意力结果
                head_attn = attention_module.apply(q=q_h, k=k_h, v=v_h, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, **kwargs)
                head_attns.append(head_attn)

            # 合并当前进程的所有 head 的 attn
            attn = torch.cat(head_attns, dim=1)

        else:
            if use_qkv_fusion:
                img_qkv = img_qkv.permute(2, 1, 0, 3, 4).contiguous()  # (world_size, img_qkv_len, 3, shard_heads, hidden_dims)
            else:
                img_q = img_q.permute(1, 0, 2, 3).contiguous()  # (world_size, img_qkv_len, q_shard_heads, hidden_dims)
                img_k = img_k.permute(1, 0, 2, 3).contiguous()  # (world_size, img_qkv_len, kv_shard_heads, hidden_dims)
                img_v = img_v.permute(1, 0, 2, 3).contiguous()

            # 通信图像的查询、键和值
            if use_qkv_fusion:
                if use_fp8_comm or use_fp4_comm:
                    if use_fp8_comm:
                        img_qkv_quant, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                        img_qkv_quant = img_qkv_quant.reshape(world_size, img_qkv_len, shard_heads, 3, hidden_dims)
                        img_qkv_scale = img_qkv_scale.reshape(world_size, img_qkv_len, shard_heads, 3, 1)
                    else:
                        img_qkv_quant, img_qkv_scale = quant_fp4_sage3(img_qkv.reshape(1, 1, -1, hidden_dims))
                        img_qkv_quant = img_qkv_quant.reshape(world_size, img_qkv_len, shard_heads, 3, hidden_dims // 2)
                        img_qkv_scale = img_qkv_scale.reshape(world_size, img_qkv_len, shard_heads, 3, hidden_dims // 16)
                    output_qkv_quant = torch.empty_like(img_qkv_quant)
                    output_qkv_scale = torch.empty_like(img_qkv_scale)
                    dist.all_to_all_single(output_qkv_quant, img_qkv_quant, group=seq_p_group)
                    dist.all_to_all_single(output_qkv_scale, img_qkv_scale, group=seq_p_group)
                    if use_fp8_comm:
                        output_qkv = dequant_fp8_vllm(output_qkv_quant, output_qkv_scale, original_dtype)
                    else:
                        output_qkv = dequant_fp4_sage3(output_qkv_quant.reshape(1, 1, -1, hidden_dims // 2), output_qkv_scale.reshape(1, 1, -1, hidden_dims // 16))
                else:
                    output_qkv = torch.empty_like(img_qkv)
                    dist.all_to_all_single(output_qkv, img_qkv, group=seq_p_group)

                qkv = output_qkv.reshape(global_img_seqlen, 3, shard_heads, hidden_dims).transpose(0, 1)
                shard_img_q = qkv[0]  # (global_img_seqlen, shard_heads, hidden_dims)
                shard_img_k = qkv[1]
                shard_img_v = qkv[2]
            else:
                # 非 fusion 路径：q 与 kv 分别做 all-to-all，支持 GQA 下头数不同
                if use_fp8_comm or use_fp4_comm:
                    if use_fp8_comm:
                        img_q_quant, img_q_scale = quant_fp8_vllm(img_q.reshape(-1, hidden_dims))
                        img_k_quant, img_k_scale = quant_fp8_vllm(img_k.reshape(-1, hidden_dims))
                        img_v_quant, img_v_scale = quant_fp8_vllm(img_v.reshape(-1, hidden_dims))
                        img_q_quant = img_q_quant.reshape(world_size, img_qkv_len, q_shard_heads, hidden_dims)
                        img_k_quant = img_k_quant.reshape(world_size, img_qkv_len, kv_shard_heads, hidden_dims)
                        img_v_quant = img_v_quant.reshape(world_size, img_qkv_len, kv_shard_heads, hidden_dims)
                        img_q_scale = img_q_scale.reshape(world_size, img_qkv_len, q_shard_heads, 1)
                        img_k_scale = img_k_scale.reshape(world_size, img_qkv_len, kv_shard_heads, 1)
                        img_v_scale = img_v_scale.reshape(world_size, img_qkv_len, kv_shard_heads, 1)
                    else:
                        img_q_quant, img_q_scale = quant_fp4_sage3(img_q.reshape(1, 1, -1, hidden_dims))
                        img_k_quant, img_k_scale = quant_fp4_sage3(img_k.reshape(1, 1, -1, hidden_dims))
                        img_v_quant, img_v_scale = quant_fp4_sage3(img_v.reshape(1, 1, -1, hidden_dims))
                        img_q_quant = img_q_quant.reshape(world_size, img_qkv_len, q_shard_heads, hidden_dims // 2)
                        img_k_quant = img_k_quant.reshape(world_size, img_qkv_len, kv_shard_heads, hidden_dims // 2)
                        img_v_quant = img_v_quant.reshape(world_size, img_qkv_len, kv_shard_heads, hidden_dims // 2)
                        img_q_scale = img_q_scale.reshape(world_size, img_qkv_len, q_shard_heads, hidden_dims // 16)
                        img_k_scale = img_k_scale.reshape(world_size, img_qkv_len, kv_shard_heads, hidden_dims // 16)
                        img_v_scale = img_v_scale.reshape(world_size, img_qkv_len, kv_shard_heads, hidden_dims // 16)
                    output_q_quant = torch.empty_like(img_q_quant)
                    output_k_quant = torch.empty_like(img_k_quant)
                    output_v_quant = torch.empty_like(img_v_quant)
                    output_q_scale = torch.empty_like(img_q_scale)
                    output_k_scale = torch.empty_like(img_k_scale)
                    output_v_scale = torch.empty_like(img_v_scale)
                    dist.all_to_all_single(output_q_quant, img_q_quant, group=seq_p_group)
                    dist.all_to_all_single(output_k_quant, img_k_quant, group=seq_p_group)
                    dist.all_to_all_single(output_v_quant, img_v_quant, group=seq_p_group)
                    dist.all_to_all_single(output_q_scale, img_q_scale, group=seq_p_group)
                    dist.all_to_all_single(output_k_scale, img_k_scale, group=seq_p_group)
                    dist.all_to_all_single(output_v_scale, img_v_scale, group=seq_p_group)
                    if use_fp8_comm:
                        output_q = dequant_fp8_vllm(output_q_quant, output_q_scale, original_dtype)
                        output_k = dequant_fp8_vllm(output_k_quant, output_k_scale, original_dtype)
                        output_v = dequant_fp8_vllm(output_v_quant, output_v_scale, original_dtype)
                    else:
                        output_q = dequant_fp4_sage3(output_q_quant.reshape(1, 1, -1, hidden_dims // 2), output_q_scale.reshape(1, 1, -1, hidden_dims // 16))
                        output_k = dequant_fp4_sage3(output_k_quant.reshape(1, 1, -1, hidden_dims // 2), output_k_scale.reshape(1, 1, -1, hidden_dims // 16))
                        output_v = dequant_fp4_sage3(output_v_quant.reshape(1, 1, -1, hidden_dims // 2), output_v_scale.reshape(1, 1, -1, hidden_dims // 16))
                else:
                    output_q = torch.empty_like(img_q)
                    output_k = torch.empty_like(img_k)
                    output_v = torch.empty_like(img_v)
                    dist.all_to_all_single(output_q, img_q, group=seq_p_group)
                    dist.all_to_all_single(output_k, img_k, group=seq_p_group)
                    dist.all_to_all_single(output_v, img_v, group=seq_p_group)
                # q 与 kv 使用各自对应的 shard_heads 进行 reshape
                shard_img_q = output_q.reshape(global_img_seqlen, q_shard_heads, hidden_dims)
                shard_img_k = output_k.reshape(global_img_seqlen, kv_shard_heads, hidden_dims)
                shard_img_v = output_v.reshape(global_img_seqlen, kv_shard_heads, hidden_dims)

            if q_only_img:
                # q 只含图像 token：q 直接用图像侧结果，k/v 需拼接文本部分
                shard_txt_k = txt_k[:, cur_rank * kv_shard_heads : (cur_rank + 1) * kv_shard_heads, :]
                shard_txt_v = txt_v[:, cur_rank * kv_shard_heads : (cur_rank + 1) * kv_shard_heads, :]
                q = shard_img_q
                if img_first:
                    k = torch.cat((shard_img_k, shard_txt_k), dim=0)
                    v = torch.cat((shard_img_v, shard_txt_v), dim=0)
                else:
                    k = torch.cat((shard_txt_k, shard_img_k), dim=0)
                    v = torch.cat((shard_txt_v, shard_img_v), dim=0)
            else:
                # 处理文本的查询、键和值，选择当前进程的头（GQA 下 q 和 kv 使用各自的 shard_heads）
                shard_txt_q = txt_q[:, cur_rank * q_shard_heads : (cur_rank + 1) * q_shard_heads, :]
                shard_txt_k = txt_k[:, cur_rank * kv_shard_heads : (cur_rank + 1) * kv_shard_heads, :]
                shard_txt_v = txt_v[:, cur_rank * kv_shard_heads : (cur_rank + 1) * kv_shard_heads, :]
                if img_first:
                    q = torch.cat((shard_img_q, shard_txt_q), dim=0)
                    k = torch.cat((shard_img_k, shard_txt_k), dim=0)
                    v = torch.cat((shard_img_v, shard_txt_v), dim=0)
                else:
                    q = torch.cat((shard_txt_q, shard_img_q), dim=0)
                    k = torch.cat((shard_txt_k, shard_img_k), dim=0)
                    v = torch.cat((shard_txt_v, shard_img_v), dim=0)

            # 调用注意力函数计算注意力结果
            attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, **kwargs)

        if q_only_img:
            # q 只含图像 token：attn 全部是图像侧结果，无 txt_attn，直接还原通信格式
            img_attn = self._reshape_img_attn(attn, world_size, shard_seqlen, q_shard_heads, hidden_dims, seq_p_group, use_fp8_comm)
            return img_attn

        # 分割图像和文本的注意力结果
        if img_first:
            img_attn, txt_attn = attn[:global_img_seqlen, :], attn[global_img_seqlen:]
        else:
            txt_attn, img_attn = attn[:txt_qkv_len, :], attn[txt_qkv_len:]

        # 通信所有进程的图像注意力结果
        img_attn = self._reshape_img_attn(img_attn, world_size, shard_seqlen, q_shard_heads, hidden_dims, seq_p_group, use_fp8_comm)

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
            img_attn_quant, attn_scale = quant_fp8_vllm(img_attn.reshape(-1, original_shape[-1]))
            img_attn_quant = all2all_head2seq(img_attn_quant.reshape(original_shape), group=seq_p_group)
            attn_scale = all2all_head2seq(attn_scale.reshape(original_shape[0], original_shape[1], 1), group=seq_p_group)
            img_attn = dequant_fp8_vllm(img_attn_quant, attn_scale, original_dtype)
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

    def apply(
        self,
        q,
        k,
        v,
        slice_qkv_len,
        cu_seqlens_qkv,
        attention_module=None,
        seq_p_group=None,
        use_fp8_comm=False,
        use_fp4_comm=False,
        enable_head_parallel=False,
        img_first=True,
        **kwargs,
    ):
        """
        执行 Ulysses 注意力机制，结合图像和文本的查询、键和值。

        参数:
            q (torch.Tensor): 查询张量，形状为 [shard_seqlen, heads, hidden_dims]
            k (torch.Tensor): 键张量，形状为 [shard_seqlen, heads, hidden_dims]
            v (torch.Tensor): 值张量，形状为 [shard_seqlen, heads, hidden_dims]
            slice_qkv_len (int): 图像或者文本查询、键和值的长度，根据img_first确定谁在前半部分
            cu_seqlens_qkv (torch.Tensor): 累积序列长度，包含文本和图像的长度信息

        返回:
            torch.Tensor: 计算得到的注意力结果
        """
        assert not enable_head_parallel, "Ulysses-4090 can't support head parallel mode."
        assert not (use_fp8_comm and use_fp4_comm), "use_fp8_comm and use_fp4_comm can't be enabled at the same time."

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
            qkv_quant_byte_tensors = []
            qkv_quant_bytes = 0
            qkv_quant_dtype = None
            qkv_scale_dtype = None
            for i in range(world_size):
                qkv_quant, qkv_scale = quant_fp8_vllm(qkv_shards[i].reshape(-1, hidden_dims))
                if i == 0:
                    qkv_quant_bytes = qkv_quant.numel() * qkv_quant.element_size()
                    qkv_quant_dtype = qkv_quant.dtype
                    qkv_scale_dtype = qkv_scale.dtype
                qkv_quant_byte_tensors.append(torch.cat([qkv_quant.contiguous().reshape(-1).view(torch.uint8), qkv_scale.contiguous().reshape(-1).view(torch.uint8)], dim=0))

            gathered_qkv_quant_byte_tensors = self.load_balanced_all_to_all(qkv_quant_byte_tensors, seq_p_group)

            gathered_q_shards = []
            gathered_k_shards = []
            gathered_v_shards = []
            for i in range(world_size):
                qkv_quant_byte_tensor = gathered_qkv_quant_byte_tensors[i]
                qkv_quant = qkv_quant_byte_tensor[:qkv_quant_bytes].view(qkv_quant_dtype).reshape(3, -1, hidden_dims)
                qkv_scale = qkv_quant_byte_tensor[qkv_quant_bytes:].view(qkv_scale_dtype).reshape(3, -1, 1)
                q_shards_new = dequant_fp8_vllm(qkv_quant[0], qkv_scale[0], qkv_dtype).reshape(-1, shard_heads, hidden_dims)
                k_shards_new = dequant_fp8_vllm(qkv_quant[1], qkv_scale[1], qkv_dtype).reshape(-1, shard_heads, hidden_dims)
                v_shards_new = dequant_fp8_vllm(qkv_quant[2], qkv_scale[2], qkv_dtype).reshape(-1, shard_heads, hidden_dims)
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
        attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, **kwargs)

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
            attn_quant_byte_tensors = []
            attn_quant_bytes = 0
            attn_quant_dtype = None
            attn_scale_dtype = None
            for i in range(world_size):
                attn_quant, attn_scale = quant_fp8_vllm(attn_shards[i].reshape(-1, hidden_dims))
                if i == 0:
                    attn_quant_bytes = attn_quant.numel() * attn_quant.element_size()
                    attn_quant_dtype = attn_quant.dtype
                    attn_scale_dtype = attn_scale.dtype
                attn_quant_byte_tensors.append(torch.cat([attn_quant.contiguous().reshape(-1).view(torch.uint8), attn_scale.contiguous().reshape(-1).view(torch.uint8)], dim=0))

            gathered_attn_quant_byte_tensors = self.load_balanced_all_to_all(attn_quant_byte_tensors, seq_p_group)

            gathered_attn_shards = []
            for i in range(world_size):
                attn_quant_byte_tensor = gathered_attn_quant_byte_tensors[i]
                attn_quant = attn_quant_byte_tensor[:attn_quant_bytes].view(attn_quant_dtype).reshape(-1, hidden_dims)
                attn_scale = attn_quant_byte_tensor[attn_quant_bytes:].view(attn_scale_dtype).reshape(-1, 1)
                attn_shards_new = dequant_fp8_vllm(attn_quant, attn_scale, attn_dtype).reshape(-1, shard_heads, hidden_dims)
                gathered_attn_shards.append(attn_shards_new)

        else:
            gathered_attn_shards = self.load_balanced_all_to_all(attn_shards, seq_p_group)

        # 拼接所有分片 (在头维度上)
        img_attn = torch.cat(gathered_attn_shards, dim=1)
        img_attn = img_attn.reshape(shard_seqlen, -1)  # 重塑为 [shard_seqlen, -1] 形状

        return img_attn
