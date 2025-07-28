import torch
import torch.distributed as dist
from lightx2v.attentions import attention
from lightx2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.attentions.distributed.comm.all2all import all2all_seq2head, all2all_head2seq


class DistriFusionWanTransformerInferWrapper:
    def __init__(self, transformer_infer: WanTransformerInfer, config):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.transformer_infer = transformer_infer
        self.config = config
        
        self._wrap_attn(self.transformer_infer)

    def _wrap_attn(self, transformer_infer):
        transformer_infer.parallel_attention = self.parallel_attention

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self.transformer_weights, name)
        
    def parallel_attention(self, q, k, v, img_qkv_len, cu_seqlens_qkv, attention_type="flash_attn2"):
        # 获取查询张量的头数和隐藏维度
        _, heads, hidden_dims = q.shape
        shard_heads = heads // self.world_size  # 每个进程处理的头数
        shard_seqlen = img_qkv_len  # 每个进程处理的序列长度
        
        # 将图像的查询、键和值转换为头的格式
        q = all2all_seq2head(q)
        k = all2all_seq2head(k)
        v = all2all_seq2head(v)
        torch.cuda.synchronize()  # 确保CUDA操作完成
        
        # 初始化累积序列长度张量
        cu_seqlens_qkv = torch.zeros([2], dtype=torch.int32, device="cuda")
        s = q.shape[0]  # 计算文本和图像的总长度
        s1 = s  # 当前样本的结束位置
        cu_seqlens_qkv[1] = s1  # 设置累积序列长度
        max_seqlen_qkv = q.shape[0] # 最大序列长度

        # 调用注意力函数计算注意力结果
        attn = attention(attention_type=attention_type, q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv)
        
        # 处理图像注意力结果
        attn = attn.reshape(self.world_size * shard_seqlen, shard_heads, hidden_dims)  # 重塑图像注意力结果
        attn = all2all_head2seq(attn)  # 将头的格式转换回序列格式
        attn = attn.reshape(shard_seqlen, -1)  # 重塑为 [shard_seqlen, -1] 形状
        
        torch.cuda.synchronize()  # 确保CUDA操作完成
        
        return attn  # 返回最终的注意力结果