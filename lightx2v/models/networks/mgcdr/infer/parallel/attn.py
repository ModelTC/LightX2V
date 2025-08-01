import torch
import torch._dynamo as dynamo
import torch.distributed as dist
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v.common.ops.attn.attn_weight import AttnWeightTemplate
import xformers.ops
from loguru import logger

try:
    from flash_attn import flash_attn_func
except ImportError:
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")


def sequence_split_before_transformer_infer(x, group, dim):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    dim_size = x.shape[dim]
    
    splited_x_list = torch.split(x, dim_size//world_size, dim=dim)
    x = splited_x_list[rank].contiguous()
    
    return x
    

def sequence_gather_after_transformer_infer(x, group, dim):
    world_size = dist.get_world_size(group)
    
    x = x.contiguous()
    
    gatherd_x_buffers = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gatherd_x_buffers, x, group=group)
    
    x = torch.cat(gatherd_x_buffers, dim=dim).contiguous()
    
    return x
    

@dynamo.disable
def all2all_seq2head(input, group):
    world_size = dist.get_world_size(group)
    
    batch_size, shard_seq_len, heads, hidden_dims = input.shape
    seq_len = shard_seq_len * world_size
    shard_heads = heads // world_size
    
    # [batch_size, shard_seq_len, heads, hidden_dims]
    # -->
    # [batch_size, shard_seq_len, world_size, shard_heads, hidden_dims]
    # -->
    # [world_size, shard_seq_len, batch_size, shard_heads, hidden_dims]
    # -->
    # [world_size, shard_seq_len, shard_heads, batch_size, hidden_dims]
    # --> all2all
    input_t = (
        input.reshape(batch_size, shard_seq_len, world_size, shard_heads, hidden_dims)
        .transpose(0, 2)
        .transpose(2, 3)
        .contiguous()
    )
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)
    
    # [world_size, shard_seq_len, shard_heads, batch_size, hidden_dims]
    # -->
    # [world_size, shard_seq_len, batch_size, shard_heads, hidden_dims]
    # -->
    # [batch_size, shard_seq_len, world_size, shard_heads, hidden_dims]
    # -->
    # [batch_size, seq_len, shard_heads, hidden_dims]
    output = (
        output
        .transpose(2, 3)
        .transpose(0, 2)
        .reshape(batch_size, seq_len, shard_heads, hidden_dims)
    )
    return output
    
    
@dynamo.disable
def all2all_head2seq(input, group):
    world_size = dist.get_world_size(group)
    
    batch_size, seq_len, shard_heads, hidden_dims = input.shape
    heads = shard_heads * world_size
    shard_seq_len = seq_len // world_size
    
    # [batch_size, seq_len, shard_heads, hidden_dims]
    # -->
    # [batch_size, shard_seq_len, world_size, shard_heads, hidden_dims]
    # -->
    # [world_size, shard_seq_len, batch_size, shard_heads, hidden_dims]
    # -->
    # [world_size, shard_seq_len, shard_heads, batch_size, hidden_dims]
    input_t = (
        input.reshape(batch_size, shard_seq_len, world_size, shard_heads, hidden_dims)
        .transpose(0, 2)
        .transpose(2, 3)
        .contiguous()
    )
    
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)
    
    # [world_size, shard_seq_len, shard_heads, batch_size, hidden_dims]
    # -->
    # [world_size, shard_seq_len, batch_size, shard_heads, hidden_dims]
    # -->
    # [batch_size, shard_seq_len, world_size, shard_heads, hidden_dims]
    # -->
    # [batch_size, shard_seq_len, heads, hidden_dims]
    output = (
        output
        .transpose(2, 3)
        .transpose(0, 2)
        .reshape(batch_size, shard_seq_len, heads, hidden_dims)
    )
    return output


def all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()
    
    
class SequenceParallelAttnWeightTemplate(AttnWeightTemplate):
    def __init__(self):
        pass


@ATTN_WEIGHT_REGISTER("sp_naive")
class SequenceParallelAttn(SequenceParallelAttnWeightTemplate):
    def __init__(self):
        super().__init__()
        
    def apply(self, q, k, v, softmax_scale, is_causal, model_cls=None, sp_group: dist.distributed_c10d.ProcessGroup=None, scatter_dim=1, gather_dim=0):
        q = all_to_all(q, dist.get_world_size(sp_group), sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
        k = all_to_all(k, dist.get_world_size(sp_group), sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
        v = all_to_all(v, dist.get_world_size(sp_group), sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
        dtype = q.dtype
        q = q * softmax_scale
        attn = q @ k.transpose(-2, -1)  # translate attn to float32
        attn = attn.to(torch.float32)
        if is_causal:
            causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
            causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
            attn += causal_mask
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)  # cast back attn to original dtype
        x = attn @ v
        x = x.transpose(1, 2).contiguous()
        x = all_to_all(x, dist.get_world_size(sp_group), sp_group, scatter_dim=gather_dim, gather_dim=scatter_dim)
        return x
    
    
@ATTN_WEIGHT_REGISTER("sp_flash_attn2_base")
class FlashAttn2Weight(SequenceParallelAttnWeightTemplate):
    def __init__(self):
        super().__init__()

    def apply(self, q, k, v, dropout_p, softmax_scale, is_causal, model_cls=None, sp_group: dist.distributed_c10d.ProcessGroup=None, scatter_dim=2, gather_dim=1):
        q = all_to_all(q, dist.get_world_size(sp_group), sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
        k = all_to_all(k, dist.get_world_size(sp_group), sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
        v = all_to_all(v, dist.get_world_size(sp_group), sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
        if torch.distributed.get_rank() == 0:
            import pdb; pdb.set_trace()
        x = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=is_causal,
        )
        x = all_to_all(x, dist.get_world_size(sp_group), sp_group, scatter_dim=gather_dim, gather_dim=scatter_dim)
        return x
    
    
@ATTN_WEIGHT_REGISTER("sp_xformers")
class FlashAttn2Weight(SequenceParallelAttnWeightTemplate):
    def __init__(self):
        super().__init__()

    def apply(self, q, k, v, dropout_p, softmax_scale, attn_bias, model_cls=None, sp_group: dist.distributed_c10d.ProcessGroup=None, scatter_dim=2, gather_dim=1):
        q = all_to_all(q, dist.get_world_size(sp_group), sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
        k = all_to_all(k, dist.get_world_size(sp_group), sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
        v = all_to_all(v, dist.get_world_size(sp_group), sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
        x = xformers.ops.memory_efficient_attention(
            q, 
            k, 
            v, 
            p=dropout_p,
            attn_bias=attn_bias, 
            scale=softmax_scale
        )
        x = all_to_all(x, dist.get_world_size(sp_group), sp_group, scatter_dim=gather_dim, gather_dim=scatter_dim)
        return x