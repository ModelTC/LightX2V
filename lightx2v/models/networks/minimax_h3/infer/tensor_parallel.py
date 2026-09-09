import torch
import torch.distributed as dist

from lightx2v.common.ops.mm.mm_weight import unwrap_tp_weight


def all_gather_last_dim(tensor, group, world_size):
    if world_size == 1:
        return tensor
    tensor = tensor.contiguous()
    input_shape = list(tensor.shape)
    gathered_shape = input_shape.copy()
    gathered_shape[0] *= world_size
    gathered = torch.empty(gathered_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(gathered, tensor, group=group)
    gathered = gathered.reshape([world_size] + input_shape)
    gathered = gathered.movedim(0, tensor.dim() - 1)
    output_shape = input_shape.copy()
    output_shape[-1] *= world_size
    return gathered.reshape(output_shape)


def row_parallel_linear(module, tensor, group, rank, world_size):
    if world_size == 1:
        return module.apply(tensor)

    # MMWeightTP keeps row-parallel bias outside its concrete backend. Run the
    # backend locally so quantization, diff, and LoRA are all included, then
    # place the replicated bias on rank 0 before the reduction to match H3's
    # reference operation order.
    concrete = unwrap_tp_weight(module)
    output = concrete.apply(tensor)
    if rank == 0:
        bias = concrete._get_actual_bias(module._row_split_bias)
        if bias is not None:
            output = output + bias
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
    return output


__all__ = ["all_gather_last_dim", "row_parallel_linear"]
