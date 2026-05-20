import torch
import torch.distributed as dist


def all2all(tensor, scatter_dim, gather_dim, cur_group, async_op):
    group_size = dist.get_world_size(group=cur_group)
    scatter_tensor_list = list(chunk.contiguous() for chunk in torch.chunk(tensor, chunks=group_size, dim=scatter_dim))
    gather_tensor_list = [torch.zeros_like(x) for x in scatter_tensor_list]
    comm = dist.all_to_all(gather_tensor_list, scatter_tensor_list, group=cur_group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            recieved_tensor = torch.cat(gather_tensor_list, dim=gather_dim).contiguous()
            return recieved_tensor

        return wait()
    recieved_tensor = torch.cat(gather_tensor_list, dim=gather_dim).contiguous()
    return recieved_tensor


def all_gather(tensor, gather_dim, cur_group, async_op):
    tensor = tensor.contiguous()
    group_size = dist.get_world_size(group=cur_group)
    gather_list = [torch.zeros_like(tensor) for _ in range(group_size)]
    comm = dist.all_gather(gather_list, tensor, group=cur_group, async_op=async_op)
    gather_tensor = torch.cat(gather_list, dim=gather_dim)
    if async_op:

        def wait():
            comm.wait()
            gather_tensor = torch.cat(gather_list, dim=gather_dim)
            return gather_tensor

        return wait()
    return gather_tensor


class _All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, scatter_dim, gather_dim, cur_group, async_op):
        ctx.cur_group = cur_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all2all(tensor=tensor, scatter_dim=scatter_dim, gather_dim=gather_dim, cur_group=cur_group, async_op=async_op)

    @staticmethod
    def backward(ctx, grad_outputs):
        input_t = grad_outputs
        return (all2all(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.cur_group, False), None, None, None, None)


class _Allgather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, gather_dim, cur_group, async_op):
        ctx.gather_dim = gather_dim
        ctx.cur_group = cur_group
        ctx.async_op = async_op
        return all_gather(tensor=tensor, gather_dim=gather_dim, cur_group=cur_group, async_op=async_op)

    @staticmethod
    def backward(ctx, grad_outputs):
        sp_group = ctx.cur_group
        sp_group_size = dist.get_world_size(group=sp_group)
        rank = dist.get_rank()
        rank_in_group = dist.get_group_rank(group=sp_group, global_rank=rank)
        return (grad_outputs.split(grad_outputs.shape[ctx.gather_dim] // sp_group_size, dim=ctx.gather_dim)[rank_in_group], None, None, None)
