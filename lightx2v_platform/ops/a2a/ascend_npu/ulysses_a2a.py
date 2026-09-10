import torch
import torch.distributed as dist

from lightx2v_platform.registry_factory import PLATFORM_A2A_BACKEND_REGISTER


@PLATFORM_A2A_BACKEND_REGISTER("hccl_eager")
class HcclEagerUlyssesA2A:
    """HCCL A2A kept outside compiled graphs to retain fixed-size collectives."""

    @staticmethod
    @torch.compiler.disable
    def exchange(input_tensor, group=None, async_op=False):
        output_tensor = torch.empty_like(input_tensor)
        work = dist.all_to_all_single(output_tensor, input_tensor, group=group, async_op=async_op)
        return output_tensor, work
