"""Four-rank NCCL smoke test for HunyuanImage3 phase-aware parallelism.

Run with::

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        -m test_cases.test_hunyuan_image3_phase_topology_distributed
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from lightx2v.models.networks.hunyuan_image3.parallel import build_hunyuan_image3_parallel_context


def _assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    config = {
        "parallel": {
            "phase_aware": True,
            "storage_tensor_p_size": 2,
            "ar": {"tensor_p_size": 4, "seq_p_size": 1},
            "denoise": {"tensor_p_size": 2, "seq_p_size": 2},
        }
    }
    context = build_hunyuan_image3_parallel_context(config)

    expected_logical_rank = (0, 2, 1, 3)[rank]
    _assert_equal(context.storage_tp_rank, rank % 2, "storage_tp_rank")
    _assert_equal(context.local_micro_shard_id, rank // 2, "local_micro_shard_id")
    _assert_equal(context.logical_tp_rank, rank % 2, "initial denoise logical_tp_rank")

    context.activate_phase("ar")
    _assert_equal(context.active_tp_size, 4, "AR TP size")
    _assert_equal(context.active_seq_size, 1, "AR SP size")
    _assert_equal(context.logical_tp_rank, expected_logical_rank, "AR logical TP rank")
    _assert_equal(context.logical_gather_order, (0, 2, 1, 3), "AR gather order")

    logical = torch.tensor([context.logical_tp_rank], device="cuda", dtype=torch.int64)
    gathered = [torch.empty_like(logical) for _ in range(4)]
    dist.all_gather(gathered, logical, group=context.active_tp_group)
    canonical = [int(gathered[index].item()) for index in context.logical_gather_order]
    _assert_equal(canonical, [0, 1, 2, 3], "canonical AR gather")

    context.activate_phase("denoise")
    _assert_equal(context.active_tp_size, 2, "denoise TP size")
    _assert_equal(context.active_seq_size, 2, "denoise SP size")

    # A stale rank-local phase must fail on every rank before any subset can
    # early-return while the rest enters a transition barrier.
    if rank == 0:
        context._phase = "ar"
    try:
        context.activate_phase("denoise")
    except RuntimeError as error:
        if "phase state diverged" not in str(error):
            raise
    else:
        raise AssertionError("Diverged phase state was not rejected on every rank.")
    context._phase = "denoise"

    invalid_target = "invalid" if rank == 0 else "denoise"
    try:
        context.activate_phase(invalid_target)
    except RuntimeError as error:
        if "phase state diverged" not in str(error):
            raise
    else:
        raise AssertionError("Invalid rank-local phase target was not rejected on every rank.")

    tp_value = torch.tensor([rank], device="cuda", dtype=torch.int64)
    dist.all_reduce(tp_value, group=context.active_tp_group)
    _assert_equal(int(tp_value.item()), (1, 1, 5, 5)[rank], "denoise TP pair sum")

    sp_value = torch.tensor([rank], device="cuda", dtype=torch.int64)
    dist.all_reduce(sp_value, group=context.active_seq_group)
    _assert_equal(int(sp_value.item()), (2, 4, 2, 4)[rank], "denoise SP pair sum")

    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "physical_to_logical": [0, 2, 1, 3],
                    "ar_tp": [0, 1, 2, 3],
                    "denoise_tp": [[0, 1], [2, 3]],
                    "denoise_sp": [[0, 2], [1, 3]],
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
