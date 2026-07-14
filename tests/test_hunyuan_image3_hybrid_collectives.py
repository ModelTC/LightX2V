import multiprocessing as mp
import queue
import time
import traceback
from datetime import timedelta
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist

_HYBRID_WORLD_SIZE = 4
_PROCESS_TIMEOUT_SECONDS = 60


def _hybrid_collectives_worker(rank, rendezvous_path, result_queue):
    error = None
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous_path}",
            rank=rank,
            world_size=_HYBRID_WORLD_SIZE,
            timeout=timedelta(seconds=30),
        )

        from torch.distributed.device_mesh import init_device_mesh

        from lightx2v.models.networks.hunyuan_image3.model import HunyuanImage3Model
        from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cfg_p", "seq_p"))
        seq_group = mesh.get_group(mesh_dim="seq_p")
        cfg_group = mesh.get_group(mesh_dim="cfg_p")

        # Row-major mesh: sequence groups are [0, 1] and [2, 3].
        rank_tensor = torch.tensor([rank], dtype=torch.int64)
        seq_ranks = [torch.empty_like(rank_tensor) for _ in range(2)]
        dist.all_gather(seq_ranks, rank_tensor, group=seq_group)
        expected_seq_ranks = [0, 1] if rank < 2 else [2, 3]
        assert [int(value.item()) for value in seq_ranks] == expected_seq_ranks

        # CFG groups are columns [0, 2] and [1, 3].
        cfg_ranks = [torch.empty_like(rank_tensor) for _ in range(2)]
        dist.all_gather(cfg_ranks, rank_tensor, group=cfg_group)
        expected_cfg_ranks = [rank % 2, rank % 2 + 2]
        assert [int(value.item()) for value in cfg_ranks] == expected_cfg_ranks

        # Exercise Hunyuan's real CFG gather/guidance path on each CFG column.
        model = HunyuanImage3Model.__new__(HunyuanImage3Model)
        model.config = {
            "cfg_parallel": True,
            "device_mesh": mesh,
            "sample_guide_scale": 5.0,
        }

        def fake_infer_branch(self, inputs, infer_condition=True):
            del self, inputs
            value = 10.0 if infer_condition else 2.0
            return {"diffusion_prediction": torch.tensor([value])}

        model._infer_cond_uncond = MethodType(fake_infer_branch, model)
        guided = model.infer({"_cfg_parallel_branch": True})["diffusion_prediction"]
        torch.testing.assert_close(guided, torch.tensor([42.0]))

        # Replicated runner state must span both mesh dimensions in hybrid mode.
        runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
        runner.config = {"seq_parallel": True, "cfg_parallel": True, "enable_cfg": True}
        runner.model = SimpleNamespace(seq_p_group=seq_group)
        assert runner._parallel_control_group() is dist.group.WORLD
        replicated = torch.tensor([rank], dtype=torch.int64)
        runner._broadcast_parallel_tensor(replicated)
        assert int(replicated.item()) == 0

        dist.barrier()
    except Exception:
        error = traceback.format_exc()
    finally:
        was_initialized = dist.is_initialized()
        if was_initialized:
            dist.destroy_process_group()
        result_queue.put({"rank": rank, "error": error, "destroyed": was_initialized and not dist.is_initialized()})


def test_hunyuan_cfg2_sp2_process_groups_guidance_and_control_sync(tmp_path):
    rendezvous_path = tmp_path / "hunyuan_cfg2_sp2_gloo_rendezvous"
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_hybrid_collectives_worker,
            args=(rank, str(rendezvous_path), result_queue),
        )
        for rank in range(_HYBRID_WORLD_SIZE)
    ]

    for process in processes:
        process.start()

    deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))

    timed_out = [process for process in processes if process.is_alive()]
    for process in timed_out:
        process.terminate()
    for process in timed_out:
        process.join(5)

    results = []
    for _ in range(_HYBRID_WORLD_SIZE):
        try:
            results.append(result_queue.get(timeout=5))
        except queue.Empty:
            break

    assert not timed_out, f"Distributed workers timed out: {[process.pid for process in timed_out]}"
    assert len(results) == _HYBRID_WORLD_SIZE, f"Expected {_HYBRID_WORLD_SIZE} worker results, got {results}"
    errors = [result for result in results if result["error"]]
    assert not errors, "\n".join(result["error"] for result in errors)
    assert all(result["destroyed"] for result in results)
    assert all(process.exitcode == 0 for process in processes)
