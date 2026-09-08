import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import MiniMaxH3VAEDistillationCapability
from lightx2v_train.model_zoo.native.minimax_h3.pruned_vae import MiniMaxH3PrunedVideoVAE
from tests.test_minimax_h3_pruned_vae import tiny_config, tiny_teacher


class _WindowModel:
    def __init__(self, student):
        self.student = student

    def student_decode_window(self, window, dtype, *, return_features):
        return self.student(window, return_features=return_features)


def _window_loss(student, windows, features):
    owner = _WindowModel(student)
    capability = MiniMaxH3VAEDistillationCapability(owner, {
        "student_tile_batch_size": 1,
        "student_window_checkpointing": True,
        "feature_weight": float(features),
    })
    capability._test_model_owner = owner
    raw, hidden = capability._decode_student_tiles(windows, torch.float32)
    return raw.square().mean() + 0.01 * sum(value.square().mean() for value in hidden)


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        for search in (True, False):
            torch.manual_seed(21)
            config = tiny_config()
            settings = {"search": {"group_size": 3, "lora_rank": 2, "gate_scale": 1.0}} if search else {"kept_layers": [1, 5]}
            student = MiniMaxH3PrunedVideoVAE(config, **settings)
            student.initialize_from_teacher(tiny_teacher(config))
            student.enable_gradient_checkpointing()
            reference = copy.deepcopy(student)
            distributed = DistributedDataParallel(student, find_unused_parameters=False)
            optimizer = torch.optim.SGD((p for p in student.parameters() if p.requires_grad), lr=1e-4)
            reference_optimizer = torch.optim.SGD((p for p in reference.parameters() if p.requires_grad), lr=1e-4)
            torch.manual_seed(42 + rank)
            for _ in range(2):
                for micro in range(2):
                    windows = torch.randn(2, 24, 2, 2, 2)
                    distributed.require_backward_grad_sync = micro == 1
                    if search:
                        student.prepare_search_step(1)
                        reference.decoder._search_noise = student.decoder._search_noise.clone()
                    (_window_loss(distributed, windows, not search) / 2).backward()
                    (_window_loss(reference, windows, not search) / 2).backward()
                for actual, expected in zip(student.parameters(), reference.parameters(), strict=True):
                    if actual.requires_grad:
                        assert actual.grad is not None and expected.grad is not None
                        dist.all_reduce(expected.grad)
                        expected.grad.div_(2)
                        torch.testing.assert_close(actual.grad, expected.grad)
                optimizer.step()
                reference_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
            del distributed
    finally:
        dist.destroy_process_group()


class PruningDistributedTest(unittest.TestCase):
    def test_search_and_recovery_multi_window_accumulation(self):
        with TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
