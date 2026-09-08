import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import MiniMaxH3VAEDistillationCapability
from lightx2v_train.model_zoo.native.minimax_h3.pruned_vae import MiniMaxH3PrunedVideoVAE, decode_teacher_suffix
from tests.test_minimax_h3_pruned_vae import tiny_config, tiny_teacher


class _WindowModel:
    def __init__(self, student, teacher):
        self.student = student
        self.teacher = teacher

    def student_decode_window(self, window, dtype, *, return_features):
        return self.student(window, return_features=return_features)

    def student_decode_window_with_aux(self, window, dtype, *, auxiliary_feature_index, return_features):
        return self.student(window, return_features=return_features, auxiliary_feature_index=auxiliary_feature_index)

    def teacher_decode_suffix(self, tokens, shape, index, *, gradient_checkpointing):
        return decode_teacher_suffix(self.teacher.decoder, tokens, shape, index, gradient_checkpointing=gradient_checkpointing)


def _window_loss(owner, windows, *, auxiliary, features):
    capability = MiniMaxH3VAEDistillationCapability(owner, {
        "student_tile_batch_size": 1, "teacher_tile_batch_size": 1,
        "student_window_checkpointing": True,
        "feature_weight": float(features), "teacher_feature_indices": [2, 5],
    })
    result = capability._decode_student_tiles(windows, torch.float32, auxiliary_feature_index=0 if auxiliary else None)
    raw, hidden = result[:2]
    loss = raw.square().mean() + 0.01 * sum(value.square().mean() for value in hidden)
    if auxiliary:
        aux_raw = capability._decode_auxiliary_tiles(result[2], windows.shape[2:], 0)
        aux_loss, _ = capability._auxiliary_loss(aux_raw, torch.zeros_like(aux_raw))
        loss = loss + 0.1 * aux_loss
    return loss


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(23)
        config = tiny_config()
        teacher = tiny_teacher(config).requires_grad_(False).eval()
        student = MiniMaxH3PrunedVideoVAE(config, kept_layers=[1, 5])
        student.initialize_from_teacher(teacher)
        student.enable_gradient_checkpointing()
        reference = copy.deepcopy(student)
        distributed = DistributedDataParallel(student, find_unused_parameters=False)
        actual_owner = _WindowModel(distributed, teacher)
        expected_owner = _WindowModel(reference, teacher)
        optimizer = torch.optim.SGD(student.parameters(), lr=1e-4)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-4)
        torch.manual_seed(43 + rank)
        for auxiliary, features in ((False, True), (True, True), (True, False)):
            for micro in range(2):
                windows = torch.randn(2, 24, 2, 2, 2)
                distributed.require_backward_grad_sync = micro == 1
                (_window_loss(actual_owner, windows, auxiliary=auxiliary, features=features) / 2).backward()
                (_window_loss(expected_owner, windows, auxiliary=auxiliary, features=features) / 2).backward()
            for actual, expected in zip(student.parameters(), reference.parameters(), strict=True):
                assert actual.grad is not None and expected.grad is not None
                dist.all_reduce(expected.grad)
                expected.grad.div_(2)
                torch.testing.assert_close(actual.grad, expected.grad)
            assert all(parameter.grad is None for parameter in teacher.parameters())
            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        del distributed
    finally:
        dist.destroy_process_group()


class AuxiliaryDistributedTest(unittest.TestCase):
    def test_multi_window_gradient_accumulation_and_auxiliary_activation(self):
        with TemporaryDirectory() as directory:
            mp.spawn(_worker, args=((Path(directory) / "rendezvous").as_uri(),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
