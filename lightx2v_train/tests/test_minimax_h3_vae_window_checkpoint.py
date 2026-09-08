import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import (
    MiniMaxH3VAEDistillationCapability,
)


class _Model:
    device = torch.device("cpu")

    def __init__(self, module):
        self.module = module
        self.batch_sizes = []

    def student_decode_window(self, inputs, running_dtype, *, return_features):
        self.batch_sizes.append(inputs.shape[0])
        value = self.module(inputs)
        return (value, (value * 2, value * 3)) if return_features else value


def _decode(model, inputs, checkpointing, features=False):
    capability = MiniMaxH3VAEDistillationCapability(model, {
        "student_tile_batch_size": 1 if checkpointing else None,
        "student_window_checkpointing": checkpointing,
        "feature_weight": float(features),
    })
    raw, hidden = capability._decode_student_tiles(inputs, torch.float32)
    loss = raw.square().mean() + sum(item.square().mean() for item in hidden)
    return raw, loss


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(21)
        layer = nn.Conv3d(1, 2, 1)
        reference = copy.deepcopy(layer)
        distributed = DistributedDataParallel(layer)
        model = _Model(distributed)
        inputs = torch.randn(4, 1, 2, 2, 2) + rank
        for micro in range(2):
            distributed.require_backward_grad_sync = micro == 1
            _, actual = _decode(model, inputs + micro, True)
            (actual / 2).backward()
            (reference(inputs + micro).square().mean() / 2).backward()
        for actual, expected in zip(layer.parameters(), reference.parameters(), strict=True):
            dist.all_reduce(expected.grad)
            expected.grad.div_(2)
            torch.testing.assert_close(actual.grad, expected.grad)
    finally:
        dist.destroy_process_group()


class WindowCheckpointTest(unittest.TestCase):
    def test_chunked_checkpoint_matches_batched_output_and_gradients(self):
        torch.set_num_threads(1)
        for features in (False, True):
            with self.subTest(features=features):
                torch.manual_seed(7)
                reference = _Model(nn.Conv3d(1, 2, 1))
                model = _Model(copy.deepcopy(reference.module))
                inputs = torch.randn(4, 1, 2, 3, 3)
                expected, reference_loss = _decode(reference, inputs, False, features)
                actual, loss = _decode(model, inputs, True, features)
                torch.testing.assert_close(actual, expected)
                reference_loss.backward()
                loss.backward()
                self.assertEqual(set(model.batch_sizes), {1})
                self.assertGreater(len(model.batch_sizes), 4)
                for parameter, expected_parameter in zip(
                    model.module.parameters(), reference.module.parameters(), strict=True,
                ):
                    torch.testing.assert_close(parameter.grad, expected_parameter.grad)

    def test_checkpoint_multiple_windows_with_ddp_gradient_accumulation(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(_distributed_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
