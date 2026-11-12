"""CPU checks for the replicated H3 GAN head and weights-only role loading."""

import copy
import os
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from lightx2v_train.trainers.dmd.dmd2_gan_head import H3GanHead, clip_dmd2_gan_grad_norm, frozen_dmd2_gan_critic, setup_dmd2_gan_head
from lightx2v_train.trainers.dmd.fake_warmstart import load_fake_checkpoint
from torch.utils.checkpoint import checkpoint

from lightx2v_train.trainers.dmd.tests import test_dmd_checkpoint_metadata as checkpoint_tests


def _joint_clip_worker(rank, init_file):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        backbone = torch.nn.Linear(2, 2, bias=False)
        fully_shard(backbone, mesh=init_device_mesh("cpu", (2,)), reshard_after_forward=False)
        backbone(torch.full((1, 2), float(rank + 1))).sum().backward()
        head = torch.nn.Parameter(torch.tensor([0.0]))
        head.grad = torch.tensor([float(1 + 2 * rank)])
        original_fake_gradient = backbone.weight.grad.full_tensor().clone()
        owner = SimpleNamespace(fake_trainable_params=list(backbone.parameters()), dmd2_gan_trainable_params=[head])
        norm = clip_dmd2_gan_grad_norm(owner, 1.0)
        expected = (original_fake_gradient.square().sum() + 4).sqrt()
        torch.testing.assert_close(norm, expected)
        torch.testing.assert_close(head.grad, torch.tensor([2.0]) / (expected + 1e-6))
        torch.testing.assert_close(backbone.weight.grad.full_tensor(), original_fake_gradient / (expected + 1e-6))
    finally:
        dist.destroy_process_group()


def _frozen_first_mixed_precision_worker(rank, init_file):
    """Resume at G before any D forward: FSDP must retain FP32 master grads."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(33)
        mesh = init_device_mesh("cpu", (2,))
        backbone = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU(), torch.nn.Linear(4, 3))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for module in (backbone[0], backbone[2], backbone):
            fully_shard(module, mesh=mesh, mp_policy=policy)
        parameters = list(backbone.parameters())
        head = H3GanHead(3, hidden_dim=4, temporal_bins=2)
        owner = SimpleNamespace(
            fake_trainable_params=parameters,
            dmd2_gan_trainable_params=list(head.parameters()),
            fake=SimpleNamespace(denoiser=lambda: backbone, set_training=backbone.train),
        )
        groups = [backbone[index]._get_fsdp_state()._fsdp_param_group for index in (0, 2)]
        inputs = torch.randn(1, 4, 4, requires_grad=True)
        with frozen_dmd2_gan_critic(owner):
            head(backbone(inputs)).sum().backward()
            assert all(group._orig_dtype is None and group._reduce_dtype is None for group in groups)
        assert inputs.grad is not None and inputs.grad.abs().sum() > 0
        assert all(parameter.grad is None and parameter.dtype == torch.float32 for parameter in parameters)
        assert all(group._orig_dtype == torch.float32 and group._reduce_dtype == torch.float32 for group in groups)
        # Before the fix this failed assigning BF16 gradients to FP32 DTensors.
        head(backbone(inputs.detach())).sum().backward()
        assert all(parameter.grad is not None and parameter.grad.dtype == torch.float32 for parameter in parameters)
        assert all(torch.isfinite(parameter.grad.to_local()).all() for parameter in parameters)
        torch.optim.SGD(parameters, lr=0.01).step()
        assert all(parameter.dtype == torch.float32 for parameter in parameters)
    finally:
        dist.destroy_process_group()


class H3GanHeadTest(unittest.TestCase):
    def test_ordered_video_bins_and_fp32_output_retain_input_gradients(self):
        head = H3GanHead(2, hidden_dim=1, temporal_bins=2)
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            head.classifier[0].weight[0, 0] = 1
            head.classifier[2].weight[0, 0] = 1
        inputs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = head(inputs)
            reversed_logits = head(inputs.flip(1))
        self.assertEqual(logits.shape, (1,))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertGreater(float(logits.detach()), float(reversed_logits.detach()))
        logits.sum().backward()
        self.assertGreater(float(inputs.grad.abs().sum()), 0)

    def test_frozen_backbone_and_head_restore_flags_after_checkpointed_backward(self):
        backbone = torch.nn.Linear(3, 4)
        head = H3GanHead(4, hidden_dim=3, temporal_bins=2)
        owner = SimpleNamespace(fake_trainable_params=list(backbone.parameters()), dmd2_gan_trainable_params=list(head.parameters()), fake=SimpleNamespace(denoiser=lambda: backbone, set_training=Mock()))
        inputs = torch.randn(1, 5, 3, requires_grad=True)
        with frozen_dmd2_gan_critic(owner):
            features = checkpoint(backbone, inputs, use_reentrant=False)
            torch.nn.functional.softplus(-head(features)).mean().backward()
            self.assertTrue(all(not p.requires_grad for p in [*backbone.parameters(), *head.parameters()]))
        self.assertTrue(all(p.grad is None and p.requires_grad for p in [*backbone.parameters(), *head.parameters()]))
        self.assertGreater(float(inputs.grad.abs().sum()), 0)
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with frozen_dmd2_gan_critic(owner):
                raise RuntimeError("test exception")
        self.assertTrue(all(p.requires_grad for p in [*backbone.parameters(), *head.parameters()]))

    def test_joint_clip_matches_single_parameter_group(self):
        fake, head = torch.nn.Parameter(torch.tensor([0.0])), torch.nn.Parameter(torch.tensor([0.0]))
        fake.grad, head.grad = torch.tensor([3.0]), torch.tensor([4.0])
        owner = SimpleNamespace(fake_trainable_params=[fake], dmd2_gan_trainable_params=[head])
        norm = clip_dmd2_gan_grad_norm(owner, 1.0)
        torch.testing.assert_close(norm, torch.tensor(5.0))
        torch.testing.assert_close(fake.grad, torch.tensor([0.6]))
        torch.testing.assert_close(head.grad, torch.tensor([0.8]))

    def test_replicated_head_reduction_precedes_joint_clip(self):
        fake, head = torch.nn.Parameter(torch.tensor([0.0])), torch.nn.Parameter(torch.tensor([0.0]))
        fake.grad, head.grad = torch.tensor([3.0]), torch.tensor([2.0])
        owner = SimpleNamespace(fake_trainable_params=[fake], dmd2_gan_trainable_params=[head])
        # Second rank contributes six, so the replicated gradient averages four.
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_reduce", side_effect=lambda gradient: gradient.add_(6)) as reduce,
        ):
            norm = clip_dmd2_gan_grad_norm(owner, 1.0)
        reduce.assert_called_once()
        torch.testing.assert_close(norm, torch.tensor(5.0))
        torch.testing.assert_close(head.grad, torch.tensor([0.8]))

    def test_setup_preserves_rng_and_uses_fake_optimizer_recipe(self):
        backbone = SimpleNamespace(config=SimpleNamespace(hidden_size=4), transformer_blocks=[object(), object()])
        owner = SimpleNamespace(
            dmd_config={"official_pdmd": True, "gan": {"enabled": True, "feature_block": 1, "head_hidden_dim": 3, "head_temporal_bins": 2}},
            config={"seed": 7},
            fake=SimpleNamespace(denoiser=lambda: backbone),
            student=SimpleNamespace(device="cpu"),
            fake_optimizer_config={"learning_rate": 2e-6},
            max_train_iters=30,
            fake_update_ratio=10,
            fake_optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-6),
            _build_optimizer=lambda parameters, config: torch.optim.AdamW(parameters, lr=config["learning_rate"]),
            _build_lr_scheduler=lambda optimizer, **kwargs: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        )
        before = torch.get_rng_state()
        setup_dmd2_gan_head(owner)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(owner.dmd2_gan_optimizer.param_groups[0]["lr"], 2e-6)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in owner.dmd2_gan_trainable_params))

    def test_two_rank_cpu_fsdp_joint_norm_does_not_double_count_replicated_head(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(_joint_clip_worker, args=(str(Path(directory) / "init"),), nprocs=2, join=False)
            deadline = time.monotonic() + 50
            try:
                while time.monotonic() < deadline:
                    if context.join(timeout=1):
                        return
                self.fail("Two-rank CPU FSDP GAN joint-clipping test timed out.")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=2)

    def test_first_frozen_forward_then_fake_update_preserves_fp32_masters(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(_frozen_first_mixed_precision_worker, args=(str(Path(directory) / "init"),), nprocs=2, join=False)
            deadline = time.monotonic() + 50
            try:
                while time.monotonic() < deadline:
                    if context.join(timeout=1):
                        return
                self.fail("Two-rank G-first FSDP mixed-precision test timed out.")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=2)


class GanCheckpointTest(unittest.TestCase):
    def fixture(self):
        owner, manager, state = checkpoint_tests.DmdCheckpointMetadataTest().fixture()
        owner.dmd2_gan_head = H3GanHead(4, 3, 2)
        owner.dmd2_gan_optimizer = torch.optim.AdamW(owner.dmd2_gan_head.parameters(), lr=0.01)
        owner.dmd2_gan_lr_scheduler = torch.optim.lr_scheduler.StepLR(owner.dmd2_gan_optimizer, step_size=1)
        owner.dmd2_gan_metadata = {"version": 1, "options": {"enabled": True, "feature_block": 49}}
        owner.dmd2_gan_head(torch.randn(1, 5, 4)).sum().backward()
        owner.dmd2_gan_optimizer.step()
        owner.dmd2_gan_lr_scheduler.step()
        state.update(manager._extra_checkpoint_metadata())
        state.update(copy.deepcopy(manager._extra_residual_head_training_state()))
        return owner, manager, state

    def test_head_optimizer_schedule_round_trip(self):
        owner, manager, state = self.fixture()
        manager._validate_checkpoint_state(state, "training_state.pt", "checkpoint-000000007")
        expected = copy.deepcopy(owner.dmd2_gan_head.state_dict())
        with torch.no_grad():
            for parameter in owner.dmd2_gan_head.parameters():
                parameter.zero_()
        owner.dmd2_gan_optimizer.state.clear()
        owner.dmd2_gan_lr_scheduler.last_epoch = 0
        manager._load_residual_head_training_state(state)
        for name, value in owner.dmd2_gan_head.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        self.assertEqual(owner.dmd2_gan_lr_scheduler.last_epoch, 1)
        self.assertTrue(owner.dmd2_gan_optimizer.state)

    def test_full_resume_rejects_missing_changed_or_baseline_gan_state(self):
        owner, manager, state = self.fixture()
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        for key in ("dmd2_gan_config", "dmd2_gan_head", "dmd2_gan_optimizer", "dmd2_gan_lr_scheduler"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "missing required state"):
                manager._validate_dmd2_gan_state({k: v for k, v in state.items() if k != key}, "state")
        with self.assertRaisesRegex(RuntimeError, "dmd2_gan_config"):
            manager._validate_dmd2_gan_state({**state, "dmd2_gan_config": {}}, "state")
        with self.assertRaisesRegex(RuntimeError, "weights-only"):
            manager._validate_dmd2_gan_state({}, "state")
        owner.dmd2_gan_head = None
        with self.assertRaisesRegex(RuntimeError, "weights-only"):
            manager._validate_dmd2_gan_state(state, "state")
        manager._validate_dmd2_gan_state({}, "legacy baseline")

    def test_fake_weights_only_dcp_load_leaves_optimizer_and_student_untouched(self):
        fake, student = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
        target = {name: torch.full_like(value, 0.25) for name, value in fake.state_dict().items()}
        optimizer = torch.optim.AdamW(fake.parameters(), lr=0.01)
        before_student = copy.deepcopy(student.state_dict())
        owner = SimpleNamespace(
            fake_model=SimpleNamespace(ensure_capabilities=lambda: SimpleNamespace(require=lambda _: SimpleNamespace(state_module=lambda: fake))),
            fake_optimizer=optimizer,
            checkpoint_manager=SimpleNamespace(_get_checkpoint_process_group=lambda: None),
        )
        with tempfile.TemporaryDirectory() as directory:
            dcp.save({"fake_model": target, "student_model": student.state_dict(), "fake_optimizer": {"old_step": torch.tensor(5700)}}, checkpoint_id=str(Path(directory) / "dist_state"))
            load_fake_checkpoint(owner, directory)
        for name, value in fake.state_dict().items():
            torch.testing.assert_close(value, target[name], rtol=0, atol=0)
        for name, value in student.state_dict().items():
            torch.testing.assert_close(value, before_student[name], rtol=0, atol=0)
        self.assertFalse(optimizer.state)


if __name__ == "__main__":
    unittest.main()
