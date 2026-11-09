"""Native H3 CPU FSDP coverage for PDMD + GAN role changes and accumulation."""

import math
import os
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lightx2v_train.model_zoo.native.minimax_h3.modeling import install_minimax_h3_dmad_features
from lightx2v_train.trainers.dmad.tests import test_dmad as native_fixtures
from lightx2v_train.trainers.dmd.dmd2_gan import Dmd2GanObjective
from lightx2v_train.trainers.dmd.dmd2_gan_head import H3GanHead
from lightx2v_train.trainers.dmd.official_pdmd_core import full_rollout, rollout_grid
from lightx2v_train.trainers.dmd.official_pdmd_training import OfficialH3PdmdTraining


def _native_gan_fsdp_worker(rank, init_file):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2,))
        models, capabilities = {}, {}
        for name in ("student", "fake", "teacher"):
            model, capability, latents, condition = native_fixtures.DmadH3FeaturesTest().fixture(references=False, checkpointing=True)
            capability.options = replace(capability.options, official_pdmd=True, projected_dmd=True)
            capability.set_training = lambda enabled, module=model: module.train(enabled)
            if name == "teacher":
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(0.01)
                model.requires_grad_(False)
            install_minimax_h3_dmad_features(model)
            for block in [*model.token_refiner.refiner_blocks, *model.transformer_blocks]:
                fully_shard(block, mesh=mesh)
            fully_shard(model, mesh=mesh, reshard_after_forward=False)
            models[name], capabilities[name] = model, capability

        # Keep optimizer-owned DTensors: module.parameters() can expose ordinary
        # unsharded tensors after a no-grad root forward with reshard=False.
        role_parameters = {name: list(model.parameters()) for name, model in models.items()}
        trainer = SimpleNamespace(**capabilities)
        trainer.config = {"seed": 17}
        # Block zero omits a later FSDP block and output projections in GAN
        # calls, while the score/FM backward must still train those parameters.
        trainer.dmd_config = {"gan": {"feature_block": 0, "generator_weight": 0.005, "discriminator_weight": 0.01}}
        trainer.trainable_params = role_parameters["student"]
        trainer.fake_trainable_params = role_parameters["fake"]
        torch.manual_seed(99)
        trainer.dmd2_gan_head = H3GanHead(16, hidden_dim=8, temporal_bins=2)
        trainer.dmd2_gan_trainable_params = list(trainer.dmd2_gan_head.parameters())
        for name, parameters in (
            ("optimizer", trainer.trainable_params),
            ("fake_optimizer", trainer.fake_trainable_params),
            ("dmd2_gan_optimizer", trainer.dmd2_gan_trainable_params),
        ):
            setattr(trainer, name, torch.optim.AdamW(parameters, lr=0.0001, betas=(0.0, 0.9)))
        for scheduler, optimizer in (("lr_scheduler", "optimizer"), ("fake_lr_scheduler", "fake_optimizer"), ("dmd2_gan_lr_scheduler", "dmd2_gan_optimizer")):
            setattr(trainer, scheduler, torch.optim.lr_scheduler.LambdaLR(getattr(trainer, optimizer), lambda _: 1))

        def set_sync(name, enabled):
            models[name].set_requires_gradient_sync(enabled)
            models[name].set_is_last_backward(enabled)

        trainer._set_student_gradient_sync = lambda enabled: set_sync("student", enabled)
        trainer._set_fake_gradient_sync = lambda enabled: set_sync("fake", enabled)
        trainer._sync_sequence_parallel_grads = lambda parameters: None
        trainer._after_student_optimizer_step = lambda role: None
        trainer.max_grad_norm, trainer.num_inference_steps = 1.0, 2
        latents = trainer.student.detach(latents)
        trajectory = full_rollout(trainer.student, latents, condition, rollout_grid(2, 12), rollout_grid(2, 3))
        real_stream = SimpleNamespace(load=lambda student, cursor: (latents, condition))
        engine = OfficialH3PdmdTraining.__new__(OfficialH3PdmdTraining)
        engine.trainer, engine.dmd, engine.gan = trainer, trainer.dmd_config, Dmd2GanObjective(trainer, real_stream)
        engine.critic_steps, engine.accum, engine.seed = 10, 2, 17
        engine.cached, engine.consumed_microbatches = [], 18
        engine.route_loss_scale = lambda _: 1.0
        engine._trajectory = lambda sample, iteration, slot: trajectory

        # Each fake update performs FM, GAN-negative and GAN-positive backwards
        # per microbatch. Exercise G after D, then D again after frozen G.
        for iteration in (9, 10, 11, 12):
            before = {name: [parameter.detach().to_local().clone() for parameter in parameters] for name, parameters in role_parameters.items()}
            head_before = [parameter.detach().clone() for parameter in trainer.dmd2_gan_trainable_params]
            role, metrics = engine._update(iter([{}, {}]), iteration)
            assert all(math.isfinite(value) for value in metrics.values()), metrics
            assert all(parameter.requires_grad for parameter in trainer.fake_trainable_params)
            if role == "student":
                assert all(torch.equal(parameter.to_local(), saved) for parameter, saved in zip(trainer.fake_trainable_params, before["fake"]))
                assert all(torch.equal(parameter, saved) for parameter, saved in zip(trainer.dmd2_gan_trainable_params, head_before))
                assert any(not torch.equal(parameter.to_local(), saved) for parameter, saved in zip(trainer.trainable_params, before["student"]))
                assert all(parameter.grad is None for parameter in [*trainer.fake_trainable_params, *trainer.dmd2_gan_trainable_params])
            else:
                assert any(not torch.equal(parameter.to_local(), saved) for parameter, saved in zip(trainer.fake_trainable_params, before["fake"]))
                assert any(not torch.equal(parameter, saved) for parameter, saved in zip(trainer.dmd2_gan_trainable_params, head_before))
                assert all(torch.equal(parameter.to_local(), saved) for parameter, saved in zip(trainer.trainable_params, before["student"]))
            assert all(torch.equal(parameter.to_local(), saved) for parameter, saved in zip(role_parameters["teacher"], before["teacher"]))
    finally:
        dist.destroy_process_group()


class NativeH3GanFsdpTest(unittest.TestCase):
    def test_accumulated_score_and_gan_backwards_across_role_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(_native_gan_fsdp_worker, args=(str(Path(directory) / "init"),), nprocs=2, join=False)
            deadline = time.monotonic() + 90
            try:
                while time.monotonic() < deadline:
                    if context.join(timeout=1):
                        return
                self.fail("Two-rank native H3 PDMD + GAN FSDP test timed out.")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
