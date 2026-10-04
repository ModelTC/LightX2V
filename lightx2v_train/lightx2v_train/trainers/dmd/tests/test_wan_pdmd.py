import unittest
from types import SimpleNamespace

import torch

from lightx2v_train.model_zoo.capability_adapters.common import GenericDistributionMatchingCapability
from lightx2v_train.model_zoo.wan.capability_adapters.wan_distribution_matching_capability import (
    WanDistributionMatchingCapability,
)
from lightx2v_train.trainers.dmd.checkpoint import DmdCheckpointManager


class WanPdmdTest(unittest.TestCase):
    def fixture(self, projected=None):
        model = torch.nn.Linear(1, 1)
        model.config = {"model": {}}
        if projected is not None:
            model.config["model"]["capabilities"] = {
                "distribution_matching": {"projected_dmd": projected},
            }
        return model, WanDistributionMatchingCapability(model)

    def test_default_is_original_dmd(self):
        model, capability = self.fixture()
        self.assertFalse(capability.projected_dmd)
        self.assertEqual(capability.dmd_metrics(), {})
        self.assertEqual(capability.extra_checkpoint_metadata(), capability.legacy_extra_checkpoint_metadata())

    def test_disabled_projection_preserves_wan_float32_loss_and_gradient(self):
        model, capability = self.fixture(False)
        generator = torch.Generator().manual_seed(42)
        latents = torch.randn(2, 3, 4, 5, 6, generator=generator, requires_grad=True)
        fake = torch.randn(latents.shape, generator=generator)
        teacher = torch.randn(latents.shape, generator=generator)
        expected = GenericDistributionMatchingCapability.dmd_loss(latents, fake, teacher)
        actual = capability.dmd_loss(latents, fake, teacher)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual, latents)[0],
            torch.autograd.grad(expected, latents)[0],
            rtol=0,
            atol=0,
        )

    def test_projected_wan_gradient_is_orthogonal_per_video(self):
        model, capability = self.fixture(True)
        generator = torch.Generator().manual_seed(7)
        latents = torch.randn(2, 3, 4, 5, 6, generator=generator, requires_grad=True)
        fake = torch.randn(latents.shape, generator=generator, requires_grad=True)
        teacher = torch.randn(latents.shape, generator=generator, requires_grad=True)
        loss = capability.dmd_loss(latents, fake, teacher)
        gradient = torch.autograd.grad(loss, latents)[0]
        residual = (fake - latents).detach()
        dot_products = (gradient * residual).flatten(1).sum(dim=1)
        torch.testing.assert_close(dot_products, torch.zeros(2), rtol=0, atol=1e-7)
        self.assertIsNone(fake.grad)
        self.assertIsNone(teacher.grad)

    def test_projection_removes_parallel_direction_but_not_orthogonal_direction(self):
        model, capability = self.fixture(True)
        latents = torch.tensor([[[[[0.0, 1.0]]]]], requires_grad=True)
        fake = torch.tensor([[[[[1.0, 1.0]]]]])
        teacher = torch.tensor([[[[[0.0, -1.0]]]]])
        loss = capability.dmd_loss(latents, fake, teacher)
        loss.backward()
        torch.testing.assert_close(latents.grad, torch.tensor([[[[[0.0, 1.0]]]]]))
        torch.testing.assert_close(loss, torch.tensor(1.0))

    def test_diagnostics_are_detached_and_float32(self):
        model, capability = self.fixture(True)
        latents = torch.tensor([[[[[0.25, 0.5, -1.0]]]]], dtype=torch.bfloat16, requires_grad=True)
        fake = torch.tensor([[[[[1.25, 2.0, -2.0]]]]], dtype=torch.bfloat16)
        teacher = torch.zeros_like(fake)
        loss = capability.dmd_loss(latents, fake, teacher)
        self.assertEqual(loss.dtype, torch.float32)
        for value in capability.dmd_metrics().values():
            self.assertFalse(value.requires_grad)
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value))
        loss.backward()
        self.assertEqual(latents.grad.dtype, torch.bfloat16)

    def checkpoint_fixture(self, projected):
        model, capability = self.fixture(projected)
        roles = {
            "student": SimpleNamespace(model=model, train_type="lora"),
            "fake": SimpleNamespace(model=object(), train_type="full"),
        }
        owner = SimpleNamespace(
            model=model,
            student=capability,
            config={},
            dataloader_train=SimpleNamespace(sampler=None),
            role_registry=SimpleNamespace(runtimes=lambda: roles),
            _validate_checkpoint_metadata=lambda *args: None,
        )
        manager = DmdCheckpointManager(owner)
        state = {
            "dmd_checkpoint_version": 2,
            "student_train_type": "lora",
            "fake_train_type": "full",
            **capability.extra_checkpoint_metadata(),
        }
        return owner, manager, state

    def test_saved_pdmd_switch_cannot_silently_change_on_resume(self):
        owner, manager, state = self.checkpoint_fixture(True)
        manager._validate_checkpoint_state(state, "training_state.pt", "checkpoint-100")
        state["wan_distribution_matching"]["projected_dmd"] = False
        with self.assertRaisesRegex(RuntimeError, "wan_distribution_matching.*allow_distribution_matching_transition"):
            manager._validate_checkpoint_state(state, "training_state.pt", "checkpoint-100")
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        manager._validate_checkpoint_state(state, "training_state.pt", "checkpoint-100")

    def test_legacy_checkpoint_can_resume_original_dmd(self):
        owner, manager, state = self.checkpoint_fixture(False)
        state.pop("wan_distribution_matching")
        manager._validate_checkpoint_state(state, "training_state.pt", "checkpoint-100")

    def test_legacy_checkpoint_requires_explicit_transition_to_pdmd(self):
        owner, manager, state = self.checkpoint_fixture(True)
        state.pop("wan_distribution_matching")
        with self.assertRaisesRegex(RuntimeError, "wan_distribution_matching.*allow_distribution_matching_transition"):
            manager._validate_checkpoint_state(state, "training_state.pt", "checkpoint-100")


if __name__ == "__main__":
    unittest.main()
