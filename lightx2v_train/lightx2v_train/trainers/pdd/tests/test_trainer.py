"""CPU autograd tests for staggered PDD carry, exact resume and fused sampling."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from lightx2v_train.trainers.base import BaseTrainer
from lightx2v_train.trainers.pdd.config import PDDConfig
from lightx2v_train.trainers.pdd.core import advance_blocks
from lightx2v_train.trainers.pdd.sampler import sample_pdd
from lightx2v_train.trainers.pdd.trainer import PDDTrainer
from lightx2v_train.trainers.turbo.cursor import TurboCursorSampler
from torch.utils.data import DataLoader, Dataset


class Conditions(Dataset):
    def __init__(self):
        self.samples = [{"id": i, "conditioning": {"prompt": f"sample {i}"}} for i in range(3)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4, 0.5]))
        self.device = torch.device("cpu")
        self.head_calls, self.block_calls, self.teacher_calls = [], [], []

    def denoiser_module(self):
        return self

    def pdd_sigma_schedule(self, sigma):
        return sigma

    def predict_pdd_heads(self, state, start, condition, sigma):
        self.head_calls.append(start)
        return state[:, None] * self.weight[None, :, None] + condition["value"]

    def predict_pdd_block(self, state, start, end, condition, grid):
        self.block_calls.append((start, end))
        increments = grid.diff()[start:end].to(state.dtype)
        # Fuse the affine heads before applying them: only one state projection.
        scale = (self.weight[start:end] * increments).sum()
        return state * scale + condition["value"] * increments.sum()

    def predict_pdd_teacher(self, state, sigma, condition):
        self.teacher_calls.append((torch.is_grad_enabled(), sigma.detach().clone()))
        return state * 0.7 + sigma[:, None].to(state.dtype) * 0.1 + condition["value"]


class ToyCapability:
    def __init__(self):
        self.condition_ids = []

    def encode_conditions(self, sample, *args):
        self.condition_ids.append(sample["id"])
        return {"id": sample["id"], "value": torch.tensor(float(sample["id"] + 1) / 10)}, None

    def latent_shape(self, *args):
        return (1, 3)

    def initial_latents(self, shape, dtype, broadcast):
        return broadcast(torch.randn(shape, dtype=dtype))


def fixture(output):
    trainer = PDDTrainer.__new__(PDDTrainer)
    trainer.pdd = PDDConfig(grid_size=4, num_inference_steps=2, video_shift=3, audio_shift=1, block_size_min=1, block_size_max=2, num_supervised_heads=1)
    trainer.config = {
        "seed": 42,
        "model": {"name": "minimax_h3_image"},
        "training": {"method": "pdd", "pdd": asdict(trainer.pdd), "output_dir": str(output)},
        "data": {"train": {"name": "toy_cache", "shuffle": False}},
    }
    trainer.training_config = trainer.config["training"]
    trainer.model_config = trainer.config["model"]
    trainer.model, trainer.teacher_model, trainer.student = ToyModel(), ToyModel(), ToyCapability()
    trainer.teacher_model.requires_grad_(False)
    trainer.trainable_params = list(trainer.model.parameters())
    trainer.optimizer = torch.optim.AdamW(trainer.trainable_params, lr=0.003, betas=(0.9, 0.99), weight_decay=0)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1, gamma=0.99)
    trainer.grid = trainer.pdd.base_grid(trainer.model.device)
    trainer.max_grad_norm = 1
    trainer.generation_shapes = None
    dataset = Conditions()
    trainer.dataloader_train = DataLoader(dataset, batch_size=1, sampler=TurboCursorSampler(dataset, shuffle=False), collate_fn=lambda samples: samples[0])
    trainer.consumed_microbatches, trainer.carry = 0, None
    trainer.output_train_dir = str(output)
    trainer.parallel = SimpleNamespace(is_fsdp=lambda: False)
    trainer.train_type = "lora"
    trainer.auto_resume = True
    return trainer


def save_toy_base(trainer, iteration, save_total_limit):
    del save_total_limit
    directory = Path(trainer.output_train_dir) / f"checkpoint-{iteration:09d}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict(), "scheduler": trainer.lr_scheduler.state_dict()}, directory / "toy_state.pt")


class PDDTrainerTests(unittest.TestCase):
    def test_staggered_initialization_keeps_forward_counts_equal(self):
        starts = []
        for rank in range(4):
            trainer = fixture("unused")
            with patch("lightx2v_train.trainers.pdd.trainer.get_data_parallel_rank", return_value=rank):
                carry = trainer._initialize_staggered(trainer._samples())
            self.assertEqual(carry["step"], rank)
            self.assertEqual(trainer.model.head_calls, [0, 1, 2])
            self.assertEqual(trainer.consumed_microbatches, 1)
            self.assertFalse(carry["state"].requires_grad)
            starts.append(carry["state"])
        self.assertFalse(torch.equal(starts[0], starts[3]))

    def test_conditions_advance_only_after_trajectory_terminal_without_fake(self):
        trainer = fixture("unused")
        samples = trainer._samples()
        with patch("lightx2v_train.trainers.pdd.trainer.get_data_parallel_rank", return_value=2):
            trainer.carry = trainer._initialize_staggered(samples)
        trainer.update(samples, 0)
        self.assertEqual(trainer.carry["step"], 3)
        self.assertEqual(trainer.student.condition_ids, [0])
        trainer.update(samples, 1)
        self.assertIsNone(trainer.carry)
        trainer.update(samples, 2)
        self.assertEqual(trainer.student.condition_ids, [0, 1])
        self.assertEqual(trainer.consumed_microbatches, 2)
        self.assertFalse(hasattr(trainer, "fake"))
        self.assertFalse(hasattr(trainer, "fake_optimizer"))
        self.assertEqual(len(trainer.teacher_model.teacher_calls), 3)
        self.assertTrue(all(not enabled for enabled, _ in trainer.teacher_model.teacher_calls))
        self.assertTrue(all(parameter.grad is None for parameter in trainer.teacher_model.parameters()))

    def test_mid_trajectory_checkpoint_resume_preserves_updates_and_data_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = fixture(directory)
            samples = baseline._samples()
            with patch("lightx2v_train.trainers.pdd.trainer.get_data_parallel_rank", return_value=2):
                baseline.carry = baseline._initialize_staggered(samples)
            baseline.update(samples, 0)
            with patch.object(BaseTrainer, "save_checkpoint", save_toy_base):
                baseline.save_checkpoint(1, 2)
            checkpoint, iteration = baseline._resolve_resume()
            self.assertEqual(iteration, 1)
            expected_losses = [float(baseline.update(samples, i)["total_loss"]) for i in range(1, 7)]
            resumed = fixture(directory)
            state = torch.load(Path(checkpoint) / "toy_state.pt", weights_only=True)
            resumed.model.load_state_dict(state["model"])
            resumed.optimizer.load_state_dict(state["optimizer"])
            resumed.lr_scheduler.load_state_dict(state["scheduler"])
            resumed._restore_carry(checkpoint, iteration)
            self.assertEqual(resumed.carry["step"], 3)
            self.assertEqual(resumed.consumed_microbatches, 1)
            continued = resumed._samples()
            actual_losses = [float(resumed.update(continued, i)["total_loss"]) for i in range(1, 7)]
            self.assertEqual(actual_losses, expected_losses)
            torch.testing.assert_close(resumed.model.weight, baseline.model.weight, rtol=0, atol=0)
            self.assertEqual(resumed.consumed_microbatches, baseline.consumed_microbatches)
            self.assertEqual(resumed.student.condition_ids, baseline.student.condition_ids[1:])
            self.assertEqual(resumed.carry["step"], baseline.carry["step"])
            torch.testing.assert_close(resumed.carry["state"], baseline.carry["state"], rtol=0, atol=0)
            changed = fixture(directory)
            changed.training_config["pdd"]["teacher_solver"] = "midpoint"
            with self.assertRaisesRegex(ValueError, "recipe/world size"):
                changed._resolve_resume()

    def test_nonfinite_teacher_target_does_not_update_student(self):
        trainer = fixture("unused")
        before = trainer.model.weight.detach().clone()
        trainer.teacher_model.predict_pdd_teacher = lambda state, *args: torch.full_like(state, float("nan"))
        with self.assertRaises(FloatingPointError):
            trainer.update(trainer._samples(), 0)
        torch.testing.assert_close(trainer.model.weight, before, rtol=0, atol=0)

    def test_fused_sampler_matches_materialized_heads_and_restores_mode(self):
        trainer = fixture("unused")
        initial, condition = torch.randn(1, 3), {"value": torch.tensor(0.2)}
        expected = initial.clone()
        for start in (0, 2):
            heads = trainer.model.predict_pdd_heads(expected, start, condition, trainer.grid[start])
            expected = advance_blocks(expected, heads, start, start + 2, trainer.grid)
        trainer.model.head_calls.clear()
        trainer.model.train()
        actual = sample_pdd(trainer.model, initial, condition, trainer.pdd)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(trainer.model.block_calls, [(0, 2), (2, 4)])
        self.assertEqual(trainer.model.head_calls, [])
        self.assertTrue(trainer.model.training)
        self.assertFalse(actual.requires_grad)
        with self.assertRaises(ValueError):
            sample_pdd(trainer.model, initial, condition, trainer.pdd, num_inference_steps=1)


if __name__ == "__main__":
    unittest.main()
