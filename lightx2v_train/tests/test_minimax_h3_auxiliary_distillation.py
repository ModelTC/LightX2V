import copy
import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from lightx2v_train.model_capabilities import VAEDistillationStepContext
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import MiniMaxH3VAEDistillationCapability
from lightx2v_train.model_zoo.minimax_h3.minimax_h3_pruned_vae import MiniMaxH3PrunedVAEModel
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.vae.distillation import VAEDistillationTrainer
from tests.test_minimax_h3_pruned_vae import tiny_config, tiny_teacher


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = "lightx2v_train.model_zoo.minimax_h3.minimax_h3_pruned_vae"


@contextmanager
def auxiliary_model(*, route="temporal_pair", feature_weight=0.01, unpruned=False, checkpointing=True):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        teacher_config = tiny_config()
        (root / "config.json").write_text(json.dumps(teacher_config))
        teacher = tiny_teacher(teacher_config)
        teacher.config = SimpleNamespace(**teacher_config)
        config = load_config(str(ROOT / "configs/train/vae/minimax_h3_vit_prune_recover_aux_3k_4gpu_ddp.yaml"))
        config["model"].update(pretrained_model_name_or_path=str(root), running_dtype="fp32", student_autocast=False)
        config["model"]["pruned_decoder"] = {
            "kept_layers": list(range(6)) if unpruned else [1, 5],
            "tile_sample_min_height": 64, "tile_sample_min_width": 64,
            "tile_sample_min_overlap_height": 16, "tile_sample_min_overlap_width": 16,
        }
        config["training"].update(max_train_iters=2, gradient_accumulation_iters=2, lr_warmup_iters=0,
                                  save_every_iters=1, output_dir=str(root / "recover"))
        loss_config = {
            "teacher_output_weight": 0.0, "temporal_velocity_weight": 0.0,
            "teacher_feature_indices": list(range(6)) if unpruned else [2, 5],
            "teacher_tile_batch_size": 1, "student_tile_batch_size": 1,
            "student_window_checkpointing": checkpointing,
            "auxiliary_decoder": {
                "student_feature_indices": [1] if unpruned else [0],
                "ramp_iters": 0, "gradient_checkpointing": checkpointing,
            },
            "stages": [{"name": "test", "start_iter": 0, "routes": {route: 1.0},
                        "weights": {"reconstruction": 1.0, "feature": feature_weight, "auxiliary": 0.1}}],
            "gan": {"enabled": False},
        }
        config["training"]["vae_distillation"] = loss_config
        config["resume"]["auto_resume"] = False
        config["inference"] = {"infer_every_iters": 0}
        with patch(f"{WRAPPER}.load_minimax_h3_video_vae", side_effect=lambda *args, **kwargs: copy.deepcopy(teacher)):
            model = MiniMaxH3PrunedVAEModel(config)
            model.load_components(load_transformer=True, load_vae=True, load_condition_encoder=False)
        model.log_model_structure = lambda: None
        model.transformer.decoder.gradient_checkpointing = checkpointing
        capability = MiniMaxH3VAEDistillationCapability(model, loss_config)
        yield config, model, capability


def sample():
    return {
        "inputs": {"video": torch.rand(1, 3, 124, 96, 96), "latents": torch.randn(1, 24, 37, 6, 6)},
        "meta": {"source_num_frames": torch.tensor([124])},
    }


class AuxiliaryDistillationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(51)

    def test_mixed_difference_formula_and_teacher_stop_gradient(self):
        owner = torch.nn.Module()
        capability = MiniMaxH3VAEDistillationCapability(owner, {})
        left = torch.randn(1, 3, 5, 4, 6, requires_grad=True)
        right = torch.randn_like(left, requires_grad=True)
        error = left - right.detach()
        differences = [error.diff(dim=2).diff(dim=axis) for axis in (-2, -1)]
        epsilon = capability.charbonnier_epsilon
        expected = sum(((value.square() + epsilon**2).sqrt() - epsilon).mean() for value in differences) / 2
        actual = capability._spatiotemporal_loss(left, right)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(right.grad)
        self.assertGreater(left.grad.abs().sum().item(), 0)
        self.assertEqual(capability._spatiotemporal_loss(left.detach(), left.detach()).item(), 0)
        self.assertEqual(capability._spatiotemporal_loss(left[:, :, :1], right[:, :, :1]).item(), 0)
        static_error = torch.randn(1, 3, 1, 4, 6).expand(1, 3, 5, 4, 6)
        self.assertEqual(capability._spatiotemporal_loss(static_error, torch.zeros_like(static_error)).item(), 0)

    def test_auxiliary_component_weights(self):
        owner = torch.nn.Module()
        capability = MiniMaxH3VAEDistillationCapability(owner, {})
        prediction = torch.randn(1, 3, 4, 5, 6, requires_grad=True)
        target = torch.randn_like(prediction, requires_grad=True)
        actual, metrics = capability._auxiliary_loss(prediction, target)
        expected = metrics["auxiliary_reconstruction"] + 0.1 * metrics["auxiliary_spatial_gradient"] + 0.1 * metrics["auxiliary_spatiotemporal"]
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(target.grad)

    def test_ramp_starts_once_and_anchor_sampling_is_reproducible(self):
        owner = torch.nn.Module()
        capability = MiniMaxH3VAEDistillationCapability(owner, {
            "teacher_feature_indices": [2, 5, 8],
            "auxiliary_decoder": {"student_feature_indices": [0, 1], "ramp_iters": 500},
            "stages": [
                {"name": "warmup", "start_iter": 0, "weights": {"auxiliary": 0}},
                {"name": "joint", "start_iter": 2000, "weights": {"auxiliary": 0.1}},
                {"name": "refine", "start_iter": 7000, "weights": {"auxiliary": 0.02}},
            ],
        })
        self.assertEqual(capability._auxiliary_ramp(1999), 0)
        self.assertEqual(capability._auxiliary_ramp(2000), 1 / 500)
        self.assertEqual(capability._auxiliary_ramp(2499), 1)
        self.assertEqual(capability._auxiliary_ramp(7000), 1)
        contexts = [VAEDistillationStepContext(running_dtype=torch.float32, iteration=2000, micro_step=i) for i in range(16)]
        first = [capability._auxiliary_anchor(context) for context in contexts]
        self.assertEqual(first, [capability._auxiliary_anchor(context) for context in contexts])
        self.assertEqual(set(first), {0, 1})

    def test_unpruned_auxiliary_matches_teacher_after_spatial_and_temporal_stitching(self):
        for route in ("quad", "temporal_pair", "temporal_quad"):
            with self.subTest(route=route), auxiliary_model(route=route, unpruned=True) as (_, model, capability):
                result = capability.compute_loss(sample(), VAEDistillationStepContext(running_dtype=torch.float32, iteration=0, micro_step=0))
                self.assertLess(result.metrics["auxiliary"].item(), 1e-7)
                self.assertEqual(result.metrics["auxiliary_student_layer"], 2)
                self.assertEqual(result.metrics["auxiliary_teacher_layer"], 2)
                self.assertTrue(all(parameter.grad is None for parameter in model.teacher_vae.parameters()))

    def test_auxiliary_only_supervision_keeps_teacher_when_feature_mse_disabled(self):
        with auxiliary_model(feature_weight=0) as (_, model, capability):
            self.assertIsNotNone(model.teacher_vae.decoder)
            first_block_calls = []
            handle = model.teacher_vae.decoder.transformer_blocks[0].register_forward_hook(
                lambda _module, _inputs, _output: first_block_calls.append(1)
            )
            result = capability.compute_loss(sample(), VAEDistillationStepContext(running_dtype=torch.float32, iteration=0, micro_step=0))
            handle.remove()
            self.assertEqual(len(first_block_calls), 2)  # One target pass for each temporal window, not a second teacher pass.
            result.loss.backward()
            self.assertGreater(result.metrics["auxiliary"].item(), 0)
            self.assertTrue(all(parameter.grad is None for parameter in model.teacher_vae.parameters()))
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in model.transformer.parameters()))

    def test_auxiliary_recovery_checkpoint_and_resume(self):
        with auxiliary_model() as (config, model, _):
            data = DataLoader([sample()], batch_size=1, collate_fn=lambda samples: samples[0])
            trainer = VAEDistillationTrainer(config)
            trainer.set_model(model)
            trainer.set_data(data)
            trainer.train()
            self.assertEqual(trainer.current_train_iteration, 2)
            self.assertTrue((Path(config["training"]["output_dir"]) / "checkpoint-000000002/minimax_h3_pruned_vae.safetensors").is_file())
            config["resume"]["auto_resume"] = True
            resumed = VAEDistillationTrainer(config)
            resumed.set_model(model)
            resumed.set_data(data)
            resumed.train()
            self.assertEqual(resumed.current_train_iteration, 2)
            self.assertTrue(all(parameter.grad is None for parameter in model.teacher_vae.parameters()))


if __name__ == "__main__":
    unittest.main()
