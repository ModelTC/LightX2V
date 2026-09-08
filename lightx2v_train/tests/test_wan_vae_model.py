"""Real Wan VAE wrapper/native integration on temporary, tiny CPU checkpoints."""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from lightx2v_train.model_capabilities import VAEDistillationCapability, VAEDistillationStepContext
from lightx2v_train.model_zoo.native.wan.modules.vae import WanVAE_
from lightx2v_train.model_zoo.native.wan.pruned_vae import WAN_VAE_CONFIG, normalized_posterior_stats
from lightx2v_train.model_zoo.wan.wan_pruned_vae import WanPrunedDecoderModel, WanPrunedEncoderModel


class WanVAEModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(41)
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.architecture = {**WAN_VAE_CONFIG, "dim": 4}
        self.source = WanVAE_(**self.architecture).eval()
        self.source_path = self.root / "Wan2.1_VAE.pth"
        torch.save(self.source.state_dict(), self.source_path)
        self.video = torch.rand(1, 3, 5, 16, 16)
        self.batch = {"inputs": {"video": self.video}}
        self.context = VAEDistillationStepContext(running_dtype=torch.float32)

    def config(self, component, *, search=False, grouped=False, weights=None):
        keep = 3 if component == "encoder" else 5
        pruning = {"search": {"keep_residuals": keep, "lora_rank": 2, "lora_alpha": 4, "gate_scale": 1.0}}
        if not search:
            pruning = {"kept_residual_indices": [1, 5, 8] if component == "encoder" else [0, 3, 6, 9, 12]}
        elif grouped:
            pruning["search"].pop("keep_residuals")
            pruning["search"].update(grouping="stage", keep_per_group=1)
        losses = {"reconstruction": 1.0, "posterior": float(component == "encoder")}
        if weights is not None:
            losses = weights
        return {
            "model": {
                "pretrained_model_name_or_path": str(self.source_path), "running_dtype": "fp32",
                "student_param_dtype": "fp32", "teacher_autocast_dtype": "fp32", "student_autocast": False,
                "vae_architecture": self.architecture, f"pruned_{component}": pruning,
            },
            "training": {
                "vae_distillation": {
                    "posterior_weight": 0.0, "reconstruction_weight": 0.0,
                    "auxiliary_decoder": {
                        "student_feature_indices": [5] if component == "encoder" else [8], "ramp_iters": 0,
                    },
                    "stages": [{
                        "name": "tiny", "start_iter": 0, "crop_num_frames": 5, "crop_size": 16,
                        "weights": losses,
                    }],
                },
            },
        }

    def model(self, component, config):
        cls = WanPrunedEncoderModel if component == "encoder" else WanPrunedDecoderModel
        with patch("torch.cuda.is_available", return_value=False):
            model = cls(config)
        model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
        model.set_full_trainable()
        model.enable_gradient_checkpointing()
        model.ensure_capabilities()
        self.assertEqual(model.device, torch.device("cpu"))
        return model

    def loss(self, model):
        return model.capabilities.require(VAEDistillationCapability).compute_loss(self.batch, self.context)

    def assert_teacher_frozen(self, model):
        for teacher in (model.teacher_encoder, model.teacher_decoder):
            self.assertFalse(teacher.training)
            self.assertTrue(teacher.gradient_checkpointing)
            for parameter in teacher.parameters():
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)

    def assert_nonzero_finite_gradients(self, parameters):
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)

    def test_search_backpropagates_gates_and_lora_through_real_components(self):
        self.assert_search_backpropagation(grouped=False)

    def test_stage_search_backpropagates_gates_and_lora_through_real_components(self):
        self.assert_search_backpropagation(grouped=True)

    def assert_search_backpropagation(self, *, grouped):
        for component in ("encoder", "decoder"):
            with self.subTest(component=component):
                model = self.model(component, self.config(component, search=True, grouped=grouped))
                student = model.transformer
                student.prepare_search_step(1)
                if grouped:
                    self.assertEqual(student.search_grouping, "stage")
                    self.assertEqual(student.keep_residuals, 5)
                    self.assertEqual(len(student.candidate_masks), 32 if component == "encoder" else 162)
                    mask = student._search_mask(1)
                    for group in student.residual_groups:
                        torch.testing.assert_close(mask[:, group].sum(-1), torch.ones(1))
                result = self.loss(model)
                self.assertTrue(torch.isfinite(result.loss))
                result.loss.backward()
                self.assert_nonzero_finite_gradients([student.gate_logits])
                if grouped:
                    for group in student.residual_groups:
                        self.assertGreater(student.gate_logits.grad[list(group)].abs().sum().item(), 0)
                named = dict(student.named_parameters())
                self.assert_nonzero_finite_gradients([parameter for name, parameter in named.items() if "lora_B" in name])
                self.assertTrue(all(name == "gate_logits" or "lora_" in name for name, parameter in named.items() if parameter.requires_grad))
                self.assertTrue(all(parameter.grad is None for parameter in named.values() if not parameter.requires_grad))
                self.assert_teacher_frozen(model)

                # A receives its gradient once the initially zero B adapter is nonzero.
                with torch.no_grad():
                    for name, parameter in named.items():
                        if "lora_B" in name:
                            parameter.add_(0.01)
                student.zero_grad(set_to_none=True)
                self.loss(model).loss.backward()
                self.assert_nonzero_finite_gradients([parameter for name, parameter in named.items() if "lora_A" in name])
                self.assert_teacher_frozen(model)
                student.clear_search_step()

    def test_wrapper_teacher_posterior_and_prediction_match_native_streaming(self):
        for component, count, anchor in (("encoder", 10, 5), ("decoder", 14, 8)):
            with self.subTest(component=component):
                model = self.model(component, self.config(component))
                result = model.distillation_forward(
                    self.video, running_dtype=torch.float32, return_features=True, auxiliary_feature_index=anchor,
                )
                self.assertEqual(tuple(result["teacher_latents"].shape), (1, 16, 2, 2, 2))
                self.assertEqual(len(result["student_features"]), count)
                self.assertEqual(len(result["teacher_features"]), count)
                for student, teacher in zip(result["student_features"], result["teacher_features"], strict=True):
                    self.assertEqual(student.shape, teacher.shape)
                    self.assertFalse(teacher.requires_grad)
                with torch.no_grad():
                    pixels = self.video * 2 - 1
                    scale = [model.teacher_encoder.latents_mean, model.teacher_encoder.latents_std.reciprocal()]
                    expected_latents = self.source.encode(pixels, scale)
                    expected_prediction = self.source.decode(expected_latents, scale)
                    moments = model.teacher_encoder(pixels)
                    expected_mu, expected_std = normalized_posterior_stats(moments)
                torch.testing.assert_close(result["teacher_latents"], expected_latents, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(result["teacher_prediction"], expected_prediction, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(result["teacher_mu"], expected_mu)
                torch.testing.assert_close(result["teacher_std"], expected_std)
                for key in ("teacher_latents", "teacher_prediction", "teacher_mu", "teacher_std"):
                    self.assertFalse(result[key].requires_grad)
                self.assertEqual(result["prediction"].shape, self.video.shape)
                self.assertEqual(result["auxiliary_prediction"].shape, self.video.shape)
                self.assertTrue(result["auxiliary_prediction"].requires_grad)
                self.assert_teacher_frozen(model)

    def test_recovery_reconstruction_feature_posterior_and_auxiliary_backward(self):
        for component in ("encoder", "decoder"):
            with self.subTest(component=component):
                weights = {"reconstruction": 1, "feature": 0.01, "auxiliary": 0.1, "posterior": float(component == "encoder")}
                model = self.model(component, self.config(component, weights=weights))
                result = self.loss(model)
                self.assertGreater(result.metrics["reconstruction"].item(), 0)
                self.assertGreater(result.metrics["feature"].item(), 0)
                self.assertGreater(result.metrics["auxiliary"].item(), 0)
                if component == "encoder":
                    self.assertGreater(result.metrics["posterior"].item(), 0)
                result.loss.backward()
                self.assert_nonzero_finite_gradients(model.transformer.parameters())
                self.assertTrue(all(parameter.grad is not None for parameter in model.transformer.parameters()))
                self.assert_teacher_frozen(model)

    def test_auxiliary_only_updates_student_prefix_through_frozen_teacher_suffix(self):
        for component in ("encoder", "decoder"):
            with self.subTest(component=component):
                model = self.model(component, self.config(component, weights={"auxiliary": 1}))
                result = self.loss(model)
                result.loss.backward()
                self.assert_nonzero_finite_gradients([model.transformer.network.conv1.weight])
                self.assertIsNone(model.transformer.network.head[-1].weight.grad)
                self.assert_teacher_frozen(model)

    def test_export_uses_original_weights_and_recovery_loads_selection(self):
        self.assert_export_and_recovery(grouped=False)

    def test_stage_export_preserves_group_budget_teacher_weights_and_recovery_gradients(self):
        self.assert_export_and_recovery(grouped=True)

    def assert_export_and_recovery(self, *, grouped):
        original = self.source.state_dict()
        for component in ("encoder", "decoder"):
            with self.subTest(component=component):
                model = self.model(component, self.config(component, search=True, grouped=grouped))
                with torch.no_grad():
                    for name, parameter in model.transformer.named_parameters():
                        if "lora_" in name:
                            parameter.fill_(0.75)
                        elif name != "gate_logits":
                            parameter.add_(0.25)
                gate_logits = torch.arange(model.transformer.depth, dtype=torch.float32)
                expected_kept = model.transformer.selected_layers(gate_logits)
                export_dir = self.root / component / "export"
                model.export_pruned_component(export_dir, gate_logits)
                selection_path = export_dir / "kept_layers.json"
                selection = json.loads(selection_path.read_text())
                self.assertEqual(selection["kept_residual_indices"], expected_kept)
                self.assertEqual(selection["original_residual_count"], 10 if component == "encoder" else 14)
                self.assertEqual(selection["initialization"], "original_teacher_weights_without_search_lora")
                self.assertEqual(selection["teacher_architecture"], self.architecture)
                self.assertEqual(selection["search_grouping"], "stage" if grouped else "global")
                self.assertEqual(selection["residual_groups"], [list(group) for group in model.transformer.residual_groups])
                self.assertEqual(selection["kept_per_group"], [
                    len(set(group).intersection(expected_kept)) for group in model.transformer.residual_groups
                ])
                if grouped:
                    self.assertEqual(len(expected_kept), 5)
                    self.assertEqual(selection["kept_per_group"], [1] * 5)
                self.assertFalse((export_dir / "kept_layers.json.tmp").exists())
                exported_state = load_file(str(export_dir / model.transformer.weights_name))
                self.assertFalse(any("lora_" in name or "gate_logits" in name for name in exported_state))
                for name, value in exported_state.items():
                    if name not in ("latents_mean", "latents_std"):
                        torch.testing.assert_close(value, original[name], rtol=0, atol=0)

                config = self.config(component)
                config["model"][f"pruned_{component}"] = {"selection_path": str(selection_path)}
                restored = self.model(component, config)
                self.assertEqual(restored.transformer.kept_residual_indices, expected_kept)
                for name, value in restored.transformer.state_dict().items():
                    torch.testing.assert_close(value, exported_state[name], rtol=0, atol=0)
                self.loss(restored).loss.backward()
                self.assert_nonzero_finite_gradients(restored.transformer.parameters())
                self.assertTrue(all(parameter.grad is not None for parameter in restored.transformer.parameters()))
                self.assert_teacher_frozen(restored)

    def test_reconstruct_has_native_shape_and_rgb_range_for_single_and_multiple_frames(self):
        for component in ("encoder", "decoder"):
            with self.subTest(component=component):
                model = self.model(component, self.config(component))
                for frames in (1, 5, 9):
                    video = torch.rand(1, 3, frames, 16, 24)
                    actual = model.reconstruct(video)
                    self.assertEqual(actual.shape, video.shape)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue((actual >= 0).all() and (actual <= 1).all())
                    self.assertFalse(actual.requires_grad)

    def test_consolidated_metadata_and_initial_weights_roundtrip(self):
        for component in ("encoder", "decoder"):
            with self.subTest(component=component):
                config = self.config(component)
                model = self.model(component, config)
                with torch.no_grad():
                    model.transformer.network.conv1.weight.add_(0.02)
                checkpoint_path = self.root / f"{component}_recovery.safetensors"
                metadata = model.consolidated_safetensors_metadata()
                self.assertEqual(metadata["model_type"], f"wan21_pruned_{component}")
                self.assertEqual(json.loads(metadata["architecture"]), model.transformer.architecture_config)
                save_file(model.transformer.state_dict(), str(checkpoint_path), metadata=metadata)
                config["model"]["initial_weights_path"] = str(checkpoint_path)
                restored = self.model(component, config)
                for name, value in restored.transformer.state_dict().items():
                    torch.testing.assert_close(value, model.transformer.state_dict()[name], rtol=0, atol=0)
                torch.testing.assert_close(restored.reconstruct(self.video), model.reconstruct(self.video), rtol=0, atol=0)
                bad_config = copy.deepcopy(config)
                bad_config["model"][f"pruned_{component}"]["kept_residual_indices"] = [0]
                with self.assertRaises(ValueError):
                    self.model(component, bad_config)


if __name__ == "__main__":
    unittest.main()
