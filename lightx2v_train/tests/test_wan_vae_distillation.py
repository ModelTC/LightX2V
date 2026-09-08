"""CPU loss/gradient contracts without released weights or LPIPS downloads."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from lightx2v_train.model_capabilities import VAEDistillationStepContext
from lightx2v_train.model_zoo.wan.capability_adapters.wan_vae_distillation_capability import (
    WanVAEDistillationCapability,
)
from lightx2v_train.trainers.vae.adversarial import VAEAdversarialObjective


class TinyLPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, prediction, target):
        self.seen.append((prediction.detach(), target.detach()))
        return (prediction - target).square().mean((1, 2, 3), keepdim=True)


class TinyWrapper(nn.Module):
    def __init__(self, component):
        super().__init__()
        self.component = component
        self.teacher_encoder = nn.Conv3d(3, 4, 1).requires_grad_(False)
        self.teacher_decoder = nn.Conv3d(2, 3, 1).requires_grad_(False)
        self.teacher_suffix = nn.Conv3d(3, 3, 1).requires_grad_(False)
        self.student = nn.Conv3d(3, 4, 1) if component == "encoder" else nn.Conv3d(2, 3, 1)
        self.calls = []

    @property
    def device(self):
        return self.student.weight.device

    def distillation_forward(self, video, *, running_dtype, return_features, auxiliary_feature_index):
        self.calls.append((video.detach().clone(), running_dtype, return_features, auxiliary_feature_index))
        pixels = F.avg_pool3d(video * 2 - 1, (1, 8, 8))
        with torch.no_grad():
            teacher_moments = self.teacher_encoder(pixels)
            teacher_mu, teacher_std = teacher_moments.chunk(2, 1)
            teacher_std = F.softplus(teacher_std)
            teacher_raw = self.teacher_decoder(teacher_mu)
            teacher_prediction = F.interpolate(teacher_raw, size=video.shape[2:], mode="nearest").tanh()
        if self.component == "encoder":
            student_moments = self.student(pixels)
            student_mu, student_std = student_moments.chunk(2, 1)
            student_std = F.softplus(student_std)
            # Frozen decoder weights must still transmit gradients into the encoder.
            student_raw = self.teacher_decoder(student_mu)
            student_features, teacher_features = (student_moments,), (teacher_moments,)
        else:
            student_raw = self.student(teacher_mu)
            student_mu, student_std = None, None
            student_features, teacher_features = (student_raw,), (teacher_raw,)
        prediction = F.interpolate(student_raw, size=video.shape[2:], mode="nearest").tanh()
        auxiliary = None
        if auxiliary_feature_index is not None:
            auxiliary = F.interpolate(self.teacher_suffix(student_raw), size=video.shape[2:], mode="nearest").tanh()
        return {
            "prediction": prediction, "teacher_prediction": teacher_prediction,
            "student_features": student_features if return_features else (),
            "teacher_features": teacher_features if return_features else (),
            "auxiliary_prediction": auxiliary,
            "student_mu": student_mu, "student_std": student_std,
            "teacher_mu": teacher_mu, "teacher_std": teacher_std,
            "teacher_latents": teacher_mu[:, :, ::4],
        }


def tiny_config(**weights):
    return {
        "reconstruction_weight": 0, "posterior_weight": 0,
        "perceptual_gradient_checkpointing": True,
        "perceptual_frame_batch_size": 1,
        "auxiliary_decoder": {"student_feature_indices": [1], "ramp_iters": 0},
        "stages": [{"name": "tiny", "start_iter": 0, "crop_num_frames": 5, "crop_size": 16, "weights": weights}],
    }


class WanVAEDistillationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.video = torch.rand(1, 3, 9, 24, 24)
        self.batch = {"inputs": {"video": self.video, "latents": torch.full((1,), float("nan"))}}
        self.context = VAEDistillationStepContext(running_dtype=torch.float32)

    def check_gradients(self, model, result):
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        gradients = [parameter.grad for parameter in model.student.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
        for name, parameter in model.named_parameters():
            if name.startswith("teacher_"):
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)

    def test_default_search_and_recovery_schedules(self):
        for component in ("encoder", "decoder"):
            model = TinyWrapper(component)
            search = WanVAEDistillationCapability(model, {"phase": "search"})
            stage = search.stages[0]
            self.assertEqual((stage.crop_num_frames, stage.crop_size), (33, 256))
            self.assertEqual(stage.weights["perceptual"], 0.05)
            self.assertEqual(stage.weights["posterior"], float(component == "encoder"))
            self.assertFalse(search.uses_auxiliary_loss)
            recover = WanVAEDistillationCapability(model, {})
            early = recover._stage_for(599)[1]
            late = recover._stage_for(600)[1]
            self.assertEqual((early.crop_num_frames, early.crop_size), (33, 256))
            self.assertEqual((late.crop_num_frames, late.crop_size), (65, 384))
            self.assertEqual((early.weights["perceptual"], early.weights["feature"]), (0.1, 0.01))
            self.assertEqual((late.weights["perceptual"], late.weights["feature"]), (0.2, 0.005))
            self.assertEqual((late.weights["auxiliary"], late.weights["adversarial"]), (0.1, 0.5))
            self.assertEqual(recover._auxiliary_ramp(599), 0)
            self.assertAlmostEqual(recover._auxiliary_ramp(600), 1 / 200)
            self.assertEqual(recover._auxiliary_ramp(799), 1)

    def test_shipped_configs_use_supported_stage_and_auxiliary_contracts(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs/train/vae"
        for component, keep in (("encoder", 3), ("decoder", 5)):
            for phase in ("search", "recover"):
                with self.subTest(component=component, phase=phase):
                    path = config_dir / f"wan21_{component}_prune_{phase}_keep{keep}_ddp.yaml"
                    with path.open() as handle:
                        config = yaml.safe_load(handle)["training"]["vae_distillation"]
                    model = TinyWrapper(component)
                    capability = WanVAEDistillationCapability(model, config)
                    first = capability.stages[0]
                    self.assertEqual((first.crop_num_frames, first.crop_size), (33, 256))
                    self.assertEqual(first.weights["posterior"], float(component == "encoder"))
                    if phase == "recover":
                        late = capability._stage_for(600)[1]
                        self.assertEqual((late.crop_num_frames, late.crop_size), (65, 384))
                        self.assertEqual(config["gan"]["spatial_compression_ratio"], 8)
                        self.assertEqual(config["gan"]["discriminator_warmup_iters"], 100)
                        self.assertEqual(config["gan"]["generator_ramp_iters"], 300)
                        expected = (5, 7, 8) if component == "encoder" else (8, 11, 12)
                        self.assertEqual(capability.auxiliary_feature_indices, expected)

    def test_each_sample_is_cropped_in_rgb_and_reencoded_without_cached_latents(self):
        model = TinyWrapper("decoder")
        capability = WanVAEDistillationCapability(model, tiny_config(reconstruction=1))
        batch = {"inputs": {"video": self.video.expand(2, -1, -1, -1, -1), "latents": self.batch["inputs"]["latents"]}}
        first = capability.compute_loss(batch, self.context)
        first_crop = model.calls[-1][0]
        capability.compute_loss(batch, self.context)
        torch.testing.assert_close(model.calls[-1][0], first_crop, rtol=0, atol=0)
        self.assertEqual(tuple(first_crop.shape), (2, 3, 5, 16, 16))
        self.assertFalse(torch.equal(first_crop[0], first_crop[1]))
        capability.compute_loss(batch, VAEDistillationStepContext(running_dtype=torch.float32, micro_step=1))
        self.assertFalse(torch.equal(model.calls[-1][0], first_crop))
        self.assertEqual(first.metrics["supervised_frames"], 5)
        self.assertEqual(model.calls[-1][1], torch.float32)
        self.assertFalse(model.calls[-1][2])
        self.assertIsNone(model.calls[-1][3])

    def test_raw_reconstruction_gradient_reaches_each_component_through_frozen_modules(self):
        for component in ("encoder", "decoder"):
            with self.subTest(component=component):
                model = TinyWrapper(component)
                capability = WanVAEDistillationCapability(model, tiny_config(reconstruction=1))
                result = capability.compute_loss(self.batch, self.context)
                crop, dtype, need_features, anchor = model.calls[-1]
                output = model.distillation_forward(crop, running_dtype=dtype, return_features=need_features, auxiliary_feature_index=anchor)
                expected = capability._charbonnier(output["prediction"], crop * 2 - 1, capability.charbonnier_epsilon)
                torch.testing.assert_close(result.loss, expected)
                self.check_gradients(model, result)

    def test_feature_loss_is_direct_mse_and_preserves_student_gradient(self):
        model = TinyWrapper("decoder")
        capability = WanVAEDistillationCapability(model, tiny_config(feature=1))
        result = capability.compute_loss(self.batch, self.context)
        self.assertTrue(model.calls[-1][2])
        student = torch.tensor([1.0, 4.0], requires_grad=True)
        teacher = torch.tensor([2.0, 2.0], requires_grad=True)
        feature_loss = capability._matching_mse(student, teacher)
        self.assertEqual(feature_loss.item(), 2.5)
        feature_loss.backward()
        self.assertIsNone(teacher.grad)
        with self.assertRaises(ValueError):
            capability._matching_mse(student, teacher[:1])
        self.check_gradients(model, result)

    def test_normalized_posterior_alignment_uses_mean_and_std_without_rescaling(self):
        model = TinyWrapper("encoder")
        capability = WanVAEDistillationCapability(model, tiny_config(posterior=1))
        result = capability.compute_loss(self.batch, self.context)
        crop, dtype, features, anchor = model.calls[-1]
        outputs = model.distillation_forward(crop, running_dtype=dtype, return_features=features, auxiliary_feature_index=anchor)
        mean = F.mse_loss(outputs["student_mu"], outputs["teacher_mu"])
        std = F.mse_loss(outputs["student_std"], outputs["teacher_std"])
        torch.testing.assert_close(result.loss, mean + std)
        torch.testing.assert_close(result.metrics["posterior_mean"], mean)
        torch.testing.assert_close(result.metrics["posterior_std"], std)
        self.check_gradients(model, result)

    def test_shared_perceptual_helper_with_checkpointing_needs_no_download(self):
        model = TinyWrapper("decoder")
        capability = WanVAEDistillationCapability(model, tiny_config(perceptual=1))
        capability._perceptual_model = TinyLPIPS()
        result = capability.compute_loss(self.batch, self.context)
        self.assertEqual(len(capability._perceptual_model.seen), 2)
        for prediction, target in capability._perceptual_model.seen:
            self.assertEqual(tuple(prediction.shape), (1, 3, 16, 16))
            self.assertTrue((prediction.abs() <= 1).all())
            self.assertTrue((target.abs() <= 1).all())
        self.check_gradients(model, result)

    def test_auxiliary_target_is_teacher_not_source_and_suffix_transmits_gradients(self):
        for component in ("encoder", "decoder"):
            with self.subTest(component=component):
                model = TinyWrapper(component)
                capability = WanVAEDistillationCapability(model, tiny_config(auxiliary=1))
                result = capability.compute_loss(self.batch, self.context)
                crop, dtype, features, anchor = model.calls[-1]
                self.assertEqual(anchor, 1)
                outputs = model.distillation_forward(crop, running_dtype=dtype, return_features=features, auxiliary_feature_index=anchor)
                expected, parts = capability._auxiliary_loss(outputs["auxiliary_prediction"], outputs["teacher_prediction"])
                torch.testing.assert_close(result.loss, expected)
                for name, value in parts.items():
                    torch.testing.assert_close(result.metrics[name], value)
                self.check_gradients(model, result)

    def test_auxiliary_differences_detach_teacher_and_have_zero_identity_loss(self):
        model = TinyWrapper("decoder")
        capability = WanVAEDistillationCapability(model, tiny_config(auxiliary=1))
        target = torch.rand(1, 3, 5, 8, 8, requires_grad=True)
        identity, _ = capability._auxiliary_loss(target, target)
        self.assertEqual(identity.item(), 0)
        prediction = torch.rand_like(target, requires_grad=True)
        value, metrics = capability._auxiliary_loss(prediction, target)
        value.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        self.assertGreater(metrics["auxiliary_spatiotemporal"].item(), 0)

    def test_stage_crops_and_gan_use_rgb_detached_teacher_latents_and_stage_relative_step(self):
        model = TinyWrapper("decoder")
        config = tiny_config(reconstruction=1)
        config["stages"].append({
            "name": "late", "start_iter": 2, "crop_num_frames": 9, "crop_size": 24,
            "weights": {"reconstruction": 1, "adversarial": 0.5},
        })
        capability = WanVAEDistillationCapability(model, config)
        calls = []

        def adversarial(fake, real, latent, stage_step):
            calls.append((fake, real, latent, stage_step))
            return (fake - real).square().mean(), {"adversarial_ramp": 0.25}

        early = capability.compute_loss(self.batch, self.context)
        self.assertEqual(early.metrics["supervised_frames"], 5)
        self.assertEqual(calls, [])
        late_context = VAEDistillationStepContext(running_dtype=torch.float32, iteration=3, adversarial_objective=adversarial)
        result = capability.compute_loss(self.batch, late_context)
        fake, real, latent, step = calls[0]
        self.assertEqual(step, 1)
        self.assertEqual(tuple(fake.shape), (1, 3, 9, 24, 24))
        self.assertTrue((fake >= 0).all() and (fake <= 1).all())
        torch.testing.assert_close(real, self.video)
        self.assertFalse(latent.requires_grad)
        self.assertEqual(result.metrics["adversarial_ramp"], 0.25)
        torch.testing.assert_close(result.metrics["weighted_adversarial"], 0.5 * (fake - real).square().mean())
        self.check_gradients(model, result)

    def test_missing_gan_and_invalid_schedule_fail_before_forward(self):
        model = TinyWrapper("decoder")
        capability = WanVAEDistillationCapability(model, tiny_config(adversarial=1))
        with self.assertRaises(RuntimeError):
            capability.compute_loss(self.batch, self.context)
        self.assertEqual(model.calls, [])
        for frames, size in ((4, 16), (5, 15)):
            config = tiny_config(reconstruction=1)
            config["stages"][0].update(crop_num_frames=frames, crop_size=size)
            with self.assertRaises(ValueError):
                WanVAEDistillationCapability(model, config)
        with self.assertRaises(ValueError):
            WanVAEDistillationCapability(model, tiny_config(posterior=1))
        with self.assertRaises(ValueError):
            WanVAEDistillationCapability(model, tiny_config(unknown=1))
        capability = WanVAEDistillationCapability(model, tiny_config(reconstruction=1))
        with self.assertRaises(ValueError):
            capability.compute_loss({"inputs": {"video": self.video[:, :, :1]}}, self.context)

    def test_real_adversarial_objective_wan_crop_warmup_and_generator_ramp(self):
        model = TinyWrapper("decoder")
        capability = WanVAEDistillationCapability(model, tiny_config(adversarial=1))
        objective = VAEAdversarialObjective(
            {
                "spatial_compression_ratio": 8, "crop_size": 16, "max_frames": 5,
                "discriminator_warmup_iters": 100, "generator_ramp_iters": 300,
                "discriminator": {
                    "base_channels": 2, "condition_channels": 2, "channel_multipliers": [1],
                    "temporal_strides": [1], "group_norm_groups": 1, "spectral_normalization": False,
                },
            },
            device=torch.device("cpu"), latent_channels=2, gradient_accumulation_iters=1,
        )
        for iteration, ramp in ((99, 0), (100, 1 / 300), (399, 1)):
            context = VAEDistillationStepContext(
                running_dtype=torch.float32, iteration=iteration, adversarial_objective=objective,
            )
            result = capability.compute_loss(self.batch, context)
            self.assertAlmostEqual(result.metrics["adversarial_ramp"], ramp)
            self.assertEqual(result.metrics["adversarial_crop_height"], 16)
            self.assertEqual(result.metrics["adversarial_crop_width"], 16)
            self.assertTrue(any(parameter.grad is not None for parameter in objective.discriminator_parameters))
            if ramp:
                self.check_gradients(model, result)
            else:
                self.assertEqual(result.loss.item(), 0)
            objective.step()
            model.student.zero_grad(set_to_none=True)

    def test_gan_condition_uses_first_token_once_and_later_tokens_four_times(self):
        model = TinyWrapper("decoder")
        config = tiny_config(adversarial=1)
        config["stages"][0]["crop_num_frames"] = 9
        capability = WanVAEDistillationCapability(model, config)
        forward = model.distillation_forward

        def tracked(*args, **kwargs):
            outputs = forward(*args, **kwargs)
            outputs["teacher_latents"] = torch.arange(3.0).view(1, 1, 3, 1, 1).requires_grad_(True)
            return outputs

        def adversarial(fake, real, condition, stage_step):
            self.assertEqual(condition.shape[2], 9)
            self.assertFalse(condition.requires_grad)
            torch.testing.assert_close(condition.flatten(), torch.tensor([0., 1., 1., 1., 1., 2., 2., 2., 2.]))
            self.assertEqual([condition[0, 0, index, 0, 0].item() for index in (0, 1, 4, 5)], [0, 1, 1, 2])
            return (fake - real).square().mean(), {}

        context = VAEDistillationStepContext(running_dtype=torch.float32, adversarial_objective=adversarial)
        with patch.object(model, "distillation_forward", side_effect=tracked):
            result = capability.compute_loss(self.batch, context)
        self.check_gradients(model, result)

    def test_teacher_feature_shape_and_count_cannot_broadcast_silently(self):
        model = TinyWrapper("decoder")
        capability = WanVAEDistillationCapability(model, tiny_config(feature=1))
        forward = model.distillation_forward

        def mismatched(*args, **kwargs):
            outputs = forward(*args, **kwargs)
            outputs["teacher_features"] = ()
            return outputs

        with patch.object(model, "distillation_forward", side_effect=mismatched):
            with self.assertRaises(ValueError):
                capability.compute_loss(self.batch, self.context)


if __name__ == "__main__":
    unittest.main()
