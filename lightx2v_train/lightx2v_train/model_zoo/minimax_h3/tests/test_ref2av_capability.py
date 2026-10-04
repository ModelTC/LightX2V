import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import default_collate

from lightx2v_train.data.minimax_h3_cache_dataset import _strip_condition_batch
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import (
    MiniMaxH3DistributionMatchingCapability,
)
from lightx2v_train.model_zoo.minimax_h3.minimax_h3_ref2av import MiniMaxH3Ref2AVModel
from lightx2v_train.model_zoo.native.minimax_h3 import build_row_timesteps
from lightx2v_train.model_zoo.native.minimax_h3.modeling import _transformer_class
from lightx2v_train.utils.registry import build_model
from lightx2v_train.utils.utils import is_train_cache_dataset


def model_wrapper(reference=True):
    model = build_model(
        {
            "model": {
                "name": "minimax_h3_ref2av" if reference else "minimax_h3_t2av",
                "running_dtype": "bf16",
            }
        }
    )
    model.device = torch.device("cpu")
    model.patch_size = (1, 2, 2)
    model.video_latent_channels = 24
    model.audio_latent_channels = 32
    model.vae_spatial_scale_factor = 16
    model.use_autocast = False
    return model


def reference_condition(audio=True):
    condition = {
        "task": "ref2av",
        "prompt_embeds": torch.randn(1, 4, 5120),
        "text_token_tags": torch.tensor([1, 0, 0, 1]),
        "target_height": 64,
        "target_width": 96,
        "target_num_frames": 124,
        "references": [
            {
                "kind": "image",
                "normalized": True,
                "num_latent_frames": 1,
                "latent_height": 4,
                "latent_width": 6,
                "video_latents": torch.randn(6, 96),
            }
        ],
    }
    if audio:
        condition["references"].append(
            {
                "kind": "audio",
                "normalized": True,
                "num_audio_latents": 2,
                "audio_latents": torch.randn(4, 32),
            }
        )
    return condition


class Ref2AVCapabilityTest(unittest.TestCase):
    def capability(self, model=None, **options):
        self.model = model or model_wrapper()
        return MiniMaxH3DistributionMatchingCapability(self.model, {"geometry_from_metadata": True, **options})

    def prepare(self, capability, condition):
        sample = default_collate(
            [
                {
                    "inputs": {},
                    "conditioning": {"positive": _strip_condition_batch(condition)},
                    "meta": {"target_height": condition["target_height"], "target_width": condition["target_width"], "target_num_frames": condition["target_num_frames"]},
                }
            ]
        )
        return sample, capability.encode_conditions(sample, "", 1.0, lambda x: x)[0]

    def test_registry_selects_reference_partition(self):
        model = model_wrapper()
        self.assertIsInstance(model, MiniMaxH3Ref2AVModel)
        self.assertEqual(model.transformer_component, "transformer_ref")
        self.assertTrue(is_train_cache_dataset({"data": {"train": {"name": "minimax_h3_ref_cache_dataset"}}}))

    def test_precise_reference_rows_and_shared_geometry(self):
        capability = self.capability(allowed_resolutions=[[64, 96], [96, 64]])
        cached = reference_condition()
        sample, condition = self.prepare(capability, cached)
        torch.testing.assert_close(condition["condition_video_latents"], cached["references"][0]["video_latents"], rtol=0, atol=0)
        self.assertEqual(condition["prompt_embeds"].dtype, torch.bfloat16)
        self.assertEqual(condition["condition_video_latents"].dtype, torch.float32)
        self.assertEqual(condition["condition_audio_latents"].dtype, torch.float32)
        shape = capability.latent_shape(sample, [{"value": [124, 64, 96]}], lambda x: x)
        self.assertEqual(shape.video_tokens, (1, 222, 96))
        self.assertEqual(shape.audio_tokens, (1, 414, 32))
        layout = capability._layout(condition, shape)
        self.assertEqual(layout.num_condition_video_rows, 6)
        self.assertEqual(layout.num_condition_audio_rows, 4)
        values, indices = build_row_timesteps(layout, 0.2, 0.3)
        times = values[indices]
        torch.testing.assert_close(times[layout.video_indices[:6]], torch.full((6,), 0.999))
        torch.testing.assert_close(times[layout.audio_indices[:4]], torch.ones(4))
        self.assertIs(capability._layout(condition, shape), layout)

    def test_prediction_excludes_reference_rows_from_losses(self):
        model = model_wrapper()
        capability = self.capability(model)
        sample, condition = self.prepare(capability, reference_condition())
        shape = capability.latent_shape(sample, [{"value": [124, 64, 96]}], lambda x: x)
        generated = capability.initial_latents(shape, torch.float32, lambda x: x)
        captured = {}

        def transformer(**kwargs):
            captured.update(kwargs)
            return kwargs["hidden_states"] + 2, kwargs["audio_hidden_states"] + 3

        model.transformer = transformer
        predicted = capability.predict_velocity(generated, torch.tensor([0.5]), condition)
        torch.testing.assert_close(predicted.video, generated.video + 2)
        torch.testing.assert_close(predicted.audio, generated.audio + 3)
        self.assertEqual(captured["hidden_states"].shape[1], generated.video.shape[1] + 6)
        self.assertEqual(captured["audio_hidden_states"].shape[1], generated.audio.shape[1] + 4)

    def test_fixed_duration_rejects_audio_and_preserves_image_cache(self):
        capability = self.capability(fixed_num_frames=141)
        with self.assertRaisesRegex(ValueError, "image-only"):
            self.prepare(capability, reference_condition())
        cached = reference_condition(audio=False)
        sample, condition = self.prepare(capability, cached)
        self.assertEqual(cached["target_num_frames"], 124)
        self.assertEqual(condition["target_num_frames"], 141)
        shape = capability.latent_shape(sample, [], lambda x: x)
        self.assertEqual(shape.num_frames, 141)

    def test_geometry_mismatch_rejected(self):
        capability = self.capability()
        sample, condition = self.prepare(capability, reference_condition())
        sample["meta"]["target_height"] = torch.tensor([96])
        shape = capability.latent_shape(sample, [], lambda x: x)
        with self.assertRaisesRegex(ValueError, "cached target_height"):
            capability._layout(condition, shape)

    def test_keyframe_tasks_keep_conditions_fixed(self):
        model = model_wrapper(reference=False)
        capability = self.capability(model)
        for task, anchors in [("i2av", [0]), ("l2av", [1]), ("fl2av", [0, 1])]:
            cached = {
                "task": task,
                "prompt_embeds": torch.randn(1, 4, 5120),
                "text_token_tags": torch.ones(4, dtype=torch.long),
                "target_height": 64,
                "target_width": 96,
                "target_num_frames": 124,
                "keyframe_anchors": torch.tensor(anchors),
                "condition_video_latents": torch.randn(6 * len(anchors), 96),
            }
            sample, condition = self.prepare(capability, cached)
            shape = capability.latent_shape(sample, [], lambda x: x)
            layout = capability._layout(condition, shape)
            self.assertEqual(layout.num_condition_video_rows, len(anchors) * 6)

    def test_limited_layout_cache(self):
        capability = self.capability(layout_cache_size=1)
        sample, condition = self.prepare(capability, reference_condition())
        shape = capability.latent_shape(sample, [], lambda x: x)
        first = capability._layout(condition, shape)
        second_condition = dict(condition)
        second_condition["text_token_tags"] = torch.tensor([0, 1, 0, 1])
        second = capability._layout(second_condition, shape)
        self.assertIsNot(first, second)
        self.assertEqual(len(capability._layout_cache), 1)

    def test_projected_loss_routes_modalities_and_excludes_teacher_gradients(self):
        capability = self.capability(projected_dmd=True, dmd_normalization_epsilon=1e-6)
        sample, condition = self.prepare(capability, reference_condition())
        shape = capability.latent_shape(sample, [], lambda x: x)
        generated = MiniMaxH3JointLatents(torch.randn(1, 2, 3, requires_grad=True), torch.randn(1, 2, 2, requires_grad=True), shape)
        fake = MiniMaxH3JointLatents(torch.randn_like(generated.video, requires_grad=True), torch.randn_like(generated.audio, requires_grad=True), shape)
        teacher = MiniMaxH3JointLatents(torch.randn_like(generated.video, requires_grad=True), torch.randn_like(generated.audio, requires_grad=True), shape)
        loss = capability.dmd_loss(generated, fake, teacher)
        loss.backward()
        self.assertIsNotNone(generated.video.grad)
        self.assertIsNotNone(generated.audio.grad)
        self.assertIsNone(fake.video.grad)
        self.assertIsNone(teacher.audio.grad)

    def test_disabled_regularization_preserves_dmd_only_path(self):
        capability = self.capability()
        self.assertIsNone(capability.student_regularization(None, None, None, None, None))
        self.assertEqual(capability.extra_training_state(), {})
        capability.after_optimizer_step("student")

    def test_reference_training_rejects_target_regularization(self):
        with self.assertRaisesRegex(ValueError, "condition-only"):
            self.capability(adaptive_video_regularization={"enabled": True})

    def test_sparse_attention_changes_only_student_role(self):
        model = model_wrapper(reference=False)
        model.transformer = object()
        capability = self.capability(model, student_sparse_attention={"enabled": True})
        with patch("lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability.install_minimax_h3_student_sla") as install:
            capability.prepare_role("fake")
            capability.prepare_role("teacher")
            install.assert_not_called()
            capability.prepare_role("student")
            install.assert_called_once()

    def test_adaptive_joint_regression_packing_commit_and_resume(self):
        model = model_wrapper(reference=False)
        capability = self.capability(
            model,
            adaptive_video_regularization={
                "enabled": True,
                "regression": {"enabled": True, "weight": 2.0},
                "audio_regression": {"enabled": True, "weight": 0.25},
                "temporal": {"enabled": False},
            },
        )
        shape = capability.latent_shape({"meta": {"target_num_frames": 124, "target_height": 64, "target_width": 96}}, [], lambda x: x)
        condition = {
            "prompt_embeds": torch.randn(1, 4, 5120),
            "text_token_tags": torch.ones(4, dtype=torch.long),
        }
        model.transformer = lambda **kwargs: (
            torch.zeros_like(kwargs["hidden_states"]),
            torch.zeros_like(kwargs["audio_hidden_states"]),
        )
        capability.random_noise_like = lambda clean, dtype, broadcast: MiniMaxH3JointLatents(torch.zeros_like(clean.video), torch.zeros_like(clean.audio), clean.shape)
        generated = capability.initial_latents(shape, torch.float32, lambda x: x)
        sample = {
            "inputs": {
                "video_latents": {"normalized": True, "latents": torch.ones(1, 24, 37, 4, 6)},
                "audio_latents": 2 * torch.ones(1, 2, 32, 207),
            }
        }
        result = capability.student_regularization(generated, sample, condition, None, lambda x: x)
        torch.testing.assert_close(result.loss, torch.tensor(2.0))
        torch.testing.assert_close(result.metrics["adv_reg_raw"], torch.tensor(1.0))
        torch.testing.assert_close(result.metrics["adv_audio_reg_raw"], torch.tensor(4.0))
        with self.assertRaisesRegex(RuntimeError, "middle of a student optimizer"):
            capability.extra_training_state()
        capability.after_optimizer_step("student")
        saved = capability.extra_training_state()
        self.assertEqual(saved["adaptive_video_regularization"]["regression_ema"], [1.0])
        capability.adv_regularizer.regression_ema[0] = 5.0
        capability.load_extra_training_state(saved)
        self.assertEqual(capability.adv_regularizer.regression_ema, [1.0])

    def test_metadata_and_legacy_default_normalization(self):
        capability = self.capability(projected_dmd=True, dmd_reduction="sum", dmd_normalization_epsilon=1e-6)
        metadata = capability.extra_checkpoint_metadata()
        self.assertTrue(metadata["minimax_h3_distribution_matching"]["projected_dmd"])
        legacy = capability.legacy_extra_checkpoint_metadata()
        self.assertFalse(legacy["minimax_h3_distribution_matching"]["projected_dmd"])
        self.assertNotIn("minimax_h3_parallel_topology", legacy)
        self.assertNotIn("minimax_h3_target_geometry", legacy)
        self.assertEqual(legacy["minimax_h3_per_modality_normalization"], {"enabled": True, "epsilon": 0.0, "reduction": "mean"})
        capability.student_sla = SimpleNamespace(
            enabled=True,
            checkpoint_metadata=lambda: {"enabled": True, "attention_backend": "sla"},
        )
        self.assertEqual(capability.legacy_extra_checkpoint_metadata()["student_sparse_attention"], {"enabled": False})

    def test_native_reference_forward_matches_explicit_packing_and_backward(self):
        model = model_wrapper()
        model.running_dtype = torch.float32
        model.text_dim = 12
        model.video_latent_channels = 2
        model.audio_latent_channels = 2
        model.transformer = _transformer_class()(
            num_attention_heads=2,
            attention_head_dim=8,
            hidden_size=16,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=2,
            patch_size=(1, 2, 2),
            text_dim=12,
            freq_dim=4,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=1,
        )
        cached = reference_condition()
        cached["prompt_embeds"] = torch.randn(1, 4, 12)
        cached["references"][0]["video_latents"] = torch.randn(6, 8)
        cached["references"][1]["audio_latents"] = torch.randn(4, 2)
        capability = self.capability(model)
        sample, condition = self.prepare(capability, cached)
        shape = capability.latent_shape(sample, [], lambda x: x)
        generated = capability.initial_latents(shape, torch.float32, lambda x: x)
        sigma = torch.tensor([0.5])
        predicted = capability.predict_velocity(generated, sigma, condition)
        layout = capability._layout(condition, shape)
        times, time_indices = build_row_timesteps(layout, *capability._modality_sigmas(sigma))
        explicit_video, explicit_audio = model.transformer(
            hidden_states=torch.cat((condition["noised_condition_video_latents"].unsqueeze(0), generated.video), dim=1),
            audio_hidden_states=torch.cat((condition["condition_audio_latents"].unsqueeze(0), generated.audio), dim=1),
            encoder_hidden_states=condition["prompt_embeds"],
            timestep=times,
            timestep_indices=time_indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            return_dict=False,
        )
        torch.testing.assert_close(predicted.video, explicit_video[:, 6:], rtol=0, atol=0)
        torch.testing.assert_close(predicted.audio, explicit_audio[:, 4:], rtol=0, atol=0)
        (predicted.video.square().mean() + predicted.audio.square().mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.transformer.parameters()))


if __name__ == "__main__":
    unittest.main()
