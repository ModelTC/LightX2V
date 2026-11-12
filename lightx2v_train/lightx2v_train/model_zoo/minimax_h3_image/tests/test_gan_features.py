"""Image fake-backbone features used by the shared DMD2 GAN classifier."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3_image.model import MiniMaxH3ImageModel
from lightx2v_train.model_zoo.minimax_h3_image.tests.test_model import model_config, save_tiny_model
from lightx2v_train.model_zoo.native.minimax_h3_image import transformer_class, unpatchify_video_latents
from lightx2v_train.trainers.dmd.dmd2_gan_head import H3GanHead


def gan_fixture(directory, *, checkpointing=False):
    save_tiny_model(directory)
    with patch("torch.cuda.is_available", return_value=False):
        model = MiniMaxH3ImageModel(model_config(directory))
        model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
    if checkpointing:
        model.enable_gradient_checkpointing()
    condition = model.prepare_text_condition({"prompt_embeds": torch.randn(1, 3, 12), "text_token_tags": torch.ones(3, dtype=torch.long)})
    return model, torch.randn(1, 2, 1, 4, 4), condition


class ImageGanFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_features_match_native_residual_block_and_exclude_text(self):
        with tempfile.TemporaryDirectory() as directory:
            model, state, condition = gan_fixture(directory)
            captured = []
            handle = model.transformer.transformer_blocks[0].register_forward_hook(lambda module, inputs, output: captured.append(output.detach().clone()))
            expected_velocity = model.predict_denoiser_output(state, torch.tensor(0.75), condition)
            handle.remove()
            image_rows = model.prepare_denoiser_input(state, condition)["layout"]["image_indices"]
            expected = captured[0].index_select(1, image_rows)
            with patch.object(model.transformer.transformer_blocks[1], "forward", side_effect=AssertionError("later block should not run")):
                with patch.object(model.transformer.norm_out, "forward", side_effect=AssertionError("output normalization should not run")):
                    features = model.ensure_capabilities().require(DistributionMatchingCapability).predict_dmad_features(state, torch.tensor(0.75), condition, feature_block=0)
            self.assertEqual(set(features), {"image"})
            self.assertEqual(features["image"].shape, (1, 4, 16))
            torch.testing.assert_close(features["image"], expected, rtol=0, atol=0)
            torch.testing.assert_close(model.predict_denoiser_output(state, torch.tensor(0.75), condition), expected_velocity, rtol=0, atol=0)

    def test_frozen_generator_input_gradient_and_trainable_discriminator_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            model, state, condition = gan_fixture(directory, checkpointing=True)
            head = H3GanHead(16, hidden_dim=8, temporal_bins=2)
            model.transformer.requires_grad_(False)
            head.requires_grad_(False)
            generated = state.detach().requires_grad_()
            features = model.predict_dmad_features(generated, torch.tensor(0.4), condition, feature_block=0)
            torch.nn.functional.softplus(-head(features["image"])).mean().backward()
            self.assertTrue(torch.isfinite(generated.grad).all())
            self.assertGreater(float(generated.grad.abs().sum()), 0)
            self.assertTrue(all(parameter.grad is None for parameter in model.transformer.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))
            model.transformer.requires_grad_(True)
            head.requires_grad_(True)
            real = state.detach()
            features = model.predict_dmad_features(real, torch.tensor(0.9), condition, feature_block=0)
            torch.nn.functional.softplus(head(features["image"])).mean().backward()
            self.assertGreater(float(model.transformer.proj_in.weight.grad.abs().sum()), 0)
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in model.transformer.transformer_blocks[1].parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in model.transformer.pdd_proj_out.parameters()))

    def test_invalid_feature_block_and_audio_sigma_pair_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model, state, condition = gan_fixture(directory)
            for index in (-1, 2, 1.5, True):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    model.predict_dmad_features(state, torch.tensor(0.5), condition, feature_block=index)
            with self.assertRaises(ValueError):
                model.predict_dmad_features(state, torch.tensor([0.5, 0.5]), condition, feature_block=0)

    def test_official_six_step_teacher_matches_native_head_sequence(self):
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            model, state, condition = gan_fixture(directory)
            native = transformer_class().from_pretrained(Path(directory) / "transformer_diffusers", local_files_only=True).eval()
            reference = state.clone()
            sigmas = model.pdd_schedule.physical_sigmas()
            for index in range(6):
                data = model.prepare_denoiser_input(reference, condition)
                raw = native(
                    hidden_states=data["rows"],
                    encoder_hidden_states=condition["prompt_embeds"],
                    timestep=(1 - sigmas[index]).reshape(1),
                    pdd_step_index=index,
                    **data["layout"],
                    return_dict=False,
                )[0]
                reference += (sigmas[index] - sigmas[index + 1]) * unpatchify_video_latents(raw, reference.shape)
                state += (sigmas[index + 1] - sigmas[index]) * model.predict_denoiser_output(state, sigmas[index], condition)
                torch.testing.assert_close(state, reference, rtol=0, atol=0)

    def test_gan_real_latent_identity_precision_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            model, state, _ = gan_fixture(directory)
            capability = model.ensure_capabilities().require(DistributionMatchingCapability)
            path = Path(directory) / "real.pt"
            identity = {"real_sample_key": "sample-0", "real_cache_fingerprint": "teacher-six-step-v1"}
            sample = {"meta": {"image_latent_path": str(path), "vae_normalized": True, **identity}}
            payload = {"latents": state.squeeze(0), **identity}
            torch.save(payload, path)
            torch.testing.assert_close(capability.extract_turbo_latents(sample, state.shape), state, rtol=0, atol=0)
            for changed in (
                {"vae_normalized": False},
                {"real_sample_key": "other-sample"},
                {"real_cache_fingerprint": "other-teacher"},
                {"latents": state.bfloat16()},
                {"latents": state[:, :, :, :2]},
                {"latents": torch.full_like(state, float("nan"))},
            ):
                with self.subTest(changed=list(changed)):
                    torch.save({**payload, **changed}, path)
                    with self.assertRaises(ValueError):
                        capability.extract_turbo_latents(sample, state.shape)
            sample["meta"]["vae_normalized"] = False
            with self.assertRaises(ValueError):
                capability.extract_turbo_latents(sample, state.shape)


if __name__ == "__main__":
    unittest.main()
