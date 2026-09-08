import unittest

import torch
from diffusers import AutoencoderKLMiniMaxH3
from torch import nn

from lightx2v_train.model_capabilities import VAEDistillationStepContext
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_encoder_distillation_capability import (
    MiniMaxH3EncoderDistillationCapability,
    posterior_alignment,
)
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import _SampleGroup
from lightx2v_train.model_zoo.native.minimax_h3.pruned_encoder import (
    MiniMaxH3PrunedVideoEncoder,
    teacher_encoder_forward,
    teacher_encoder_suffix,
)
from lightx2v_train.model_zoo.native.minimax_h3.vae_geometry import spatial_tile_layout, temporal_decode_windows
from lightx2v_train.model_zoo.native.minimax_h3.video_vae import imagenet_postprocess, imagenet_preprocess
from tests.test_minimax_h3_pruned_encoder import tiny_encoder_config


class EncoderGeometryOwner(nn.Module):
    def __init__(self, *, pruned=False):
        super().__init__()
        config = tiny_encoder_config()
        config.update(
            out_channels=3, block_out_channels=[4] * 6, norm_num_groups=1,
            decoder_num_layers=2, decoder_num_attention_heads=2, decoder_attention_head_dim=8,
            decoder_ffn_mult=1, decoder_num_register_tokens=4,
        )
        self.teacher_vae = AutoencoderKLMiniMaxH3(**config).requires_grad_(False).eval()
        self.teacher_vae.enable_tiling(64, 64, 16, 16)
        self.student = MiniMaxH3PrunedVideoEncoder(
            config, kept_residual_indices=[0, 4, 10] if pruned else None,
            tile_sample_min_height=64, tile_sample_min_width=64,
            tile_sample_min_overlap_height=16, tile_sample_min_overlap_width=16,
        )
        self.student.initialize_from_teacher(self.teacher_vae)
        self.student.enable_gradient_checkpointing()
        self.device = torch.device("cpu")
        self.encoder_calls = 0

    def denoiser_module(self):
        return self.student

    preprocess_video = staticmethod(imagenet_preprocess)
    postprocess_raw_video = staticmethod(imagenet_postprocess)

    def student_encode_clip(self, pixels, running_dtype, **kwargs):
        self.encoder_calls += 1
        return self.student(pixels, **kwargs)

    @torch.no_grad()
    def teacher_encode_clip(self, pixels, *, return_features=False):
        return teacher_encoder_forward(self.teacher_vae, pixels, return_features=return_features)

    def teacher_encoder_suffix(self, feature, index):
        return teacher_encoder_suffix(self.teacher_vae, feature, index)

    def decode_student_latent_window(self, latents, running_dtype=None):
        return self.teacher_vae.decoder(self.teacher_vae.post_quant_conv(self.student.denormalize_latents(latents)))

    @torch.no_grad()
    def teacher_decode_window(self, latents):
        return self.decode_student_latent_window(latents)


def loss_config(*, auxiliary=True):
    return {
        "teacher_feature_indices": list(range(6)),
        "teacher_output_weight": 0,
        "temporal_velocity_weight": 0,
        "feature_weight": 0.01,
        "auxiliary_weight": 0.1 if auxiliary else 0,
        "student_window_checkpointing": True,
        "auxiliary_decoder": {"student_feature_indices": [2, 3, 4], "ramp_iters": 0},
    }


class EncoderDistillationGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(83)
        self.video = torch.rand(1, 3, 124, 160, 160)
        self.layout = spatial_tile_layout(10, 10, tile_height=64, tile_width=64, overlap_height=16, overlap_width=16)
        self.windows = temporal_decode_windows(37)
        self.groups = (
            _SampleGroup("single", 0, 1, 0, 1, 0, 1),
            _SampleGroup("single", 6, 1, 2, 1, 2, 1),
            _SampleGroup("horizontal_pair", 1, 1, 0, 1, 1, 2),
            _SampleGroup("vertical_pair", 3, 1, 1, 2, 0, 1),
            _SampleGroup("quad", 2, 1, 1, 2, 1, 2),
            _SampleGroup("temporal_pair", 2, 2, 0, 1, 2, 1),
            _SampleGroup("temporal_quad", 3, 4, 1, 1, 0, 1),
        )

    def test_sparse_all_routes_match_full_native_posterior_including_tail_and_halo(self):
        owner = EncoderGeometryOwner()
        capability = MiniMaxH3EncoderDistillationCapability(owner, loss_config())
        with torch.no_grad():
            full = owner.teacher_vae._encode(imagenet_preprocess(self.video))
            self.assertEqual(full.shape, (1, 4, 37, 10, 10))
            for group in self.groups:
                with self.subTest(route=group.route, temporal_start=group.temporal_start):
                    owner.encoder_calls = 0
                    moments, feature, region = capability._encode_group(
                        self.video, self.layout, self.windows, group, torch.float32, True, 3,
                    )
                    self.assertLess(owner.encoder_calls, 8 * self.layout.num_tiles)
                    index = (slice(None), slice(None), *region)
                    for values in moments:
                        torch.testing.assert_close(values[index], full[index], rtol=0, atol=0)
                    self.assertEqual(feature.item(), 0)

    def test_unpruned_full_loss_has_zero_posterior_feature_auxiliary_for_all_routes(self):
        owner = EncoderGeometryOwner()
        capability = MiniMaxH3EncoderDistillationCapability(owner, loss_config())
        batch = {"inputs": {"video": self.video}, "meta": {"source_num_frames": torch.tensor([124])}}
        with torch.no_grad():
            for group in self.groups:
                with self.subTest(route=group.route, temporal_start=group.temporal_start):
                    capability._sample_training_group = lambda *args: (self.layout, self.windows, group)
                    result = capability.compute_loss(batch, VAEDistillationStepContext(torch.float32, 0, 0))
                    for key in ("posterior", "posterior_mean", "posterior_std", "feature", "auxiliary"):
                        self.assertLess(result.metrics[key].item(), 1e-7, msg=key)
                    self.assertGreater(result.metrics["reconstruction"].item(), 0)
                    self.assertTrue(torch.isfinite(result.loss))

    def test_pixel_and_auxiliary_gradients_reach_only_student_encoder(self):
        owner = EncoderGeometryOwner(pruned=True)
        config = loss_config()
        config.update(posterior_weight=0, feature_weight=0)
        capability = MiniMaxH3EncoderDistillationCapability(owner, config)
        group = self.groups[0]
        capability._sample_training_group = lambda *args: (self.layout, self.windows, group)
        batch = {"inputs": {"video": self.video}, "meta": {"source_num_frames": torch.tensor([124])}}
        result = capability.compute_loss(batch, VAEDistillationStepContext(torch.float32, 0, 0))
        self.assertGreater(result.metrics["auxiliary"].item(), 0)
        result.loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in owner.teacher_vae.parameters()))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in owner.student.parameters()))
        self.assertGreater(owner.student.encoder.conv_in.weight.grad.abs().sum().item(), 0)
        self.assertGreater(owner.student.quant_conv.weight.grad.abs().sum().item(), 0)

    def test_posterior_loss_uses_mean_and_std_with_teacher_detached(self):
        student = torch.randn(1, 4, 5, 2, 2, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        std = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1, 1)
        loss, mean_loss, std_loss = posterior_alignment(student, teacher, std)
        expected_mean = ((student[:, :2] - teacher[:, :2]) / std).square().mean()
        expected_std = ((student[:, 2:].mul(0.5).exp() - teacher[:, 2:].mul(0.5).exp()) / std).square().mean()
        torch.testing.assert_close(mean_loss, expected_mean)
        torch.testing.assert_close(std_loss, expected_std)
        torch.testing.assert_close(loss, expected_mean + expected_std)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad[:, :2].abs().sum(), 0)
        self.assertGreater(student.grad[:, 2:].abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
