import copy
import unittest
from tempfile import TemporaryDirectory

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoEncoder3d
from torch import nn

from lightx2v_train.model_zoo.native.minimax_h3.pruned_encoder import (
    MiniMaxH3PrunedVideoEncoder,
    SearchLoRAConv3d,
    encoder_config,
    teacher_encoder_forward,
    teacher_encoder_suffix,
)


def tiny_encoder_config():
    return {
        "in_channels": 3,
        "latent_channels": 2,
        "block_out_channels": [4, 8, 8, 16, 16, 32],
        "layers_per_block": 2,
        "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
        "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
        "norm_num_groups": 4,
        "norm_eps": 1e-6,
        "spatial_padding_mode": "reflect",
        "clip_length": 17,
        "token_drop": 3,
        "latents_mean": [0.2, -0.1],
        "latents_std": [1.1, 0.9],
    }


def tiny_teacher(config):
    teacher = nn.Module()
    teacher.encoder = MiniMaxH3VideoEncoder3d(**encoder_config(config))
    teacher.quant_conv = nn.Conv3d(4, 4, 1)
    return teacher


class MiniMaxH3PrunedEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.config = tiny_encoder_config()
        self.teacher = tiny_teacher(self.config)
        self.pixels = torch.randn(1, 3, 17, 32, 32)

    def test_no_pruning_matches_teacher_and_six_stage_features(self):
        student = MiniMaxH3PrunedVideoEncoder(self.config)
        student.initialize_from_teacher(self.teacher)
        with torch.no_grad():
            expected, teacher_features = teacher_encoder_forward(self.teacher, self.pixels, return_features=True)
            actual, features = student(self.pixels, return_features=True)
        self.assertEqual(actual.shape, (1, 4, 5, 2, 2))
        self.assertEqual(len(features), 6)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for feature, expected_feature in zip(features, teacher_features, strict=True):
            torch.testing.assert_close(feature, expected_feature, rtol=0, atol=0)

    def test_three_kept_branches_preserve_shape_and_all_shortcuts(self):
        student = MiniMaxH3PrunedVideoEncoder(self.config, kept_residual_indices=[0, 4, 10])
        student.initialize_from_teacher(self.teacher)
        with torch.no_grad():
            output = student(self.pixels)
        self.assertEqual(output.shape, (1, 4, 5, 2, 2))
        self.assertEqual(sum(isinstance(module, nn.Conv3d) for module in self.teacher.modules()), 34)
        self.assertEqual(sum(isinstance(module, nn.Conv3d) for module in student.modules()), 16)
        for index in (1, 3, 5):
            self.assertIsNotNone(student.encoder.down_blocks[index].resnets[0].conv_shortcut)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(student.selected_layers(), [0, 4, 10])
        self.assertTrue(student.fsdp_modules())

    def test_search_exact_budget_backward_and_lora_only_adaptation(self):
        student = MiniMaxH3PrunedVideoEncoder(
            self.config, search={"keep_residuals": 3, "lora_rank": 2, "lora_alpha": 4, "gate_scale": 1},
        )
        student.initialize_from_teacher(self.teacher)
        student.enable_gradient_checkpointing()
        self.assertIn("SearchLoRAConv3d", repr(student))
        student.prepare_search_step(1)
        mask = student.encoder._search_mask(1)
        self.assertEqual(tuple(mask.shape), (1, 12))
        torch.testing.assert_close(mask.sum(-1), torch.tensor([3.0]))
        self.assertTrue(torch.equal(mask, student.encoder._search_mask(1)))
        student(self.pixels).float().square().mean().backward()
        self.assertGreater(student.encoder.gate_logits.grad.abs().sum(), 0)
        trainable = {name: parameter for name, parameter in student.named_parameters() if parameter.requires_grad}
        self.assertTrue(all("lora_" in name or name.endswith("gate_logits") for name in trainable))
        self.assertTrue(any("lora_B" in name and parameter.grad.abs().sum() > 0 for name, parameter in trainable.items()))
        self.assertTrue(all(parameter.grad is None for parameter in student.parameters() if not parameter.requires_grad))
        parameter_names = tuple(student.state_dict())
        student.configure_search_trainable()
        self.assertEqual(tuple(student.state_dict()), parameter_names)

    def test_zero_search_lora_retains_original_causal_conv_padding(self):
        convolution = self.teacher.encoder.down_blocks[1].resnets[0].conv1
        wrapped = SearchLoRAConv3d(copy.deepcopy(convolution), 2, 4)
        inputs = torch.randn(1, convolution.in_channels, 5, 8, 8)
        torch.testing.assert_close(wrapped(inputs), convolution(inputs), rtol=0, atol=0)
        with torch.no_grad():
            wrapped.lora_B.fill_(0.1)
            padded = torch.nn.functional.pad(inputs, (1, 1, 1, 1, 0, 0), mode="reflect")
            padded = torch.nn.functional.pad(padded, (0, 0, 0, 0, 2, 0))
            delta = (wrapped.lora_B.flatten(1) @ wrapped.lora_A.flatten(1)).view_as(wrapped.weight)
            expected = torch.nn.functional.conv3d(padded, wrapped.weight + 2 * delta, wrapped.bias)
        torch.testing.assert_close(wrapped(inputs), expected, rtol=1e-5, atol=2e-6)

    def test_grouped_search_keeps_one_per_stage_and_trains_each_gate_group(self):
        student = MiniMaxH3PrunedVideoEncoder(
            self.config, search={"grouping": "stage", "keep_per_group": 1, "lora_rank": 2, "gate_scale": 1},
        )
        student.initialize_from_teacher(self.teacher)
        student.enable_gradient_checkpointing()
        pruning = student.encoder
        self.assertEqual(pruning.residual_groups, [(i, i + 1) for i in range(0, 12, 2)])
        self.assertEqual(pruning.candidate_masks.shape, (64, 12))
        self.assertEqual(pruning.keep_residuals, 6)
        student.prepare_search_step(1)
        mask = pruning._search_mask(1)
        for group in pruning.residual_groups:
            torch.testing.assert_close(mask[:, list(group)].sum(-1), torch.ones(1))
        student(self.pixels).float().square().mean().backward()
        for group in pruning.residual_groups:
            self.assertGreater(pruning.gate_logits.grad[list(group)].abs().sum(), 0)
        torch.testing.assert_close(mask, pruning._search_mask(1), rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in student.parameters() if not parameter.requires_grad))
        for _ in range(10):
            student.prepare_search_step(4)
            sampled = pruning._search_mask(4)
            for group in pruning.residual_groups:
                torch.testing.assert_close(sampled[:, list(group)].sum(-1), torch.ones(4))

    def test_grouped_selection_eval_and_search_checkpoint_roundtrip(self):
        search = MiniMaxH3PrunedVideoEncoder(
            self.config, search={"grouping": "stage", "keep_per_group": 1, "lora_rank": 2},
        )
        search.initialize_from_teacher(self.teacher)
        with torch.no_grad():
            search.encoder.gate_logits.copy_(torch.tensor([20, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        search.eval()
        kept = [0, 3, 5, 7, 9, 11]
        self.assertEqual(search.selected_layers(), kept)
        student = MiniMaxH3PrunedVideoEncoder(self.config, kept_residual_indices=kept)
        student.initialize_from_teacher(self.teacher)
        with torch.no_grad():
            torch.testing.assert_close(search(self.pixels), student(self.pixels), rtol=0, atol=0)
        with TemporaryDirectory() as directory:
            search.save_pretrained(directory)
            loaded = MiniMaxH3PrunedVideoEncoder.from_pretrained(directory).eval()
            self.assertEqual(loaded.selected_layers(), kept)
            self.assertEqual(loaded.encoder.search_grouping, "stage")
            torch.testing.assert_close(loaded.encoder.candidate_masks, search.encoder.candidate_masks, rtol=0, atol=0)
            with torch.no_grad():
                torch.testing.assert_close(loaded(self.pixels), student(self.pixels), rtol=0, atol=0)

    def test_eval_search_matches_exported_pruned_topology(self):
        search = MiniMaxH3PrunedVideoEncoder(self.config, search={"keep_residuals": 3, "lora_rank": 2})
        search.initialize_from_teacher(self.teacher)
        with torch.no_grad():
            search.encoder.gate_logits[[0, 4, 10]] = 1
        search.eval()
        student = MiniMaxH3PrunedVideoEncoder(self.config, kept_residual_indices=search.selected_layers())
        student.initialize_from_teacher(self.teacher)
        with torch.no_grad():
            torch.testing.assert_close(search(self.pixels), student(self.pixels), rtol=0, atol=0)
        self.assertEqual(search.selected_layers(), [0, 4, 10])

    def test_frozen_teacher_suffix_transmits_input_gradients(self):
        self.teacher.requires_grad_(False)
        with torch.no_grad():
            expected, features = teacher_encoder_forward(self.teacher, self.pixels, return_features=True)
        for stage_index in (2, 3, 4):
            with self.subTest(stage=stage_index):
                feature = features[stage_index].detach().requires_grad_()
                actual = teacher_encoder_suffix(self.teacher, feature, stage_index)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.square().mean().backward()
                self.assertGreater(feature.grad.abs().sum(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.teacher.parameters()))

    def test_auxiliary_feature_is_matching_post_downsample_stage(self):
        student = MiniMaxH3PrunedVideoEncoder(self.config, kept_residual_indices=[0, 4, 10])
        moments, features, auxiliary = student(self.pixels, return_features=True, auxiliary_index=3)
        self.assertIs(auxiliary, features[3])
        self.assertEqual(moments.shape, (1, 4, 5, 2, 2))

    def test_checkpoint_roundtrip_and_bfloat16_compute(self):
        student = MiniMaxH3PrunedVideoEncoder(self.config, kept_residual_indices=[0, 4, 10])
        student.initialize_from_teacher(self.teacher)
        student.enable_gradient_checkpointing()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = student(self.pixels)
        output.float().square().mean().backward()
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in student.parameters()))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in student.parameters()))
        with TemporaryDirectory() as directory:
            student.save_pretrained(directory)
            loaded = MiniMaxH3PrunedVideoEncoder.from_pretrained(directory)
            self.assertEqual(loaded.architecture_config, student.architecture_config)
            for name, parameter in loaded.state_dict().items():
                torch.testing.assert_close(parameter, student.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
