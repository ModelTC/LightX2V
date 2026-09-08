import unittest
from tempfile import TemporaryDirectory

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoViTDecoder3d
from torch import nn
from torch.utils.checkpoint import checkpoint

from lightx2v_train.model_zoo.native.minimax_h3.pruned_vae import (
    MiniMaxH3PrunedVideoVAE,
    _decoder_config,
)


def tiny_config():
    return {
        "latent_channels": 24,
        "out_channels": 3,
        "spatial_downsample_factors": [2, 2, 2, 2],
        "temporal_downsample_factors": [1, 2, 2, 1],
        "decoder_num_layers": 6,
        "decoder_num_attention_heads": 2,
        "decoder_attention_head_dim": 8,
        "decoder_num_register_tokens": 4,
        "decoder_ffn_mult": 1,
        "decoder_rope_theta": 100.0,
        "decoder_rope_dim_ratio": 0.75,
        "decoder_norm_eps": 1e-5,
        "clip_length": 17,
        "token_drop": 3,
        "latents_mean": [0.2] * 24,
        "latents_std": [1.1] * 24,
    }


def tiny_teacher(config):
    teacher = nn.Module()
    teacher.post_quant_conv = nn.Conv3d(24, 24, 1)
    teacher.decoder = MiniMaxH3VideoViTDecoder3d(**_decoder_config(config))
    with torch.no_grad():
        for index, block in enumerate(teacher.decoder.transformer_blocks):
            block.scale1.fill_(0.1 * (index + 1))
            block.scale2.fill_(0.1 * (index + 1))
    return teacher


class MiniMaxH3PrunedVideoVAETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.config = tiny_config()
        self.teacher = tiny_teacher(self.config)

    def test_no_pruning_matches_original_decoder_and_features(self):
        student = MiniMaxH3PrunedVideoVAE(teacher_config=self.config)
        student.initialize_from_teacher(self.teacher)
        latents = torch.randn(2, 24, 2, 2, 3)
        captures = []
        handles = [
            block.register_forward_hook(lambda _module, _args, result: captures.append(result))
            for block in self.teacher.decoder.transformer_blocks
        ]
        with torch.no_grad():
            expected = self.teacher.decoder(self.teacher.post_quant_conv(student.denormalize_latents(latents)))
            actual, features = student(latents, return_features=True)
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (2, 3, 8, 32, 48))
        self.assertEqual(len(features), 6)
        for feature, capture in zip(features, captures, strict=True):
            self.assertEqual(capture.shape, (2, 2 * 2 * 3 + 5, 16))
            expected_feature = capture[:, :12].reshape(2, 2, 2, 3, 16).permute(0, 4, 1, 2, 3)
            torch.testing.assert_close(feature, expected_feature, rtol=0, atol=0)

    def test_search_only_updates_lora_and_gates(self):
        student = MiniMaxH3PrunedVideoVAE(
            teacher_config=self.config,
            search={"group_size": 3, "lora_rank": 2, "temperature": 1.0, "gate_scale": 1.0},
        )
        student.initialize_from_teacher(self.teacher)
        student.prepare_search_step(2)
        mask = student.decoder._search_mask(2)
        torch.testing.assert_close(mask.reshape(2, 2, 3).sum(-1), torch.ones(2, 2))
        self.assertTrue(torch.equal(mask, student.decoder._search_mask(2)))
        student(torch.randn(2, 24, 2, 2, 2)).square().mean().backward()
        gate = student.decoder.gate_logits
        self.assertIsNotNone(gate.grad)
        self.assertGreater(gate.grad.abs().sum().item(), 0)
        trainable = {name: param for name, param in student.named_parameters() if param.requires_grad}
        self.assertTrue(all("lora_" in name or name.endswith("gate_logits") for name in trainable))
        self.assertTrue(any("lora_B" in name and param.grad.abs().sum() > 0 for name, param in trainable.items()))
        self.assertTrue(all(param.grad is None for param in student.parameters() if not param.requires_grad))
        keys = list(student.state_dict())
        student.decoder.configure_search_trainable()
        self.assertEqual(keys, list(student.state_dict()))
        self.assertEqual(set(trainable), {name for name, param in student.named_parameters() if param.requires_grad})

    def test_bfloat16_autocast_keeps_fp32_weights_and_normalization(self):
        latents = torch.randn(2, 24, 2, 2, 2)
        for search in (None, {"group_size": 3, "lora_rank": 2, "gate_scale": 1.0}):
            with self.subTest(search=search):
                model = MiniMaxH3PrunedVideoVAE(self.config, search=search)
                model.initialize_from_teacher(self.teacher)
                model.enable_gradient_checkpointing()
                model.prepare_search_step(2)
                norm_input_dtypes = []
                handles = [
                    block.norm1.register_forward_pre_hook(
                        lambda _module, inputs: norm_input_dtypes.append(inputs[0].dtype)
                    )
                    for block in model.decoder.transformer_blocks
                ]
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output, features = model(latents, return_features=True)
                    loss = output.float().square().mean()
                loss.backward()
                for handle in handles:
                    handle.remove()
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(all(feature.dtype == torch.float32 for feature in features))
                self.assertEqual(set(norm_input_dtypes), {torch.float32})
                self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
                gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
                self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
                self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))
                if search is not None:
                    self.assertGreater(model.decoder.gate_logits.grad.abs().sum().item(), 0)
                else:
                    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                        reference = self.teacher.decoder(
                            self.teacher.post_quant_conv(model.denormalize_latents(latents))
                        )
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)

    def test_export_reloads_original_layers_without_search_updates(self):
        search = MiniMaxH3PrunedVideoVAE(
            teacher_config=self.config, search={"group_size": 3, "lora_rank": 2}
        )
        search.initialize_from_teacher(self.teacher)
        with torch.no_grad():
            search.decoder.gate_logits.copy_(torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]))
            for name, parameter in search.named_parameters():
                if "lora_" in name:
                    parameter.fill_(12.0)
        self.assertEqual(search.selected_layers(), [1, 5])
        student = MiniMaxH3PrunedVideoVAE(teacher_config=self.config, kept_layers=search.selected_layers())
        student.initialize_from_teacher(self.teacher)
        self.assertFalse(any("lora_" in name or "gate_logits" in name for name in student.state_dict()))
        self.assertEqual(len(student.fsdp_modules()), 2)
        for student_index, teacher_index in enumerate([1, 5]):
            actual = student.decoder.transformer_blocks[student_index].state_dict()
            original = self.teacher.decoder.transformer_blocks[teacher_index].state_dict()
            for name in actual:
                torch.testing.assert_close(actual[name], original[name], rtol=0, atol=0)

    def test_checkpoint_recomputes_without_resampling_gates(self):
        search_config = {"group_size": 3, "lora_rank": 2, "temperature": 1.0, "gate_scale": 1.0}
        baseline = MiniMaxH3PrunedVideoVAE(teacher_config=self.config, search=search_config)
        baseline.initialize_from_teacher(self.teacher)
        checkpointed = MiniMaxH3PrunedVideoVAE(teacher_config=self.config, search=search_config)
        checkpointed.load_state_dict(baseline.state_dict())
        checkpointed.enable_gradient_checkpointing()
        baseline.prepare_search_step(2)
        checkpointed.decoder._search_noise = baseline.decoder._search_noise.clone()
        latents = torch.randn(2, 24, 2, 2, 2)
        baseline_output = baseline(latents)
        checkpointed_output = checkpointed(latents)
        baseline_output.square().mean().backward()
        checkpointed_output.square().mean().backward()
        torch.testing.assert_close(checkpointed_output, baseline_output, rtol=0, atol=0)
        for (name, original), (_, recomputed) in zip(baseline.named_parameters(), checkpointed.named_parameters()):
            if original.requires_grad:
                self.assertIsNotNone(original.grad, name)
                torch.testing.assert_close(recomputed.grad, original.grad)

    def test_temporal_stitching_and_checkpoint_round_trip(self):
        model = MiniMaxH3PrunedVideoVAE(teacher_config=self.config, kept_layers=[1, 5]).eval()
        model.initialize_from_teacher(self.teacher)
        with torch.inference_mode():
            for latent_frames, output_frames in ((32, 107), (37, 124), (107, 362)):
                output = model.decode_raw(torch.randn(1, 24, latent_frames, 1, 1))
                self.assertEqual(output.shape, (1, 3, output_frames, 16, 16))
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = MiniMaxH3PrunedVideoVAE.from_pretrained(directory)
        self.assertEqual(restored.architecture_config, model.architecture_config)
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], parameter)

    def test_whole_window_checkpoint_reuses_search_noise(self):
        search_config = {"group_size": 3, "lora_rank": 2, "temperature": 1.0, "gate_scale": 1.0}
        baseline = MiniMaxH3PrunedVideoVAE(self.config, search=search_config)
        baseline.initialize_from_teacher(self.teacher)
        recomputed = MiniMaxH3PrunedVideoVAE(self.config, search=search_config)
        recomputed.load_state_dict(baseline.state_dict())
        recomputed.enable_gradient_checkpointing()
        baseline.prepare_search_step(1)
        recomputed.decoder._search_noise = baseline.decoder._search_noise.clone()
        first_window = torch.randn(1, 24, 2, 2, 2)
        second_window = torch.randn(1, 24, 2, 2, 2)
        original_loss = sum(baseline(window).square().mean() for window in (first_window, second_window))
        recomputed_loss = sum(
            checkpoint(recomputed, window, use_reentrant=False).square().mean()
            for window in (first_window, second_window)
        )
        original_loss.backward()
        recomputed_loss.backward()
        torch.testing.assert_close(recomputed_loss, original_loss, rtol=0, atol=0)
        for (name, original), (_, checkpointed) in zip(baseline.named_parameters(), recomputed.named_parameters()):
            if original.requires_grad:
                self.assertIsNotNone(original.grad, name)
                torch.testing.assert_close(checkpointed.grad, original.grad)

    def test_search_eval_is_deterministic_and_checkpoint_loadable(self):
        model = MiniMaxH3PrunedVideoVAE(
            teacher_config=self.config, search={"group_size": 3, "lora_rank": 2}
        ).eval()
        model.initialize_from_teacher(self.teacher)
        latents = torch.randn(1, 24, 2, 1, 1)
        with torch.inference_mode():
            expected = model(latents)
            torch.testing.assert_close(model(latents), expected, rtol=0, atol=0)
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = MiniMaxH3PrunedVideoVAE.from_pretrained(directory).eval()
        with torch.inference_mode():
            torch.testing.assert_close(restored(latents), expected, rtol=0, atol=0)

    def test_invalid_layer_indices_are_rejected(self):
        for indices in ([], [0, 0], [2, 1], [-1, 2], [0, 6]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                MiniMaxH3PrunedVideoVAE(teacher_config=self.config, kept_layers=indices)
        with self.assertRaises(ValueError):
            MiniMaxH3PrunedVideoVAE(teacher_config=self.config, search={"group_size": 4})


if __name__ == "__main__":
    unittest.main()
