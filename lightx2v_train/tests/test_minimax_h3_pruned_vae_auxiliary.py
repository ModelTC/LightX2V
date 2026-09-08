import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from lightx2v_train.model_zoo.minimax_h3.minimax_h3_pruned_vae import MiniMaxH3PrunedVAEModel
from lightx2v_train.model_zoo.minimax_h3.minimax_h3_pruned_vae import teacher_config_fingerprint
from lightx2v_train.model_zoo.native.minimax_h3.pruned_vae import MiniMaxH3PrunedVideoVAE, decode_teacher_suffix
from lightx2v_train.runtime.config import load_config
from tests.test_minimax_h3_pruned_vae import tiny_config, tiny_teacher


class PrunedVAEAuxiliaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        self.config = tiny_config()
        self.teacher = tiny_teacher(self.config).requires_grad_(False).eval()
        self.latents = torch.randn(2, 24, 2, 2, 3)

    def student(self, kept_layers=None):
        student = MiniMaxH3PrunedVideoVAE(self.config, kept_layers=kept_layers)
        student.initialize_from_teacher(self.teacher)
        return student

    def test_teacher_prefix_suffix_matches_original_for_every_anchor(self):
        student = self.student()
        captures = []
        handles = [
            block.register_forward_hook(lambda _module, _args, output: captures.append(output))
            for block in self.teacher.decoder.transformer_blocks
        ]
        with torch.no_grad():
            expected = self.teacher.decoder(self.teacher.post_quant_conv(student.denormalize_latents(self.latents)))
        for handle in handles:
            handle.remove()
        for anchor, full_tokens in enumerate(captures):
            with self.subTest(anchor=anchor):
                actual = decode_teacher_suffix(self.teacher.decoder, full_tokens, (2, 2, 3), anchor)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_optional_capture_keeps_default_output_and_complete_auxiliary_tokens(self):
        student = self.student()
        with torch.no_grad():
            expected, expected_features = student(self.latents, return_features=True)
            actual, features, tokens = student(self.latents, return_features=True, auxiliary_feature_index=2)
            no_features, empty_features, second_tokens = student(self.latents, auxiliary_feature_index=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(no_features, expected, rtol=0, atol=0)
        self.assertEqual(empty_features, ())
        self.assertEqual(tokens.shape, (2, 12 + 5, 16))
        torch.testing.assert_close(tokens, second_tokens, rtol=0, atol=0)
        torch.testing.assert_close(tokens[:, :12].reshape(2, 2, 2, 3, 16).permute(0, 4, 1, 2, 3), features[2])
        for actual_feature, expected_feature in zip(features, expected_features, strict=True):
            torch.testing.assert_close(actual_feature, expected_feature, rtol=0, atol=0)

    def test_auxiliary_updates_student_prefix_but_not_teacher_or_student_suffix(self):
        student = self.student([0, 2, 5])
        _, _, tokens = student(self.latents, auxiliary_feature_index=1)
        tokens.retain_grad()
        output = decode_teacher_suffix(self.teacher.decoder, tokens, (2, 2, 3), 3)
        output.square().mean().backward()
        self.assertGreater(tokens.grad.abs().sum().item(), 0)
        self.assertGreater(tokens.grad[:, 12:].abs().sum().item(), 0)
        self.assertGreater(student.post_quant_conv.weight.grad.abs().sum().item(), 0)
        for index, block in enumerate(student.decoder.transformer_blocks):
            gradients = [parameter.grad for parameter in block.parameters()]
            if index <= 1:
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
            else:
                self.assertTrue(all(gradient is None for gradient in gradients))
        self.assertIsNone(student.decoder.proj_out.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.teacher.parameters()))

    def test_frozen_teacher_suffix_input_gradient_matches_direct_original_graph(self):
        student = self.student()
        with torch.no_grad():
            _, _, tokens = student(self.latents, auxiliary_feature_index=2)
        suffix_input = tokens.detach().requires_grad_()
        actual = decode_teacher_suffix(self.teacher.decoder, suffix_input, (2, 2, 3), 2)
        actual.square().mean().backward()

        reference_input = tokens.detach().requires_grad_()

        def replace_prefix(_module, _args, _output):
            return reference_input

        handle = self.teacher.decoder.transformer_blocks[2].register_forward_hook(replace_prefix)
        reference = self.teacher.decoder(self.teacher.post_quant_conv(student.denormalize_latents(self.latents)))
        handle.remove()
        reference.square().mean().backward()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(suffix_input.grad, reference_input.grad, rtol=0, atol=0)

    def test_main_plus_auxiliary_window_and_block_checkpoint_gradients_match(self):
        plain = self.student([0, 2, 5])
        checked = copy.deepcopy(plain)
        checked.enable_gradient_checkpointing()

        def objective(model, use_checkpoint):
            total = 0
            for window in (self.latents, self.latents * 0.75):
                if use_checkpoint:
                    raw, _, tokens = checkpoint(model, window, auxiliary_feature_index=1, use_reentrant=False)
                else:
                    raw, _, tokens = model(window, auxiliary_feature_index=1)
                auxiliary = decode_teacher_suffix(
                    self.teacher.decoder, tokens, window.shape[-3:], 3, gradient_checkpointing=use_checkpoint
                )
                total = total + raw.square().mean() + auxiliary.square().mean() * 0.2
            return total

        original_loss = objective(plain, False)
        checked_loss = objective(checked, True)
        original_loss.backward()
        checked_loss.backward()
        torch.testing.assert_close(checked_loss, original_loss, rtol=0, atol=0)
        for (name, original), (_, recomputed) in zip(plain.named_parameters(), checked.named_parameters(), strict=True):
            self.assertIsNotNone(original.grad, name)
            self.assertTrue(torch.isfinite(original.grad).all(), name)
            torch.testing.assert_close(recomputed.grad, original.grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.teacher.parameters()))

    def test_bfloat16_autocast_keeps_frozen_teacher_differentiable(self):
        student = self.student([0, 2, 5])
        student.enable_gradient_checkpointing()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            raw, _, tokens = student(self.latents, auxiliary_feature_index=1)
            auxiliary = decode_teacher_suffix(self.teacher.decoder, tokens, (2, 2, 3), 3)
            loss = raw.float().square().mean() + auxiliary.float().square().mean()
        loss.backward()
        self.assertEqual(tokens.dtype, torch.float32)
        self.assertEqual(auxiliary.dtype, torch.bfloat16)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in self.teacher.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.teacher.parameters()))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in student.parameters()))

    def test_bfloat16_suffix_preserves_small_auxiliary_ramp_gradients(self):
        student = self.student([0, 2, 5])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, _, tokens = student(self.latents, auxiliary_feature_index=1)
            tokens.retain_grad()
            auxiliary = decode_teacher_suffix(self.teacher.decoder, tokens, self.latents.shape[-3:], 3)
        # Large RGB means and the early auxiliary ramp produce very small output gradients.
        (auxiliary.float().square().mean() * 1e-5).backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(tokens.grad.abs().sum().item(), 0)
        self.assertGreater(student.post_quant_conv.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.teacher.parameters()))

    def test_wrapper_uses_normal_forward_and_preserves_auxiliary_input_gradient(self):
        student = self.student([0, 2, 5])
        wrapper = object.__new__(MiniMaxH3PrunedVAEModel)
        wrapper.device = torch.device("cpu")
        wrapper.student_autocast = True
        wrapper.teacher_autocast_dtype = torch.float16
        wrapper.teacher_vae = self.teacher
        wrapper.denoiser_module = lambda: student
        calls = []
        handle = student.register_forward_hook(lambda _module, _args, _result: calls.append(True))
        raw, features, tokens = wrapper.student_decode_window_with_aux(
            self.latents, torch.float32, auxiliary_feature_index=1, return_features=True
        )
        handle.remove()
        output = wrapper.teacher_decode_suffix(tokens, self.latents.shape[-3:], 3)
        (raw.square().mean() + output.square().mean()).backward()
        self.assertEqual(calls, [True])
        self.assertEqual(len(features), 3)
        self.assertGreater(student.post_quant_conv.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.teacher.parameters()))

    def test_fp32_teacher_suffix_accepts_bfloat16_student_and_keeps_small_gradients(self):
        student = self.student([0, 2, 5])
        student.enable_gradient_checkpointing()
        wrapper = object.__new__(MiniMaxH3PrunedVAEModel)
        wrapper.device = torch.device("cpu")
        wrapper.teacher_autocast_dtype = torch.float32
        wrapper.teacher_vae = self.teacher
        self.teacher.config = SimpleNamespace(**self.config)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, _, tokens = student(self.latents, auxiliary_feature_index=1)
        self.assertEqual(tokens.dtype, torch.float32)
        tokens.retain_grad()
        teacher_dtypes = []
        handles = [
            module.register_forward_hook(lambda _module, _args, output: teacher_dtypes.append(output.dtype))
            for module in self.teacher.modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv3d))
        ]
        try:
            target = wrapper.teacher_decode_window(self.latents)
            auxiliary = wrapper.teacher_decode_suffix(tokens, self.latents.shape[-3:], 3)
            (auxiliary.square().mean() * 1e-5).backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(set(teacher_dtypes), {torch.float32})
        self.assertEqual(target.dtype, torch.float32)
        self.assertFalse(target.requires_grad)
        self.assertEqual(auxiliary.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(tokens.grad.abs().sum().item(), 0)
        self.assertGreater(student.post_quant_conv.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.teacher.parameters()))

    def test_anchor_and_complete_sequence_boundaries(self):
        student = self.student([0, 2, 5])
        for index in (-1, 3):
            with self.subTest(student_index=index), self.assertRaises(ValueError):
                student(self.latents, auxiliary_feature_index=index)
        _, _, tokens = student(self.latents, auxiliary_feature_index=1)
        for index in (-1, 6):
            with self.subTest(teacher_index=index), self.assertRaises(ValueError):
                decode_teacher_suffix(self.teacher.decoder, tokens, (2, 2, 3), index)
        with self.assertRaisesRegex(ValueError, "full video, register, and auxiliary"):
            decode_teacher_suffix(self.teacher.decoder, tokens[:, :12], (2, 2, 3), 3)

    def test_auxiliary_only_recovery_loads_teacher_and_selection_anchors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(self.config))
            exported = root / "export"
            self.student([0, 2, 5]).save_pretrained(exported)
            selection = {
                "teacher_config_sha256": teacher_config_fingerprint(self.config),
                "kept_layers": [0, 2, 5],
                "teacher_feature_indices": [1, 3, 5],
            }
            (exported / "kept_layers.json").write_text(json.dumps(selection))
            config_path = Path(__file__).resolve().parents[1] / "configs/train/vae/minimax_h3_vit_prune_recover_aux_3k_4gpu_ddp.yaml"
            config = load_config(str(config_path))
            config["model"]["pretrained_model_name_or_path"] = directory
            config["model"]["pruned_decoder"]["selection_path"] = str(exported / "kept_layers.json")
            config["model"].pop("load_teacher_decoder")
            config["training"]["vae_distillation"] = {"feature_weight": 0, "auxiliary_weight": 0.2}
            wrapper = MiniMaxH3PrunedVAEModel(config)
            with patch(
                "lightx2v_train.model_zoo.minimax_h3.minimax_h3_pruned_vae.load_minimax_h3_video_vae",
                return_value=copy.deepcopy(self.teacher),
            ):
                wrapper.load_components(load_transformer=True, load_vae=True, load_condition_encoder=False)
            self.assertIsNotNone(wrapper.teacher_vae.decoder)
            self.assertIsNone(wrapper.teacher_vae.encoder)
            self.assertTrue(all(not parameter.requires_grad for parameter in wrapper.teacher_vae.parameters()))
            self.assertEqual(config["training"]["vae_distillation"]["teacher_feature_indices"], [1, 3, 5])
            self.assertEqual(wrapper.auxiliary_distillation_weight, 0.2)


if __name__ == "__main__":
    unittest.main()
