import copy
import itertools
import unittest
from tempfile import TemporaryDirectory

import torch

from lightx2v_train.model_zoo.native.wan.modules.vae import CausalConv3d, Resample, WanVAE_
from lightx2v_train.model_zoo.native.wan.pruned_vae import (
    WAN_VAE_CONFIG,
    ResidualShortcut,
    SearchLoRAConv3d,
    WanPrunedDecoder,
    WanPrunedEncoder,
    normalized_posterior_stats,
    posterior_stats,
    resample_full_sequence,
    teacher_suffix,
)


def tiny_config():
    return {**WAN_VAE_CONFIG, "dim": 4}


def streaming_moments(teacher, video):
    teacher.clear_cache()
    chunks = [video[:, :, :1]]
    chunks.extend(video[:, :, start:start + 4] for start in range(1, video.shape[2], 4))
    outputs = []
    for chunk in chunks:
        outputs.append(teacher.encoder(chunk, feat_cache=teacher._enc_feat_map, feat_idx=[0]))
    teacher.clear_cache()
    return teacher.conv1(torch.cat(outputs, dim=2))


class Wan21PrunedVAETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(4)
        self.config = tiny_config()
        self.teacher = WanVAE_(**self.config).eval()

    def component(self, cls, **kwargs):
        model = cls(self.config, **kwargs)
        model.initialize_from_teacher(self.teacher)
        return model

    def test_encoder_full_sequence_matches_native_streaming_moments(self):
        student = self.component(WanPrunedEncoder).eval()
        for frames in (1, 5, 9, 17):
            with self.subTest(frames=frames), torch.no_grad():
                video = torch.randn(1, 3, frames, 16, 24)
                expected = streaming_moments(self.teacher, video)
                actual, features = student(video, return_features=True)
                self.assertEqual(len(features), 10)
                self.assertEqual(actual.shape, (1, 32, (frames - 1) // 4 + 1, 2, 3))
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                native_mode = self.teacher.encode(video, [student.latents_mean, 1 / student.latents_std])
                torch.testing.assert_close(student.posterior_mode(actual), native_mode, rtol=2e-5, atol=2e-6)

    def test_decoder_full_sequence_matches_native_streaming(self):
        student = self.component(WanPrunedDecoder).eval()
        for frames in (1, 2, 3, 5):
            with self.subTest(frames=frames), torch.no_grad():
                latents = torch.randn(1, 16, frames, 2, 3)
                expected = self.teacher.decode(latents, [student.latents_mean, 1 / student.latents_std])
                actual, features = student(latents, return_features=True)
                self.assertEqual(len(features), 14)
                self.assertEqual(actual.shape, (1, 3, 4 * (frames - 1) + 1, 16, 24))
                # Batched full-sequence convolutions and streamed convolutions round differently.
                torch.testing.assert_close(actual, expected, rtol=5e-5, atol=1e-5)

    def test_stateless_temporal_resamplers_match_streaming_in_float64(self):
        for mode, frames, chunk_size in (("upsample3d", 5, 1), ("downsample3d", 17, 4)):
            module = Resample(4, mode).double()
            inputs = torch.randn(1, 4, frames, 4, 6, dtype=torch.float64)
            cache = [None]
            chunks = [inputs[:, :, :1]]
            chunks.extend(inputs[:, :, start:start + chunk_size] for start in range(1, frames, chunk_size))
            with torch.no_grad():
                expected = torch.cat([module(chunk, feat_cache=cache, feat_idx=[0]) for chunk in chunks], dim=2)
                actual = resample_full_sequence(module, inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_native_full_sequence_is_causal(self):
        for cls, shape, prefix in (
            (WanPrunedEncoder, (1, 3, 9, 16, 16), 5),
            (WanPrunedDecoder, (1, 16, 3, 2, 2), 2),
        ):
            model = self.component(cls).eval()
            inputs = torch.randn(shape)
            altered = inputs.clone()
            altered[:, :, prefix:] = torch.randn_like(altered[:, :, prefix:]) * 10
            output_prefix = (prefix - 1) // 4 + 1 if cls is WanPrunedEncoder else 4 * (prefix - 1) + 1
            with torch.no_grad():
                before, after = model(inputs), model(altered)
            torch.testing.assert_close(before[:, :, :output_prefix], after[:, :, :output_prefix], rtol=0, atol=0)

    def test_fixed_pruning_removes_only_main_branches(self):
        for cls, kept, shape, expected in (
            (WanPrunedEncoder, [1, 5, 8], (1, 3, 9, 16, 24), (1, 32, 3, 2, 3)),
            (WanPrunedDecoder, [0, 3, 6, 9, 12], (1, 16, 3, 2, 3), (1, 3, 9, 16, 24)),
        ):
            full = self.component(cls)
            student = self.component(cls, kept_residual_indices=kept)
            self.assertEqual(student.depth, 10 if cls is WanPrunedEncoder else 14)
            shortcuts = [module for module in student.modules() if isinstance(module, ResidualShortcut)]
            self.assertEqual(len(shortcuts), student.depth - len(kept))
            self.assertLess(sum(p.numel() for p in student.parameters()), sum(p.numel() for p in full.parameters()))
            original = full.state_dict()
            for name, value in student.state_dict().items():
                torch.testing.assert_close(value, original[name])
            result, features = student(torch.randn(shape), return_features=True)
            self.assertEqual(result.shape, expected)
            self.assertEqual(len(features), student.depth)

    def test_search_budget_gradients_and_frozen_original_parameters(self):
        for cls, keep, shape in (
            (WanPrunedEncoder, 3, (1, 3, 5, 16, 16)),
            (WanPrunedDecoder, 5, (1, 16, 2, 2, 2)),
        ):
            model = self.component(cls, search={"keep_residuals": keep, "lora_rank": 2, "gate_scale": 1.0})
            model.enable_gradient_checkpointing()
            model.prepare_search_step(1)
            mask = model._search_mask(1)
            torch.testing.assert_close(mask.sum(-1), torch.tensor([float(keep)]))
            self.assertTrue(torch.equal(mask, model._search_mask(1)))
            model(torch.randn(shape)).square().mean().backward()
            self.assertGreater(model.gate_logits.grad.abs().sum().item(), 0)
            trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
            self.assertTrue(all(name == "gate_logits" or "lora_" in name for name in trainable))
            self.assertTrue(any("lora_B" in name and value.grad.abs().sum() > 0 for name, value in trainable.items()))
            self.assertTrue(all(value.grad is None for value in model.parameters() if not value.requires_grad))
            keys = list(model.state_dict())
            model.configure_search_trainable()
            self.assertEqual(keys, list(model.state_dict()))

    def test_search_groups_follow_actual_resampling_topology(self):
        for residual_count in (1, 2, 3):
            config = {**self.config, "num_res_blocks": residual_count}
            for cls, sizes in (
                (WanPrunedEncoder, [residual_count] * 4 + [2]),
                (WanPrunedDecoder, [2] + [residual_count + 1] * 4),
            ):
                with self.subTest(component=cls.component_name, residual_count=residual_count):
                    model = cls(config)
                    self.assertEqual([len(group) for group in model.residual_groups], sizes)
                    self.assertEqual(
                        [index for group in model.residual_groups for index in group], list(range(model.depth)),
                    )

    def test_global_search_preserves_legacy_candidate_order(self):
        for cls, keep in ((WanPrunedEncoder, 3), (WanPrunedDecoder, 5)):
            model = self.component(cls, search={"keep_residuals": keep, "lora_rank": 2})
            choices = list(itertools.combinations(range(model.depth), keep))
            expected = torch.zeros(len(choices), model.depth)
            for index, choice in enumerate(choices):
                expected[index, list(choice)] = 1
            self.assertEqual(model.search_grouping, "global")
            torch.testing.assert_close(model.candidate_masks, expected, rtol=0, atol=0)

    def test_stage_search_masks_gradients_and_checkpoint_noise_reuse(self):
        for cls, candidates, shape in (
            (WanPrunedEncoder, 32, (1, 3, 5, 16, 16)),
            (WanPrunedDecoder, 162, (1, 16, 2, 2, 2)),
        ):
            with self.subTest(component=cls.component_name):
                plain = self.component(cls, search={
                    "grouping": "stage", "keep_per_group": 1, "lora_rank": 2, "gate_scale": 1.0,
                })
                self.assertEqual(plain.search_grouping, "stage")
                self.assertEqual(plain.keep_residuals, 5)
                self.assertEqual(plain.candidate_masks.shape, (candidates, plain.depth))
                for group in plain.residual_groups:
                    torch.testing.assert_close(plain.candidate_masks[:, group].sum(-1), torch.ones(candidates))
                plain.prepare_search_step(1)
                checked = copy.deepcopy(plain)
                checked.enable_gradient_checkpointing()
                mask = plain._search_mask(1)
                self.assertTrue(torch.all((mask == 0) | (mask == 1)))
                for group in plain.residual_groups:
                    torch.testing.assert_close(mask[:, group].sum(-1), torch.ones(1))
                torch.testing.assert_close(mask, checked._search_mask(1), rtol=0, atol=0)
                inputs = torch.randn(shape)
                expected, actual = plain(inputs), checked(inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected.square().mean().backward()
                actual.square().mean().backward()
                for group in plain.residual_groups:
                    self.assertGreater(plain.gate_logits.grad[list(group)].abs().sum().item(), 0)
                for original, recomputed in zip(plain.parameters(), checked.parameters()):
                    if original.requires_grad:
                        torch.testing.assert_close(original.grad, recomputed.grad, rtol=0, atol=0)
                    else:
                        self.assertIsNone(original.grad)
                        self.assertIsNone(recomputed.grad)
                torch.testing.assert_close(checked._search_mask(1), mask, rtol=0, atol=0)
                checked.clear_search_step()
                self.assertIsNone(checked._search_noise)

    def test_stage_search_checkpoint_and_physical_export_round_trip(self):
        for cls, shape in (
            (WanPrunedEncoder, (1, 3, 5, 16, 16)),
            (WanPrunedDecoder, (1, 16, 2, 2, 2)),
        ):
            with self.subTest(component=cls.component_name):
                model = self.component(cls, search={
                    "grouping": "stage", "keep_per_group": 1, "lora_rank": 2,
                }).eval()
                with torch.no_grad():
                    model.gate_logits.copy_(torch.arange(model.depth))
                kept = [group[-1] for group in model.residual_groups]
                self.assertEqual(model.selected_layers(), kept)
                ema = -model.gate_logits.detach()
                self.assertEqual(model.selected_layers(logits=ema), [group[0] for group in model.residual_groups])
                exported = self.component(cls, kept_residual_indices=kept).eval()
                inputs = torch.randn(shape)
                with torch.no_grad():
                    torch.testing.assert_close(model(inputs), exported(inputs), rtol=5e-5, atol=2e-6)
                    for layer in model.modules():
                        if isinstance(layer, SearchLoRAConv3d):
                            layer.lora_B.normal_(std=0.01)
                with TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    restored_search = cls.from_pretrained(directory).eval()
                    exported.save_pretrained(directory)
                    restored_export = cls.from_pretrained(directory).eval()
                self.assertEqual(restored_search.search_grouping, "stage")
                self.assertEqual(restored_search.residual_groups, model.residual_groups)
                self.assertEqual(restored_search.selected_layers(), kept)
                torch.testing.assert_close(restored_search.candidate_masks, model.candidate_masks, rtol=0, atol=0)
                with torch.no_grad():
                    torch.testing.assert_close(restored_search(inputs), model(inputs), rtol=0, atol=0)
                    torch.testing.assert_close(restored_export(inputs), exported(inputs), rtol=0, atol=0)
                original_state = self.teacher.state_dict()
                for name, value in restored_export.state_dict().items():
                    if name not in ("latents_mean", "latents_std"):
                        torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
                self.assertFalse(any("lora_" in name or name == "gate_logits" for name in restored_export.state_dict()))

    def test_search_eval_matches_physical_topology_and_export_uses_original_weights(self):
        for cls, keep, shape in (
            (WanPrunedEncoder, 3, (1, 3, 5, 16, 16)),
            (WanPrunedDecoder, 5, (1, 16, 2, 2, 2)),
        ):
            search = self.component(cls, search={"keep_residuals": keep, "lora_rank": 2}).eval()
            with torch.no_grad():
                search.gate_logits.copy_(torch.randn_like(search.gate_logits))
            kept = search.selected_layers()
            exported = self.component(cls, kept_residual_indices=kept).eval()
            inputs = torch.randn(shape)
            with torch.no_grad():
                torch.testing.assert_close(search(inputs), exported(inputs), rtol=0, atol=0)
                for layer in search.modules():
                    if isinstance(layer, SearchLoRAConv3d):
                        layer.lora_B.fill_(0.1)
            original = self.teacher.state_dict()
            with TemporaryDirectory() as directory:
                exported.save_pretrained(directory)
                restored = cls.from_pretrained(directory)
            self.assertEqual(restored.selected_layers(), kept)
            for name, value in restored.state_dict().items():
                if name not in ("latents_mean", "latents_std"):
                    torch.testing.assert_close(value, original[name])
            self.assertFalse(any("lora_" in name or "gate_logits" in name for name in restored.state_dict()))

    def test_every_teacher_suffix_reproduces_complete_forward(self):
        for cls, shape in (
            (WanPrunedEncoder, (1, 3, 9, 16, 24)),
            (WanPrunedDecoder, (1, 16, 3, 2, 3)),
        ):
            teacher = self.component(cls).eval().requires_grad_(False)
            with torch.no_grad():
                expected, features = teacher(torch.randn(shape), return_features=True)
                for index, feature in enumerate(features):
                    actual = teacher_suffix(teacher, feature, index)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_search_checkpoint_round_trip_preserves_logits_and_lora(self):
        for cls, keep, shape in (
            (WanPrunedEncoder, 3, (1, 3, 5, 16, 16)),
            (WanPrunedDecoder, 5, (1, 16, 2, 2, 2)),
        ):
            model = self.component(cls, search={"keep_residuals": keep, "lora_rank": 2}).eval()
            with torch.no_grad():
                model.gate_logits.copy_(torch.randn_like(model.gate_logits))
                for layer in model.modules():
                    if isinstance(layer, SearchLoRAConv3d):
                        layer.lora_B.normal_(std=0.01)
            with TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                restored = cls.from_pretrained(directory).eval()
            self.assertEqual(model.selected_layers(), restored.selected_layers())
            self.assertTrue(torch.equal(model.candidate_masks, restored.candidate_masks))
            self.assertNotIn("candidate_masks", restored.state_dict())
            inputs = torch.randn(shape)
            with torch.no_grad():
                torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)

    def test_functional_checkpointing_preserves_full_sequence_gradients(self):
        for cls, shape in (
            (WanPrunedEncoder, (1, 3, 9, 16, 16)),
            (WanPrunedDecoder, (1, 16, 3, 2, 2)),
        ):
            plain = self.component(cls)
            checked = copy.deepcopy(plain)
            checked.enable_gradient_checkpointing()
            inputs = torch.randn(shape)
            plain(inputs).square().mean().backward()
            checked(inputs).square().mean().backward()
            for original, recomputed in zip(plain.parameters(), checked.parameters()):
                torch.testing.assert_close(original.grad, recomputed.grad, rtol=0, atol=0)

    def test_auxiliary_gradients_update_prefix_not_teacher_or_student_suffix(self):
        for cls, shape, anchor in (
            (WanPrunedEncoder, (1, 3, 5, 16, 16), 5),
            (WanPrunedDecoder, (1, 16, 2, 2, 2), 8),
        ):
            student = self.component(cls)
            teacher = copy.deepcopy(student).requires_grad_(False).eval()
            student.enable_gradient_checkpointing()
            _, _, auxiliary = student(torch.randn(shape), auxiliary_feature_index=anchor)
            output = teacher_suffix(teacher, auxiliary, anchor)
            output.square().mean().backward()
            self.assertIsNotNone(student.network.conv1.weight.grad)
            self.assertGreater(student.network.conv1.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
            self.assertIsNone(student.network.head[-1].weight.grad)

    def test_checkpointing_and_bfloat16_keep_gradients_finite(self):
        for cls, shape in (
            (WanPrunedEncoder, (1, 3, 5, 16, 16)),
            (WanPrunedDecoder, (1, 16, 2, 2, 2)),
        ):
            model = self.component(cls)
            model.enable_gradient_checkpointing()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = model(torch.randn(shape))
                loss = output.float().square().mean()
            loss.backward()
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_search_lora_preserves_causal_padding_and_zero_initial_update(self):
        convolution = CausalConv3d(3, 4, 3, padding=1)
        adapted = SearchLoRAConv3d(convolution, 2, 4)
        inputs = torch.randn(1, 3, 5, 8, 8)
        for cache in (None, torch.randn(1, 3, 2, 8, 8)):
            torch.testing.assert_close(adapted(inputs, cache), convolution(inputs, cache), rtol=0, atol=0)

    def test_posterior_clamp_and_scale(self):
        mean, std = posterior_stats(torch.tensor([0.25, 80.0]).view(1, 2, 1, 1, 1))
        self.assertEqual(mean.item(), 0.25)
        torch.testing.assert_close(std.flatten(), torch.exp(torch.tensor([10.0])))
        model = self.component(WanPrunedEncoder)
        latents = torch.randn(1, 16, 3, 2, 2)
        torch.testing.assert_close(model.denormalize_latents(model.normalize_latents(latents)), latents)
        moments = torch.randn(1, 32, 3, 2, 2)
        mu, sigma = posterior_stats(moments)
        normalized_mu, normalized_sigma = normalized_posterior_stats(moments)
        torch.testing.assert_close(normalized_mu, model.normalize_latents(mu))
        torch.testing.assert_close(normalized_sigma, sigma / model.latents_std.view(1, -1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
