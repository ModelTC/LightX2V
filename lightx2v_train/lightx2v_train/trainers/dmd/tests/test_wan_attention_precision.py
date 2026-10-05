import importlib
import importlib.machinery
import importlib.util
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

# Native Wan's package also imports T5, which probes CUDA in a default argument.
# These transformer-only CPU tests never load/tokenize text or initialize CUDA.
_optional_modules = {}
if importlib.util.find_spec("ftfy") is None:
    _ftfy = types.ModuleType("ftfy")
    _ftfy.__spec__ = importlib.machinery.ModuleSpec("ftfy", loader=None)
    _ftfy.fix_text = Mock(side_effect=AssertionError("Attention tests must not tokenize text"))
    _optional_modules["ftfy"] = _ftfy
sys.modules.update(_optional_modules)
try:
    with patch("torch.cuda.current_device", return_value=0):
        from lightx2v_train.model_zoo.native.wan.modules.attention import flash_attention, sdpa_attention
        from lightx2v_train.model_zoo.native.wan.modules.model import WanModel
        from lightx2v_train.model_zoo.wan.wan_t2v import WanT2VModel
finally:
    for _name in _optional_modules:
        del sys.modules[_name]

wan_attention = importlib.import_module("lightx2v_train.model_zoo.native.wan.modules.attention")


class WanFlashAttention3Test(unittest.TestCase):
    def test_hopper_detection_recognizes_h200_without_name_matching(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.get_device_capability", return_value=(9, 0)) as capability:
            self.assertTrue(wan_attention.is_hopper_gpu(torch.device("cuda:3")))
            capability.assert_called_once_with(torch.device("cuda:3"))

    def test_forced_fa3_requires_extension_even_when_fa2_is_available(self):
        with patch.object(wan_attention, "FLASH_ATTN_3_AVAILABLE", False), patch.object(wan_attention, "FLASH_ATTN_2_AVAILABLE", True):
            with self.assertRaisesRegex(RuntimeError, "explicitly requested.*unavailable"):
                wan_attention.resolve_flash_attention_version("flash_attention_3", None, "cuda:0")

    def test_forced_fa3_rejects_cpu_and_non_hopper_devices(self):
        with patch.object(wan_attention, "FLASH_ATTN_3_AVAILABLE", True), patch("torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "requires a CUDA Hopper"):
                wan_attention.resolve_flash_attention_version("flash_attention_3", None, "cpu")
            with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
                with self.assertRaisesRegex(RuntimeError, "requires a Hopper GPU"):
                    wan_attention.resolve_flash_attention_version("flash_attention_3", None, "cuda:0")

    def test_forced_fa3_resolves_on_h200_and_reports_hardware(self):
        with (
            patch.object(wan_attention, "FLASH_ATTN_3_AVAILABLE", True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch("torch.cuda.get_device_name", return_value="NVIDIA H200"),
        ):
            self.assertEqual(wan_attention.resolve_flash_attention_version("flash_attention_3", None, "cuda:1"), 3)
            hardware = wan_attention.require_flash_attention_3("cuda:1")
            self.assertEqual(hardware, {"device": "cuda:1", "gpu": "NVIDIA H200", "compute_capability": (9, 0)})

    def test_forced_fa3_rejects_ignored_options_and_version_conflicts(self):
        for options in ({"dropout_p": 0.1}, {"window_size": (10, 0)}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "requires dropout_p=0"):
                wan_attention.resolve_flash_attention_version("flash_attention_3", None, "cuda:0", **options)
        with self.assertRaisesRegex(ValueError, "different attention version"):
            wan_attention.resolve_flash_attention_version("flash_attention_3", 2, "cuda:0")

    def test_legacy_auto_remains_auto_and_explicit_fa2_stays_fa2(self):
        with patch.object(wan_attention, "FLASH_ATTN_3_AVAILABLE", True), patch.object(wan_attention, "is_hopper_gpu", return_value=True):
            self.assertEqual(wan_attention.resolve_flash_attention_version("flash_attention", None, "cuda:0"), 3)
            self.assertEqual(wan_attention.resolve_flash_attention_version("flash_attention", 2, "cuda:0"), 2)
        with patch.object(wan_attention, "FLASH_ATTN_3_AVAILABLE", False):
            self.assertEqual(wan_attention.resolve_flash_attention_version("flash_attention", None, "cuda:0"), 2)
            with self.assertWarnsRegex(UserWarning, "use flash attention 2"):
                self.assertEqual(wan_attention.resolve_flash_attention_version("flash_attention", 3, "cuda:0"), 2)

    def test_legacy_auto_uses_fa2_for_options_not_forwarded_to_fa3(self):
        with patch.object(wan_attention, "FLASH_ATTN_3_AVAILABLE", True), patch.object(wan_attention, "is_hopper_gpu", return_value=True):
            for options in ({"dropout_p": 0.1}, {"window_size": (10, 0)}):
                with self.subTest(options=options):
                    self.assertEqual(wan_attention.resolve_flash_attention_version("flash_attention", None, "cuda:0", **options), 2)
                    with self.assertWarnsRegex(UserWarning, "device/options.*use flash attention 2"):
                        self.assertEqual(wan_attention.resolve_flash_attention_version("flash_attention", 3, "cuda:0", **options), 2)


class WanAttentionPrecisionTest(unittest.TestCase):
    def inputs(self, batch=2, lq=5, lk=6):
        generator = torch.Generator().manual_seed(17)
        return [torch.randn(batch, length, 2, 4, generator=generator, requires_grad=True) for length in (lq, lk, lk)]

    def reference(self, q, k, v, q_lens, k_lens, *, causal=False, window_size=(-1, -1), scale=None, q_scale=1.0):
        outputs = []
        for index, (lq, lk) in enumerate(zip(q_lens, k_lens)):
            query = q[index, :lq].transpose(0, 1) * q_scale
            key = k[index, :lk].transpose(0, 1)
            value = v[index, :lk].transpose(0, 1)
            logits = query @ key.transpose(-1, -2) * (q.shape[-1] ** -0.5 if scale is None else scale)
            q_positions = torch.arange(lq)[:, None] + lk - lq
            k_positions = torch.arange(lk)[None, :]
            mask = torch.ones(lq, lk, dtype=torch.bool)
            if causal:
                mask &= k_positions <= q_positions
            if window_size[0] >= 0:
                mask &= k_positions >= q_positions - window_size[0]
            if window_size[1] >= 0:
                mask &= k_positions <= q_positions + window_size[1]
            valid_rows = mask.any(dim=-1, keepdim=True)
            # Avoid NaN gradients from softmax on an entirely masked row.
            logits = logits.masked_fill(~mask, -torch.inf)
            logits = torch.where(valid_rows, logits, torch.zeros_like(logits))
            probabilities = torch.softmax(logits, dim=-1) * valid_rows
            output = (probabilities @ value).transpose(0, 1)
            outputs.append(F.pad(output, (0, 0, 0, 0, 0, q.shape[1] - lq)))
        return torch.stack(outputs)

    def test_explicit_sdpa_preserves_fp32_even_under_outer_autocast(self):
        q, k, v = self.inputs()
        with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as call:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = flash_attention(q, k, v, backend="sdpa")
        self.assertEqual(actual.dtype, torch.float32)
        for tensor in call.call_args.args[:3]:
            self.assertEqual(tensor.dtype, torch.float32)
        expected = self.reference(q, k, v, [5, 5], [6, 6])
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        for tensor in (q, k, v):
            self.assertEqual(tensor.grad.dtype, torch.float32)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_lengths_scale_outputs_and_gradients_match_reference(self):
        q, k, v = self.inputs()
        q_lens, k_lens = [2, 5], [4, 3]
        actual = sdpa_attention(q, k, v, q_lens=torch.tensor(q_lens), k_lens=torch.tensor(k_lens), softmax_scale=0.7, q_scale=1.3)
        expected = self.reference(q, k, v, q_lens, k_lens, scale=0.7, q_scale=1.3)
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)
        self.assertEqual(actual[0, 2:].count_nonzero().item(), 0)
        self.assertEqual(actual_grads[0][0, 2:].count_nonzero().item(), 0)
        self.assertEqual(actual_grads[1][1, 3:].count_nonzero().item(), 0)
        self.assertEqual(actual_grads[2][1, 3:].count_nonzero().item(), 0)

    def test_causal_rectangular_masks_use_flash_attention_bottom_right_alignment(self):
        for lq, lk in [(5, 3), (3, 5), (4, 4)]:
            with self.subTest(lq=lq, lk=lk):
                q, k, v = self.inputs(batch=1, lq=lq, lk=lk)
                actual = sdpa_attention(q, k, v, causal=True)
                expected = self.reference(q, k, v, [lq], [lk], causal=True)
                torch.testing.assert_close(actual, expected)
                actual.sum().backward()
                self.assertTrue(all(torch.isfinite(tensor.grad).all() for tensor in (q, k, v)))
                if lq > lk:
                    self.assertEqual(actual[:, : lq - lk].count_nonzero().item(), 0)

    def test_window_and_variable_lengths_combine_with_causal_mask(self):
        q, k, v = self.inputs()
        actual = sdpa_attention(q, k, v, q_lens=[3, 5], k_lens=[6, 4], causal=True, window_size=(1, 2))
        expected = self.reference(q, k, v, [3, 5], [6, 4], causal=True, window_size=(1, 2))
        torch.testing.assert_close(actual, expected)

    def test_empty_sequences_have_zero_outputs_and_gradients(self):
        q, k, v = self.inputs()
        actual = sdpa_attention(q, k, v, q_lens=[0, 5], k_lens=[6, 0])
        self.assertEqual(actual.count_nonzero().item(), 0)
        actual.sum().backward()
        for tensor in (q, k, v):
            self.assertEqual(tensor.grad.count_nonzero().item(), 0)

    def test_bf16_value_activations_allow_teacher_fallback(self):
        q, k, v = self.inputs(batch=1)
        with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as call:
            actual = sdpa_attention(q, k, v.to(torch.bfloat16))
        self.assertEqual(actual.dtype, q.dtype)
        for tensor in call.call_args.args[:3]:
            self.assertEqual(tensor.dtype, torch.bfloat16)

    def test_bad_backend_and_lengths_fail_explicitly(self):
        q, k, v = self.inputs()
        with self.assertRaisesRegex(ValueError, "Unsupported Wan attention backend"):
            flash_attention(q, k, v, backend="unknown")
        for lengths in ([1], [1, 7], [1, -1], [1, 2.5]):
            with self.subTest(lengths=lengths), self.assertRaisesRegex(ValueError, "lengths"):
                sdpa_attention(q, k, v, k_lens=lengths)


class WanAttentionConfigurationTest(unittest.TestCase):
    def tiny_model(self, **kwargs):
        return WanModel(
            patch_size=(1, 1, 1),
            text_len=4,
            in_dim=4,
            dim=16,
            ffn_dim=32,
            freq_dim=8,
            text_dim=12,
            out_dim=4,
            num_heads=2,
            num_layers=1,
            **kwargs,
        )

    def test_backend_propagates_per_model_without_changing_default(self):
        sdpa = self.tiny_model(attention_backend="sdpa")
        fa3 = self.tiny_model(attention_backend="flash_attention_3")
        default = self.tiny_model()
        self.assertEqual(sdpa.config.attention_backend, "sdpa")
        self.assertEqual(default.config.attention_backend, "flash_attention")
        for model, backend in [(sdpa, "sdpa"), (fa3, "flash_attention_3"), (default, "flash_attention")]:
            self.assertEqual(model.blocks[0].self_attn.attention_backend, backend)
            self.assertEqual(model.blocks[0].cross_attn.attention_backend, backend)

    def test_forced_fa3_model_rejects_local_window(self):
        with self.assertRaisesRegex(ValueError, "global window_size"):
            self.tiny_model(attention_backend="flash_attention_3", window_size=(2, 2))

    def test_forced_fa3_wrapper_validates_before_loading_weights(self):
        wrapper = WanT2VModel({"model": {"name": "wan_t2v", "running_dtype": "bf16", "pretrained_model_name_or_path": "/unused", "attention_backend": "flash_attention_3"}})
        with patch("lightx2v_train.model_zoo.wan.wan_t2v.require_flash_attention_3", side_effect=RuntimeError("strict FA3 unavailable")), patch.object(WanModel, "from_pretrained") as loader:
            with self.assertRaisesRegex(RuntimeError, "strict FA3 unavailable"):
                wrapper.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
            loader.assert_not_called()

    def test_tiny_wan_forward_backward_uses_fp32_self_and_cross_attention(self):
        model = self.tiny_model(attention_backend="sdpa")
        with torch.no_grad():
            model.head.head.weight.normal_(std=0.1)
        latent = torch.randn(4, 1, 2, 2, requires_grad=True)
        with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as call:
            output = model(x=[latent], t=torch.tensor([0.5]), context=[torch.randn(3, 12)], seq_len=4)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(call.call_count, 2)
        for args in call.call_args_list:
            for tensor in args.args[:3]:
                self.assertEqual(tensor.dtype, torch.float32)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(model.blocks[0].self_attn.q.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.blocks[0].cross_attn.q.weight.grad.abs().sum().item(), 0)

    def test_wrapper_passes_configured_backend_into_pretrained_loader(self):
        for backend in ("flash_attention", "sdpa"):
            config = {"model": {"name": "wan_t2v", "running_dtype": "fp32", "pretrained_model_name_or_path": "/unused"}}
            if backend != "flash_attention":
                config["model"]["attention_backend"] = backend
            wrapper = WanT2VModel(config)
            native = self.tiny_model(attention_backend=backend)
            with patch.object(WanModel, "from_pretrained", return_value=native) as loader:
                wrapper.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
            loader.assert_called_once_with("/unused", torch_dtype=torch.float32, attention_backend=backend)
            self.assertIs(wrapper.transformer, native)

    def test_causal_model_does_not_silently_ignore_fp32_backend(self):
        wrapper = WanT2VModel({"model": {"name": "wan_t2v_ar", "running_dtype": "fp32", "pretrained_model_name_or_path": "/unused", "attention_backend": "sdpa"}})
        with self.assertRaisesRegex(ValueError, "non-causal WanModel only"):
            wrapper.load_components(load_transformer=False, load_vae=False, load_condition_encoder=False)


if __name__ == "__main__":
    unittest.main()
