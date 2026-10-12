"""Run with PYTHONPATH=lightx2v_train python -m unittest discover -s lightx2v_train/tests -p test_mlu_runtime.py."""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from lightx2v_train.runtime.attention import prepare_diffusers_attention
from lightx2v_train.runtime.backend import factory
from lightx2v_train.runtime.backend.mlu import MluBackend
from lightx2v_train.runtime.ops.attention import training_attention
from lightx2v_train.runtime.ops.rope import apply_rotary_pairs, positional_compute_dtype, prepare_rotary_frequencies


class BackendSelectionTests(unittest.TestCase):
    def test_explicit_platform_uses_inference_name(self):
        with patch.dict(os.environ, {"PLATFORM": "cambricon_mlu"}):
            name = factory._resolve_backend_name()
        self.assertIs(factory._load_backend_class(name), MluBackend)

    def test_auto_does_not_load_optional_mlu_extension(self):
        with patch.dict(os.environ, {"PLATFORM": "auto"}), patch("torch.cuda.is_available", return_value=False), patch.object(MluBackend, "_load_extension") as load:
            self.assertEqual(factory._resolve_backend_name(), "cpu")
        load.assert_not_called()

    def test_missing_extension_fails_explicit_initialization(self):
        with patch("lightx2v_train.runtime.backend.mlu.importlib.import_module", side_effect=ImportError("torch_mlu")):
            self.assertFalse(MluBackend.is_available())
            with self.assertRaisesRegex(RuntimeError, "requires torch_mlu"):
                MluBackend().initialize()

    def test_unavailable_device_does_not_fall_back_to_cpu(self):
        with patch.object(MluBackend, "load_device_module", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "cambricon_mlu.*not available"):
                MluBackend().initialize()

    def test_diffusers_attention_policy(self):
        backend = MluBackend()
        for policy in (None, "auto", "AUTO", "native", "sdpa", "torch_sdpa"):
            with self.subTest(policy=policy):
                self.assertEqual(backend.resolve_attention_backend(policy), "native")


class RotaryTests(unittest.TestCase):
    def test_cpu_precision_is_preserved(self):
        self.assertEqual(positional_compute_dtype("cpu"), torch.float64)
        frequencies = torch.polar(torch.ones(5, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64))
        torch.testing.assert_close(prepare_rotary_frequencies(frequencies, "cpu"), frequencies)

    def test_real_pairs_match_complex_outputs_and_gradients(self):
        torch.manual_seed(17)
        frequencies = torch.polar(torch.ones(5, 4), torch.randn(5, 4)).unsqueeze(0).unsqueeze(2)
        actual_input = torch.randn(2, 5, 3, 8, requires_grad=True)
        expected_input = actual_input.detach().clone().requires_grad_()
        actual = apply_rotary_pairs(actual_input, torch.view_as_real(frequencies))
        expected = torch.view_as_real(torch.view_as_complex(expected_input.reshape(2, 5, 3, 4, 2)) * frequencies).flatten(-2)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


def _reference_attention(q, k, v, q_lens, k_lens, causal, window_size):
    outputs = []
    for i, (qlen, klen) in enumerate(zip(q_lens, k_lens)):
        if not qlen or not klen:
            output = (q[i] + k[i].sum() + v[i].sum()) * 0
        else:
            query = q[i, :qlen].transpose(0, 1) * 0.75
            key = k[i, :klen].repeat_interleave(q.size(2) // k.size(2), dim=1).transpose(0, 1)
            value = v[i, :klen].repeat_interleave(q.size(2) // k.size(2), dim=1).transpose(0, 1)
            rows = torch.arange(qlen).unsqueeze(1)
            cols = torch.arange(klen).unsqueeze(0)
            center = rows + klen - qlen
            visible = torch.ones(qlen, klen, dtype=torch.bool)
            if causal:
                visible &= cols <= center
            if window_size[0] >= 0:
                visible &= cols >= center - window_size[0]
            if window_size[1] >= 0:
                visible &= cols <= center + window_size[1]
            logits = (query @ key.transpose(-2, -1) * 0.3).masked_fill(~visible, float("-inf"))
            valid_rows = visible.any(-1, keepdim=True)
            logits = torch.where(valid_rows, logits, 0)
            weights = logits.softmax(-1) * valid_rows
            output = (weights @ value).transpose(0, 1)
            output = torch.cat([output, output.new_zeros(q.size(1) - qlen, q.size(2), v.size(-1))])
        outputs.append(output)
    return torch.stack(outputs)


class MluOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MluBackend.is_available():
            raise unittest.SkipTest("Requires torch_mlu and a visible MLU device")
        cls.backend = MluBackend().initialize(0)

    def test_autocast_and_device_operations(self):
        backend = self.backend
        backend.manual_seed_all(42)
        first = torch.randn(4, 4, device=backend.device)
        backend.manual_seed_all(42)
        torch.testing.assert_close(torch.randn_like(first), first)
        with backend.autocast(torch.bfloat16):
            output = first @ first
        self.assertEqual(output.dtype, torch.bfloat16)
        with backend.autocast():
            output = first @ first
        self.assertEqual(output.dtype, torch.get_autocast_dtype("mlu"))
        backend.synchronize()
        self.assertIsNotNone(backend.memory_gb())
        backend.empty_cache()

    def test_real_rotary_preparation_and_backward(self):
        frequencies = torch.polar(torch.ones(5, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64))
        pairs = prepare_rotary_frequencies(frequencies, self.backend.device)
        self.assertEqual(positional_compute_dtype(self.backend.device), torch.float32)
        self.assertEqual(pairs.dtype, torch.float32)
        self.assertEqual(pairs.shape, (5, 4, 2))
        torch.testing.assert_close(pairs.cpu(), torch.view_as_real(frequencies).float())
        x = torch.randn(2, 5, 3, 8, requires_grad=True)
        device_x = x.detach().to(self.backend.device).requires_grad_()
        actual = apply_rotary_pairs(device_x, pairs[None, :, None])
        expected = apply_rotary_pairs(x, pairs.cpu()[None, :, None])
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual.cpu(), expected)
        torch.testing.assert_close(device_x.grad.cpu(), x.grad)

    def test_attention_outputs_and_gradients(self):
        cases = [
            (4, 6, [3, 4], [5, 6], False, (-1, -1)),
            (6, 3, [6, 5], [3, 2], True, (-1, -1)),
            (5, 5, [4, 5], [5, 4], False, (1, 0)),
            (4, 3, [0, 4], [2, 0], False, (-1, -1)),
        ]
        for dtype in (torch.float32, torch.bfloat16):
            for qlen, klen, q_lens, k_lens, causal, window in cases:
                with self.subTest(dtype=dtype, qlen=qlen, klen=klen, causal=causal, window=window):
                    cpu = [torch.randn(2, qlen, 4, 8), torch.randn(2, klen, 2, 8), torch.randn(2, klen, 2, 8)]
                    device = [t.to(device=self.backend.device, dtype=dtype).requires_grad_() for t in cpu]
                    reference = [t.detach().cpu().float().requires_grad_() for t in device]
                    actual = training_attention(*device, q_lens=q_lens, k_lens=k_lens, causal=causal, window_size=window, softmax_scale=0.3, q_scale=0.75)
                    expected = _reference_attention(*reference, q_lens, k_lens, causal, window)
                    gradient = torch.randn_like(expected)
                    actual.backward(gradient.to(device=actual.device, dtype=dtype))
                    expected.backward(gradient.to(dtype).float())
                    tolerance = 0.03 if dtype == torch.bfloat16 else 1e-4
                    torch.testing.assert_close(actual.cpu().float(), expected, atol=tolerance, rtol=tolerance)
                    for tensor, ref in zip(device, reference):
                        self.assertTrue(torch.isfinite(tensor.grad).all().item())
                        torch.testing.assert_close(tensor.grad.cpu().float(), ref.grad, atol=tolerance, rtol=tolerance)

    def test_attention_dropout_backward(self):
        inputs = [torch.randn(2, 5, 2, 8, device=self.backend.device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        output = training_attention(*inputs, dropout_p=0.25)
        self.assertTrue(torch.isfinite(output).all().item())
        output.float().square().mean().backward()
        for tensor in inputs:
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_qwen_complex_rotary_backward(self):
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

        x = torch.randn(2, 5, 3, 8, requires_grad=True)
        frequencies = torch.polar(torch.ones(5, 4), torch.randn(5, 4))
        device_x = x.detach().to(self.backend.device).requires_grad_()
        actual = apply_rotary_emb_qwen(device_x, frequencies.to(self.backend.device), use_real=False)
        expected = apply_rotary_emb_qwen(x, frequencies, use_real=False)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual.cpu(), expected)
        torch.testing.assert_close(device_x.grad.cpu(), x.grad)

    def test_qwen_lora_steps_and_save_reload(self):
        from diffusers import QwenImageTransformer2DModel
        from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file, save_file

        model = QwenImageTransformer2DModel(
            in_channels=8,
            out_channels=2,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            axes_dims_rope=(2, 2, 4),
        ).to(device=self.backend.device, dtype=torch.bfloat16)
        model.requires_grad_(False)
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        with patch.object(factory, "_BACKEND", self.backend):
            prepare_diffusers_attention(model)
        model.enable_gradient_checkpointing()
        model.train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().cpu().clone() for p in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=0.01)
        for _ in range(3):
            output = model(
                hidden_states=torch.randn(1, 4, 8, device=self.backend.device, dtype=torch.bfloat16),
                encoder_hidden_states=torch.randn(1, 3, 16, device=self.backend.device, dtype=torch.bfloat16),
                timestep=torch.ones(1, device=self.backend.device, dtype=torch.bfloat16),
                img_shapes=[[(1, 2, 2)]],
                encoder_hidden_states_mask=torch.tensor([[True, True, False]], device=self.backend.device),
                return_dict=False,
            )[0]
            self.assertEqual(output.shape, (1, 4, 8))
            loss = output.float().square().mean()
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in parameters))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.assertTrue(any(not torch.equal(p.detach().cpu(), previous) for p, previous in zip(parameters, before)))
        state = {key: value.detach().cpu().contiguous() for key, value in get_peft_model_state_dict(model).items()}
        with tempfile.TemporaryDirectory(prefix="lightx2v_mlu_lora_") as directory:
            filename = os.path.join(directory, "adapter.safetensors")
            save_file(state, filename)
            with torch.no_grad():
                for parameter in parameters:
                    parameter.zero_()
            set_peft_model_state_dict(model, load_file(filename))
        restored = get_peft_model_state_dict(model)
        for key, value in state.items():
            torch.testing.assert_close(restored[key].cpu(), value)


if __name__ == "__main__":
    unittest.main()
