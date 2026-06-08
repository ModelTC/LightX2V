import unittest

import torch
import torch.nn.functional as F

try:
    from lightx2v_kernel.gemm import (
        cutlass_scaled_mxfp8_mm,
        cutlass_scaled_mxfp8_mm_residual_gate,
        scaled_mxfp8_gelu_quant,
        scaled_mxfp8_modulate_quant,
        scaled_mxfp8_quant,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - test reports extension availability.
    cutlass_scaled_mxfp8_mm = None
    cutlass_scaled_mxfp8_mm_residual_gate = None
    scaled_mxfp8_gelu_quant = None
    scaled_mxfp8_modulate_quant = None
    scaled_mxfp8_quant = None
    _IMPORT_ERROR = exc


def _first_visible_sm120_device():
    if not torch.cuda.is_available():
        return None
    for device_index in range(torch.cuda.device_count()):
        major, _minor = torch.cuda.get_device_capability(device_index)
        if major == 12:
            return torch.device("cuda", device_index)
    return None


def _skip_cuda_unavailable():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if _first_visible_sm120_device() is None:
        caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
        return f"MXFP8 fused FFN kernels require a visible SM120/SM120a CUDA device, got {caps}"
    return None


class TestMxfp8FusedFfn(unittest.TestCase):
    def setUp(self):
        skip_reason = _skip_cuda_unavailable()
        if skip_reason is not None:
            self.skipTest(skip_reason)
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"Failed to import MXFP8 fused FFN symbols: {_IMPORT_ERROR}") from _IMPORT_ERROR
        self.device = _first_visible_sm120_device()
        torch.cuda.set_device(self.device)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

    def _quantized_inputs(self, m=257, k=256, n=384):
        activation = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.5
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.5
        bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.1
        activation_quant, activation_scale = scaled_mxfp8_quant(activation)
        weight_quant, weight_scale = scaled_mxfp8_quant(weight)
        alpha = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        return activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias

    def assert_close_enough(self, actual, expected):
        actual_f = actual.float().flatten()
        expected_f = expected.float().flatten()
        cosine = F.cosine_similarity(actual_f, expected_f, dim=0).item()
        max_abs = (actual_f - expected_f).abs().max().item()
        mean_abs = (actual_f - expected_f).abs().mean().item()
        self.assertGreater(cosine, 0.999, f"cosine={cosine}, max_abs={max_abs}, mean_abs={mean_abs}")
        self.assertLess(max_abs, 0.08, f"cosine={cosine}, max_abs={max_abs}, mean_abs={mean_abs}")

    def assert_fallback_close(self, actual, expected):
        actual_f = actual.float()
        expected_f = expected.float()
        max_abs = (actual_f - expected_f).abs().max().item()
        mean_abs = (actual_f - expected_f).abs().mean().item()
        self.assertLessEqual(max_abs, 1e-3, f"max_abs={max_abs}, mean_abs={mean_abs}")
        self.assertLessEqual(mean_abs, 1e-4, f"max_abs={max_abs}, mean_abs={mean_abs}")

    def _residual_gate_reference(self, residual, gemm_out, gate):
        return (residual.float() + gemm_out.float() * gate.float()).to(torch.bfloat16)

    def _misaligned_2d_tensor(self, rows, cols, offset):
        buf = torch.empty(rows * cols + offset + 8, dtype=torch.bfloat16, device="cuda")
        tensor = buf[offset : offset + rows * cols].view(rows, cols)
        self.assertTrue(tensor.is_contiguous())
        return tensor

    def test_mxfp8_gelu_quant_matches_baseline(self):
        activation = torch.randn(257, 512, dtype=torch.bfloat16, device="cuda") * 0.5
        baseline_quant, baseline_scale = scaled_mxfp8_quant(F.gelu(activation, approximate="tanh"))
        fused_quant, fused_scale = scaled_mxfp8_gelu_quant(activation)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(fused_quant, baseline_quant))

        # Scale factors use the CUTLASS tiled layout, so row-major slicing is not
        # meaningful for partial M tiles. Use a full tile for raw scale equality.
        activation = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda") * 0.5
        baseline_quant, baseline_scale = scaled_mxfp8_quant(F.gelu(activation, approximate="tanh"))
        fused_quant, fused_scale = scaled_mxfp8_gelu_quant(activation)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(fused_quant, baseline_quant))
        self.assertTrue(torch.equal(fused_scale.view(torch.uint8), baseline_scale.view(torch.uint8)))

    def test_mxfp8_quant_ops_validate_explicit_outputs(self):
        activation = torch.randn(257, 512, dtype=torch.bfloat16, device="cuda") * 0.5
        m, n = activation.shape
        output = torch.empty((m, n), dtype=torch.uint8, device="cuda")
        output_sf = torch.empty(((m + 127) // 128 * 128, (n // 32 + 3) // 4), dtype=torch.int32, device="cuda")
        scale = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        shift = torch.randn(n, dtype=torch.bfloat16, device="cuda")

        with self.assertRaisesRegex(RuntimeError, "output dtype must be uint8"):
            torch.ops.lightx2v_kernel.scaled_mxfp8_gelu_quant_sm120.default(
                torch.empty_like(activation),
                activation,
                output_sf,
            )
        with self.assertRaisesRegex(RuntimeError, "output_sf dtype must be int32"):
            torch.ops.lightx2v_kernel.scaled_mxfp8_modulate_quant_sm120.default(
                output,
                activation,
                scale,
                shift,
                torch.empty_like(output_sf, dtype=torch.float32),
            )
        with self.assertRaisesRegex(RuntimeError, "output_sf shape must be"):
            torch.ops.lightx2v_kernel.scaled_mxfp8_quant_sm120.default(
                output,
                activation,
                torch.empty((128, output_sf.shape[1]), dtype=torch.int32, device="cuda"),
            )
        with self.assertRaisesRegex(RuntimeError, "scale must have shape"):
            torch.ops.lightx2v_kernel.scaled_mxfp8_modulate_quant_sm120.default(
                output,
                activation,
                scale.reshape(n, 1),
                shift,
                output_sf,
            )
        with self.assertRaisesRegex(RuntimeError, "shift must have shape"):
            torch.ops.lightx2v_kernel.scaled_mxfp8_modulate_quant_sm120.default(
                output,
                activation,
                scale,
                shift.reshape(n, 1),
                output_sf,
            )

    def _modulate_reference(self, activation, scale, shift):
        return (activation.float() * (1.0 + scale.float()) + shift.float()).to(activation.dtype)

    def test_mxfp8_modulate_quant_matches_1d_baseline(self):
        activation = torch.randn(257, 512, dtype=torch.bfloat16, device="cuda") * 0.5
        scale = torch.randn(512, dtype=torch.bfloat16, device="cuda") * 0.1
        shift = torch.randn(512, dtype=torch.bfloat16, device="cuda") * 0.1
        baseline_quant, baseline_scale = scaled_mxfp8_quant(self._modulate_reference(activation, scale, shift))
        fused_quant, fused_scale = scaled_mxfp8_modulate_quant(activation, scale, shift)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(fused_quant, baseline_quant))

        activation = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda") * 0.5
        baseline_quant, baseline_scale = scaled_mxfp8_quant(self._modulate_reference(activation, scale, shift))
        fused_quant, fused_scale = scaled_mxfp8_modulate_quant(activation, scale.view(1, 1, -1), shift.view(1, 1, -1))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(fused_quant, baseline_quant))
        self.assertTrue(torch.equal(fused_scale.view(torch.uint8), baseline_scale.view(torch.uint8)))

    def test_mxfp8_modulate_quant_matches_2d_baseline(self):
        activation = torch.randn(257, 512, dtype=torch.bfloat16, device="cuda") * 0.5
        scale = torch.randn(257, 512, dtype=torch.bfloat16, device="cuda") * 0.1
        shift = torch.randn(257, 512, dtype=torch.bfloat16, device="cuda") * 0.1
        baseline_quant, baseline_scale = scaled_mxfp8_quant(self._modulate_reference(activation, scale, shift))
        fused_quant, fused_scale = scaled_mxfp8_modulate_quant(activation, scale, shift)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(fused_quant, baseline_quant))

        activation = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda") * 0.5
        scale = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda") * 0.1
        shift = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda") * 0.1
        baseline_quant, baseline_scale = scaled_mxfp8_quant(self._modulate_reference(activation, scale, shift))
        fused_quant, fused_scale = scaled_mxfp8_modulate_quant(activation, scale, shift)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(fused_quant, baseline_quant))
        self.assertTrue(torch.equal(fused_scale.view(torch.uint8), baseline_scale.view(torch.uint8)))

    def test_mxfp8_gemm_residual_gate_matches_baseline(self):
        activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias = self._quantized_inputs()
        m, n = activation_quant.shape[0], weight_quant.shape[0]
        residual = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.25
        gemm_out = cutlass_scaled_mxfp8_mm(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            bias=bias,
        )
        baseline = residual + gemm_out * gate
        fused = cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual.clone(),
            gate=gate,
            bias=bias,
        )
        torch.cuda.synchronize()
        self.assert_close_enough(fused, baseline)

    def test_mxfp8_gemm_residual_gate_without_bias_matches_baseline(self):
        activation_quant, weight_quant, activation_scale, weight_scale, alpha, _ = self._quantized_inputs()
        m, n = activation_quant.shape[0], weight_quant.shape[0]
        residual = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.25
        gemm_out = cutlass_scaled_mxfp8_mm(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            bias=None,
        )
        baseline = residual + gemm_out * gate
        fused = cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual.clone(),
            gate=gate,
            bias=None,
        )
        torch.cuda.synchronize()
        self.assert_close_enough(fused, baseline)

    def test_mxfp8_gemm_residual_gate_2d_fallback_matches_baseline(self):
        activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias = self._quantized_inputs()
        m, n = activation_quant.shape[0], weight_quant.shape[0]
        residual = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(m, n, dtype=torch.bfloat16, device="cuda") * 0.25
        gemm_out = cutlass_scaled_mxfp8_mm(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            bias=bias,
        )
        baseline = residual + gemm_out * gate
        fused = cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual.clone(),
            gate=gate,
            bias=bias,
        )
        torch.cuda.synchronize()
        self.assert_close_enough(fused, baseline)
        self.assert_fallback_close(fused, self._residual_gate_reference(residual, gemm_out, gate))

    def test_mxfp8_gemm_residual_gate_1d_fast_path_matches_2d_fallback_contract(self):
        activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias = self._quantized_inputs()
        m, n = activation_quant.shape[0], weight_quant.shape[0]
        residual = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.25
        fused_1d = cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual.clone(),
            gate=gate,
            bias=bias,
        )
        fused_2d = cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual.clone(),
            gate=gate.expand(m, n).contiguous(),
            bias=bias,
        )
        torch.cuda.synchronize()
        self.assert_close_enough(fused_1d, fused_2d)

    def test_mxfp8_gemm_residual_gate_2d_fallback_misaligned_x2_path_matches_reference(self):
        activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias = self._quantized_inputs()
        m, n = activation_quant.shape[0], weight_quant.shape[0]
        residual = self._misaligned_2d_tensor(m, n, offset=2).normal_(0, 1)
        gate = self._misaligned_2d_tensor(m, n, offset=2).normal_(0, 0.25)
        self.assertNotEqual(residual.data_ptr() % 16, 0)
        self.assertEqual(residual.data_ptr() % 4, 0)
        gemm_out = cutlass_scaled_mxfp8_mm(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            bias=bias,
        )
        expected = self._residual_gate_reference(residual, gemm_out, gate)
        fused = cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual,
            gate=gate,
            bias=bias,
        )
        torch.cuda.synchronize()
        self.assert_fallback_close(fused, expected)

    def test_mxfp8_gemm_residual_gate_2d_fallback_misaligned_x1_path_matches_reference(self):
        activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias = self._quantized_inputs()
        m, n = activation_quant.shape[0], weight_quant.shape[0]
        residual = self._misaligned_2d_tensor(m, n, offset=1).normal_(0, 1)
        gate = self._misaligned_2d_tensor(m, n, offset=1).normal_(0, 0.25)
        self.assertNotEqual(residual.data_ptr() % 4, 0)
        gemm_out = cutlass_scaled_mxfp8_mm(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            bias=bias,
        )
        expected = self._residual_gate_reference(residual, gemm_out, gate)
        fused = cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual,
            gate=gate,
            bias=bias,
        )
        torch.cuda.synchronize()
        self.assert_fallback_close(fused, expected)

    def test_mxfp8_gemm_residual_gate_2d_fallback_zero_gate_matches_residual(self):
        activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias = self._quantized_inputs()
        m, n = activation_quant.shape[0], weight_quant.shape[0]
        residual = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        gate = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
        fused = cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual.clone(),
            gate=gate,
            bias=bias,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(fused, residual))

    def test_mxfp8_gemm_residual_gate_2d_fallback_cancellation_case_matches_reference(self):
        activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias = self._quantized_inputs()
        m, n = activation_quant.shape[0], weight_quant.shape[0]
        gemm_out = cutlass_scaled_mxfp8_mm(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            bias=bias,
        )
        gate = torch.where(torch.arange(n, device="cuda") % 2 == 0, -torch.ones(n, device="cuda"), torch.ones(n, device="cuda")).to(torch.bfloat16)
        gate = gate.expand(m, n).contiguous()
        residual = (-gemm_out.float() * gate.float() + 1e-2).to(torch.bfloat16)
        fused = cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual.clone(),
            gate=gate,
            bias=bias,
        )
        torch.cuda.synchronize()
        self.assert_fallback_close(fused, self._residual_gate_reference(residual, gemm_out, gate))

    def _assert_residual_gate_rejects(self, mat_a, mat_b, scales_a, scales_b, alpha, residual, gate, pattern, bias=None):
        with self.assertRaisesRegex(RuntimeError, pattern):
            cutlass_scaled_mxfp8_mm_residual_gate(
                mat_a,
                mat_b,
                scales_a,
                scales_b,
                alpha,
                residual=residual,
                gate=gate,
                bias=bias,
            )

    def test_mxfp8_gemm_residual_gate_fast_path_validates_gemm_inputs(self):
        activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias = self._quantized_inputs()
        m, n = activation_quant.shape[0], weight_quant.shape[0]
        residual = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.25

        bad_weight = torch.randn(n, activation_quant.shape[1] + 32, dtype=torch.bfloat16, device="cuda")
        bad_weight_quant, bad_weight_scale = scaled_mxfp8_quant(bad_weight)
        bad_residual = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        self._assert_residual_gate_rejects(
            activation_quant,
            bad_weight_quant,
            activation_scale,
            bad_weight_scale,
            alpha,
            bad_residual,
            gate,
            "shapes cannot be multiplied",
            bias=bias,
        )

        unaligned_weight = torch.randn(130, activation_quant.shape[1], dtype=torch.bfloat16, device="cuda")
        unaligned_weight_quant, unaligned_weight_scale = scaled_mxfp8_quant(unaligned_weight)
        unaligned_residual = torch.randn(m, 130, dtype=torch.bfloat16, device="cuda")
        unaligned_gate = torch.randn(130, dtype=torch.bfloat16, device="cuda")
        self._assert_residual_gate_rejects(
            activation_quant,
            unaligned_weight_quant,
            activation_scale,
            unaligned_weight_scale,
            alpha,
            unaligned_residual,
            unaligned_gate,
            "Expected n to be divisible by 128",
        )

        self._assert_residual_gate_rejects(
            activation_quant,
            weight_quant,
            activation_scale[:128],
            weight_scale,
            alpha,
            residual.clone(),
            gate,
            "scale_a must be padded",
            bias=bias,
        )
        self._assert_residual_gate_rejects(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale[:128],
            alpha,
            residual.clone(),
            gate,
            "scale_b must be padded",
            bias=bias,
        )
        self._assert_residual_gate_rejects(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha.double(),
            residual.clone(),
            gate,
            "Inconsistency of Tensor type:alpha",
            bias=bias,
        )
        self._assert_residual_gate_rejects(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            torch.ones(2, dtype=torch.float32, device="cuda"),
            residual.clone(),
            gate,
            "alpha must contain exactly one scalar",
            bias=bias,
        )

    def test_mxfp8_gemm_residual_gate_fast_path_validates_device_mismatch(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("device mismatch test requires at least two visible CUDA devices")
        other_device = None
        for device_index in range(torch.cuda.device_count()):
            if device_index != self.device.index:
                other_device = torch.device("cuda", device_index)
                break
        if other_device is None:
            self.skipTest("device mismatch test requires another visible CUDA device")
        activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias = self._quantized_inputs()
        m, n = activation_quant.shape[0], weight_quant.shape[0]
        residual = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(n, dtype=torch.bfloat16, device=other_device)
        self._assert_residual_gate_rejects(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual,
            gate,
            "same CUDA device",
            bias=bias,
        )


if __name__ == "__main__":
    unittest.main()
