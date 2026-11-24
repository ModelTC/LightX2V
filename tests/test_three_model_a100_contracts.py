"""Small A100 regression checks for the new bindings and request contracts."""

# Runtime imports follow the CUDA availability guard.
# ruff: noqa: E402
import ast
import unittest
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("A100 contract checks require CUDA", allow_module_level=True)
import torch.nn.functional as F

import lightx2v.common.ops  # noqa: F401 - populate the shared operator registries
from lightx2v.models.networks.hymotion.weights.positional_encoding import RotaryEmbedding
from lightx2v.models.networks.three_models.ops import Attention, LayerNorm, Linear, RMSNorm
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.hymotion.hymotion_runner import HYMotionRunner
from lightx2v.models.runners.sdxl.sdxl_runner import SDXLRunner
from lightx2v.models.runners.sf3d.sf3d_runner import SF3DRunner
from lightx2v.models.runners.three_models import validate_a100_config
from lightx2v.models.schedulers.hymotion.flow import odeint
from lightx2v.models.schedulers.sdxl import SDXLEulerScheduler


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        torch.set_grad_enabled(False)

    def test_linear_precision_and_leading_dimensions(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.randn(2, 3, 7, 32, device="cuda", dtype=dtype)
            weight = torch.randn(48, 32, device="cuda", dtype=dtype)
            bias = torch.randn(48, device="cuda", dtype=dtype)
            torch.testing.assert_close(Linear(weight, bias)(x), F.linear(x, weight, bias), rtol=0.01, atol=0.01)

    def test_sf3d_autocast_linear(self):
        x = torch.randn(1, 11, 32, device="cuda")
        weight = torch.randn(64, 32, device="cuda")
        bias = torch.randn(64, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = Linear(weight, bias)(x)
            reference = F.linear(x, weight, bias)
        self.assertEqual(value.dtype, torch.bfloat16)
        torch.testing.assert_close(value, reference, rtol=0.01, atol=0.01)

    def test_strided_linear_preserves_native_matmul_dispatch(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for requires_grad in (False, True):
                x = torch.randn(2, 64, 23, device="cuda", dtype=dtype).transpose(1, 2)
                weight = torch.nn.Parameter(torch.randn(80, 64, device="cuda", dtype=dtype), requires_grad=requires_grad)
                bias = torch.nn.Parameter(torch.randn(80, device="cuda", dtype=dtype), requires_grad=requires_grad)
                reference = F.linear(x, weight, bias)
                value = Linear(weight.detach(), bias.detach(), native_requires_grad=requires_grad)(x)
                self.assertTrue(torch.equal(value, reference), (dtype, requires_grad))

    def test_strided_autocast_linear_preserves_bias_rounding(self):
        x = torch.randn(2, 64, 23, device="cuda").transpose(1, 2)
        weight = torch.nn.Parameter(torch.randn(80, 64, device="cuda"))
        bias = torch.nn.Parameter(torch.randn(80, device="cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = F.linear(x, weight, bias)
            value = Linear(weight.detach(), bias.detach(), native_requires_grad=True)(x)
        self.assertTrue(torch.equal(value, reference))

    def test_norm_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.randn(2, 5, 32, device="cuda", dtype=dtype)
            weight = torch.randn(32, device="cuda", dtype=dtype)
            bias = torch.randn(32, device="cuda", dtype=dtype)
            torch.testing.assert_close(LayerNorm(weight, bias, 1e-6)(x), F.layer_norm(x, (32,), weight, bias, 1e-6))
            rms = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(dtype) * weight
            torch.testing.assert_close(RMSNorm(weight, 1e-6)(x), rms)

    def test_attention_masks_and_batch_one(self):
        for batch in (1, 2):
            q = torch.randn(batch, 3, 5, 16, device="cuda")
            k, v = [torch.randn(batch, 3, 7, 16, device="cuda") for _ in range(2)]
            mask = torch.ones(batch, 1, 5, 7, device="cuda", dtype=torch.bool)
            mask[:, :, :, -2:] = False
            for current_mask in (mask, torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, float("-inf"))):
                value = Attention()(q, k, v, mask=current_mask)
                self.assertEqual(value.shape, q.shape)
                torch.testing.assert_close(value, F.scaled_dot_product_attention(q, k, v, attn_mask=current_mask))

    def test_hymotion_interleaved_rope(self):
        rope = RotaryEmbedding(8, 32, use_real=True)
        q, k = [torch.randn(2, 9, 3, 8, device="cuda") for _ in range(2)]
        actual = rope.apply_rotary_emb(q, k)
        cos, sin = [t[:9].cuda()[None, :, None] for t in rope.freqs_cis]
        for x, y in zip((q, k), actual):
            first, second = x.reshape(2, 9, 3, 4, 2).unbind(-1)
            rotated = torch.stack((-second, first), dim=-1).flatten(-2)
            torch.testing.assert_close(y, x * cos + rotated * sin)

    def test_request_isolation_and_motion_frame_count(self):
        for cls, task, request in [
            (HYMotionRunner, "t2m", dict(prompt="walk", num_frames=120)),
            (SDXLRunner, "t2i", dict(prompt="apple", size=[1024, 1024])),
            (SF3DRunner, "i23d", dict(image_path="first.png")),
        ]:
            runner = object.__new__(cls)
            BaseRunner.__init__(runner, dict(model_cls={"t2m": "hymotion", "t2i": "sdxl", "i23d": "sf3d"}[task], task=task, enable_cfg=True))
            a = runner.prepare_request(request)
            b = runner.prepare_request({})
            self.assertIsNot(a, b)
            if task == "t2m":
                self.assertEqual(a.num_frames, 120)
            with self.assertRaises(ValueError):
                runner.prepare_request(dict(audio_path="unsupported.wav"))

    def test_hymotion_keeps_native_short_and_long_request_clamping(self):
        from unittest.mock import patch

        from lightx2v.models.networks.hymotion.model import HYMotionModel
        from lightx2v.models.networks.hymotion.weights.motion import MotionFlowMatching

        class ZeroFlow(torch.nn.Module):
            def forward(self, x, **kwargs):
                return torch.zeros_like(x)

        weights = object.__new__(MotionFlowMatching)
        torch.nn.Module.__init__(weights)
        weights.register_parameter("device_marker", torch.nn.Parameter(torch.zeros(1, device="cuda")))
        weights.output_mesh_fps = 30
        weights.train_frames = 360
        weights.uncondition_mode = False
        weights.enable_ctxt_null_feat = True
        weights.random_generator_on_gpu = False
        weights._network_module_args = {"input_dim": 201}
        weights._noise_scheduler_cfg = {"method": "euler"}
        weights.null_vtxt_feat = torch.nn.Parameter(torch.zeros(1, 1, 3, device="cuda"))
        weights.null_ctxt_input = torch.nn.Parameter(torch.zeros(1, 1, 4, device="cuda"))
        weights.motion_transformer = ZeroFlow()
        weights.decode_motion_from_latent = lambda value, **kwargs: {"latent": value}
        model = object.__new__(HYMotionModel)
        model.weights = weights
        model.text_weights = None
        features = (torch.zeros(1, 1, 3, device="cuda"), torch.zeros(1, 4, 4, device="cuda"), torch.tensor([2], device="cuda"))
        with patch("lightx2v.models.networks.hymotion.model.encode_text", return_value=features):
            for requested, actual in [(1, 20), (120, 120), (4999, 360)]:
                result = model.generate("walk", 42, requested, 2, 5.0)
                self.assertEqual(result["latent"].shape[1], actual)

    def test_unsupported_features_fail_before_loading(self):
        for cfg in [dict(cpu_offload=True), dict(parallel={"seq_p_size": 2}), dict(attn_type="flash_attn2"), dict(dit_quant_scheme="fp8"), dict(implementation_status="placeholder")]:
            with self.assertRaises(NotImplementedError):
                validate_a100_config(cfg)

    def test_euler_grid(self):
        times = torch.linspace(0, 1, 51, device="cuda")
        result = odeint(lambda t, x: torch.ones_like(x), torch.zeros(1, device="cuda"), times)
        torch.testing.assert_close(result[-1], torch.ones(1, device="cuda"))
        cfg = dict(beta_schedule="scaled_linear", prediction_type="epsilon", num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, timestep_spacing="leading", steps_offset=1)
        scheduler = SDXLEulerScheduler(cfg, 30)
        self.assertEqual(scheduler.timesteps[0].item(), 958)
        self.assertEqual(scheduler.timesteps[-1].item(), 1)
        self.assertEqual(scheduler.sigmas[-1].item(), 0)
        # Native Euler uses CPU FP32 scalars; CUDA scalar promotion changes
        # FP16 samples even when the cached numbers themselves are identical.
        noise = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16)
        sigma = torch.tensor(float(scheduler.sigmas[0]), dtype=torch.float32, device="cpu")
        expected = noise / (sigma**2 + 1).sqrt()
        self.assertTrue(torch.equal(scheduler.scale_model_input(noise, 0), expected))

    def test_no_upstream_neural_runtime_imports(self):
        import lightx2v.models.networks as networks

        root = Path(networks.__file__).parent
        for name in ("hymotion", "sf3d", "sdxl", "three_models"):
            for path in (root / name).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom) and node.level == 0:
                        module = node.module or ""
                        self.assertFalse(module.startswith(("hymotion", "sf3d", "open_clip", "diffusers")), str(path))
                        if module == "transformers":
                            self.assertTrue(all(alias.name.endswith("Tokenizer") for alias in node.names), str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
