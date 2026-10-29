"""Wan Turbo native-tensor boundary and exact FP32 latent/time JVP."""

import unittest
from unittest.mock import Mock

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from lightx2v_train.model_zoo.native.wan.modules.model import WanModel
from lightx2v_train.model_zoo.wan.capability_adapters.wan_distribution_matching_capability import WanDistributionMatchingCapability
from lightx2v_train.model_zoo.wan.capability_adapters.wan_turbo_jvp import fp32_attention_jvp
from lightx2v_train.model_zoo.wan.wan_t2v import WanT2VModel


def fixture(device="cpu"):
    model = WanT2VModel({"model": {"name": "wan_t2v", "running_dtype": "fp32"}, "training": {"method": "turbo"}})
    model.device = torch.device(device)
    model.running_dtype = torch.float32
    model.use_causal_transformer = False
    model.sp_size = 1
    model.patch_size = (1, 1, 1)
    model.num_train_timesteps = 1000
    model.max_sequence_length = 4
    model.vae_scale_factor_temporal = 4
    model.vae_scale_factor_spatial = 8
    model.vae_dtype = torch.float32
    model.transformer = (
        WanModel(patch_size=(1, 1, 1), text_len=4, in_dim=4, dim=32, ffn_dim=64, freq_dim=8, text_dim=12, out_dim=4, num_heads=2, num_layers=1, attention_backend="sdpa").to(device).eval()
    )
    # Pretrained heads are nonzero; avoid a vacuous zero-output JVP test.
    torch.nn.init.normal_(model.transformer.head.head.weight, std=0.05)
    state = torch.randn(1, 4, 2, 2, 2, device=device)
    condition = {"prompt_embed": torch.randn(1, 4, 12, device=device)}
    return model, WanDistributionMatchingCapability(model), state, condition


def check_jvp_identity(test, device):
    torch.manual_seed(215)
    model, capability, state, condition = fixture(device)
    sigma = torch.tensor(0.45, device=device)
    tangent, time_tangent = torch.randn_like(state), torch.tensor(0.1, device=device)
    primal, derivative = capability.predict_velocity_jvp(state, sigma, condition, tangent, time_tangent)
    test.assertEqual(primal.dtype, torch.float32)
    test.assertEqual(derivative.dtype, torch.float32)
    test.assertFalse(primal.requires_grad or derivative.requires_grad)
    x, t = state.detach().requires_grad_(), sigma.detach().requires_grad_()
    with sdpa_kernel(SDPBackend.MATH if str(device) == "cpu" else SDPBackend.EFFICIENT_ATTENTION):
        reference = capability.predict_velocity_at_sigmas(x, t, condition)
        cotangent = torch.randn_like(reference)
        gradients = torch.autograd.grad((reference * cotangent).sum(), (x, t))
    lhs = (derivative * cotangent).sum()
    rhs = (gradients[0] * tangent).sum() + gradients[1] * time_tangent
    torch.testing.assert_close(primal, reference, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(lhs, rhs, rtol=3e-4, atol=3e-4)
    test.assertGreater(float(derivative.abs().sum()), 0)
    test.assertGreater(float(gradients[1].abs()), 0)
    test.assertTrue(all(parameter.grad is None for parameter in model.transformer.parameters()))


class WanTurboTests(unittest.TestCase):
    def test_velocity_sign_and_direct_physical_sigma(self):
        model, capability, state, condition = fixture()
        sigma = torch.tensor(0.6)
        native = capability.predict_velocity(state, sigma, condition)
        cleanward = capability.predict_velocity_at_sigmas(state, sigma, condition)
        torch.testing.assert_close(cleanward, -native, rtol=0, atol=0)

    def test_cpu_exact_native_wan_joint_latent_time_jvp(self):
        check_jvp_identity(self, "cpu")

    def test_clean_target_uses_cached_wan_latents_or_online_encoder_without_writes(self):
        model, capability, state, _ = fixture()
        result = capability.extract_turbo_latents({"inputs": {"latents": state}}, state.shape)
        torch.testing.assert_close(result, state)
        model.vae = Mock()
        model.vae.encode.return_value = [state[0]]
        pixels = torch.randn(1, 3, 5, 16, 16)
        encoded = capability.extract_turbo_latents({"inputs": {"video": pixels}}, state.shape)
        torch.testing.assert_close(encoded, state)
        model.vae.encode.assert_called_once()
        with self.assertRaisesRegex(ValueError, "shape"):
            capability.extract_turbo_latents({"inputs": {"latents": state}}, (1, 4, 1, 2, 2))

    def test_turbo_lora_kaiming_initialization_and_fp32_backward(self):
        model, capability, state, condition = fixture()
        model.add_lora(4, 4, ["q", "k", "v", "o", "ffn.0", "ffn.2"])
        self.assertIs(model.transformer.peft_config["default"].init_lora_weights, True)
        model.set_lora_trainable()
        model.transformer.enable_gradient_checkpointing()
        output = capability.predict_velocity_at_sigmas(state, torch.tensor(0.4), condition)
        output.square().mean().backward()
        parameters = [parameter for parameter in model.transformer.parameters() if parameter.requires_grad]
        self.assertTrue(parameters)
        self.assertTrue(all(parameter.dtype == torch.float32 and parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters))

    @unittest.skipUnless(torch.cuda.is_available(), "FP32 CUDA exact JVP")
    def test_cuda_fp32_cross_attention_kernel_matches_exact_autograd(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(216)
        values = tuple(torch.randn(1, length, 2, 128, device="cuda") * 0.4 for length in (17, 29, 29, 17, 29, 29))
        output, tangent = fp32_attention_jvp(*values)

        def attention(q, k, v):
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
            return ((q @ k.transpose(-1, -2) * (128**-0.5)).softmax(-1) @ v).transpose(1, 2)

        reference, derivative = torch.func.jvp(attention, values[:3], values[3:])
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, reference, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(tangent, derivative, rtol=3e-4, atol=3e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "FP32 CUDA native Wan JVP")
    def test_cuda_fp32_native_wan_joint_latent_time_jvp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        check_jvp_identity(self, "cuda")


if __name__ == "__main__":
    unittest.main()
