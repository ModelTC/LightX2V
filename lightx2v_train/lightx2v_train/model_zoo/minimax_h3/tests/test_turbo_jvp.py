"""Exact H3 consistency JVP, including independent AV times and FSDP hooks."""

import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.attention import SDPBackend, sdpa_kernel

from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents, MiniMaxH3LatentShape
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.turbo_jvp import _math_attention, differentiable_row_timesteps
from lightx2v_train.model_zoo.minimax_h3.tests.test_ref2av_capability import model_wrapper
from lightx2v_train.model_zoo.native.minimax_h3.modeling import _transformer_class
from lightx2v_train.model_zoo.native.minimax_h3.packing import build_packed_sequence, build_row_timesteps


def tiny_setup(device="cpu", dtype=torch.float32):
    model = model_wrapper(False)
    model.device = torch.device(device)
    model.running_dtype = dtype
    model.patch_size = (1, 1, 1)
    model.video_latent_channels = model.audio_latent_channels = 2
    model.vae_spatial_scale_factor = 1
    model.transformer = _transformer_class()(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=12,
        freq_dim=4,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
    ).to(device=device, dtype=dtype)
    for name in ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out", "time_embedder"):
        getattr(model.transformer, name).float()
    model.transformer.enable_gradient_checkpointing()
    capability = model.ensure_capabilities().require(DistributionMatchingCapability)
    shape = MiniMaxH3LatentShape(22, 2, 2, 2, 3, (1, 8, 2), (1, 6, 2))
    condition = {"prompt_embeds": torch.randn(1, 4, 12, device=device, dtype=dtype), "text_token_tags": torch.ones(4, device=device, dtype=torch.long)}
    state = MiniMaxH3JointLatents(torch.randn(shape.video_tokens, device=device), torch.randn(shape.audio_tokens, device=device), shape)
    tangent = MiniMaxH3JointLatents(torch.randn_like(state.video), torch.randn_like(state.audio), shape)
    return model, capability, state, tangent, condition


def fsdp_jvp_worker(rank, init_file, cuda=False):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=2, init_method="file://" + init_file, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(35)
        model, capability, state, tangent, condition = tiny_setup(device, torch.bfloat16 if cuda else torch.float32)
        model.add_lora(
            4,
            4,
            [*capability.default_lora_target_modules, "adaln_proj.linear", "norm_out.linear"],
            rank_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2},
            alpha_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2},
        )
        model.set_lora_param_dtype("fp32")
        model.set_lora_trainable()
        mesh = init_device_mesh(device.type, (2,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if cuda else torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False)
        for block in [*model.transformer.token_refiner.refiner_blocks, *model.transformer.transformer_blocks]:
            fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
        fully_shard(model.transformer, mesh=mesh, reshard_after_forward=False, mp_policy=policy)
        sigma = torch.tensor([0.6, 0.3], device=device)
        primal, derivative = capability.predict_velocity_jvp(state, sigma, condition, tangent, torch.tensor([0.1, -0.2], device=device))
        assert torch.isfinite(derivative.video).all() and torch.isfinite(derivative.audio).all()
        predicted = capability.predict_velocity_at_sigmas(state, sigma, condition)
        tolerance = {"atol": 0.02, "rtol": 0.03} if cuda else {}
        torch.testing.assert_close(primal.video, predicted.video, **tolerance)
        torch.testing.assert_close(primal.audio, predicted.audio, **tolerance)
        (predicted.video.square().mean() + predicted.audio.square().mean()).backward()
        assert any(parameter.grad is not None for parameter in model.transformer.parameters())
    finally:
        dist.destroy_process_group()


class TurboJVPTests(unittest.TestCase):
    def test_fixed_time_slots_match_legacy_and_preserve_equal_sigma_directions(self):
        layout = build_packed_sequence(torch.ones(4, dtype=torch.long), 2, 2, 2, 3, (1, 1, 1))
        sigma = torch.tensor([0.4, 0.4], requires_grad=True)
        times, indices = differentiable_row_timesteps(layout, *sigma.unbind())
        old_times, old_indices = build_row_timesteps(layout, *sigma.unbind())
        torch.testing.assert_close(times[indices], old_times[old_indices], rtol=0, atol=0)
        video = times[indices][layout.video_indices].sum()
        audio = times[indices][layout.audio_indices].sum()
        torch.testing.assert_close(torch.autograd.grad(video - audio, sigma)[0], torch.tensor([-8.0, 6.0]))

    def test_native_jvp_satisfies_reverse_mode_identity_and_keeps_time_derivative(self):
        torch.manual_seed(31)
        model, capability, state, tangent, condition = tiny_setup()
        sigma, sigma_tangent = torch.tensor([0.6, 0.3]), torch.tensor([0.1, -0.2])
        processors = [block.attn.processor for block in model.transformer.transformer_blocks]
        primal, derivative = capability.predict_velocity_jvp(state, sigma, condition, tangent, sigma_tangent)
        self.assertEqual(processors, [block.attn.processor for block in model.transformer.transformer_blocks])
        self.assertFalse(primal.video.requires_grad or derivative.video.requires_grad)
        self.assertTrue(model.transformer.gradient_checkpointing)
        self.assertTrue(all(parameter.grad is None for parameter in model.transformer.parameters()))
        xvideo, xaudio, times = state.video.detach().requires_grad_(), state.audio.detach().requires_grad_(), sigma.requires_grad_()
        with sdpa_kernel(SDPBackend.MATH):
            output = capability.predict_velocity_at_sigmas(MiniMaxH3JointLatents(xvideo, xaudio, state.shape), times, condition)
            video_cotangent, audio_cotangent = torch.randn_like(output.video), torch.randn_like(output.audio)
            grads = torch.autograd.grad((output.video * video_cotangent).sum() + (output.audio * audio_cotangent).sum(), (xvideo, xaudio, times))
        lhs = (derivative.video * video_cotangent).sum() + (derivative.audio * audio_cotangent).sum()
        rhs = (grads[0] * tangent.video).sum() + (grads[1] * tangent.audio).sum() + (grads[2] * sigma_tangent).sum()
        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-4)
        self.assertGreater(float(grads[2].abs().sum()), 0)
        torch.testing.assert_close(primal.video, output.video, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(primal.audio, output.audio, rtol=1e-5, atol=1e-5)

    def test_two_rank_fsdp_jvp_then_checkpointed_parameter_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(fsdp_jvp_worker, args=(os.path.join(directory, "init"),), nprocs=2, join=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-only exact Triton JVP")
    def test_cuda_streaming_kernel_matches_exact_autograd_reference(self):
        from lightx2v_train.model_zoo.minimax_h3.capability_adapters.turbo_flash_attention_jvp import attention_with_jvp

        torch.manual_seed(45)
        values = tuple(torch.randn(1, 117, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.5 for _ in range(6))
        primal, tangent = attention_with_jvp(*values)
        reference, reference_tangent = torch.func.jvp(lambda q, k, v: _math_attention(q, k, v, None), values[:3], values[3:])
        torch.testing.assert_close(primal, reference, rtol=0.02, atol=0.01)
        torch.testing.assert_close(tangent, reference_tangent, rtol=0.03, atol=0.02)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-only exact native H3 JVP")
    def test_cuda_h3_mixed_rank_jvp_then_checkpointed_parameter_backward(self):
        torch.manual_seed(41)
        model, capability, state, tangent, condition = tiny_setup("cuda", torch.bfloat16)
        model.add_lora(
            4,
            4,
            [*capability.default_lora_target_modules, "adaln_proj.linear", "norm_out.linear"],
            rank_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2},
            alpha_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2},
        )
        model.set_lora_param_dtype("fp32")
        model.set_lora_trainable()
        sigma, sigma_tangent = torch.tensor([0.6, 0.3], device="cuda"), torch.tensor([0.1, -0.2], device="cuda")
        primal, derivative = capability.predict_velocity_jvp(state, sigma, condition, tangent, sigma_tangent)
        self.assertTrue(torch.isfinite(derivative.video).all() and torch.isfinite(derivative.audio).all())
        with sdpa_kernel(SDPBackend.MATH):
            prediction = capability.predict_velocity_at_sigmas(state, sigma, condition)
            torch.testing.assert_close(primal.video, prediction.video, rtol=0.03, atol=0.02)
            torch.testing.assert_close(primal.audio, prediction.audio, rtol=0.03, atol=0.02)
            (prediction.video.square().mean() + prediction.audio.square().mean()).backward()
        trainable = [parameter for parameter in model.transformer.parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two-GPU NCCL FSDP2 exact JVP")
    def test_cuda_two_rank_fsdp_mixed_rank_jvp_then_parameter_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(fsdp_jvp_worker, args=(os.path.join(directory, "init"), True), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
