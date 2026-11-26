"""FA3 forward/backward against SDPA at H3's head count and head dimension."""

from pathlib import Path

import torch
from diffusers.models.attention_dispatch import _AttentionBackendRegistry, attention_backend, dispatch_attention_fn
from omegaconf import OmegaConf

from .encoding import OnlineAVEncoder, StageTimer
from .runtime import load_attention_kernel


def test_fa3_backward():
    recipe = Path(__file__).parents[3] / "configs/train/dense2moe/minimax_h3_t2av_fsdp4.yaml"
    config = OmegaConf.to_container(OmegaConf.load(recipe), resolve=True)
    load_attention_kernel(config)
    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 257, 56, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    actual = dispatch_attention_fn(q, k, v, backend=config["attention_backend"])
    expected = dispatch_attention_fn(q, k, v, backend="native")
    actual_grad = torch.autograd.grad(actual.float().square().sum(), (q, k, v), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.float().square().sum(), (q, k, v))
    for a, b in zip((actual, *actual_grad), (expected, *expected_grad)):
        relative_error = (a - b).float().norm() / b.float().norm()
        assert relative_error < 0.01
        torch.testing.assert_close(a, b, atol=0.006, rtol=0.07)


def test_online_vae_keeps_fp32_attention_separate_from_dit_fa3():
    recipe = Path(__file__).parents[3] / "configs/train/dense2moe/minimax_h3_t2av_fsdp4.yaml"
    config = OmegaConf.to_container(OmegaConf.load(recipe), resolve=True)
    load_attention_kernel(config)

    class EncoderModel:
        @torch.inference_mode()
        def _encode_video_latents(self, pixels):
            return torch.zeros(1, 24, 1, 2, 2, device="cuda")

        @torch.inference_mode()
        def _encode_audio_latents(self, audio):
            query = torch.randn(1, 7, 2, 32, device="cuda", dtype=torch.float32)
            return dispatch_attention_fn(query, query, query).reshape(1, 14, 32)

    encoder = OnlineAVEncoder.__new__(OnlineAVEncoder)
    encoder.model = EncoderModel()
    with attention_backend(config["attention_backend"]):
        backend = _AttentionBackendRegistry._active_backend
        outputs = encoder.encode(torch.zeros(1), torch.zeros(2, 800), StageTimer("cuda"))
        assert _AttentionBackendRegistry._active_backend == backend
    assert all(value.dtype == torch.float32 and not value.is_inference() for value in outputs.values())
