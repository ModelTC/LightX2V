"""Encoder graph replay preserves pixels, features, and tensor ownership."""

import os
from pathlib import Path

import pytest
import torch

from lightx2v.models.input_encoders.edenlive.proxy_cuda_graph import ProxyCUDAGraph
from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO


def test_cpu_fallback_and_disabled_encoder_dispatch():
    calls = []

    def forward(value):
        calls.append(value)
        return value.square().add(3)

    wrapper = ProxyCUDAGraph(forward)
    value = torch.arange(9).reshape(3, 3)
    assert torch.equal(wrapper(value), forward(value))
    assert calls[0] is value and not wrapper._entries
    encoder = ProxyDINO.__new__(ProxyDINO)
    encoder._cuda_graph = None
    encoder._infer_eager = forward
    assert torch.equal(encoder.infer(value), forward(value))


@pytest.mark.parametrize("kwargs", [{"warmup_iters": 0}, {"max_graphs": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError, match="must be positive"):
        ProxyCUDAGraph(lambda value: value, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replay_changes_inputs_and_owns_output_with_noncontiguous_sources():
    calls = []

    def forward(value):
        calls.append(True)
        return value.square().add(3)

    wrapper = ProxyCUDAGraph(forward)
    outputs = []
    originals = []
    for index in range(5):
        value = torch.arange(192, device="cuda", dtype=torch.float32).reshape(8, 24)[:, ::2].add_(index)
        assert not value.is_contiguous()
        before = value.clone()
        output = wrapper(value)
        assert torch.equal(output, value.square().add(3))
        assert torch.equal(value, before)
        outputs.append(output)
        originals.append(output.clone())
    assert len(calls) == 3  # Two warmups, one capture; then no Python forward.
    assert all(torch.equal(output, original) for output, original in zip(outputs, originals))
    assert len({output.data_ptr() for output in outputs}) == len(outputs)
    assert next(iter(wrapper._entries.values())).graph is not None
    wrapper.clear()
    assert not wrapper._entries
    assert all(torch.equal(output, original) for output, original in zip(outputs, originals))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shape_eviction_and_cross_stream_handoff():
    wrapper = ProxyCUDAGraph(lambda value: value.mul(2).add(1), warmup_iters=1, max_graphs=1)
    producer = torch.cuda.Stream()
    retained = []
    for size in (32, 64, 32):
        with torch.cuda.stream(producer):
            value = torch.empty(size, device="cuda")
            value.fill_(size)
            retained.append(wrapper(value))
            output = wrapper(value)
        torch.cuda.current_stream().wait_stream(producer)
        assert torch.equal(output, torch.full_like(output, size * 2 + 1))
        assert len(wrapper._entries) == 1
        assert next(iter(wrapper._entries.values())).graph is not None
    for result, size in zip(retained, (32, 64, 32)):
        assert torch.equal(result, torch.full_like(result, size * 2 + 1))
    wrapper.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_proxy_dino_eager_and_graph_features_are_bit_exact():
    import lightx2v.common.ops  # noqa: F401

    root = Path(os.environ.get("EDENLIVE_PROXY_TEST_MODEL", Path(__file__).resolve().parents[2] / "EdenLive-x2v"))
    if not (root / "vision-encoder/projection.safetensors").is_file():
        pytest.skip("Set EDENLIVE_PROXY_TEST_MODEL to an EdenLive model directory")
    encoder = ProxyDINO(str(root), "bilinear", use_cuda_graph=True)
    generator = torch.Generator(device="cuda").manual_seed(20261016)
    retained = []
    for _ in range(4):
        pixels = torch.randint(0, 256, (3, 3, 704, 1280), device="cuda", dtype=torch.uint8, generator=generator)
        expected = encoder._infer_eager(pixels)
        actual = encoder.infer(pixels)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        retained.append((actual, actual.clone()))
    assert next(iter(encoder._cuda_graph._entries.values())).graph is not None
    assert all(torch.equal(actual.view(torch.int32), saved.view(torch.int32)) for actual, saved in retained)
    encoder._cuda_graph.clear()
