"""QKV storage regressions and device-side split/RMSNorm equivalence."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_helper(relative_path):
    # These helpers intentionally have no dependency on the full pipeline or
    # checkpoint loader, allowing CPU storage tests without model dependencies.
    path = Path(__file__).resolve().parents[2] / "lightx2v/models/networks/minimax_h3" / relative_path
    name = "lightx2v.models.networks.minimax_h3." + relative_path.removesuffix(".py").replace("/", ".")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FusedQKVStorage = _load_helper("weights/fused_qkv.py").FusedQKVStorage
qkv_ops = _load_helper("infer/fused_qkv.py")


@pytest.mark.parametrize("layout", ["default", "int8", "transposed_int8"])
@pytest.mark.parametrize("scale_shape", [(4,), (4, 1), (1, 4)])
def test_projection_equivalence_and_shared_storage(layout, scale_shape):
    torch.manual_seed(0)
    x = torch.randn(7, 8)
    native = [torch.randint(-8, 8, (4, 8), dtype=torch.int8) for _ in range(3)]
    scales = [torch.rand(scale_shape) for _ in range(3)]
    sources = []
    for weight, scale in zip(native, scales):
        if layout == "default":
            module = SimpleNamespace(weight=weight.float().t(), base_attrs=[("w", "weight", True)])
        else:
            module = SimpleNamespace(weight=weight.t() if layout == "transposed_int8" else weight, weight_need_transpose=layout == "transposed_int8", weight_scale=scale)
        sources.append(module)
    target = SimpleNamespace()
    storage = FusedQKVStorage(sources, target)
    storage.refresh()
    effective = target.weight.float() if storage.weight_dim == 1 else target.weight.float().t()
    if layout != "default":
        effective = effective * target.weight_scale.reshape(-1)
    expected = torch.cat([x @ w.float().t() * (s.reshape(-1) if layout != "default" else 1) for w, s in zip(native, scales)], -1)
    torch.testing.assert_close(x @ effective, expected)
    for source in sources:
        assert source.weight.untyped_storage().data_ptr() == target.weight.untyped_storage().data_ptr()
        assert (source.weight if storage.weight_dim == 0 else source.weight.t()).is_contiguous()
    allocation = target.weight.data_ptr()
    storage.refresh()
    assert target.weight.data_ptr() == allocation


def test_offload_reuses_host_storage_and_does_not_join(monkeypatch):
    sources = [SimpleNamespace(pin_weight=torch.randn(4, 8), weight=None, weight_need_transpose=False) for _ in range(3)]
    target = SimpleNamespace()
    storage = FusedQKVStorage(sources, target)
    storage.refresh()
    allocation = target.pin_weight.data_ptr()

    def no_cat(*args, **kwargs):
        raise AssertionError("offload must not concatenate QKV again")

    monkeypatch.setattr(torch, "cat", no_cat)
    storage.move("cpu")
    storage.move("cpu")
    assert target.weight.data_ptr() == allocation
    assert all(source.weight.data_ptr() == source.pin_weight.data_ptr() for source in sources)


def test_block_buffer_updates_are_visible_without_reallocation(monkeypatch):
    sources = [SimpleNamespace(weight_cuda_buffer=torch.randn(8, 4).t(), base_attrs=[("w", "weight", False)]) for _ in range(3)]
    storage = FusedQKVStorage(sources, SimpleNamespace())
    storage.refresh()
    allocation = storage.target.weight_cuda_buffer.data_ptr()
    for index, source in enumerate(sources):
        source.weight = source.weight_cuda_buffer.copy_(torch.full((4, 8), index + 1.0))

    def no_cat(*args, **kwargs):
        raise AssertionError("buffer aliases must be reused")

    monkeypatch.setattr(torch, "cat", no_cat)
    storage.refresh()
    assert storage.target.weight.data_ptr() == allocation
    for index, actual in enumerate(storage.target.weight.chunk(3, dim=0)):
        torch.testing.assert_close(actual, torch.full((4, 8), index + 1.0))
    sources[1].weight = None
    storage.refresh()
    assert storage.target.weight is None  # A partial load cannot serve stale QKV.


def test_adapter_diff_is_not_folded_into_shared_base():
    sources = [SimpleNamespace(weight=torch.randn(8, 4), weight_diff=torch.ones(8, 4), base_attrs=[("w", "weight", True)]) for _ in range(3)]
    base = torch.cat([source.weight.clone() for source in sources], -1)
    storage = FusedQKVStorage(sources, SimpleNamespace())
    storage.refresh()
    torch.testing.assert_close(storage.target.weight, base)
    for source, expected in zip(sources, base.chunk(3, -1)):
        torch.testing.assert_close(source.weight + source.weight_diff, expected + 1)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
@pytest.mark.parametrize("transpose", [False, True])
def test_xpu_offload_roundtrip(transpose, monkeypatch):
    sources = []
    for _ in range(3):
        pinned = torch.randn(4, 8).pin_memory()
        sources.append(SimpleNamespace(pin_weight=pinned.t() if transpose else pinned, pin_weight_scale=torch.rand(4).pin_memory(), weight_need_transpose=transpose))
    storage = FusedQKVStorage(sources, SimpleNamespace())
    storage.refresh()
    expected = storage.target.pin_weight.clone()
    allocation = storage.target.pin_weight.data_ptr()
    assert storage.target.pin_weight.is_pinned()
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("unexpected concatenation during offload"))
    for _ in range(2):
        storage.move("xpu", non_blocking=True)
        storage.target.weight.add_(1)
        storage.move("cpu", non_blocking=True)
        torch.xpu.synchronize()
        expected.add_(1)
        torch.testing.assert_close(storage.target.weight, expected)
        assert storage.target.weight.data_ptr() == allocation
        assert all(source.weight.untyped_storage().data_ptr() == storage.target.weight.untyped_storage().data_ptr() for source in sources)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
@pytest.mark.parametrize("shape", [(0, 1, 128), (17, 1, 128), (9, 28, 128), (5, 7, 96)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_split_qkv_norm_matches_reference(shape, dtype):
    tokens, heads, dim = shape
    packed = torch.randn(tokens, 3 * heads * dim, device="xpu", dtype=dtype)
    qw, kw = (torch.randn(dim, device="xpu", dtype=dtype) for _ in range(2))
    actual = qkv_ops.split_qkv_norm(packed, qw, kw, 1e-5, 1e-4)
    expected = list(packed.chunk(3, -1))
    expected = [x.unflatten(-1, (heads, dim)) for x in expected]
    for index, (weight, eps) in enumerate(((qw, 1e-5), (kw, 1e-4))):
        x = expected[index].float()
        expected[index] = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()).to(dtype)
    for index, (a, e) in enumerate(zip(actual, expected)):
        assert a.is_contiguous()
        torch.testing.assert_close(a, e, atol=0 if index == 2 else 2e-3, rtol=0 if index == 2 else 1e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
@pytest.mark.parametrize("tokens,heads", [(1, 1), (17, 1), (9, 28)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_xpu_qkv_norm_kernel_matches_reference(tokens, heads, dtype):
    import sycl_kernels

    packed = torch.randn(tokens, 3 * heads * 128, device="xpu", dtype=dtype)
    qw = torch.randn(128, device="xpu", dtype=dtype)
    kw = torch.randn(128, device="xpu", dtype=dtype)
    actual = sycl_kernels.minimax_h3_qkv_norm(packed, qw, kw, 1e-5, 1e-4)
    expected = [x.unflatten(-1, (heads, 128)) for x in packed.chunk(3, -1)]
    expected[0] = torch.nn.functional.rms_norm(
        expected[0].float(), (128,), qw.float(), 1e-5
    ).to(dtype)
    expected[1] = torch.nn.functional.rms_norm(
        expected[1].float(), (128,), kw.float(), 1e-4
    ).to(dtype)
    atol = 2e-5 if dtype == torch.float32 else 2e-3
    rtol = 2e-5 if dtype == torch.float32 else 1e-2
    for index, (a, e) in enumerate(zip(actual, expected)):
        assert a.is_contiguous()
        torch.testing.assert_close(
            a, e, atol=0 if index == 2 else atol, rtol=0 if index == 2 else rtol
        )


def test_split_qkv_norm_fake_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        packed = torch.empty(9, 3 * 28 * 128)
        weight = torch.empty(128)
        outputs = qkv_ops.split_qkv_norm(packed, weight, weight, 1e-5, 1e-5)
        assert all(output.shape == (9, 28, 128) and output.is_contiguous() for output in outputs)
