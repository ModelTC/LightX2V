"""Weight copy overrides stay in the selected platform, including new platforms."""

from types import SimpleNamespace

import pytest
import torch

from lightx2v.common.ops import utils
from lightx2v_platform.base import global_var
from lightx2v_platform.base.ascend_npu import NpuDevice
from lightx2v_platform.base.global_var import AI_DEVICE, PLATFORM
from lightx2v_platform.ops.mm.ascend_npu.mm_weight import MMWeightWint8channelAint8channeldynamicNpu
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

pytestmark = pytest.mark.skipif(not PLATFORM_DEVICE_REGISTER[PLATFORM].is_available(), reason="requires an accelerator")


@pytest.mark.parametrize("custom_platform", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_weight_roundtrips_preserve_host_storage(monkeypatch, custom_platform, transpose):
    calls = []
    if custom_platform:
        # A new platform can supply the same hooks without editing common utilities.
        def to_host(destination, source, non_blocking=False):
            calls.append("d2h")
            return NpuDevice.copy_to_cpu(destination, source, non_blocking)

        def to_device(tensor, device, non_blocking=False):
            calls.append("h2d")
            return NpuDevice.copy_transposed_weight_to_device(tensor, device, non_blocking)

        monkeypatch.setitem(PLATFORM_DEVICE_REGISTER, "copy_test", SimpleNamespace(copy_to_cpu=to_host, copy_transposed_weight_to_device=to_device))
        monkeypatch.setattr(global_var, "PLATFORM", "copy_test")

    parent = torch.full((16 + 32 * 64 + 16,), -17.0, pin_memory=True)
    pin_weight = parent[16:-16].reshape(32, 64)
    expected = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    if transpose:
        pin_weight = pin_weight.t()
        expected = expected.t()
    pin_weight.copy_(expected)
    owner = SimpleNamespace(base_attrs=[("w", "weight", transpose)], pin_weight=pin_weight)

    def default_move(non_blocking=False):
        utils.move_tensor_to_device(owner, "weight", AI_DEVICE, non_blocking=non_blocking)

    owner.to_cuda = default_move
    pointer = pin_weight.data_ptr()
    for _ in range(3):
        used_hook = utils.move_transposed_weight_module_to_device(owner, non_blocking=True)
        assert used_hook == (transpose and (custom_platform or PLATFORM == "ascend_npu"))
        torch.testing.assert_close(owner.weight.cpu(), expected, rtol=0, atol=0)
        owner.weight.add_(1)
        utils.move_tensor_to_device(owner, "weight", "cpu", use_copy=True)
        expected = expected + 1
        torch.testing.assert_close(owner.weight, expected, rtol=0, atol=0)
        assert owner.weight.data_ptr() == pointer
        assert owner.weight.stride() == pin_weight.stride()
        assert owner.pin_weight.is_pinned()
        assert torch.all(parent[:16] == -17) and torch.all(parent[-16:] == -17)
    if custom_platform:
        assert calls == (["h2d", "d2h"] if transpose else ["d2h"]) * 3


@pytest.mark.parametrize("pin_attribute", [False, True])
def test_unpinned_and_optional_weights_survive_roundtrip(pin_attribute):
    expected = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
    owner = SimpleNamespace(weight=expected.clone(), bias=None)
    if pin_attribute:
        owner.pin_weight = None

    utils.move_tensor_to_device(owner, "weight", AI_DEVICE, non_blocking=True)
    torch.testing.assert_close(owner.weight.cpu(), expected, rtol=0, atol=0)
    owner.weight.add_(1)
    utils.move_tensor_to_device(owner, "weight", "cpu", use_copy=True)
    torch.testing.assert_close(owner.weight, expected + 1, rtol=0, atol=0)
    assert owner.weight.device.type == "cpu"
    assert not owner.weight.is_pinned()

    utils.move_tensor_to_device(owner, "bias", AI_DEVICE)
    utils.move_tensor_to_device(owner, "absent", AI_DEVICE)
    assert owner.bias is None
    assert not hasattr(owner, "absent")


def test_quantized_template_uses_platform_copy_for_every_state(monkeypatch):
    calls = []

    def to_host(destination, source, non_blocking=False):
        calls.append(destination.data_ptr())
        return NpuDevice.copy_to_cpu(destination, source, non_blocking)

    monkeypatch.setitem(PLATFORM_DEVICE_REGISTER, "copy_test", SimpleNamespace(copy_to_cpu=to_host))
    monkeypatch.setattr(global_var, "PLATFORM", "copy_test")
    source = {
        "proj.weight": (torch.arange(32 * 64).reshape(32, 64) % 7).to(torch.int8),
        "proj.weight_scale": torch.full((32, 1), 0.125, dtype=torch.float32),
        "proj.bias": torch.zeros(32, dtype=torch.bfloat16),
    }
    owner = MMWeightWint8channelAint8channeldynamicNpu("proj.weight", "proj.bias")
    owner.load(dict(source))
    expected = {name: tensor.clone() for name, tensor in owner.state_dict().items()}
    pointers = {tensor.data_ptr() for tensor in owner.state_dict().values()}
    for _ in range(3):
        owner.to_cuda()
        owner.to_cpu()
        for name, tensor in owner.state_dict().items():
            assert tensor.is_pinned() and tensor.data_ptr() in pointers
            assert tensor.stride() == expected[name].stride()
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    assert len(calls) == 9 and set(calls) == pointers
