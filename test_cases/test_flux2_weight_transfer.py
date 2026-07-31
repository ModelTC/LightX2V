import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def _load_transfer_module():
    dependency_names = (
        "lightx2v.common.ops.utils",
        "lightx2v_platform.base.global_var",
    )
    previous_modules = {name: sys.modules.get(name) for name in dependency_names}
    generic_move_calls = []

    ops_utils_stub = types.ModuleType(dependency_names[0])

    def move_tensor_to_device(module, attr_name, device, non_blocking=False):
        generic_move_calls.append((module, attr_name, device, non_blocking))
        setattr(module, attr_name, getattr(module, f"pin_{attr_name}", None))

    ops_utils_stub.move_tensor_to_device = move_tensor_to_device
    global_var_stub = types.ModuleType(dependency_names[1])
    global_var_stub.AI_DEVICE = "npu"

    sys.modules[dependency_names[0]] = ops_utils_stub
    sys.modules[dependency_names[1]] = global_var_stub
    try:
        module_path = (
            Path(__file__).resolve().parents[1]
            / "lightx2v"
            / "models"
            / "networks"
            / "flux2"
            / "weights"
            / "transfer.py"
        )
        spec = importlib.util.spec_from_file_location(
            "flux2_weight_transfer_under_test",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._generic_move_calls = generic_move_calls
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


transfer = _load_transfer_module()


class _FakeLeaf:
    def __init__(self, pin_weight, pin_bias=None):
        self.base_attrs = [("weight", "weight", True)]
        if pin_bias is not None:
            self.base_attrs.append(("bias", "bias", False))
        self.lora_attrs = {}
        self.pin_weight = pin_weight
        self.pin_bias = pin_bias
        self.weight = None
        self.bias = None
        self.to_cuda_calls = []

    def to_cuda(self, non_blocking=True):
        self.to_cuda_calls.append(non_blocking)


class Flux2WeightTransferTest(unittest.TestCase):
    def setUp(self):
        transfer._generic_move_calls.clear()

    def test_detects_only_simple_two_dimensional_transpose_views(self):
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.assertTrue(transfer._is_transposed_cpu_view(base.t()))
        self.assertFalse(transfer._is_transposed_cpu_view(base))
        self.assertFalse(transfer._is_transposed_cpu_view(base[:, ::2]))
        self.assertFalse(
            transfer._is_transposed_cpu_view(
                torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
            )
        )

    def test_copy_order_is_contiguous_host_base_then_device_transpose(self):
        events = []
        result = object()

        class DeviceBase:
            def t(self):
                events.append("device_t")
                return result

        class HostBase:
            def to(self, device, non_blocking):
                events.append(("to", device, non_blocking))
                return DeviceBase()

        class HostTranspose:
            def t(self):
                events.append("host_t")
                return HostBase()

        actual = transfer._move_transposed_cpu_tensor_to_device(
            HostTranspose(),
            "npu",
            True,
        )

        self.assertIs(actual, result)
        self.assertEqual(events, ["host_t", ("to", "npu", True), "device_t"])

    def test_fast_path_preserves_cpu_master_and_moves_other_attrs_normally(self):
        pin_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        pin_bias = torch.arange(4, dtype=torch.float32)
        leaf = _FakeLeaf(pin_weight, pin_bias)
        device_weight = object()

        with patch.object(
            transfer,
            "_move_transposed_cpu_tensor_to_device",
            return_value=device_weight,
        ) as fast_move:
            used_fast_path = transfer.move_flux2_leaf_to_cuda(
                leaf,
                non_blocking=True,
            )

        self.assertTrue(used_fast_path)
        self.assertIs(leaf.pin_weight, pin_weight)
        self.assertIs(leaf.weight, device_weight)
        self.assertIs(leaf.bias, pin_bias)
        self.assertEqual(leaf.to_cuda_calls, [])
        fast_move.assert_called_once_with(pin_weight, "npu", True)
        self.assertEqual(
            transfer._generic_move_calls,
            [(leaf, "bias", "npu", True)],
        )

    def test_falls_back_for_contiguous_or_non_npu_sources(self):
        contiguous_leaf = _FakeLeaf(torch.arange(12).reshape(3, 4))
        self.assertFalse(
            transfer.move_flux2_leaf_to_cuda(
                contiguous_leaf,
                non_blocking=False,
            )
        )
        self.assertEqual(contiguous_leaf.to_cuda_calls, [False])

        transposed_leaf = _FakeLeaf(torch.arange(12).reshape(3, 4).t())
        with patch.object(transfer, "AI_DEVICE", "cuda"):
            self.assertFalse(
                transfer.move_flux2_leaf_to_cuda(
                    transposed_leaf,
                    non_blocking=True,
                )
            )
        self.assertEqual(transposed_leaf.to_cuda_calls, [True])


if __name__ == "__main__":
    unittest.main()
