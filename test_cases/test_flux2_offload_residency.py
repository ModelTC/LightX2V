import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class _WeightModule:
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def add_module(self, name, module):
        self._modules[name] = module
        setattr(self, name, module)


class _WeightModuleList(_WeightModule):
    def __init__(self, modules=None):
        super().__init__()
        self._list = []
        for module in modules or ():
            self.append(module)

    def append(self, module):
        self._list.append(module)
        self.add_module(str(len(self._list) - 1), module)

    def __getitem__(self, index):
        return self._list[index]

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)


def _load_transformer_weights_module():
    """Load the target file without initializing a CUDA/NPU platform."""
    dependency_names = (
        "lightx2v.common.modules.weight_module",
        "lightx2v.common.offload.block_slab",
        "lightx2v.utils.registry_factory",
    )
    previous_modules = {name: sys.modules.get(name) for name in dependency_names}

    weight_module_stub = types.ModuleType(dependency_names[0])
    weight_module_stub.WeightModule = _WeightModule
    weight_module_stub.WeightModuleList = _WeightModuleList
    block_slab_stub = types.ModuleType(dependency_names[1])
    block_slab_stub.pack_cpu_block_slab = lambda *args, **kwargs: object()
    registry_stub = types.ModuleType(dependency_names[2])
    registry_stub.ATTN_WEIGHT_REGISTER = {}
    registry_stub.LN_WEIGHT_REGISTER = {}
    registry_stub.MM_WEIGHT_REGISTER = {}
    registry_stub.RMS_WEIGHT_REGISTER = {}
    registry_stub.ROPE_REGISTER = {}

    sys.modules[dependency_names[0]] = weight_module_stub
    sys.modules[dependency_names[1]] = block_slab_stub
    sys.modules[dependency_names[2]] = registry_stub
    try:
        module_path = Path(__file__).resolve().parents[1] / "lightx2v" / "models" / "networks" / "flux2" / "weights" / "transformer_weights.py"
        spec = importlib.util.spec_from_file_location("flux2_transformer_weights_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


transformer_weights = _load_transformer_weights_module()


class _FakeBlock:
    def __init__(self, config, block_idx, create_cuda_buffer=False, create_cpu_buffer=False):
        self.block_idx = block_idx
        self.create_cuda_buffer = create_cuda_buffer


class Flux2OffloadResidencyTest(unittest.TestCase):
    def test_block_slab_capability_gate_is_strict(self):
        config = {
            "offload_use_block_slab": True,
            "cpu_offload": True,
            "offload_granularity": "block",
            "offload_use_npu_events": True,
            "dit_quant_scheme": "Default",
        }
        self.assertTrue(transformer_weights.validate_flux2_block_slab_config(config, "npu"))
        with self.assertRaisesRegex(ValueError, "LoRA disabled"):
            transformer_weights.validate_flux2_block_slab_config(
                {**config, "lora_configs": [{"path": "adapter.safetensors"}]},
                "npu",
            )
        with self.assertRaisesRegex(ValueError, "offload_use_npu_events"):
            transformer_weights.validate_flux2_block_slab_config(
                {**config, "offload_use_npu_events": False},
                "npu",
            )

    def test_unconfigured_residency_preserves_full_streaming(self):
        resolve = transformer_weights._resolve_resident_block_indices
        self.assertEqual(resolve(None, 8, "prefix", "double"), frozenset())
        self.assertEqual(resolve(0, 48, "interleaved", "single"), frozenset())

    def test_all_and_interleaved_residency(self):
        resolve = transformer_weights._resolve_resident_block_indices
        self.assertEqual(resolve("all", 8, "interleaved", "double"), frozenset(range(8)))

        resident = resolve(36, 48, "interleaved", "single")
        self.assertEqual(len(resident), 36)
        self.assertEqual(sorted(set(range(48)) - resident), list(range(3, 48, 4)))

    def test_invalid_resident_configuration_is_rejected(self):
        resolve = transformer_weights._resolve_resident_block_indices
        for value in (-1, 9, True, "half"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve(value, 8, "interleaved", "double")
        with self.assertRaises(ValueError):
            resolve(1, 8, "unknown", "double")

    def test_residency_rejects_unsupported_weight_lifecycles(self):
        weights = self._make_weights_shell((), ())
        base_config = {
            "cpu_offload": True,
            "offload_granularity": "block",
            "offload_resident_single_blocks": 1,
        }

        with self.assertRaisesRegex(NotImplementedError, "unquantized"):
            weights._configure_resident_blocks({**base_config, "dit_quantized": True})
        with self.assertRaisesRegex(NotImplementedError, "LoRA"):
            weights._configure_resident_blocks({**base_config, "lora_configs": [{"path": "adapter.safetensors"}]})

    def _make_weights_shell(self, resident_double, resident_single):
        weights = object.__new__(transformer_weights.Flux2TransformerWeights)
        _WeightModule.__init__(weights)
        weights.num_layers = 8
        weights.num_single_layers = 48
        weights.resident_double_block_indices = frozenset(resident_double)
        weights.resident_single_block_indices = frozenset(resident_single)
        return weights

    def test_all_resident_category_does_not_allocate_staging_buffers(self):
        weights = self._make_weights_shell(range(8), range(48))
        config = {"cpu_offload": True, "offload_granularity": "block"}
        with (
            patch.object(transformer_weights, "Flux2DoubleBlockWeights", _FakeBlock),
            patch.object(transformer_weights, "Flux2SingleBlockWeights", _FakeBlock),
        ):
            weights.register_offload_buffers(config)

        self.assertFalse(hasattr(weights, "offload_double_block_cuda_buffers"))
        self.assertFalse(hasattr(weights, "offload_single_block_cuda_buffers"))

    def test_partial_residency_keeps_double_staging_buffers(self):
        weights = self._make_weights_shell(range(7), range(48))
        config = {"cpu_offload": True, "offload_granularity": "block"}
        with (
            patch.object(transformer_weights, "Flux2DoubleBlockWeights", _FakeBlock),
            patch.object(transformer_weights, "Flux2SingleBlockWeights", _FakeBlock),
        ):
            weights.register_offload_buffers(config)

        self.assertEqual(len(weights.offload_double_block_cuda_buffers), 2)
        self.assertFalse(hasattr(weights, "offload_single_block_cuda_buffers"))

    def test_block_slabs_pack_only_nonresident_blocks(self):
        blocks = []
        for block_idx in range(4):
            weight = torch.full((2,), block_idx, dtype=torch.bfloat16)
            state = {f"block.{block_idx}.weight": weight}
            leaf = types.SimpleNamespace(
                _modules={},
                _parameters={},
                base_attrs=((f"block.{block_idx}.weight", "weight", False),),
                pin_weight=weight,
                weight=None,
            )
            blocks.append(
                types.SimpleNamespace(
                    _modules={"leaf": leaf},
                    _parameters={},
                    state_dict=lambda state=state: state,
                )
            )

        packed = []

        def fake_pack(state_dict, **kwargs):
            packed.append((next(iter(state_dict)), kwargs))
            return object()

        with patch.object(transformer_weights, "pack_cpu_block_slab", side_effect=fake_pack):
            slabs = transformer_weights.Flux2TransformerWeights._pack_offload_block_slabs(
                blocks,
                frozenset({1, 3}),
            )

        self.assertEqual(set(slabs), {0, 2})
        self.assertEqual([name for name, _ in packed], ["block.0.weight", "block.2.weight"])
        self.assertTrue(all(options == {"pin_memory": True, "strict_pin": True} for _, options in packed))

    def test_block_slab_ignores_non_cpu_derived_state_entries(self):
        weight = torch.ones(2, dtype=torch.bfloat16)
        derived = torch.zeros((), dtype=torch.bfloat16, device="meta")
        leaf = types.SimpleNamespace(
            _modules={},
            _parameters={},
            base_attrs=(("block.0.weight", "weight", False),),
            pin_weight=weight,
            weight=None,
            weight_diff=derived,
        )
        block = types.SimpleNamespace(
            _modules={"leaf": leaf},
            _parameters={},
            state_dict=lambda: {
                "block.0.weight": weight,
                "block.0.diff": derived,
            },
        )

        captured = []
        with patch.object(
            transformer_weights,
            "pack_cpu_block_slab",
            side_effect=lambda state_dict, **kwargs: captured.append(state_dict) or object(),
        ):
            transformer_weights.Flux2TransformerWeights._pack_offload_block_slabs(
                [block],
                frozenset(),
            )

        self.assertEqual(list(captured[0]), ["block.0.weight"])

    def test_block_slab_rejects_missing_cpu_base_tensor(self):
        weight = torch.ones(2, dtype=torch.bfloat16)
        leaf = types.SimpleNamespace(
            _modules={},
            _parameters={},
            base_attrs=(("block.0.weight", "weight", False),),
            pin_weight=weight,
            weight=None,
        )
        block = types.SimpleNamespace(
            _modules={"leaf": leaf},
            _parameters={},
            state_dict=lambda: {},
        )

        with self.assertRaisesRegex(ValueError, "missing CPU base attribute"):
            transformer_weights.Flux2TransformerWeights._pack_offload_block_slabs(
                [block],
                frozenset(),
            )

    def test_release_uses_pinned_cpu_master_without_copy_back(self):
        pin_weight = torch.ones(2, 2)
        cpu_lora = torch.full((1,), 3.0)
        leaf = types.SimpleNamespace(
            _modules={},
            base_attrs=(("weight.name", "weight", True),),
            lora_attrs={"lora_down": "lora_down_name"},
            weight=torch.zeros(2, 2),
            pin_weight=pin_weight,
            lora_down=cpu_lora,
        )
        root = types.SimpleNamespace(_modules={"leaf": leaf})

        transformer_weights.preserve_weight_module_cpu_tensors(root)
        leaf.lora_down = torch.zeros(1)  # stand-in for the device-side copy
        transformer_weights.release_weight_module_device_tensors(root)

        self.assertIsNone(leaf.weight)
        self.assertIs(leaf.pin_weight, pin_weight)
        self.assertIs(leaf.lora_down, cpu_lora)


if __name__ == "__main__":
    unittest.main()
