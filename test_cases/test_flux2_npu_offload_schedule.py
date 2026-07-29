import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def _package(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


def _load_transformer_infer_module():
    """Load the target file without importing the LightX2V or torch packages."""
    torch_module = _package("torch")
    torch_nn_module = _package("torch.nn")
    torch_functional_module = ModuleType("torch.nn.functional")
    torch_functional_module.silu = lambda value: value
    torch_npu_module = ModuleType("torch.npu")
    torch_module.nn = torch_nn_module
    torch_module.npu = torch_npu_module
    torch_nn_module.functional = torch_functional_module

    class _StubWeightAsyncStreamManager:
        pass

    manager_module = ModuleType("lightx2v.common.offload.manager")
    manager_module.WeightAsyncStreamManager = _StubWeightAsyncStreamManager

    class _StubFlux2TransformerInfer:
        def __init__(self, config):
            self.config = config

        def infer(self, *args, **kwargs):
            raise NotImplementedError

    base_infer_module = ModuleType("lightx2v.models.networks.flux2.infer.transformer_infer")
    base_infer_module.Flux2TransformerInfer = _StubFlux2TransformerInfer

    global_var_module = ModuleType("lightx2v_platform.base.global_var")
    global_var_module.AI_DEVICE = "npu"

    stub_modules = {
        "torch": torch_module,
        "torch.nn": torch_nn_module,
        "torch.nn.functional": torch_functional_module,
        "lightx2v": _package("lightx2v"),
        "lightx2v.common": _package("lightx2v.common"),
        "lightx2v.common.offload": _package("lightx2v.common.offload"),
        "lightx2v.common.offload.manager": manager_module,
        "lightx2v.models": _package("lightx2v.models"),
        "lightx2v.models.networks": _package("lightx2v.models.networks"),
        "lightx2v.models.networks.flux2": _package("lightx2v.models.networks.flux2"),
        "lightx2v.models.networks.flux2.infer": _package("lightx2v.models.networks.flux2.infer"),
        "lightx2v.models.networks.flux2.infer.transformer_infer": base_infer_module,
        "lightx2v_platform": _package("lightx2v_platform"),
        "lightx2v_platform.base": _package("lightx2v_platform.base"),
        "lightx2v_platform.base.global_var": global_var_module,
    }

    module_path = (
        Path(__file__).resolve().parents[1]
        / "lightx2v"
        / "models"
        / "networks"
        / "flux2"
        / "infer"
        / "offload"
        / "transformer_infer.py"
    )
    spec = importlib.util.spec_from_file_location("_flux2_npu_offload_schedule_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stub_modules):
        spec.loader.exec_module(module)
    return module


class _RecordingManager:
    def __init__(self, trace):
        self.trace = trace
        self.slot_contents = {}

    def prefetch_to_slot(self, slot_idx, block_idx, blocks):
        self.trace.append(("prefetch", slot_idx, block_idx))
        self.slot_contents[slot_idx] = ("staged", block_idx, slot_idx)

    def wait_ready(self, slot_idx):
        block = self.slot_contents[slot_idx]
        self.trace.append(("wait_ready", slot_idx, block[1]))
        return block

    def record_free(self, slot_idx):
        self.trace.append(("record_free", slot_idx, self.slot_contents[slot_idx][1]))


class _FailIfCalledManager:
    def __getattr__(self, name):
        raise AssertionError(f"all-resident stage must not access manager.{name}")


class Flux2NpuOffloadScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_transformer_infer_module()

    def _make_infer(self):
        infer_type = self.module.Flux2OffloadTransformerInfer
        return infer_type.__new__(infer_type)

    def test_interleaved_resident_blocks_keep_order_and_reuse_slots(self):
        infer = self._make_infer()
        blocks = [("resident", block_idx) for block_idx in range(8)]
        resident_indices = {0, 2, 3, 5, 6}
        trace = []
        manager = _RecordingManager(trace)

        def run_block(block_idx, block):
            trace.append(("run", block_idx, block))

        infer._run_npu_block_stage(manager, blocks, resident_indices, run_block)

        run_entries = [entry for entry in trace if entry[0] == "run"]
        self.assertEqual([entry[1] for entry in run_entries], list(range(8)))
        for _, block_idx, block in run_entries:
            if block_idx in resident_indices:
                self.assertIs(block, blocks[block_idx])
            else:
                expected_slot = {1: 0, 4: 1, 7: 0}[block_idx]
                self.assertEqual(block, ("staged", block_idx, expected_slot))

        manager_entries = [entry for entry in trace if entry[0] != "run"]
        self.assertEqual(
            manager_entries,
            [
                ("prefetch", 0, 1),
                ("prefetch", 1, 4),
                ("wait_ready", 0, 1),
                ("record_free", 0, 1),
                ("prefetch", 0, 7),
                ("wait_ready", 1, 4),
                ("record_free", 1, 4),
                ("wait_ready", 0, 7),
                ("record_free", 0, 7),
            ],
        )

        first_slot_free = trace.index(("record_free", 0, 1))
        first_slot_reuse = trace.index(("prefetch", 0, 7))
        self.assertLess(first_slot_free, first_slot_reuse)

    def test_all_resident_stage_does_not_touch_manager(self):
        infer = self._make_infer()
        blocks = [object() for _ in range(5)]
        executions = []

        infer._run_npu_block_stage(
            _FailIfCalledManager(),
            blocks,
            set(range(len(blocks))),
            lambda block_idx, block: executions.append((block_idx, block)),
        )

        self.assertEqual([block_idx for block_idx, _ in executions], list(range(len(blocks))))
        self.assertEqual([block for _, block in executions], blocks)


if __name__ == "__main__":
    unittest.main()
