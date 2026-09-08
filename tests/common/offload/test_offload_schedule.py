from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lightx2v.common.transformer_infer import transformer_infer as transformer_infer_module
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.runners.base_runner import BaseRunner, keep_transformer_weights_loaded


class _Stream:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait_stream(self, stream):
        self.log.append(("wait_stream", self.name, stream.name))

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event))

    def record_event(self):
        event = f"{self.name}_done"
        self.log.append(("record_event", self.name, event))
        return event


class _Device:
    def __init__(self, caller_stream):
        self.caller_stream = caller_stream

    def current_stream(self):
        return self.caller_stream

    @staticmethod
    def stream(stream):
        return nullcontext()

    def synchronize(self):
        self.caller_stream.log.append(("device_sync",))


class _Blocks(list):
    pass


class _Infer(BaseTransformerInfer):
    def infer(self):
        raise NotImplementedError


class _BindManager:
    def __init__(self, uses_events=False, load_stream=None, compute_stream=None):
        self.uses_events = uses_events
        self.cuda_load_stream = load_stream if load_stream is not None else object()
        self.compute_stream = compute_stream if compute_stream is not None else object()
        self.cuda_buffers = None

    def init_cuda_buffer(self, buffers):
        self.cuda_buffers = list(buffers)


class _BindInfer(_Infer):
    def __init__(self, uses_events=False):
        self.uses_events = uses_events
        self.created_managers = []

    def create_block_offload_manager(self, config, **kwargs):
        manager = _BindManager(self.uses_events, **kwargs)
        self.created_managers.append(manager)
        return manager


class _TransformerWeights:
    def __init__(self, groups):
        self.groups = groups

    def get_offload_block_groups(self):
        return self.groups


class _EventManager:
    uses_events = True

    def __init__(self, log, slot_count=2):
        self.log = log
        self.slot_count = slot_count
        self.compute_stream = _Stream("compute", log)
        self.pending = [False] * slot_count
        self.loaded_blocks = [None] * slot_count
        self.load_calls = []

    def prefetch_to_slot(self, slot_idx, block_idx, blocks, adapter_block_idx, state_dict_transform):
        assert not self.pending[slot_idx]
        self.pending[slot_idx] = True
        self.loaded_blocks[slot_idx] = block_idx
        self.load_calls.append((block_idx, adapter_block_idx, state_dict_transform))
        self.log.append(("prefetch", slot_idx, block_idx))

    def wait_ready(self, slot_idx, stream):
        assert self.pending[slot_idx]
        self.log.append(("ready", slot_idx, stream.name))
        return ("slot", slot_idx, self.loaded_blocks[slot_idx])

    def record_free(self, slot_idx, stream):
        assert self.pending[slot_idx]
        self.log.append(("free", slot_idx, stream.name))
        self.pending[slot_idx] = False

    def reset_slots(self):
        self.log.append(("reset_slots",))
        self.pending = [False] * self.slot_count


class _Buffer:
    def __init__(self, slot_idx):
        self.slot_idx = slot_idx
        self.block_idx = None


class _OutputTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, log):
        return torch.Tensor._make_subclass(cls, torch.empty(0), False)

    def __init__(self, log):
        self.log = log

    def record_stream(self, stream):
        self.log.append(("record_stream", stream.name))


class _StreamManager:
    uses_events = False

    def __init__(self, log, slot_count=2):
        self.log = log
        self.compute_stream = _Stream("compute", log)
        self.cuda_buffers = [_Buffer(slot_index) for slot_index in range(slot_count)]
        self.need_init_first_buffer = True
        self.loaded_state_dict_transform = None
        self.loaded_first_adapter_block_index = None
        self.load_calls = []

    def init_first_buffer(self, blocks, adapter_block_idx, block_idx, state_dict_transform):
        self.cuda_buffers[0].block_idx = block_idx
        self.need_init_first_buffer = False
        self.load_calls.append((block_idx, adapter_block_idx, state_dict_transform))
        self.log.append(("init", self.cuda_buffers[0].slot_idx, block_idx))

    def prefetch_weights_to_buffer(self, buffer_idx, block_idx, blocks, adapter_block_idx, state_dict_transform):
        self.cuda_buffers[buffer_idx].block_idx = block_idx
        self.load_calls.append((block_idx, adapter_block_idx, state_dict_transform))
        self.log.append(("prefetch", self.cuda_buffers[buffer_idx].slot_idx, block_idx))

    def swap_blocks(self):
        self.cuda_buffers[0], self.cuda_buffers[1] = self.cuda_buffers[1], self.cuda_buffers[0]
        self.log.append(("swap",))

    def wait_for_block_compute(self):
        self.log.append(("wait_compute",))


def _make_blocks(group_name="blocks"):
    blocks = _Blocks([("resident", index) for index in range(6)])
    blocks.offload_group_name = group_name
    blocks.resident_block_indices = frozenset({0, 3})
    blocks.offload_block_indices = (1, 2, 4, 5)
    return blocks


def _make_one_offloaded_block():
    blocks = _Blocks([("resident", index) for index in range(4)])
    blocks.offload_group_name = "blocks"
    blocks.resident_block_indices = frozenset({0, 1, 3})
    blocks.offload_block_indices = (2,)
    return blocks


def _make_three_offloaded_blocks():
    blocks = _Blocks([("resident", index) for index in range(5)])
    blocks.offload_group_name = "blocks"
    blocks.resident_block_indices = frozenset({0, 3})
    blocks.offload_block_indices = (1, 2, 4)
    return blocks


def _make_infer(manager, group_name="blocks"):
    infer = _Infer()
    infer._block_offload_managers = {group_name: manager}
    return infer


def _block_offload_config(use_events=False):
    return {
        "cpu_offload": True,
        "offload_plan": {
            "offload_granularity": "block",
            "use_event_offload": use_events,
        },
    }


def test_compiled_blocks_are_cached_for_each_staging_slot(monkeypatch):
    compiled = []

    def compile_block(block_runner, dynamic):
        compiled.append((block_runner, dynamic))
        return block_runner

    monkeypatch.setattr(transformer_infer_module.torch, "compile", compile_block)
    infer = _Infer()
    infer.compiled_blocks = {}
    first_slot = object()
    second_slot = object()

    first_compiled = infer.get_compiled_block(0, first_slot)
    infer.get_compiled_block(0, second_slot)

    assert infer.get_compiled_block(0, first_slot) is first_compiled
    assert len(compiled) == 2


def test_single_block_group_is_bound_to_its_buffers():
    blocks = _make_blocks()
    buffers = [_Buffer(0), _Buffer(1)]
    blocks.offload_cuda_buffers = buffers
    infer = _BindInfer()

    infer.init_block_offload(_block_offload_config(), _TransformerWeights({"blocks": blocks}))

    manager = infer.get_block_offload_manager(blocks)
    assert manager is infer.created_managers[0]
    assert manager.cuda_buffers == buffers


def test_multiple_block_groups_are_bound_independently():
    double_blocks = _make_blocks("double_blocks")
    single_blocks = _make_blocks("single_blocks")
    double_buffers = [_Buffer(0), _Buffer(1)]
    single_buffers = [_Buffer(0), _Buffer(1)]
    double_blocks.offload_cuda_buffers = double_buffers
    single_blocks.offload_cuda_buffers = single_buffers
    infer = _BindInfer()

    infer.init_block_offload(
        _block_offload_config(),
        _TransformerWeights(
            {
                "double_blocks": double_blocks,
                "single_blocks": single_blocks,
            }
        ),
    )

    double_manager = infer.get_block_offload_manager(double_blocks)
    single_manager = infer.get_block_offload_manager(single_blocks)
    assert double_manager is not single_manager
    assert double_manager.cuda_buffers == double_buffers
    assert single_manager.cuda_buffers == single_buffers


def test_fully_resident_group_is_bound_without_a_manager():
    blocks = _make_blocks()
    blocks.resident_block_indices = frozenset(range(len(blocks)))
    blocks.offload_block_indices = ()
    infer = _BindInfer()

    infer.init_block_offload(_block_offload_config(), _TransformerWeights({"blocks": blocks}))

    assert infer.get_block_offload_manager(blocks) is None
    assert infer.created_managers == []
    computed = []
    infer.run_blocks_with_offload(blocks, lambda block_idx, block: computed.append((block_idx, block)))
    assert computed == list(enumerate(blocks))


def test_block_manager_lookup_rejects_an_unbound_group():
    infer = _Infer()
    infer._block_offload_managers = {}

    with pytest.raises(KeyError):
        infer.get_block_offload_manager(_make_blocks("missing"))


def test_get_offload_managers_includes_bound_groups_and_legacy_manager():
    first = object()
    second = object()
    legacy = object()
    infer = _Infer()
    infer._block_offload_managers = {
        "first": first,
        "resident": None,
        "second": second,
    }
    infer.offload_manager = legacy

    assert infer.get_offload_managers() == [first, second, legacy]


def test_clear_offload_managers_clears_block_and_phase_managers():
    infer = _Infer()
    infer._block_offload_managers = {"blocks": object()}
    infer.offload_manager = object()

    infer.clear_offload_managers()

    assert infer._block_offload_managers == {}
    assert not hasattr(infer, "offload_manager")


def test_unregistered_block_groups_keep_the_legacy_manager_path():
    class LegacyManager:
        def init_cuda_buffer(self, block_buffers, phase_buffers):
            self.buffers = block_buffers, phase_buffers

    block_buffers = object()
    phase_buffers = object()
    model = SimpleNamespace(
        cpu_offload=True,
        offload_granularity="block",
        lazy_load=False,
        transformer_infer=SimpleNamespace(offload_manager=LegacyManager()),
        transformer_weights=SimpleNamespace(
            get_offload_block_groups=lambda: {},
            offload_block_cuda_buffers=block_buffers,
            offload_phase_cuda_buffers=phase_buffers,
        ),
    )

    BaseTransformerModel._init_offload_manager(model)

    assert model.transformer_infer.offload_manager.buffers == (block_buffers, phase_buffers)


def test_event_block_groups_share_load_and_compute_streams():
    double_blocks = _make_blocks("double_blocks")
    single_blocks = _make_blocks("single_blocks")
    double_blocks.offload_cuda_buffers = [_Buffer(0), _Buffer(1)]
    single_blocks.offload_cuda_buffers = [_Buffer(0), _Buffer(1)]
    infer = _BindInfer(uses_events=True)

    infer.init_block_offload(
        _block_offload_config(use_events=True),
        _TransformerWeights(
            {
                "double_blocks": double_blocks,
                "single_blocks": single_blocks,
            }
        ),
    )

    double_manager = infer.get_block_offload_manager(double_blocks)
    single_manager = infer.get_block_offload_manager(single_blocks)
    assert single_manager.cuda_load_stream is double_manager.cuda_load_stream
    assert single_manager.compute_stream is double_manager.compute_stream


def test_nonresident_blocks_require_staging_buffers():
    blocks = _make_blocks()
    infer = _BindInfer()

    with pytest.raises(RuntimeError, match="has no staging buffers"):
        infer.init_block_offload(
            _block_offload_config(),
            _TransformerWeights({"blocks": blocks}),
        )


def test_staging_buffer_count_must_match_the_offloaded_blocks():
    blocks = _make_blocks()
    blocks.offload_cuda_buffers = [_Buffer(0)]
    infer = _BindInfer()

    with pytest.raises(RuntimeError, match="requires 2 staging buffers, got 1"):
        infer.init_block_offload(
            _block_offload_config(),
            _TransformerWeights({"blocks": blocks}),
        )


def test_event_slots_overlap_nonresident_prefetch_and_preserve_block_order(monkeypatch):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    manager = _EventManager(log)
    computations = []

    _make_infer(manager).run_blocks_with_offload(
        _make_blocks(),
        lambda block_idx, block: computations.append((block_idx, block)),
    )

    assert [entry for entry in log if entry[0] == "prefetch"] == [
        ("prefetch", 0, 1),
        ("prefetch", 1, 2),
        ("prefetch", 0, 4),
        ("prefetch", 1, 5),
    ]
    assert computations == [
        (0, ("resident", 0)),
        (1, ("slot", 0, 1)),
        (2, ("slot", 1, 2)),
        (3, ("resident", 3)),
        (4, ("slot", 0, 4)),
        (5, ("slot", 1, 5)),
    ]
    assert ("wait_stream", "compute", "caller") in log
    assert [entry for entry in log if entry[0] in {"ready", "free"}] == [
        ("ready", 0, "compute"),
        ("free", 0, "compute"),
        ("ready", 1, "compute"),
        ("free", 1, "compute"),
        ("ready", 0, "compute"),
        ("free", 0, "compute"),
        ("ready", 1, "compute"),
        ("free", 1, "compute"),
    ]
    assert ("wait_event", "caller", "compute_done") in log
    assert manager.pending == [False, False]


def test_offload_output_is_owned_by_the_caller_stream(monkeypatch):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    output = _OutputTensor(log)

    result = _make_infer(_EventManager(log)).run_blocks_with_offload(
        _make_blocks(),
        lambda _block_idx, _block: output,
    )

    assert result is output
    assert ("wait_event", "caller", "compute_done") in log
    assert ("record_stream", "caller") in log


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_event_slots_are_reset_after_block_failure(monkeypatch, error_type):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    manager = _EventManager(log)

    def fail_on_first_offloaded_block(block_idx, _block):
        if block_idx == 1:
            raise error_type("block failed")

    with pytest.raises(error_type, match="block failed"):
        _make_infer(manager).run_blocks_with_offload(_make_blocks(), fail_on_first_offloaded_block)

    assert ("device_sync",) in log
    assert ("reset_slots",) in log
    assert manager.pending == [False, False]


def test_residency_is_independent_of_event_scheduling(monkeypatch):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    manager = _StreamManager(log)
    original_buffer_order = tuple(manager.cuda_buffers)
    computations = []

    _make_infer(manager).run_blocks_with_offload(
        _make_blocks(),
        lambda block_idx, block: computations.append((block_idx, block if isinstance(block, tuple) else (block.slot_idx, block.block_idx))),
    )

    assert [block_idx for block_idx, _ in computations] == list(range(6))
    assert computations[0][1] == ("resident", 0)
    assert computations[3][1] == ("resident", 3)
    assert [block_index for index, (_, block_index) in computations if index not in {0, 3}] == [1, 2, 4, 5]
    assert tuple(manager.cuda_buffers) == original_buffer_order


@pytest.mark.parametrize("manager_class", [_EventManager, _StreamManager])
def test_model_specific_weight_mapping_is_forwarded(monkeypatch, manager_class):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    manager = manager_class(log)

    def adapter_block_index(block_index):
        return block_index // 2

    def state_dict_transform(block_index, state_dict):
        return block_index, state_dict

    _make_infer(manager).run_blocks_with_offload(
        _make_blocks(),
        lambda _block_index, _block: None,
        adapter_block_index=adapter_block_index,
        state_dict_transform=state_dict_transform,
    )

    expected_loads = [
        (1, 0, state_dict_transform),
        (2, 1, state_dict_transform),
        (4, 2, state_dict_transform),
        (5, 2, state_dict_transform),
    ]
    if not manager.uses_events:
        expected_loads.append((1, 0, state_dict_transform))
    assert manager.load_calls == expected_loads


def test_stream_offload_prefetches_the_next_step_first_block(monkeypatch):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    manager = _StreamManager(log)
    infer = _make_infer(manager)
    blocks = _make_blocks()

    infer.run_blocks_with_offload(blocks, lambda _block_index, _block: None)
    infer.run_blocks_with_offload(blocks, lambda _block_index, _block: None)

    assert [entry for entry in log if entry[0] == "init"] == [("init", 0, 1)]
    assert [block_index for block_index, _, _ in manager.load_calls] == [1, 2, 4, 5, 1, 2, 4, 5, 1]


def test_stream_offload_preserves_staging_order_with_an_odd_block_count(monkeypatch):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    manager = _StreamManager(log)
    infer = _make_infer(manager)
    blocks = _make_three_offloaded_blocks()
    staged_blocks = []

    def record_staged_block(block_index, block):
        if block_index in blocks.offload_block_indices:
            staged_blocks.append(block.block_idx)

    infer.run_blocks_with_offload(blocks, record_staged_block)
    infer.run_blocks_with_offload(blocks, record_staged_block)

    assert staged_blocks == [1, 2, 4, 1, 2, 4]


def test_stream_offload_reuses_a_single_staging_block(monkeypatch):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    manager = _StreamManager(log, slot_count=1)
    infer = _make_infer(manager)
    blocks = _make_one_offloaded_block()
    computed = []

    infer.run_blocks_with_offload(blocks, lambda block_index, _block: computed.append(block_index))
    infer.run_blocks_with_offload(blocks, lambda block_index, _block: computed.append(block_index))

    assert computed == [0, 1, 2, 3, 0, 1, 2, 3]
    assert manager.load_calls == [(2, None, None)]


def test_event_offload_reuses_a_single_slot_across_steps(monkeypatch):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    manager = _EventManager(log, slot_count=1)
    infer = _make_infer(manager)
    blocks = _make_one_offloaded_block()
    staged_blocks = []

    infer.run_blocks_with_offload(blocks, lambda block_index, block: staged_blocks.append(block[2]) if block_index == 2 else None)
    infer.run_blocks_with_offload(blocks, lambda block_index, block: staged_blocks.append(block[2]) if block_index == 2 else None)

    assert staged_blocks == [2, 2]
    assert manager.pending == [False]


def test_stream_offload_reloads_first_block_when_adapter_mapping_changes(monkeypatch):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    manager = _StreamManager(log)
    infer = _make_infer(manager)
    blocks = _make_blocks()

    infer.run_blocks_with_offload(blocks, lambda _block_index, _block: None, adapter_block_index=lambda _index: 0)
    infer.run_blocks_with_offload(blocks, lambda _block_index, _block: None, adapter_block_index=lambda _index: 1)

    assert [entry for entry in log if entry[0] == "init"] == [
        ("init", 0, 1),
        ("init", 0, 1),
    ]
    assert manager.load_calls[0][1] == 0
    assert manager.load_calls[5][1] == 1


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_stream_slots_are_restored_after_block_failure(monkeypatch, error_type):
    log = []
    caller_stream = _Stream("caller", log)
    monkeypatch.setattr(transformer_infer_module, "torch_device_module", _Device(caller_stream))
    monkeypatch.setattr(transformer_infer_module, "AI_DEVICE", "cuda")
    manager = _StreamManager(log)
    original_buffer_order = tuple(manager.cuda_buffers)

    def fail_on_second_offloaded_block(block_idx, _block):
        if block_idx == 2:
            raise error_type("block failed")

    with pytest.raises(error_type, match="block failed"):
        _make_infer(manager).run_blocks_with_offload(_make_blocks(), fail_on_second_offloaded_block)

    assert ("device_sync",) in log
    assert manager.need_init_first_buffer is True
    assert tuple(manager.cuda_buffers) == original_buffer_order


def test_runner_entry_keeps_offload_weights_active_for_the_whole_loop():
    class Model:
        def __init__(self):
            self.active = False
            self.prepare_count = 0
            self.cleanup_count = 0

        def prepare_offload_weights(self):
            if self.active:
                return False
            self.active = True
            self.prepare_count += 1
            return True

        def cleanup_offload_weights(self):
            self.active = False
            self.cleanup_count += 1

    class Runner(BaseRunner):
        @keep_transformer_weights_loaded
        def run(self):
            assert self.model.active
            return "done"

    runner = Runner({})
    runner.model = Model()

    assert runner.run() == "done"
    assert runner.model.prepare_count == 1
    assert runner.model.cleanup_count == 1


def test_runner_does_not_extend_offload_session_across_warmup():
    class Model:
        def __init__(self):
            self.active = False
            self.prepare_count = 0
            self.cleanup_count = 0

        def prepare_offload_weights(self):
            self.active = True
            self.prepare_count += 1
            return True

        def cleanup_offload_weights(self):
            self.active = False
            self.cleanup_count += 1

    class Runner(BaseRunner):
        def init_modules(self):
            return None

        def warmup(self):
            assert not self.model.active

    runner = Runner({"warmup": True})
    runner.model = Model()
    runner.init_modules()

    assert runner.model.prepare_count == 0
    assert runner.model.cleanup_count == 0
