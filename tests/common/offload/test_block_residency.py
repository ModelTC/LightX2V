import pytest
import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList, resolve_resident_block_indices


def _blocks(count):
    return WeightModuleList([WeightModule() for _ in range(count)])


@pytest.mark.parametrize(
    ("resident_count", "blocks_num", "expected"),
    [
        (0, 10, frozenset()),
        (3, 10, frozenset({0, 3, 6})),
        ("all", 10, frozenset(range(10))),
    ],
)
def test_resident_block_indices(resident_count, blocks_num, expected):
    assert resolve_resident_block_indices(resident_count, blocks_num) == expected


@pytest.mark.parametrize("resident_count", [-1, 11])
def test_resident_block_count_out_of_range(resident_count):
    with pytest.raises(ValueError):
        resolve_resident_block_indices(resident_count, 10)


def test_no_offload_means_all_blocks_are_resident():
    weights = WeightModule()
    blocks = _blocks(6)

    slot_count = weights.register_offload_block_group({"cpu_offload": False}, "blocks", blocks)

    assert blocks.resident_block_indices == frozenset(range(6))
    assert blocks.offload_block_indices == ()
    assert slot_count == 0


def test_model_offload_has_no_resident_blocks():
    weights = WeightModule()
    blocks = _blocks(6)
    config = {
        "cpu_offload": True,
        "offload_plan": {
            "offload_granularity": "model",
            "resident_blocks": {"blocks": 4},
        },
    }

    slot_count = weights.register_offload_block_group(config, "blocks", blocks)

    assert blocks.resident_block_indices == frozenset()
    assert blocks.offload_block_indices == tuple(range(6))
    assert slot_count == 0


def test_unknown_resident_block_group_is_rejected():
    weights = WeightModule()
    weights.register_offload_block_group(
        {"cpu_offload": True, "offload_plan": {"offload_granularity": "block"}},
        "blocks",
        _blocks(6),
    )

    with pytest.raises(ValueError, match="missing_blocks"):
        weights.validate_offload_block_groups(
            {
                "cpu_offload": True,
                "offload_plan": {
                    "offload_granularity": "block",
                    "resident_blocks": {"missing_blocks": 2},
                },
            }
        )


def test_legacy_block_offload_without_residency_does_not_require_a_group():
    weights = WeightModule()

    weights.validate_offload_block_groups(
        {
            "cpu_offload": True,
            "offload_plan": {"offload_granularity": "block"},
        }
    )


@pytest.mark.parametrize(
    ("resident_count", "expected_resident", "expected_offloaded", "expected_slots"),
    [
        (2, frozenset({0, 3}), (1, 2, 4, 5), 2),
        (5, frozenset({0, 1, 2, 3, 4}), (5,), 1),
        ("all", frozenset(range(6)), (), 0),
    ],
)
def test_block_offload_group_metadata(resident_count, expected_resident, expected_offloaded, expected_slots):
    weights = WeightModule()
    blocks = _blocks(6)
    config = {
        "cpu_offload": True,
        "offload_plan": {
            "offload_granularity": "block",
            "resident_blocks": {"blocks": resident_count},
        },
    }

    slot_count = weights.register_offload_block_group(config, "blocks", blocks)

    assert blocks.resident_block_indices == expected_resident
    assert blocks.offload_block_indices == expected_offloaded
    assert slot_count == expected_slots


def test_release_uses_the_weight_objects_cpu_contract():
    class AttentionBackend:
        def __init__(self):
            self.runtime_cache = torch.empty(1, device="meta")

        def to_cpu(self):
            pass

    weights = WeightModule()
    attention = AttentionBackend()
    weights.add_module("attention", attention)

    weights.release_device_weights()

    assert attention.runtime_cache.device.type == "meta"
