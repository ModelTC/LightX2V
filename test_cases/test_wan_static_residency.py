import math

import pytest
import torch

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.offload.block_loader import OffloadGroup, prepare_contiguous_group
from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.offload.resident import ResidentBlocks, uniform_resident_indices
from lightx2v.common.ops.tensor.tensor import DefaultTensor
from lightx2v.models.networks.wan.infer.offload.transformer_infer import validate_wan_residency_config
from lightx2v.models.networks.wan.infer.utils import image_latent_size
from lightx2v_platform.base.global_var import AI_DEVICE
from tools.estimate_offload_residency import estimate_resident_count, write_json


@pytest.mark.parametrize("compute,transfer,expected", [(5, 3, 0), (3, 3, 0), (2, 4, 20), (1, 4, 30), (7, 10, 12), (9, 10, 4), (1, 100, 40)])
def test_estimate(compute, transfer, expected):
    assert estimate_resident_count(40, compute, transfer) == expected
    assert (40 - expected) * transfer <= 40 * compute
    if expected:
        assert (41 - expected) * transfer > 40 * compute


@pytest.mark.parametrize("compute,transfer", [(0, 1), (1, 0), (-1, 1), (math.nan, 1), (1, math.inf)])
def test_invalid_measurements(compute, transfer):
    with pytest.raises(ValueError):
        estimate_resident_count(40, compute, transfer)


@pytest.mark.parametrize("count", range(41))
def test_uniform_layout(count):
    indices = uniform_resident_indices(40, count)
    assert len(set(indices)) == count
    assert all(0 <= index < 40 for index in indices)
    if count:
        gaps = [b - a for a, b in zip(indices, (*indices[1:], indices[0] + 40))]
        assert max(gaps) - min(gaps) <= 1
    if count == 30:
        assert sorted(set(range(40)) - set(indices)) == list(range(3, 40, 4))


@pytest.mark.parametrize("count", [-1, 41, True, 1.5])
def test_invalid_count(count):
    with pytest.raises(ValueError):
        uniform_resident_indices(40, count)


def test_config_boundary():
    config = dict(model_cls="wan2.1", task="i2v", num_layers=40, cpu_offload=True, offload_granularity="block", offload_resident_blocks=0)
    validate_wan_residency_config(config)
    for overrides in ({"lazy_load": True}, {"cpu_offload": False}, {"offload_granularity": "phase"}, {"feature_caching": "Tea"}, {"model_cls": "wan2.2"}):
        with pytest.raises(ValueError):
            validate_wan_residency_config({**config, **overrides})


def test_atomic_config_update(tmp_path):
    import json

    path = tmp_path / "config.json"
    config = {"attention": "existing", "offload_resident_blocks": 30}
    write_json(path, config)
    assert json.loads(path.read_text()) == config
    with pytest.raises(TypeError):
        write_json(path, {"bad": object()})
    assert json.loads(path.read_text()) == config
    assert list(tmp_path.iterdir()) == [path]


def test_image_size_matches_wan_area_rule():
    assert image_latent_size((720, 544), (480, 832), (4, 8, 8), (1, 2, 2)) == (90, 68)


class TensorBlock(WeightModule):
    def __init__(self, index, device_slot=False):
        super().__init__()
        self.register_parameter("weight", DefaultTensor(f"blocks.{index}.weight", create_cuda_buffer=device_slot))


def make_stack(layout, count=7):
    blocks = [TensorBlock(i) for i in range(count)]
    slots = [TensorBlock(i, device_slot=True) for i in range(2)]
    weights = {f"blocks.{i}.weight": torch.eye(32, dtype=torch.bfloat16) + torch.randn(32, 32, generator=torch.Generator().manual_seed(i), dtype=torch.bfloat16) * 0.03 for i in range(count)}
    group = OffloadGroup(blocks, slots, tuple(f"blocks.{i}." for i in range(count)))
    if layout == "contiguous":
        prepare_contiguous_group(group, weights)
    for block in (*slots, *blocks):
        block.load(dict(weights))
    manager = EventSlotWeightAsyncStreamManager("block")
    manager.init_cuda_buffer(slots)
    if layout == "contiguous":
        manager.init_contiguous_group(group)
    return blocks, manager


@pytest.mark.skipif(AI_DEVICE != "cuda" or not torch.cuda.is_available(), reason="CUDA stream test")
@pytest.mark.parametrize("layout", ["per_tensor", "contiguous"])
@pytest.mark.parametrize("count", [0, 1, 4, 6, 7])
def test_static_execution_and_release(layout, count):
    blocks, manager = make_stack(layout)
    residents = ResidentBlocks(count)
    x = torch.ones(16, 32, device="cuda", dtype=torch.bfloat16)
    reference = x.clone()
    for block in blocks:
        reference = reference @ block.weight.pin_tensor.to("cuda")
    cpu_pointers = [block.weight.pin_tensor.data_ptr() for block in blocks]
    copied = []
    copy = manager.copy_block_to_buffer

    def counted_copy(target, index, sources, adapter=None):
        copied.append(index)
        return copy(target, index, sources, adapter)

    manager.copy_block_to_buffer = counted_copy
    for _ in range(2):
        residents.prepare(blocks)
        saved_count = len(residents.originals)
        residents.prepare(blocks)
        assert len(residents.originals) == saved_count
        for _ in range(3):
            actual = manager.run_blocks(blocks, residents.indices, x.clone(), lambda index, block, hidden: hidden @ block.weight.tensor)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        residents.release()
        manager.reset_slots()
        assert not residents.originals
        assert all(getattr(block.weight, "tensor", None) is None or block.weight.tensor.device.type == "cpu" for block in blocks)
    assert copied == [i for i in range(7) if i not in residents.indices] * 6
    assert cpu_pointers == [block.weight.pin_tensor.data_ptr() for block in blocks]


@pytest.mark.skipif(AI_DEVICE != "cuda" or not torch.cuda.is_available(), reason="CUDA stream test")
def test_execution_exception_can_restart():
    blocks, manager = make_stack("per_tensor")
    residents = ResidentBlocks(3)
    residents.prepare(blocks)

    def fail(index, block, state):
        if index == 2:
            raise RuntimeError("cancelled")
        return state @ block.weight.tensor

    x = torch.ones(16, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="cancelled"):
        manager.run_blocks(blocks, residents.indices, x, fail)
    assert not any(manager._slot_pending)
    residents.release()
    residents.prepare(blocks)
    manager.run_blocks(blocks, residents.indices, x, lambda index, block, state: state @ block.weight.tensor)
    torch.cuda.synchronize()
    residents.release()


@pytest.mark.skipif(AI_DEVICE != "cuda" or not torch.cuda.is_available(), reason="CUDA lifetime test")
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("phase", ["denoise", "warmup"])
def test_runner_releases_before_decode(monkeypatch, cancelled, phase):
    from types import SimpleNamespace

    from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
    from lightx2v.models.runners.default_runner import DefaultRunner
    from lightx2v.models.runners.wan.wan_runner import WanRunner

    blocks, manager = make_stack("per_tensor")
    infer = object.__new__(WanOffloadTransformerInfer)
    infer.offload_manager = manager
    infer.resident_blocks = ResidentBlocks(3)
    infer.resident_blocks.prepare(blocks)
    runner = object.__new__(WanRunner)
    runner.config = {"model_cls": "wan2.1", "task": "i2v", "cpu_offload": True, "offload_granularity": "block"}
    runner.model = SimpleNamespace(transformer_infer=infer)

    def run_segment(self, segment_idx=0):
        assert infer.resident_blocks.active
        if cancelled:
            raise RuntimeError("cancelled")
        return "latents"

    if phase == "denoise":
        monkeypatch.setattr(DefaultRunner, "run_segment", run_segment)
        run = runner.run_segment
    else:
        runner._WARMUP_RESOLUTIONS = [(64, 64)]
        runner.run_text_encoder = lambda input_info: None
        runner.get_warmup_latent_shape = lambda height, width: (16, 1, 8, 8)
        runner.get_warmup_image_encoder_output = lambda shape: {}
        runner.clear_warmup_state = lambda: None
        runner.model.scheduler = SimpleNamespace(sample_guide_scale=5, prepare=lambda **kwargs: None, step_pre=lambda step_index: None, step_post=lambda: None, latents="latents")
        runner.model.infer = lambda inputs: run_segment(runner)

        def decode(latents):
            assert latents == "latents"
            assert not infer.resident_blocks.active

        runner.run_vae_decoder = decode
        run = runner._run_warmup

    if cancelled:
        with pytest.raises(RuntimeError, match="cancelled"):
            run()
    else:
        result = run()
        if phase == "denoise":
            assert result == "latents"
    assert not infer.resident_blocks.active
    assert not infer.resident_blocks.originals
    assert all(getattr(block.weight, "tensor", None) is None or block.weight.tensor.device.type == "cpu" for block in blocks)
