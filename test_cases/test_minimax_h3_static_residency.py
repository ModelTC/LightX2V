"""Focused H3 residency tests; GPU cases use small weights, not a full checkpoint."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.offload.resident import ResidentBlocks, uniform_resident_indices
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer, validate_minimax_h3_residency_config
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3.weights.transformer_weights import MiniMaxH3TransformerBlockWeights, MiniMaxH3TransformerWeights
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v_platform.base.global_var import AI_DEVICE
from test_cases.test_wan_static_residency import make_stack
from tools.estimate_offload_residency import load_block

GPU = pytest.mark.skipif(AI_DEVICE != "cuda" or not torch.cuda.is_available(), reason="CUDA stream/lifetime test")


def config():
    return dict(model_cls="minimax_h3", model_variant="fl2av", num_layers=50, cpu_offload=True, offload_granularity="block", offload_resident_blocks=0, use_adaln_cache=True)


@pytest.mark.parametrize(
    "override",
    [
        {"model_variant": "ref2av"},
        {"use_adaln_cache": False},
        {"dit_quantized": True},
        {"dit_disk_streaming": True},
        {"use_fused_qkv": True},
        {"enable_cfg": True},
        {"shared_cpu_weights": True},
        {"use_compile": True},
        {"feature_caching": "DPCache"},
        {"cpu_offload": False},
        {"offload_granularity": "model"},
        {"offload_resident_blocks": 51},
        {"cpu_offload_layout": "contiguous", "dit_release_block_offload_buffers": True},
    ],
)
def test_config_rejects_unsupported_modes(override):
    with pytest.raises(ValueError):
        validate_minimax_h3_residency_config({**config(), **override})


def test_config_accepts_normalized_config_without_task():
    validate_minimax_h3_residency_config(config())
    validate_minimax_h3_residency_config({"model_variant": "ref2av"})  # Legacy path.
    assert uniform_resident_indices(50, 25) == tuple(range(0, 50, 2))
    missing_granularity = config()
    del missing_granularity["offload_granularity"]
    with pytest.raises(ValueError, match="offload_granularity"):
        validate_minimax_h3_residency_config(missing_granularity)


def make_infer(manager, count):
    infer = object.__new__(MiniMaxH3OffloadTransformerInfer)
    infer.offload_manager = manager
    infer.resident_blocks = ResidentBlocks(count)
    return infer


@GPU
@pytest.mark.parametrize("layout", ["per_tensor", "contiguous"])
@pytest.mark.parametrize("count", [0, 1, 4, 6, 7])
def test_logical_indices_copies_and_repeated_requests(layout, count):
    blocks, manager = make_stack(layout)
    infer = make_infer(manager, count)
    hidden = torch.ones(16, 32, device="cuda", dtype=torch.bfloat16)
    pre = SimpleNamespace(temb=None, timestep_indices=hidden, adaln_indices=hidden, rotary_emb=(hidden, hidden))
    expected = hidden.clone()
    for index, block in enumerate(blocks):
        expected = expected @ block.weight.pin_tensor.to("cuda") + index
    seen, copied = [], []
    copy = manager.copy_block_to_buffer

    def counted_copy(target, index, sources, adapter=None):
        copied.append(index)
        return copy(target, index, sources, adapter)

    def run_block(index, block, state, pre):
        assert infer.block_idx == index
        seen.append(index)
        return state @ block.weight.tensor + index

    manager.copy_block_to_buffer = counted_copy
    infer.run_block = run_block
    for _ in range(2):
        for _ in range(2):
            actual = infer.infer_with_blocks_offload(blocks, hidden.clone(), pre)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        infer.release_resident_blocks()
        infer.release_resident_blocks()
        assert not infer.resident_blocks.active and not infer.resident_blocks.originals
    assert seen == list(range(7)) * 4
    assert copied == [i for i in range(7) if i not in infer.resident_blocks.indices] * 4


@GPU
def test_failed_block_releases_and_can_restart():
    blocks, manager = make_stack("per_tensor")
    infer = make_infer(manager, 3)
    x = torch.ones(16, 32, device="cuda", dtype=torch.bfloat16)
    pre = SimpleNamespace(temb=None, timestep_indices=x, adaln_indices=x, rotary_emb=(x, x))

    def fail(index, block, state, pre):
        if index == 2:
            raise RuntimeError("cancelled")
        return state @ block.weight.tensor

    infer.run_block = fail
    with pytest.raises(RuntimeError, match="cancelled"):
        infer.infer_with_blocks_offload(blocks, x, pre)
    assert not infer.resident_blocks.active and not any(manager._slot_pending)
    infer.run_block = lambda index, block, state, pre: state @ block.weight.tensor
    infer.infer_with_blocks_offload(blocks, x, pre)
    infer.release_resident_blocks()


@GPU
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("phase", ["denoise", "warmup"])
def test_runner_releases_before_decode_and_on_failure(cancelled, phase):
    blocks, manager = make_stack("per_tensor")
    infer = make_infer(manager, 3)
    runner = object.__new__(MiniMaxH3Runner)
    runner.config = {"cpu_offload": True, "model_variant": "fl2av"}
    runner.model = SimpleNamespace(transformer_infer=infer, block_offload=True, prepost_resident=True)
    runner.maybe_empty_cache = lambda **kwargs: None
    runner._is_mps_low_memory_streaming = lambda: False
    runner.init_run = lambda: infer.resident_blocks.prepare(blocks)
    runner.end_run = lambda: None
    runner._release_low_memory_vae = lambda: None
    runner.process_images_after_vae_decoder = lambda: None

    def denoise(*args):
        assert infer.resident_blocks.active
        if cancelled:
            raise RuntimeError("cancelled")
        return "video", "audio"

    def decode(*args):
        assert not infer.resident_blocks.active
        return "video", "audio"

    runner.run_segment = denoise
    runner.run_vae_decoder = decode
    if phase == "warmup":
        runner.scheduler = SimpleNamespace(generator=None, infer_steps=1, video_latents="video", audio_latents="audio", step_pre=lambda i: None, step_post=lambda: None)
        runner.model.infer = denoise
        runner._WARMUP_SHAPES = [(32, 32, 22)]
        runner._WARMUP_STEP_COUNT = 1
        runner._prepare_warmup_inputs = lambda *args: None
        runner._run_input_encoder_local_h3 = lambda: {}
        runner.clear_warmup_state = lambda: None
        runner._maybe_freeze_gc = lambda: None
    run = runner.run_warmup if phase == "warmup" else runner.run_main
    if cancelled:
        with pytest.raises(RuntimeError, match="cancelled"):
            run()
    else:
        run()
    assert not infer.resident_blocks.active


def small_weights(cfg, count):
    weights = {}
    for i in range(count):
        block = MiniMaxH3TransformerBlockWeights(i, cfg)
        for _, leaf in block.named_weight_leaves():
            for name, _, _ in getattr(leaf, "base_attrs", ()):
                weights[name] = torch.ones(32 if "norm" in name else (32, 32), dtype=torch.bfloat16)
    return weights


@GPU
@pytest.mark.parametrize("layout", ["per_tensor", "contiguous"])
def test_loader_excludes_adaln_weights(tmp_path, layout):
    cfg = {**config(), "cpu_offload_layout": layout, "attn_type": "torch_sdpa"}
    tensors = small_weights(cfg, 1)
    expected_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    tensors["transformer_blocks.0.adaln_proj.linear.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    save_file(tensors, tmp_path / "model.safetensors")
    blocks, manager, size = load_block(cfg, str(tmp_path), MiniMaxH3TransformerBlockWeights, "transformer_blocks.0.", block_kwargs=dict(index=0, config=cfg), exclude_keys=(".adaln_proj.",))
    assert not any("adaln_proj" in name for name in blocks[0].state_dict())
    if layout == "contiguous":
        assert size >= expected_bytes  # Includes slab alignment.
    else:
        # Each RMS leaf also carries a zero scalar diff in its transfer state.
        assert size == expected_bytes + 4 * torch.tensor(0, dtype=torch.bfloat16).element_size()
    if manager.contiguous_transfer is not None:
        manager.contiguous_transfer.close()


@GPU
@pytest.mark.parametrize("event", [False, True])
def test_release_and_rebuild_slots(event):
    cfg = {**config(), "num_layers": 2, "attn_type": "torch_sdpa"}
    model = object.__new__(MiniMaxH3Model)
    model.device = torch.device("cpu")
    model.config, model.block_offload, model.lazy_load = cfg, True, False
    weights = model.transformer_weights = MiniMaxH3TransformerWeights(cfg)
    weights.load(small_weights(cfg, 2))
    infer = model.transformer_infer = make_infer(None, 1)
    if not event:
        infer.resident_blocks = None
    infer.offload_manager = infer.create_offload_manager()
    model._init_offload_manager()
    for _ in range(2):
        if event:
            infer.resident_blocks.prepare(weights.blocks)
        model.release_block_offload_buffers()
        model.release_block_offload_buffers()
        assert not weights.offload_group.device_slots
        assert weights.offload_block_cuda_buffers is None and infer.offload_manager is None
        model.ensure_block_offload_buffers()
        assert isinstance(infer.offload_manager, EventSlotWeightAsyncStreamManager) == event
        assert weights.offload_group.device_slots is weights.offload_block_cuda_buffers
        if event:
            infer.offload_manager.copy_block_to_buffer(infer.offload_manager.cuda_buffers[0], 0, weights.blocks)
        else:
            infer.offload_manager.init_first_buffer(weights.blocks)
        torch.cuda.synchronize()
    model.release_block_offload_buffers()
