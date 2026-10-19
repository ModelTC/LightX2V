from types import SimpleNamespace

import pytest
import torch

from lightx2v.common.offload.resident import ResidentBlocks
from lightx2v.models.networks.qwen_image.infer.offload.transformer_infer import QwenImageOffloadTransformerInfer, validate_qwen_residency_config
from lightx2v.models.networks.qwen_image.infer.utils import get_qwen_image_size
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner
from lightx2v_platform.base.global_var import AI_DEVICE
from test_cases.test_wan_static_residency import make_stack


def test_config_boundary():
    config = dict(model_cls="qwen_image", task="t2i", num_layers=60, cpu_offload=True, offload_granularity="block", offload_resident_blocks=0)
    validate_qwen_residency_config(config)
    for overrides in (
        {"task": "i2i"},
        {"layered": True},
        {"lazy_load": True},
        {"shared_cpu_weights": True},
        {"cpu_offload": False},
        {"offload_granularity": "phase"},
        {"use_compile": True},
        {"use_magi_compile": True},
        {"offload_resident_blocks": 61},
    ):
        with pytest.raises(ValueError):
            validate_qwen_residency_config({**config, **overrides})
    # Unrelated legacy configurations retain their existing path.
    validate_qwen_residency_config({"task": "i2i"})


def test_image_size_matches_runner():
    assert get_qwen_image_size({"aspect_ratio": "16:9"}, []) == (1664, 928)
    assert get_qwen_image_size({"aspect_ratio": "16:9"}, [], "9:16") == (928, 1664)
    assert get_qwen_image_size({}, [1024, 2048]) == (1664, 832)
    assert get_qwen_image_size({}, [64, 128]) == (256, 256)
    assert get_qwen_image_size({"aspect_ratios": {"wide": [1024, 512]}, "aspect_ratio": "wide"}, []) == (1024, 512)


@pytest.mark.skipif(AI_DEVICE != "cuda" or not torch.cuda.is_available(), reason="CUDA stream test")
@pytest.mark.parametrize("layout", ["per_tensor", "contiguous"])
@pytest.mark.parametrize("count", [0, 1, 4, 6, 7])
def test_qwen_two_stream_states(layout, count):
    blocks, manager = make_stack(layout)
    infer = object.__new__(QwenImageOffloadTransformerInfer)
    infer.offload_manager = manager
    infer.resident_blocks = ResidentBlocks(count)
    image = torch.ones(16, 32, device="cuda", dtype=torch.bfloat16)
    text = torch.full((3, 32), 0.25, device="cuda", dtype=torch.bfloat16)

    def compute(weight, image, text):
        return text @ weight + image.mean(dim=0), image @ weight + text.mean(dim=0)

    expected_text, expected_image = text.clone(), image.clone()
    for block in blocks:
        expected_text, expected_image = compute(block.weight.pin_tensor.to("cuda"), expected_image, expected_text)

    seen = []
    copied = []
    copy = manager.copy_block_to_buffer

    def counted_copy(target, index, sources, adapter=None):
        copied.append(index)
        return copy(target, index, sources, adapter)

    def run_block(index, block, image, text, *args):
        seen.append(index)
        result = compute(block.weight.tensor, image, text)
        if index == len(blocks) - 1:
            torch.testing.assert_close(result[0], expected_text, atol=0, rtol=0)
        return result

    manager.copy_block_to_buffer = counted_copy
    infer.run_block = run_block
    try:
        for _ in range(2):
            actual = infer.infer_with_blocks_offload(blocks, image.clone(), text.clone(), image, text, (image, text), (None, None), None)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected_image, atol=0, rtol=0)
        assert seen == list(range(7)) * 2
        assert copied == [i for i in range(7) if i not in infer.resident_blocks.indices] * 2
    finally:
        infer.release_resident_blocks()
    assert not infer.resident_blocks.active and not infer.resident_blocks.originals


@pytest.mark.skipif(AI_DEVICE != "cuda" or not torch.cuda.is_available(), reason="CUDA lifetime test")
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("phase", ["denoise", "warmup"])
def test_runner_releases_before_decode(cancelled, phase):
    blocks, manager = make_stack("per_tensor")
    infer = object.__new__(QwenImageOffloadTransformerInfer)
    infer.offload_manager = manager
    infer.resident_blocks = ResidentBlocks(3)
    infer.resident_blocks.prepare(blocks)
    runner = object.__new__(QwenImageRunner)
    runner.config = {"cpu_offload": True, "offload_granularity": "block"}
    scheduler = SimpleNamespace(infer_steps=1, generator=None, prepare=lambda inputs: None, step_pre=lambda step_index: None, step_post=lambda: None, latents="latents")

    def denoise(inputs):
        assert infer.resident_blocks.active
        if cancelled:
            raise RuntimeError("cancelled")

    runner.model = SimpleNamespace(transformer_infer=infer, scheduler=scheduler, infer=denoise)
    runner.inputs = {}
    runner.progress_callback = None
    if phase == "denoise":
        run = runner.run
    else:
        runner._WARMUP_RESOLUTIONS = [(256, 256)]
        runner.input_info = None
        runner._prepare_warmup_inputs = lambda *args: None
        runner.clear_warmup_state = lambda: None

        def decode(latents):
            assert latents == "latents"
            assert not infer.resident_blocks.active

        runner.run_vae_decoder = decode
        run = runner._run_warmup
    if cancelled:
        with pytest.raises(RuntimeError, match="cancelled"):
            run()
    else:
        run()
    assert not infer.resident_blocks.active and not infer.resident_blocks.originals
