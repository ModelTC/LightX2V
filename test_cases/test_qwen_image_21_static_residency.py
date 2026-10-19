"""Qwen 2.1 prefill/denoise residency, cleanup, and estimator input contracts."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from lightx2v.common.offload.resident import ResidentBlocks
from lightx2v.models.networks.qwen_image_21.infer.offload.transformer_infer import QwenImage21OffloadTransformerInfer, validate_qwen_image_21_residency_config
from lightx2v.models.networks.qwen_image_21.input_utils import prepare_condition_inputs, prepare_images
from lightx2v.models.networks.qwen_image_21.model import QwenImage21TransformerModel
from lightx2v.models.runners.qwen_image_21.qwen_image_21_runner import QwenImage21Runner
from lightx2v_platform.base.global_var import AI_DEVICE
from test_cases.test_wan_static_residency import make_stack

GPU = pytest.mark.skipif(AI_DEVICE != "cuda" or not torch.cuda.is_available(), reason="CUDA stream/lifetime test")


def config():
    return dict(model_cls="qwen_image_21", num_layers=32, causal_condition=True, cpu_offload=True, offload_granularity="block", offload_resident_blocks=0)


@pytest.mark.parametrize(
    "override",
    [
        {"cpu_offload": False},
        {"offload_granularity": "model"},
        {"offload_resident_blocks": 33},
        {"offload_resident_blocks": True},
        {"causal_condition": False},
        {"dit_quantized": True},
        {"dit_quant_scheme": "fp8-sgl"},
        {"dit_disk_streaming": True},
        {"text_encoder_disk_streaming": True},
        {"text_encoder_tensor_parallel": True},
        {"lora_configs": [{}]},
        {"enable_cfg": True},
        {"shared_cpu_weights": True},
        {"use_compile": True},
        {"feature_caching": "Tea"},
    ],
)
def test_model_rejects_unsupported_config_before_loading_weights(override):
    with pytest.raises(ValueError):
        QwenImage21TransformerModel("unused-checkpoint", {**config(), **override}, torch.device("cpu"))


def test_legacy_configs_are_unchanged():
    validate_qwen_image_21_residency_config({"dit_quantized": True, "enable_cfg": True})
    validate_qwen_image_21_residency_config(config())
    with pytest.raises(ValueError, match="LoRA"):
        QwenImage21TransformerModel("unused-checkpoint", config(), torch.device("cpu"), lora_path="adapter.safetensors")


def test_t2i_input_has_no_image_path():
    runner = object.__new__(QwenImage21Runner)
    runner.config = {"vae_scale_factor": 16, "in_channels": 64}
    runner.input_info = SimpleNamespace(task="t2i", resolution=2048, size=[1024, 1024], aspect_ratio=None)

    def encode(images, shapes):
        assert images == [] and shapes == [(1, 64, 64)]
        return {}

    runner.run_text_encoder = encode
    assert runner.run_input_encoder() == {"image_latents": None}
    assert runner.input_info.latent_shape == (1, 1, 64, 64, 64)


def test_reference_images_and_target_size(tmp_path):
    path = tmp_path / "reference.png"
    Image.new("RGBA", (200, 100), (0, 0, 0, 0)).save(path)
    images, size = prepare_images("i2i", 128, image_path=str(path))
    assert images[0].mode == "RGBA"
    assert images[0].size == (192, 96) and size == [96, 192]
    assert prepare_images("i2i", 128, [130, 100], image_path=str(path))[1] == [128, 96]
    assert prepare_images("t2i", 128, aspect_ratio="2:1")[1] == [96, 192]
    with pytest.raises(ValueError, match="image_path"):
        prepare_images("i2i", 128)


def test_processor_input_preserves_template_and_alpha():
    def processor(**kwargs):
        assert kwargs["do_resize"] is False
        assert kwargs["images"][0].getpixel((0, 0)) == (255, 255, 255)
        assert kwargs["images"][1].getpixel((0, 0)) == (0, 0, 0)
        assert kwargs["text"] == [
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n"
            "<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|>"
            "edit this<|im_end|>\n<|im_start|>assistant\n"
        ]
        return kwargs

    prepare_condition_inputs(processor, "edit this", [Image.new("RGBA", (32, 32)), Image.new("RGBA", (32, 32), (0, 0, 0, 255))])


def make_infer(manager, count):
    infer = object.__new__(QwenImage21OffloadTransformerInfer)
    infer.offload_manager = manager
    infer.resident_blocks = ResidentBlocks(count)
    return infer


@GPU
@pytest.mark.parametrize("layout", ["per_tensor", "contiguous"])
@pytest.mark.parametrize("count", [0, 1, 4, 6, 7])
def test_prefill_and_denoise_share_residents_and_logical_cache_indices(layout, count):
    blocks, manager = make_stack(layout)
    infer = make_infer(manager, count)
    x = torch.ones(16, 32, device="cuda", dtype=torch.bfloat16)
    state = SimpleNamespace(rotary=x, rotary_positions=None)
    kv, seen, copies = {}, [], []
    cache = SimpleNamespace(k_cache=lambda i: kv[i], v_cache=lambda i: kv[i])
    copy = manager.copy_block_to_buffer

    def counted_copy(target, index, sources, adapter=None):
        copies.append(index)
        return copy(target, index, sources, adapter)

    def prefill(index, block, hidden, modulation, state, cache):
        kv[index] = hidden + index
        seen.append(index)
        return hidden @ block.weight.tensor

    def denoise(index, block, hidden, modulation, rotary, positions, k, v):
        assert k is kv[index] and v is kv[index]
        seen.append(index)
        return hidden @ block.weight.tensor + k

    manager.copy_block_to_buffer = counted_copy
    infer.prefill_block, infer.run_block = prefill, denoise
    expected_prefix, expected = x.clone(), x.clone()
    for index, block in enumerate(blocks):
        weight = block.weight.pin_tensor.to("cuda")
        expected = expected @ weight + (expected_prefix + index)
        expected_prefix = expected_prefix @ weight
    for _ in range(2):
        infer._prefill_blocks(blocks, x.clone(), (x,), state, cache)
        originals = tuple(infer.resident_blocks.originals)
        for _ in range(2):
            actual = infer._infer_blocks(blocks, x.clone(), (x,), state, cache)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert tuple(infer.resident_blocks.originals) == originals
        infer.release_resident_blocks()
        infer.release_resident_blocks()
        assert not infer.resident_blocks.active and not infer.resident_blocks.originals
        kv.clear()
    assert seen == list(range(7)) * 6
    assert copies == [i for i in range(7) if i not in infer.resident_blocks.indices] * 6
    if manager.contiguous_transfer is not None:
        manager.contiguous_transfer.close()


@GPU
@pytest.mark.parametrize("phase", ["prefill", "denoise"])
def test_failed_block_releases_and_can_restart(phase):
    blocks, manager = make_stack("per_tensor")
    infer = make_infer(manager, 3)
    x = torch.ones(16, 32, device="cuda", dtype=torch.bfloat16)
    state = SimpleNamespace(rotary=x, rotary_positions=None)
    cache = SimpleNamespace(k_cache=lambda i: x, v_cache=lambda i: x)

    def fail(index, block, hidden, *args):
        if index == 2:
            raise RuntimeError("cancelled")
        return hidden @ block.weight.tensor

    infer.prefill_block = infer.run_block = fail
    run = infer._prefill_blocks if phase == "prefill" else infer._infer_blocks
    with pytest.raises(RuntimeError, match="cancelled"):
        run(blocks, x, (x,), state, cache)
    assert not infer.resident_blocks.active and not any(manager._slot_pending)
    infer.prefill_block = infer.run_block = lambda index, block, hidden, *args: hidden @ block.weight.tensor
    run(blocks, x, (x,), state, cache)
    infer.release_resident_blocks()


@GPU
@pytest.mark.parametrize("failure", [None, "prefill", "denoise"])
def test_runner_releases_before_kv_and_decode(failure):
    blocks, manager = make_stack("per_tensor")
    infer = make_infer(manager, 3)
    runner = object.__new__(QwenImage21Runner)
    runner.config = {"cpu_offload": True}
    runner.inputs = {}
    runner.init_run = lambda: None
    runner.maybe_empty_cache = lambda **kwargs: None

    def prefill(inputs):
        infer.resident_blocks.prepare(blocks)
        if failure == "prefill":
            raise RuntimeError("cancelled")

    def denoise():
        assert infer.resident_blocks.active
        if failure == "denoise":
            raise RuntimeError("cancelled")
        return "latents"

    def clear_kv():
        assert not infer.resident_blocks.active

    weight = SimpleNamespace(to_cpu=lambda: None)
    runner.model = SimpleNamespace(transformer_infer=infer, block_offload=True, pre_weight=weight, post_weight=weight, prefill_condition_kv=prefill, clear_condition_kv=clear_kv)
    runner.run_segment = denoise
    if failure:
        with pytest.raises(RuntimeError, match="cancelled"):
            runner.run_main()
    else:
        assert runner.run_main() == "latents"
    assert not infer.resident_blocks.active


@GPU
@pytest.mark.parametrize("cancelled", [False, True])
def test_warmup_releases_between_tasks_and_on_failure(cancelled):
    blocks, manager = make_stack("per_tensor")
    infer = make_infer(manager, 3)
    runner = object.__new__(QwenImage21Runner)
    runner.config = {"cpu_offload": True}
    runner._WARMUP_RESOLUTIONS = ((32, 32),)
    runner.create_input_info = lambda request: SimpleNamespace(**request)
    runner.run_input_encoder = lambda: {}
    runner.init_run = lambda: None
    runner.maybe_empty_cache = lambda **kwargs: None
    runner.scheduler = SimpleNamespace(generator=None, latents=None, step_pre=lambda **kwargs: None, step_post=lambda: None)
    finished = []

    def denoise(inputs):
        assert infer.resident_blocks.active
        if cancelled:
            raise RuntimeError("cancelled")

    def released(*args):
        assert not infer.resident_blocks.active

    def end_run():
        released()
        finished.append(runner.input_info.task)

    weight = SimpleNamespace(to_cpu=lambda: None)
    runner.model = SimpleNamespace(
        transformer_infer=infer,
        block_offload=True,
        pre_weight=weight,
        post_weight=weight,
        prefill_condition_kv=lambda inputs: infer.resident_blocks.prepare(blocks),
        infer=denoise,
        clear_condition_kv=released,
    )
    runner.run_vae_decoder = released
    runner.end_run = end_run
    if cancelled:
        with pytest.raises(RuntimeError, match="cancelled"):
            runner._run_warmup()
    else:
        runner._run_warmup()
    assert finished == (["t2i"] if cancelled else ["t2i", "i2i"])
