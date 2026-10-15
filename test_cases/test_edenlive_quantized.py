from copy import deepcopy

import pytest
import torch
from safetensors.torch import load_file, save_file

from lightx2v.models.networks.wan.edenlive_model import EdenLiveModel
from lightx2v.models.schedulers.wan.edenlive.scheduler import EdenLiveScheduler
from lightx2v.utils.set_config import get_default_config
from tools.convert.convert_edenlive import quantize_dit

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Native quantized MM requires CUDA")


@pytest.fixture(scope="module", params=[("fp8", "fp8-sgl", torch.float8_e4m3fn), ("int8-convrot", "int8-convrot", torch.int8)])
def checkpoint(request, tmp_path_factory):
    linear_type, scheme, dtype = request.param
    if linear_type == "int8-convrot":
        pytest.importorskip("comfy_kitchen")
    directory = tmp_path_factory.mktemp("edenlive_quantized")
    source = directory / "source.safetensors"
    torch.manual_seed(42)
    weights = {}

    def weight(name, shape, ones=False):
        weights[name] = torch.ones(shape, dtype=torch.bfloat16) if ones else torch.randn(shape, dtype=torch.bfloat16) * 0.02

    def linear(name, out_features, in_features):
        weight(name + ".weight", (out_features, in_features))
        weight(name + ".bias", (out_features,))

    weight("patch_embedding.weight", (256, 1072, 1, 2, 2))
    weight("patch_embedding.bias", (256,))
    for name, rows, cols in [
        ("text_embedding.0", 256, 4096),
        ("text_embedding.2", 256, 256),
        ("time_embedding.0", 256, 256),
        ("time_embedding.2", 256, 256),
        ("time_projection.1", 1536, 256),
        ("head.head", 192, 256),
    ]:
        linear(name, rows, cols)
    weight("head.modulation", (1, 2, 256))
    for layer in range(2):
        prefix = f"blocks.{layer}."
        weight(prefix + "modulation", (1, 6, 256))
        weight(prefix + "norm3.weight", (256,), ones=True)
        weight(prefix + "norm3.bias", (256,))
        for attention in ("self_attn", "cross_attn"):
            for projection in ("q", "k", "v", "o"):
                linear(prefix + attention + "." + projection, 256, 256)
            for norm in ("norm_q", "norm_k"):
                weight(prefix + attention + "." + norm + ".weight", (256,), ones=True)
        linear(prefix + "ffn.0", 512, 256)
        linear(prefix + "ffn.2", 256, 512)
    save_file(weights, str(source))
    output = directory / "quantized"
    output.mkdir()
    quantize_dit(source, output, linear_type, "tiny", "cuda", False)
    quantized = output / "tiny.safetensors"
    assert list(output.glob("*.safetensors")) == [quantized]
    state = load_file(str(quantized))
    assert state["blocks.0.self_attn.q.weight"].dtype == dtype
    assert "blocks.0.self_attn.q.weight_scale" in state
    assert state["head.head.weight"].dtype == torch.bfloat16
    if linear_type == "int8-convrot":
        assert state["blocks.0.self_attn.q.convrot_groupsize"].item() == 256
    return directory, source, quantized, scheme, dtype


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("compile_model", [False, True])
def test_single_file_quantized_model_and_block_offload(checkpoint, offload, compile_model):
    directory, source, quantized, scheme, dtype = checkpoint
    config = get_default_config()
    config.update(
        dict(
            model_cls="wan22_edenlive",
            task="i2v",
            model_path=str(directory),
            dit_original_ckpt=str(source),
            dim=256,
            num_heads=2,
            num_layers=2,
            ffn_dim=512,
            freq_dim=256,
            text_dim=4096,
            text_len=512,
            in_dim=1072,
            out_dim=48,
            num_channels_latents=48,
            infer_steps=4,
            sample_shift=5.0,
            use_image_encoder=False,
            proxy_condition_rmsnorm=False,
            proxy_unconditional_control="same",
            self_attn_1_type="flash_attn3",
            cross_attn_1_type="flash_attn3",
            cross_attn_2_type="flash_attn3",
            layer_norm_type="Triton",
            rms_norm_type="sgl-kernel",
            rope_type="flashinfer_rope",
            modulate_type="triton",
            mxfp8_fuse_enable=False,
            cpu_offload=offload,
            offload_granularity="block",
        )
    )
    torch.manual_seed(123)
    clean = torch.randn(48, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    inputs = dict(
        image_encoder_output={"clean_first_frame": clean},
        proxy_features=torch.randn(1, 1024, 3, 8, 8, device="cuda", dtype=torch.bfloat16),
        text_encoder_output={"context": torch.randn(1, 512, 4096, device="cuda", dtype=torch.bfloat16)},
    )

    def predict(settings):
        model = EdenLiveModel(str(directory), settings, torch.device("cpu" if offload else "cuda"))
        scheduler = EdenLiveScheduler(settings)
        model.set_scheduler(scheduler)
        scheduler.prepare(42, [48, 3, 8, 8], inputs["image_encoder_output"])
        scheduler.step_pre(0)
        model.infer(inputs)
        prediction = scheduler.noise_pred.clone()
        torch.cuda.synchronize()
        if settings.get("dit_quantized") and offload:
            layer = model.transformer_weights.blocks[0].compute_phases[0].self_attn_q
            assert layer.pin_weight.dtype == dtype and layer.pin_weight.is_pinned()
        return prediction

    with torch.no_grad():
        expected = predict(deepcopy(config))
        quantized_config = deepcopy(config)
        quantized_config.update(dict(dit_quantized=True, dit_quant_scheme=scheme, dit_quantized_ckpt=str(quantized), use_compile=compile_model))
        actual = predict(quantized_config)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.15, rtol=0)
