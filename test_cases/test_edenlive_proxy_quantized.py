"""ProxyDINO converted checkpoint stays native INT8 through eager and graph inference."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import lightx2v.common.ops  # noqa: F401
from lightx2v.common.ops.mm.mm_weight import MMWeightWint8ConvRot
from lightx2v.models.input_encoders.edenlive import proxy_dino as module
from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO


def _root():
    return Path(os.environ.get("EDENLIVE_PROXY_TEST_MODEL", Path(__file__).resolve().parents[2] / "EdenLive-x2v"))


def _checkpoint():
    return Path(os.environ.get("EDENLIVE_PROXY_INT8_TEST_MODEL", _root() / "quantized/proxy-dino-int8-convrot.safetensors"))


@pytest.fixture(scope="module", params=["float32", "bfloat16"])
def encoder(request):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Native ProxyDINO INT8 shared projections require SM120")
    if not _checkpoint().is_file():
        pytest.skip("Set EDENLIVE_PROXY_INT8_TEST_MODEL to a converted checkpoint")
    result = ProxyDINO(str(_root()), "bilinear", use_cuda_graph=True, attention_type="sage_attn2", quantized_ckpt=str(_checkpoint()), residual_dtype=request.param, share_qkv_quant=True)
    yield result
    result._cuda_graph.clear()


def test_all_linears_remain_native_int8_with_resident_groups(encoder):
    modules = [encoder.projection, *(m for block in encoder.blocks for m in (block.q_proj, block.k_proj, block.v_proj, block.o_proj, block.up_proj, block.down_proj))]
    assert len(modules) == 145
    assert all(type(m) is MMWeightWint8ConvRot and m.weight.dtype == torch.int8 and m.weight_scale.dtype == torch.float32 and m._convrot_groupsize == 256 for m in modules)
    assert all(block.qkv_group is not None for block in encoder.blocks)
    assert all(torch.count_nonzero(block.k_proj.bias) == 0 for block in encoder.blocks)
    assert encoder.patch.weight.dtype == torch.bfloat16
    assert encoder.cls_token.tensor.dtype == encoder.residual_dtype
    assert encoder.register_tokens.tensor.dtype == encoder.residual_dtype
    assert all(block.norm1.weight.dtype == encoder.residual_dtype and block.layer_scale1.tensor.dtype == encoder.residual_dtype for block in encoder.blocks)


@pytest.mark.parametrize("fused", [False, True])
def test_quantized_eager_graph_and_tail_inputs_match_without_sdpa(encoder, monkeypatch, fused):
    encoder._cuda_graph.clear()
    encoder.fuse_ops = fused

    def forbidden_sdpa(*args, **kwargs):
        pytest.fail("Sage2 proxy must not call torch SDPA")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden_sdpa)
    generator = torch.Generator(device="cuda").manual_seed(20261016)
    for batch in (3, 1):
        images = torch.randint(0, 256, (batch, 3, 704, 1280), device="cuda", dtype=torch.uint8, generator=generator)
        expected = encoder._infer_eager(images)
        for _ in range(4):
            actual = encoder.infer(images)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert actual.shape == (1, 1024, batch, 30, 52)
        assert actual.dtype == encoder.residual_dtype
        assert torch.isfinite(actual).all()
        changed = images.bitwise_xor(255)
        assert torch.equal(encoder.infer(changed).view(torch.uint8), encoder._infer_eager(changed).view(torch.uint8))
    assert len(encoder._cuda_graph._entries) == 2
    assert all(entry.graph is not None for entry in encoder._cuda_graph._entries.values())


def test_quantized_reload_drops_graphs_and_rebuilds_current_groups(encoder):
    pixels = torch.zeros(1, 3, 704, 1280, device="cuda", dtype=torch.uint8)
    original = encoder._infer_eager(pixels)
    for _ in range(3):
        encoder.infer(pixels)
    assert encoder._cuda_graph._entries
    state = {key.removeprefix("model.module.encoder."): value for key, value in load_file(str(_checkpoint()), device="cuda").items()}
    encoder.load(state)
    assert not encoder._cuda_graph._entries
    assert all(block.qkv_group is not None for block in encoder.blocks)
    modules = [encoder.projection, *(m for block in encoder.blocks for m in (block.q_proj, block.k_proj, block.v_proj, block.o_proj, block.up_proj, block.down_proj))]
    assert all(module.weight.dtype == torch.int8 for module in modules)
    assert torch.equal(original.view(torch.uint8), encoder._infer_eager(pixels).view(torch.uint8))


@pytest.mark.parametrize("weight_dtype,with_group", [(torch.bfloat16, True), (torch.int8, False)])
def test_invalid_quantized_checkpoint_is_rejected_before_module_load(tmp_path, monkeypatch, weight_dtype, with_group):
    config = tmp_path / "vision-encoder/backbone/config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(dict(hidden_size=1024, num_attention_heads=16, num_register_tokens=4)))
    checkpoint = tmp_path / "invalid.safetensors"
    weights = {"patch_projection.weight": torch.zeros(16, 16, dtype=weight_dtype)}
    if with_group:
        weights["patch_projection.convrot_groupsize"] = torch.tensor(256)
    save_file(weights, str(checkpoint))
    monkeypatch.setattr(module, "AI_DEVICE", "cpu")
    with pytest.raises(ValueError, match="converted INT8 ConvRot"):
        ProxyDINO(str(tmp_path), "bilinear", quantized_ckpt=str(checkpoint))


def test_invalid_residual_dtype_is_rejected_without_loading_checkpoint(tmp_path):
    config = tmp_path / "vision-encoder/backbone/config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    with pytest.raises(ValueError, match="residual_dtype"):
        ProxyDINO(str(tmp_path), "bilinear", residual_dtype="float16")
