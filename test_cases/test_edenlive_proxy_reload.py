"""Tiny real ProxyDINO forwards prove reload invalidation and precision policy."""

import gc
import json
import weakref

import pytest
import torch
from safetensors.torch import load_file, save_file

import lightx2v.common.ops  # noqa: F401
from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO
from tools.convert.convert_edenlive_proxy import PREFIX, convert_proxy_dino

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ProxyDINO reload integration requires CUDA")


def _state(seed):
    generator = torch.Generator().manual_seed(seed)
    hidden = 256
    state = {}

    def random(shape, scale=0.02):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    def linear(name, rows, columns, bias=True):
        state[PREFIX + name + ".weight"] = random((rows, columns))
        if bias:
            state[PREFIX + name + ".bias"] = random((rows,), 0.01)

    state[PREFIX + "backbone.embeddings.patch_embeddings.weight"] = random((hidden, 3, 16, 16))
    state[PREFIX + "backbone.embeddings.patch_embeddings.bias"] = random((hidden,), 0.01)
    state[PREFIX + "backbone.embeddings.cls_token"] = random((1, 1, hidden))
    state[PREFIX + "backbone.embeddings.register_tokens"] = random((1, 2, hidden))
    for norm in ("backbone.layer.0.norm1", "backbone.layer.0.norm2", "backbone.norm"):
        state[PREFIX + norm + ".weight"] = torch.ones(hidden) + random((hidden,), 0.01)
        state[PREFIX + norm + ".bias"] = random((hidden,), 0.01)
    for projection in ("q", "k", "v", "o"):
        linear("backbone.layer.0.attention." + projection + "_proj", hidden, hidden, bias=projection != "k")
    linear("backbone.layer.0.mlp.up_proj", 512, hidden)
    linear("backbone.layer.0.mlp.down_proj", hidden, 512)
    linear("patch_projection", 128, hidden)
    for scale in ("layer_scale1", "layer_scale2"):
        state[PREFIX + "backbone.layer.0." + scale + ".lambda1"] = torch.full((hidden,), 0.1) + random((hidden,), 0.01)
    return state


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    pytest.importorskip("comfy_kitchen")
    root = tmp_path_factory.mktemp("proxy_reload")
    config = dict(hidden_size=256, num_attention_heads=4, num_register_tokens=2, num_hidden_layers=1, layer_norm_eps=1e-5, patch_size=16, rope_theta=100.0)
    models = []
    for index in range(2):
        model = root / f"model_{index}"
        backbone = model / "vision-encoder/backbone"
        backbone.mkdir(parents=True)
        (backbone / "config.json").write_text(json.dumps(config))
        original = model / "vision-encoder/projection.safetensors"
        quantized = model / "proxy-int8-convrot.safetensors"
        save_file(_state(42 + index), str(original))
        convert_proxy_dino(original, quantized, device="cpu")
        models.append((model, original, quantized))
    return tuple(models)


def _gpu_state(path):
    return {name.removeprefix(PREFIX): tensor.cuda() for name, tensor in load_file(str(path)).items()}


def _encoder(checkpoint, quantized, residual_dtype, share=False, graph=False):
    model, _, quantized_path = checkpoint
    return ProxyDINO(str(model), "bilinear", use_cuda_graph=graph, quantized_ckpt=str(quantized_path) if quantized else None, residual_dtype=residual_dtype, share_qkv_quant=share)


def _linears(encoder):
    return [encoder.projection, *(module for block in encoder.blocks for module in (block.q_proj, block.k_proj, block.v_proj, block.o_proj, block.up_proj, block.down_proj))]


def _assert_dtype_policy(encoder, quantized, residual_dtype):
    assert encoder.patch.weight.dtype == torch.bfloat16
    assert encoder.patch.bias.dtype == torch.bfloat16
    for module in _linears(encoder):
        assert module.weight.dtype == (torch.int8 if quantized else torch.bfloat16)
        if module.bias is not None:
            assert module.bias.dtype == torch.bfloat16
        if quantized:
            assert module.weight_scale.dtype == torch.float32
            assert module._convrot_groupsize == 256
    dtype = getattr(torch, residual_dtype)
    for token in (encoder.cls_token, encoder.register_tokens):
        assert token.tensor.dtype == dtype
    for norm in (encoder.final_norm, *(norm for block in encoder.blocks for norm in (block.norm1, block.norm2))):
        assert norm.weight.dtype == dtype and norm.bias.dtype == dtype
    for block in encoder.blocks:
        assert block.layer_scale1.tensor.dtype == dtype and block.layer_scale2.tensor.dtype == dtype


def _images():
    generator = torch.Generator(device="cuda").manual_seed(20261016)
    return torch.randint(0, 256, (1, 3, 32, 48), generator=generator, device="cuda", dtype=torch.uint8)


@pytest.mark.parametrize("quantized,residual_dtype,share", [(False, "float32", False), (False, "bfloat16", False), (True, "float32", False), (True, "bfloat16", True)])
def test_reload_matches_fresh_encoder_and_discards_old_capture(checkpoints, quantized, residual_dtype, share):
    if share and torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 shared ConvRot group required")
    before, after = checkpoints
    encoder = _encoder(before, quantized, residual_dtype, share, graph=True)
    pixels = _images()
    for _ in range(4):
        retained = encoder.infer(pixels)
    retained_copy = retained.clone()
    entry = next(iter(encoder._cuda_graph._entries.values()))
    assert entry.graph is not None
    old_entry = weakref.ref(entry)
    del entry
    old_weight = weakref.ref(encoder.blocks[0].q_proj.weight)
    old_group = weakref.ref(encoder.blocks[0].qkv_group) if share else None
    old_bias = weakref.ref(encoder.blocks[0].k_proj.bias) if share else None
    if share:
        # A stale synthetic K bias must not survive the new checkpoint.
        encoder.blocks[0].k_proj.bias.fill_(7)
    encoder.load(_gpu_state(after[2 if quantized else 1]))
    assert not encoder._cuda_graph._entries
    gc.collect()
    assert old_entry() is None and old_weight() is None
    if share:
        assert old_group() is None and old_bias() is None
        assert encoder.blocks[0].qkv_group is not None
        assert torch.count_nonzero(encoder.blocks[0].k_proj.bias) == 0
    _assert_dtype_policy(encoder, quantized, residual_dtype)
    fresh = _encoder(after, quantized, residual_dtype, share, graph=False)
    expected = fresh.infer(pixels)
    for _ in range(4):
        actual = encoder.infer(pixels)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert next(iter(encoder._cuda_graph._entries.values())).graph is not None
    assert not torch.equal(actual, retained_copy)
    assert torch.equal(retained.view(torch.uint8), retained_copy.view(torch.uint8))
    encoder._cuda_graph.clear()


def test_reload_removes_synthetic_k_bias_when_shared_quant_is_disabled(checkpoints):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 shared ConvRot group required")
    before, after = checkpoints
    encoder = _encoder(before, True, "bfloat16", share=True)
    block = encoder.blocks[0]
    assert block.qkv_group is not None and block.k_proj.bias_name is None
    old_group = weakref.ref(block.qkv_group)
    old_bias = weakref.ref(block.k_proj.bias)
    block.k_proj.bias.fill_(7)
    encoder.share_qkv_quant = False
    encoder.load(_gpu_state(after[2]))
    gc.collect()
    assert old_group() is None and old_bias() is None
    assert block.qkv_group is None and block.k_proj.bias is None
    fresh = _encoder(after, True, "bfloat16", share=False)
    pixels = _images()
    assert torch.equal(encoder.infer(pixels), fresh.infer(pixels))


def test_rebuilt_shared_projection_matches_new_unshared_checkpoint(checkpoints):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 shared ConvRot group required")
    before, after = checkpoints
    encoder = _encoder(before, True, "float32", share=True)
    encoder.load(_gpu_state(after[2]))
    reference = _encoder(after, True, "float32", share=False)
    generator = torch.Generator(device="cuda").manual_seed(20261016)
    normalized = torch.randn((19, 256), device="cuda", dtype=torch.bfloat16, generator=generator)
    actual = encoder.blocks[0].qkv_group.apply(normalized)
    expected = tuple(module.apply(normalized) for module in (reference.blocks[0].q_proj, reference.blocks[0].k_proj, reference.blocks[0].v_proj))
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(actual, expected))
    pixels = _images()
    assert torch.equal(encoder.infer(pixels), reference.infer(pixels))
