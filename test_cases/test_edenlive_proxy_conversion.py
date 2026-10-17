"""ProxyDINO quantization precision, checkpoint compatibility and safe writes."""

import hashlib

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tools.convert.convert_edenlive_proxy import PREFIX, _atomic_checkpoint, convert_proxy_dino, is_proxy_linear_weight


def _source_state():
    torch.manual_seed(20261016)
    state = {}
    for name, rows, columns in [
        *(("backbone.layer.0.attention." + projection + "_proj", 256, 256) for projection in ("q", "k", "v", "o")),
        ("backbone.layer.0.mlp.up_proj", 512, 256),
        ("backbone.layer.0.mlp.down_proj", 256, 512),
        ("patch_projection", 128, 256),
    ]:
        state[PREFIX + name + ".weight"] = torch.randn(rows, columns, dtype=torch.float32) * 0.02
        if not name.endswith("k_proj"):
            state[PREFIX + name + ".bias"] = torch.randn(rows, dtype=torch.float32) * 0.01
    state[PREFIX + "backbone.embeddings.patch_embeddings.weight"] = torch.randn(256, 3, 2, 2)
    state[PREFIX + "backbone.embeddings.patch_embeddings.bias"] = torch.randn(256)
    state[PREFIX + "backbone.layer.0.norm1.weight"] = torch.randn(256)
    state[PREFIX + "backbone.layer.0.norm1.bias"] = torch.randn(256)
    state[PREFIX + "backbone.layer.0.layer_scale1.lambda1"] = torch.randn(256)
    state[PREFIX + "backbone.embeddings.cls_token"] = torch.randn(1, 1, 256)
    state[PREFIX + "backbone.embeddings.register_tokens"] = torch.randn(1, 4, 256)
    return state


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    pytest.importorskip("comfy_kitchen")
    root = tmp_path_factory.mktemp("proxy_convrot")
    source = root / "source.safetensors"
    output = root / "converted.safetensors"
    state = _source_state()
    save_file(state, str(source), metadata={"fixture": "preserved"})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = convert_proxy_dino(source, output, device="cpu")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    return root, source, output, state, report


def test_converter_preserves_non_linears_and_rounds_only_linear_biases(converted):
    _, _, output, original, report = converted
    actual = load_file(str(output))
    linear_names = {name for name in original if is_proxy_linear_weight(name)}
    bias_names = {name.removesuffix(".weight") + ".bias" for name in linear_names}
    assert report["linear_count"] == 7 and report["output_tensor_count"] == len(original) + 14
    assert 0 <= report["relative_rmse"] < 0.1
    for name, value in original.items():
        if name in linear_names:
            stem = name.removesuffix(".weight")
            assert actual[name].dtype == torch.int8 and actual[name].shape == value.shape
            assert actual[stem + ".weight_scale"].dtype == torch.float32
            assert actual[stem + ".weight_scale"].shape == (value.shape[0], 1)
            assert actual[stem + ".convrot_groupsize"].dtype == torch.int32
            assert actual[stem + ".convrot_groupsize"].item() == 256
        elif name in bias_names:
            assert actual[name].dtype == torch.bfloat16
            assert torch.equal(actual[name], value.to(torch.bfloat16))
        else:
            assert actual[name].dtype == value.dtype and torch.equal(actual[name], value)
    with safe_open(str(output), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["fixture"] == "preserved"
    assert metadata["quant_scheme"] == "int8-convrot"
    assert metadata["effective_linear_dtype"] == "torch.bfloat16"
    assert metadata["source_sha256"] == report["source_sha256"]


def test_converter_quantizes_effective_bf16_weights_using_official_layout(converted):
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    _, _, output, original, _ = converted
    actual = load_file(str(output))
    for name, tensor in original.items():
        if is_proxy_linear_weight(name):
            expected, params = TensorWiseINT8Layout.quantize(tensor.to(torch.bfloat16), is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256, stochastic_rounding=0)
            assert torch.equal(actual[name], expected)
            assert torch.equal(actual[name.removesuffix(".weight") + ".weight_scale"], params.scale.float().reshape(tensor.shape[0], 1))


def test_source_and_existing_outputs_are_never_overwritten(converted):
    root, source, output, _, _ = converted
    before = source.read_bytes()
    with pytest.raises(ValueError, match="distinct"):
        convert_proxy_dino(source, source, overwrite=True)
    with pytest.raises(FileExistsError, match="Refusing"):
        convert_proxy_dino(source, output)
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="group_size"):
        convert_proxy_dino(source, root / "invalid.safetensors", group_size=255)


def test_missing_linear_is_rejected_before_output_is_written(tmp_path):
    state = _source_state()
    del state[PREFIX + "backbone.layer.0.attention.k_proj.weight"]
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    save_file(state, str(source))
    with pytest.raises(ValueError, match="missing"):
        convert_proxy_dino(source, output)
    assert not output.exists() and not list(tmp_path.glob("*.tmp"))


def test_atomic_publication_does_not_overwrite_a_concurrent_writer(tmp_path):
    output = tmp_path / "checkpoint.safetensors"
    output.write_bytes(b"other writer")
    with pytest.raises(FileExistsError):
        _atomic_checkpoint({"x": torch.ones(2)}, output, {}, overwrite=False)
    assert output.read_bytes() == b"other writer"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA native ConvRot loader required")
def test_cuda_converted_checkpoint_loads_original_convrot_mm(converted):
    import comfy_kitchen

    from lightx2v.common.ops.mm.mm_weight import MMWeightWint8ConvRot

    root, source, _, _, _ = converted
    output = root / "cuda.safetensors"
    report = convert_proxy_dino(source, output, device="cuda")
    assert report["linear_count"] == 7
    state = {name.removeprefix(PREFIX): tensor.cuda() for name, tensor in load_file(str(output)).items()}
    module = MMWeightWint8ConvRot("backbone.layer.0.attention.q_proj.weight", "backbone.layer.0.attention.q_proj.bias")
    module.load(state)
    module.post_process()
    x = torch.randn(9, 256, device="cuda", dtype=torch.bfloat16)
    actual = module.apply(x)
    expected = comfy_kitchen.int8_linear(x, module.weight, module.weight_scale, module.bias, out_dtype=module.infer_dtype, convrot=True, convrot_groupsize=256)
    assert module.weight.dtype == torch.int8 and module._convrot_groupsize == 256
    assert torch.equal(actual, expected) and torch.isfinite(actual).all()
