import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).parents[3]
_SPEC = importlib.util.spec_from_file_location("minimax_h3_weights_under_test", REPO_ROOT / "lightx2v/models/video_encoders/hf/minimax_h3/weights.py")
_WEIGHTS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _WEIGHTS
_SPEC.loader.exec_module(_WEIGHTS)
_is_official_video_vae_checkpoint = _WEIGHTS._is_official_video_vae_checkpoint
load_minimax_h3_video_vae_checkpoint = _WEIGHTS.load_minimax_h3_video_vae_checkpoint
load_safetensors_subset = _WEIGHTS.load_safetensors_subset
validate_minimax_h3_video_vae_checkpoint = _WEIGHTS.validate_minimax_h3_video_vae_checkpoint


def _parameter_module(specs):
    class _IndexedModule(nn.Module):
        def __getitem__(self, index):
            return self._modules[str(index)]

    root = _IndexedModule()
    for key, tensor in specs.items():
        parent = root
        parts = key.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, _IndexedModule())
            parent = getattr(parent, part)
        parent.register_parameter(parts[-1], nn.Parameter(torch.empty_like(tensor, device="meta")))
    if hasattr(root, "decoder") and hasattr(root.decoder, "transformer_blocks"):
        attention = root.decoder.transformer_blocks[0].attn
        attention.heads = 2
        attention.dim_head = 2
    return root


def _official_tensors():
    qkv_weight = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    qkv_bias = torch.arange(12, dtype=torch.float32)
    w1_weight = torch.cat((torch.ones(4, 4), torch.full((4, 4), 2.0)))
    w1_bias = torch.cat((torch.ones(4), torch.full((4,), 2.0)))
    return {
        "encoder.conv_in.weight": torch.full((2, 3, 1, 1, 1), 3.0),
        "encoder.down.0.block.0.nin_shortcut.weight": torch.full((2, 2, 1, 1, 1), 4.0),
        "encoder.down.0.downsample.conv.bias": torch.full((2,), 5.0),
        "decoder.x_embedder.weight": torch.full((4, 2), 6.0),
        "decoder.transformer_blocks.0.attn.to_out.weight": torch.full((4, 4), 7.0),
        "decoder.transformer_blocks.0.attn.to_qkv.weight": qkv_weight,
        "decoder.transformer_blocks.0.attn.to_qkv.bias": qkv_bias,
        "decoder.transformer_blocks.0.ff.w1.weight": w1_weight,
        "decoder.transformer_blocks.0.ff.w1.bias": w1_bias,
        "decoder.transformer_blocks.0.ff.w2.weight": torch.full((4, 4), 8.0),
        "decoder.mask_token": torch.zeros(1, 1, 4),
    }


def _native_specs():
    tensors = _official_tensors()
    return {
        "encoder.conv_in.weight": tensors["encoder.conv_in.weight"],
        "encoder.down_blocks.0.resnets.0.conv_shortcut.weight": tensors["encoder.down.0.block.0.nin_shortcut.weight"],
        "encoder.down_blocks.0.downsamplers.0.conv.bias": tensors["encoder.down.0.downsample.conv.bias"],
        "decoder.proj_in.weight": tensors["decoder.x_embedder.weight"],
        "decoder.transformer_blocks.0.attn.to_out.0.weight": tensors["decoder.transformer_blocks.0.attn.to_out.weight"],
        "decoder.transformer_blocks.0.attn.to_q.weight": torch.empty(4, 4),
        "decoder.transformer_blocks.0.attn.to_k.weight": torch.empty(4, 4),
        "decoder.transformer_blocks.0.attn.to_v.weight": torch.empty(4, 4),
        "decoder.transformer_blocks.0.attn.to_q.bias": torch.empty(4),
        "decoder.transformer_blocks.0.attn.to_k.bias": torch.empty(4),
        "decoder.transformer_blocks.0.attn.to_v.bias": torch.empty(4),
        "decoder.transformer_blocks.0.ff.net.0.proj.weight": tensors["decoder.transformer_blocks.0.ff.w1.weight"],
        "decoder.transformer_blocks.0.ff.net.0.proj.bias": tensors["decoder.transformer_blocks.0.ff.w1.bias"],
        "decoder.transformer_blocks.0.ff.net.2.weight": tensors["decoder.transformer_blocks.0.ff.w2.weight"],
    }


def _write(path: Path, tensors=None):
    save_file(tensors or _official_tensors(), path)
    return path


def test_official_detection_uses_strict_key_signature(tmp_path):
    official = _write(tmp_path / "official.safetensors")
    legacy = tmp_path / "legacy.safetensors"
    save_file({"decoder.proj_in.weight": torch.zeros(4, 2)}, legacy)
    assert _is_official_video_vae_checkpoint(official)
    assert not _is_official_video_vae_checkpoint(legacy)


def test_official_mapping_qkv_ffn_and_mask_token(tmp_path):
    tensors = _official_tensors()
    module = _parameter_module(_native_specs())
    report = load_minimax_h3_video_vae_checkpoint(module, _write(tmp_path / "model.safetensors", tensors))
    state = module.state_dict()

    assert len(report.loaded_keys) == len(_native_specs())
    assert report.ignored_keys == 1
    assert torch.equal(state["encoder.down_blocks.0.resnets.0.conv_shortcut.weight"], tensors["encoder.down.0.block.0.nin_shortcut.weight"])
    assert torch.equal(state["decoder.proj_in.weight"], tensors["decoder.x_embedder.weight"])
    assert torch.equal(state["decoder.transformer_blocks.0.attn.to_out.0.weight"], tensors["decoder.transformer_blocks.0.attn.to_out.weight"])
    assert torch.equal(state["decoder.transformer_blocks.0.ff.net.2.weight"], tensors["decoder.transformer_blocks.0.ff.w2.weight"])
    for index, name in enumerate(("q", "k", "v")):
        assert torch.equal(state[f"decoder.transformer_blocks.0.attn.to_{name}.weight"], tensors["decoder.transformer_blocks.0.attn.to_qkv.weight"].reshape(2, 3, 2, 4)[:, index].reshape(4, 4))
        assert torch.equal(state[f"decoder.transformer_blocks.0.attn.to_{name}.bias"], tensors["decoder.transformer_blocks.0.attn.to_qkv.bias"].reshape(2, 3, 2)[:, index].reshape(4))
    assert torch.all(state["decoder.transformer_blocks.0.ff.net.0.proj.weight"][:4] == 2)
    assert torch.all(state["decoder.transformer_blocks.0.ff.net.0.proj.weight"][4:] == 1)
    assert torch.all(state["decoder.transformer_blocks.0.ff.net.0.proj.bias"][:4] == 2)
    assert torch.all(state["decoder.transformer_blocks.0.ff.net.0.proj.bias"][4:] == 1)


def test_mask_token_shape_is_validated(tmp_path):
    tensors = _official_tensors()
    tensors["decoder.mask_token"] = torch.zeros(1, 2, 4)
    with pytest.raises(RuntimeError, match="shape_mismatch.*decoder.mask_token"):
        validate_minimax_h3_video_vae_checkpoint(_parameter_module(_native_specs()), _write(tmp_path / "bad.safetensors", tensors))


def test_unexpected_official_key_fails(tmp_path):
    tensors = _official_tensors()
    tensors["decoder.surprise"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unknown.*decoder.surprise"):
        validate_minimax_h3_video_vae_checkpoint(_parameter_module(_native_specs()), _write(tmp_path / "bad.safetensors", tensors))


def test_missing_native_target_fails(tmp_path):
    specs = _native_specs()
    specs["decoder.proj_out.weight"] = torch.empty(4, 4)
    with pytest.raises(RuntimeError, match="missing.*decoder.proj_out.weight"):
        validate_minimax_h3_video_vae_checkpoint(_parameter_module(specs), _write(tmp_path / "bad.safetensors"))


def test_duplicate_target_fails(tmp_path):
    tensors = _official_tensors()
    tensors["decoder.proj_in.weight"] = tensors["decoder.x_embedder.weight"]
    with pytest.raises(RuntimeError, match="duplicate.*decoder.proj_in.weight"):
        validate_minimax_h3_video_vae_checkpoint(_parameter_module(_native_specs()), _write(tmp_path / "bad.safetensors", tensors))


def test_legacy_subset_loader_regression(tmp_path):
    expected = {"layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    module = _parameter_module(expected)
    path = tmp_path / "native.safetensors"
    save_file(expected, path)
    report = load_safetensors_subset(module, path)
    assert report.loaded_keys == ("layer.weight",)
    assert torch.equal(module.state_dict()["layer.weight"], expected["layer.weight"])


@pytest.mark.parametrize("is_weight", [False, True])
def test_qkv_per_head_components_and_reinterleave(is_weight):
    heads, head_dim = 3, 2
    rows = torch.tensor([100 * h + 10 * c + d for h in range(heads) for c in range(3) for d in range(head_dim)])
    source = rows.float()
    if is_weight:
        source = source[:, None] * 10 + torch.arange(5)
    parts = _WEIGHTS._split_video_vae_qkv(source, heads, head_dim)
    for c, part in enumerate(parts):
        expected = torch.tensor([100 * h + 10 * c + d for h in range(heads) for d in range(head_dim)]).float()
        if is_weight:
            expected = expected[:, None] * 10 + torch.arange(5)
        assert torch.equal(part, expected)
        assert not torch.equal(part, source.chunk(3)[c])
        assert part.dtype == source.dtype and part.device == source.device and part.is_contiguous()
    rebuilt = torch.stack([p.reshape(heads, head_dim, *source.shape[1:]) for p in parts], dim=1).reshape_as(source)
    assert torch.equal(rebuilt, source)


@pytest.mark.parametrize("shape,heads,dim", [((11,), 2, 2), ((12,), 3, 2), ((12,), 2, 0), ((12, 2, 2), 2, 2)])
def test_qkv_invalid_geometry_rejected(shape, heads, dim):
    with pytest.raises(ValueError, match="Video VAE fused QKV"):
        _WEIGHTS._split_video_vae_qkv(torch.empty(shape), heads, dim)


def test_loader_rejects_incompatible_attention_geometry(tmp_path):
    module = _parameter_module(_native_specs())
    module.decoder.transformer_blocks[0].attn.heads = 3
    with pytest.raises(ValueError, match="target num_heads"):
        load_minimax_h3_video_vae_checkpoint(module, _write(tmp_path / "model.safetensors"))
