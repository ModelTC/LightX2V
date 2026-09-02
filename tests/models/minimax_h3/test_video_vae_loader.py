import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture()
def video_vae_module(monkeypatch):
    for package_name in [
        "lightx2v",
        "lightx2v.models",
        "lightx2v.models.video_encoders",
        "lightx2v.models.video_encoders.hf",
        "lightx2v.models.video_encoders.hf.minimax_h3",
        "lightx2v.utils",
        "lightx2v_platform",
        "lightx2v_platform.base",
    ]:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    _install_module(
        monkeypatch,
        "lightx2v.models.video_encoders.hf.minimax_h3.weights",
        SafetensorsSubsetReport=object,
        load_safetensors_subset=lambda *_args, **_kwargs: None,
    )
    _install_module(monkeypatch, "lightx2v.utils.registry_factory", ATTN_WEIGHT_REGISTER={})
    _install_module(monkeypatch, "lightx2v_platform.base.global_var", AI_DEVICE="cpu")

    module_path = REPO_ROOT / "lightx2v/models/video_encoders/hf/minimax_h3/video_vae.py"
    spec = importlib.util.spec_from_file_location("minimax_h3_video_vae_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(data), encoding="utf-8")


def _wrapper_config():
    return {
        "source_path": "source",
        "source_safetensors_path": "model.safetensors",
        "latent_channels": 24,
        "latents_mean": [0.1] * 24,
        "latents_std": [1.1] * 24,
        "vae_clip_length": 17,
        "vae_token_drop": 3,
    }


def _source_config():
    return {
        "in_channels": 3,
        "out_ch": 3,
        "z_channels": 24,
        "ch": 8,
        "ch_mult": [1, 2, 4],
        "num_res_blocks": 2,
        "space_down": [2, 2, 1],
        "time_down": [1, 2, 1],
        "padding_mode": "reflect",
        "vit_decoder_kwargs": {
            "num_layers": 12,
            "heads": 4,
            "dim_head": 16,
            "rope_theta": 100.0,
            "rope_dim_ratio": 0.75,
        },
    }


def test_resolves_official_video_vae_layout_before_legacy(video_vae_module, tmp_path):
    official = tmp_path / "video_vae"
    legacy = tmp_path / "vae"
    official.mkdir()
    legacy.mkdir()

    assert video_vae_module._resolve_video_vae_dir(tmp_path) == official


def test_resolves_legacy_vae_layout(video_vae_module, tmp_path):
    legacy = tmp_path / "vae"
    legacy.mkdir()

    assert video_vae_module._resolve_video_vae_dir(tmp_path) == legacy


@pytest.mark.parametrize("name", ["video_vae", "vae"])
def test_resolves_direct_component_path(video_vae_module, tmp_path, name):
    component = tmp_path / name
    component.mkdir()

    assert video_vae_module._resolve_video_vae_dir(component) == component


def test_missing_video_vae_and_vae_reports_tried_paths(video_vae_module, tmp_path):
    with pytest.raises(FileNotFoundError, match="video_vae.*vae"):
        video_vae_module._resolve_video_vae_dir(tmp_path)


def test_official_layout_uses_source_safetensors_by_default(video_vae_module, tmp_path):
    vae_dir = tmp_path / "video_vae"
    _write_json(vae_dir / "config.json", _wrapper_config())
    _write_json(vae_dir / "source/config.json", _source_config())
    (vae_dir / "source/model.safetensors").write_bytes(b"")

    config, weight_path = video_vae_module._load_video_vae_config_and_weight_path(vae_dir, None)

    assert weight_path == vae_dir / "source/model.safetensors"
    assert config["block_out_channels"] == [8, 16, 32]


def test_explicit_checkpoint_path_wins_for_official_layout(video_vae_module, tmp_path):
    vae_dir = tmp_path / "video_vae"
    explicit = tmp_path / "quantized.safetensors"
    _write_json(vae_dir / "config.json", _wrapper_config())
    _write_json(vae_dir / "source/config.json", _source_config())
    (vae_dir / "source/model.safetensors").write_bytes(b"")

    _config, weight_path = video_vae_module._load_video_vae_config_and_weight_path(vae_dir, explicit)

    assert weight_path == explicit


def test_official_config_normalization_maps_wrapper_and_source_fields(video_vae_module):
    config = video_vae_module._normalize_official_video_vae_config(_wrapper_config(), _source_config())

    assert config["in_channels"] == 3
    assert config["out_channels"] == 3
    assert config["latent_channels"] == 24
    assert config["block_out_channels"] == [8, 16, 32]
    assert config["layers_per_block"] == 2
    assert config["spatial_downsample_factors"] == [2, 2, 1]
    assert config["temporal_downsample_factors"] == [1, 2, 1]
    assert config["spatial_padding_mode"] == "reflect"
    assert config["decoder_num_layers"] == 12
    assert config["decoder_num_attention_heads"] == 4
    assert config["decoder_attention_head_dim"] == 16
    assert config["decoder_rope_theta"] == 100.0
    assert config["decoder_rope_dim_ratio"] == 0.75
    assert config["clip_length"] == 17
    assert config["token_drop"] == 3
    assert config["latents_mean"] == [0.1] * 24
    assert config["latents_std"] == [1.1] * 24


def test_legacy_config_normalization_keeps_existing_fields(video_vae_module):
    legacy = {
        "latent_channels": 8,
        "block_out_channels": [4, 8],
        "clip_length": 9,
        "token_drop": 1,
        "custom": "kept",
    }

    assert video_vae_module._normalize_official_video_vae_config(legacy, None) == legacy


def test_legacy_layout_uses_component_dir_as_weight_path(video_vae_module, tmp_path):
    vae_dir = tmp_path / "vae"
    legacy = {"latent_channels": 8, "block_out_channels": [4, 8]}
    _write_json(vae_dir / "config.json", legacy)
    (vae_dir / "model.safetensors").write_bytes(b"")

    config, weight_path = video_vae_module._load_video_vae_config_and_weight_path(vae_dir, None)

    assert config == legacy
    assert weight_path == vae_dir
