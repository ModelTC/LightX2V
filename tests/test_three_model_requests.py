"""Exercise the new models through the shared metadata and CLI entry points."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lightx2v.models.runners.registry import get_model_config_loader, get_model_spec
from lightx2v.utils.cli import parse_cli_inputs

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("hymotion", "t2m", "hymotion_t2m_fp32"), ("sdxl", "t2i", "sdxl_t2i_fp16"), ("sf3d", "i23d", "sf3d_i23d_fp32")]


@pytest.mark.parametrize("model,task,filename", MODELS)
def test_cli_uses_nested_defaults_and_rejects_unrelated_inputs(tmp_path, model, task, filename):
    config_path = ROOT / "configs/platforms/nvidia" / f"{filename}.json"
    argv = ["--model_cls", model, "--task", task, "--model_path", str(tmp_path), "--config_json", str(config_path)]
    _, config, request = parse_cli_inputs(argv + ["--seed", "0"])
    assert config["request_defaults"]["seed"] == 42
    assert request == {"task": task, "seed": 0}
    spec = get_model_spec(model)
    assert set(spec.tasks) == {task}
    assert "audio_path" not in spec.get_fields(config, task)
    with pytest.raises(SystemExit):
        parse_cli_inputs(argv + ["--audio_path", "unsupported.wav"])
    if model == "hymotion":
        assert config["request_defaults"]["num_frames"] == 120
        for invalid in (0, 5000):
            with pytest.raises(SystemExit):
                parse_cli_inputs(argv + ["--num_frames", str(invalid)])
    elif model == "sdxl":
        _, _, request = parse_cli_inputs(argv + ["--size", "512", "768"])
        assert request["size"] == [512, 768]


@pytest.mark.parametrize("model,task,filename", MODELS)
def test_model_config_does_not_import_inference_runtime(tmp_path, model, task, filename):
    code = f"""
import sys
from lightx2v.models.runners.registry import get_model_config_loader
config = get_model_config_loader({model!r})({{'model_cls': {model!r}, 'task': {task!r}, 'model_path': {str(tmp_path)!r}}})
assert config['request_defaults']['seed'] == 42
assert not any(name in sys.modules for name in ['torch', 'rembg', 'transformers', 'texture_baker'])
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_request_fields_cannot_be_silently_misplaced():
    with pytest.raises(ValueError, match="request_defaults"):
        get_model_config_loader("hymotion")({"model_cls": "hymotion", "task": "t2m", "num_frames": 120})


def test_platform_configs_match_declared_support():
    for platform in ("nvidia", "mlu", "metax", "ascend_npu"):
        for model, task, filename in MODELS:
            config = json.loads((ROOT / "configs/platforms" / platform / f"{filename}.json").read_text())
            assert (config["model_cls"], config["task"]) == (model, task)
            assert config["implementation_status"] == ("implemented" if platform == "nvidia" else "placeholder")
            assert "num_frames" not in config and "size" not in config


def test_sf3d_asset_install_verifies_checksum_and_is_atomic(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("sf3d_assets", ROOT / "scripts/sf3d/prepare_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.npz"
    source.write_bytes(b"test grid")
    model_path = tmp_path / "weights"
    with pytest.raises(ValueError, match="SHA256"):
        module.prepare_asset(model_path, source)
    assert not (model_path / "assets/160_tets.npz").exists()
    assert not list((model_path / "assets").iterdir())
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    result = module.prepare_asset(model_path, source)
    assert result.read_bytes() == source.read_bytes()
    assert module.prepare_asset(model_path) == result
    result.write_bytes(b"corrupted grid")
    with pytest.raises(ValueError, match="Existing SF3D grid"):
        module.prepare_asset(model_path)


def test_hymotion_fixed_assets_match_validated_source():
    # Fixed geometry/statistics from HY-Motion commit 4e426f5a; preserve binary bytes.
    root = ROOT / "lightx2v/models/networks/hymotion/utils/assets"
    expected = {
        "dump_wooden/faces.bin": "777b0806d2843c797ed18644eecc11466ff822b33b02d263c22f8ad3730e9bb5",
        "dump_wooden/j_template.bin": "4f488cc9d0b650a816f5a8eda49eda7bc796f490e9130a6e0dec5be137d7b929",
        "dump_wooden/joint_names.json": "9c5b26b93bf895823e6e4d21a29f0523d52af1cea7ec0cc3eca39993237d5cce",
        "dump_wooden/kintree.bin": "98a20fa3b53193790b63d9ac3a9c917f2f70fbe6e053dca495c75317ff4b756a",
        "dump_wooden/skinIndice.bin": "846794fb90ea01e069435ad242caefb3d5c2f913fef3255c247f48836ddc1bda",
        "dump_wooden/skinWeights.bin": "343eac2902627ee6b45b547eb7d9f1526562eca7ba178d1dc71f9e466f307a77",
        "dump_wooden/v_template.bin": "b9fbe1f34bfe8a07442d11166e169318022a18da8bc62ce0a9930dfdb3171050",
        "stats/Mean.npy": "a4bbe5f68881fe793d509f262ad3dc698a91f79158a9bde9d6f24d46ef9652bd",
        "stats/Std.npy": "6cc86d1ca4b47fbd0e84b677c4397e51b6615abcb34e38fdd9daabbe6a5685cc",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
