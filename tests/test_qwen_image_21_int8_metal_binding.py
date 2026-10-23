import hashlib
import importlib
import json

import pytest


def _binding_module():
    try:
        return importlib.import_module("lightx2v.models.runners.qwen_image_21.int8_metal_binding")
    except RuntimeError as error:
        if "AI device" in str(error):
            pytest.skip(f"LightX2V platform is unavailable in this environment: {error}")
        raise


def _write_manifest(path, *, shape_counts, library_path, model_revision="revision"):
    payload = library_path.read_bytes()
    manifest = {
        "schema_version": 1,
        "model_repo": "test/model",
        "model_revision": model_revision,
        "required_shapes": [{"k": k, "n": n, "weight_count": count} for (k, n), count in sorted(shape_counts.items())],
        "libraries": [
            {
                "file": library_path.name,
                "k": k,
                "n": n,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for k, n in sorted(shape_counts)
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _config(model_path):
    return {
        "model_path": str(model_path),
        "int8_metal_manifest": "manifest.json",
        "int8_metal_manifest_required": True,
        "int8_metal_model_repo": "test/model",
        "int8_metal_model_revision": "revision",
    }


def test_validates_shape_coverage_and_library_hash(tmp_path, monkeypatch):
    module = _binding_module()
    metallib_dir = tmp_path / "metallib"
    metallib_dir.mkdir()
    library = metallib_dir / "int8_k4_n8.metallib"
    library.write_bytes(b"valid metal library")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, shape_counts={(4, 8): 3}, library_path=library)
    monkeypatch.setenv("LIGHTX2V_METALLIB_DIR", str(metallib_dir))
    monkeypatch.setattr(module, "collect_int8_weight_shapes", lambda _: {(4, 8): 3})

    result = module.validate_int8_metal_binding(_config(tmp_path))

    assert result["verified"] is True
    assert result["shape_counts"] == {(4, 8): 3}


def test_rejects_shape_coverage_mismatch(tmp_path, monkeypatch):
    module = _binding_module()
    metallib_dir = tmp_path / "metallib"
    metallib_dir.mkdir()
    library = metallib_dir / "int8_k4_n8.metallib"
    library.write_bytes(b"valid metal library")
    _write_manifest(tmp_path / "manifest.json", shape_counts={(4, 8): 3}, library_path=library)
    monkeypatch.setenv("LIGHTX2V_METALLIB_DIR", str(metallib_dir))
    monkeypatch.setattr(module, "collect_int8_weight_shapes", lambda _: {(8, 4): 3})

    with pytest.raises(ValueError, match="shape coverage mismatch"):
        module.validate_int8_metal_binding(_config(tmp_path))


def test_rejects_library_hash_mismatch(tmp_path, monkeypatch):
    module = _binding_module()
    metallib_dir = tmp_path / "metallib"
    metallib_dir.mkdir()
    library = metallib_dir / "int8_k4_n8.metallib"
    library.write_bytes(b"tampered metal library")
    _write_manifest(tmp_path / "manifest.json", shape_counts={(4, 8): 3}, library_path=library)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["libraries"][0]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("LIGHTX2V_METALLIB_DIR", str(metallib_dir))
    monkeypatch.setattr(module, "collect_int8_weight_shapes", lambda _: {(4, 8): 3})

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.validate_int8_metal_binding(_config(tmp_path))


def test_explicit_unverified_override_keeps_shape_coverage_check(tmp_path, monkeypatch):
    module = _binding_module()
    metallib_dir = tmp_path / "metallib"
    metallib_dir.mkdir()
    library = metallib_dir / "int8_k4_n8.metallib"
    library.write_bytes(b"custom metal library")
    monkeypatch.setenv("LIGHTX2V_METALLIB_DIR", str(metallib_dir))
    monkeypatch.setenv("LIGHTX2V_ALLOW_UNVERIFIED_METALLIB", "1")
    monkeypatch.setattr(module, "collect_int8_weight_shapes", lambda _: {(4, 8): 3})

    result = module.validate_int8_metal_binding(_config(tmp_path))

    assert result["verified"] is False
