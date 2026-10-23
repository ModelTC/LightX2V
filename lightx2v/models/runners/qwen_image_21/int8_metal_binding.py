"""Validate Qwen INT8 Metal libraries before loading the transformer."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from loguru import logger

from lightx2v.common.offload.checkpoint_metadata import read_safetensors_header

_MANIFEST_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64


def _allow_unverified_metallib():
    return os.environ.get("LIGHTX2V_ALLOW_UNVERIFIED_METALLIB", "").lower() in {"1", "true", "yes"}


def _resolve_manifest_path(config):
    configured = os.environ.get("LIGHTX2V_METALLIB_MANIFEST") or config.get("int8_metal_manifest")
    if not configured:
        return None
    path = Path(configured)
    if not path.is_absolute():
        path = Path(config["model_path"]) / path
    return path


def collect_int8_weight_shapes(transformer_path):
    """Return runtime ``(K, N)`` counts from INT8 checkpoint headers."""
    transformer_path = Path(transformer_path)
    files = sorted(transformer_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"INT8 transformer safetensors not found: {transformer_path}")

    tensors = {}
    for path in files:
        header = read_safetensors_header(path)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate tensor across INT8 safetensors shards: {name}")
            tensors[name] = entry

    shapes = Counter()
    for name, entry in tensors.items():
        if not name.endswith(".weight") or entry.get("dtype") != "I8":
            continue
        shape = entry.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(value, int) and value > 0 for value in shape):
            raise ValueError(f"INT8 weight must have a positive rank-2 shape: {name}")
        scale_name = f"{name[: -len('.weight')]}.weight_scale"
        if scale_name not in tensors:
            raise ValueError(f"INT8 weight is missing its weight_scale tensor: {name}")
        n, k = shape
        shapes[(k, n)] += 1

    if not shapes:
        raise ValueError(f"No INT8 transformer weights found in {transformer_path}")
    return dict(sorted(shapes.items()))


def _load_manifest(path):
    try:
        with path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid INT8 Metal manifest JSON: {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"INT8 Metal manifest must be an object: {path}")
    if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported INT8 Metal manifest schema in {path}: expected {_MANIFEST_SCHEMA_VERSION}, got {manifest.get('schema_version')!r}")
    return manifest


def _manifest_shape_counts(manifest, path):
    entries = manifest.get("required_shapes")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"INT8 Metal manifest has no required_shapes: {path}")
    shapes = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid required_shapes entry in {path}: {entry!r}")
        k, n, count = entry.get("k"), entry.get("n"), entry.get("weight_count")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (k, n, count)):
            raise ValueError(f"Invalid required shape entry in {path}: {entry!r}")
        shape = (k, n)
        if shape in shapes:
            raise ValueError(f"Duplicate required shape in {path}: K={k}, N={n}")
        shapes[shape] = count
    return shapes


def _manifest_libraries(manifest, path):
    entries = manifest.get("libraries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"INT8 Metal manifest has no libraries: {path}")
    libraries = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid libraries entry in {path}: {entry!r}")
        k, n = entry.get("k"), entry.get("n")
        filename = entry.get("file")
        size = entry.get("size")
        digest = entry.get("sha256")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (k, n, size)):
            raise ValueError(f"Invalid library shape or size in {path}: {entry!r}")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError(f"Invalid library filename in {path}: {entry!r}")
        expected_filename = f"int8_k{k}_n{n}.metallib"
        if filename != expected_filename:
            raise ValueError(f"Library filename does not match K/N in {path}: expected {expected_filename!r}, got {filename!r}")
        if not isinstance(digest, str) or len(digest) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid library SHA-256 in {path}: {entry!r}")
        shape = (k, n)
        if shape in libraries:
            raise ValueError(f"Duplicate library shape in {path}: K={k}, N={n}")
        libraries[shape] = {"file": filename, "size": size, "sha256": digest}
    return libraries


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_unverified_coverage(metallib_dir, shape_counts):
    for k, n in shape_counts:
        path = metallib_dir / f"int8_k{k}_n{n}.metallib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing precompiled INT8 Metal kernel for K={k}, N={n}: {path}")


def validate_int8_metal_binding(config):
    """Validate checkpoint shape coverage and model-bound Metal library hashes."""
    metallib_dir_value = os.environ.get("LIGHTX2V_METALLIB_DIR")
    if not metallib_dir_value:
        raise RuntimeError("int8-metal requires LIGHTX2V_METALLIB_DIR pointing to precompiled .metallib files")
    metallib_dir = Path(metallib_dir_value)
    if not metallib_dir.is_dir():
        raise FileNotFoundError(f"INT8 Metal library directory not found: {metallib_dir}")

    shape_counts = collect_int8_weight_shapes(Path(config["model_path"]) / "transformer")
    manifest_path = _resolve_manifest_path(config)
    allow_unverified = _allow_unverified_metallib()
    manifest_required = bool(config.get("int8_metal_manifest_required", False))

    if allow_unverified:
        logger.warning("LIGHTX2V_ALLOW_UNVERIFIED_METALLIB is enabled; skipping INT8 Metal manifest and hash binding")
        _validate_unverified_coverage(metallib_dir, shape_counts)
        return {"verified": False, "shape_counts": shape_counts, "metallib_dir": str(metallib_dir)}

    if manifest_path is None:
        if manifest_required:
            raise RuntimeError("int8-metal requires a model-bound manifest; set int8_metal_manifest or use the explicit unverified override")
        logger.warning("No INT8 Metal manifest configured; validating shape coverage without model/hash binding")
        _validate_unverified_coverage(metallib_dir, shape_counts)
        return {"verified": False, "shape_counts": shape_counts, "metallib_dir": str(metallib_dir)}
    if not manifest_path.is_file():
        raise FileNotFoundError(f"INT8 Metal binding manifest not found: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    expected_repo = config.get("int8_metal_model_repo")
    expected_revision = config.get("int8_metal_model_revision")
    if expected_repo is not None and manifest.get("model_repo") != expected_repo:
        raise ValueError(f"INT8 Metal manifest model_repo mismatch: expected {expected_repo!r}, got {manifest.get('model_repo')!r}")
    if expected_revision is not None and manifest.get("model_revision") != expected_revision:
        raise ValueError(f"INT8 Metal manifest model_revision mismatch: expected {expected_revision!r}, got {manifest.get('model_revision')!r}")

    expected_shapes = _manifest_shape_counts(manifest, manifest_path)
    if shape_counts != expected_shapes:
        raise ValueError(f"INT8 Metal checkpoint shape coverage mismatch: expected {expected_shapes}, got {shape_counts}")

    libraries = _manifest_libraries(manifest, manifest_path)
    if set(libraries) != set(shape_counts):
        raise ValueError(f"INT8 Metal manifest library coverage mismatch: expected shapes {sorted(shape_counts)}, got {sorted(libraries)}")
    for (k, n), metadata in sorted(libraries.items()):
        library = metallib_dir / metadata["file"]
        if not library.is_file():
            raise FileNotFoundError(f"Missing precompiled INT8 Metal kernel for K={k}, N={n}: {library}")
        actual_size = library.stat().st_size
        if actual_size != metadata["size"]:
            raise ValueError(f"INT8 Metal library size mismatch for K={k}, N={n}: expected {metadata['size']}, got {actual_size}: {library}")
        actual_digest = _sha256(library)
        if actual_digest != metadata["sha256"]:
            raise ValueError(f"INT8 Metal library SHA-256 mismatch for K={k}, N={n}: expected {metadata['sha256']}, got {actual_digest}: {library}")

    logger.info(f"[int8-metal] validated model-bound Metal libraries: revision={manifest.get('model_revision')}, shapes={sorted(shape_counts)}")
    return {
        "verified": True,
        "manifest": str(manifest_path),
        "model_revision": manifest.get("model_revision"),
        "shape_counts": shape_counts,
        "metallib_dir": str(metallib_dir),
    }


__all__ = ["collect_int8_weight_shapes", "validate_int8_metal_binding"]
