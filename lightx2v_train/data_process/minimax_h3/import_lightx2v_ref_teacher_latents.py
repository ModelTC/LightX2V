#!/usr/bin/env python3
"""Publish finalized LightX2V H3 Ref teacher exports as paired clean GAN data.

The normal path uses only the standard library and never deserializes tensors.
Final-only exports require no intermediate state files. Legacy full trajectories
are also supported; GAN rows reference only clean final AV latents and their
exact positive condition. No model or training is run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

SCHEMA = "h3_teacher_final_v1"
LEGACY_SCHEMA = "h3_teacher_trajectory_v1"
TRAIN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEDIA_ROOT = "/mnt/lm_data_afs/gushiqiao/datasets/omini-r2v"
PROMPT_FIELDS = ("enhanced_prompt",)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def jsonl(path):
    """Match the cache builder's zero-based nonblank JSON record ordinal."""
    number = 0
    with path.open(encoding="utf-8") as handle:
        for physical, line in enumerate(handle):
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), f"Expected JSON object at {path}:{physical + 1}")
                yield number, row
                number += 1


def absolute(value, base):
    require(isinstance(value, str) and bool(value.strip()), "Expected a nonempty local path")
    require(not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://|^(?:data|https?):", value), f"Expected a local path: {value}")
    path = Path(value.strip()).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def raw_identity(row, source_index):
    return str(row.get("sample_id", row.get("metadata_id", row.get("id", source_index))))


def raw_prompt(row, source):
    require(source in PROMPT_FIELDS, f"Unsupported prompt_source: {source!r}")
    value = row.get(source)
    require(isinstance(value, str) and bool(value.strip()), f"Raw row has no {source}")
    value = value.strip()
    return value


def raw_images(row, media_root):
    references = next((row[key] for key in ("references", "actual_ordered_references", "ordered_references") if row.get(key) is not None), None)
    if references is None:
        require(not row.get("reference_videos") and not row.get("reference_audios"), "Raw row must contain image-only references")
        images = row.get("reference_images", [])
        if isinstance(images, str):
            images = [images]
        require(isinstance(images, list), "reference_images must be a list")
        references = [{"kind": "image", "path": image} for image in images]
    require(isinstance(references, list) and 2 <= len(references) <= 6, "Expected 2 through 6 ordered image references")
    require(all(isinstance(reference, dict) for reference in references), "Ordered references must be objects")
    orders = [reference.get("order", reference.get("reference_index")) for reference in references]
    if any(order is not None for order in orders):
        require(
            all(type(order) is int for order in orders) and orders in (list(range(len(orders))), list(range(1, len(orders) + 1))), "Reference order must already be contiguous; refusing to reorder"
        )
    paths = []
    for reference in references:
        kind = reference.get("kind", reference.get("modality", reference.get("type", reference.get("role", ""))))
        require(str(kind).strip().lower() in {"image", "reference_image", "image_url"}, "Raw reference must be an image")
        value = next((reference[key] for key in ("local_path", "path", "media_path", "url", "rel_path") if reference.get(key) not in (None, "")), None)
        paths.append(str(absolute(value, media_root)))
    return paths


def nonempty_file(path):
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty finalized export file: {path}")
    return path


def export_file(directory, manifest, key, expected):
    require(manifest.get(key) == expected, f"Unexpected {key}; expected {expected!r}")
    path = absolute(expected, directory)
    require(path.parent == directory, f"Export file must stay within its sample directory: {path}")
    return nonempty_file(path)


def schedule(value, name):
    require(isinstance(value, list) and len(value) == 30, f"{name} must contain all 30 actual sigma endpoints")
    require(all(type(item) in (int, float) and math.isfinite(item) and 0 <= item <= 1 for item in value), f"{name} must be finite physical sigmas in [0, 1]")
    require(value[0] == 1 and value[-1] == 0 and all(left > right for left, right in zip(value, value[1:])), f"{name} must decrease strictly from one to zero")


def _leaf(relative, name):
    spec = importlib.util.spec_from_file_location(name, TRAIN_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_export(raw, index, index_path, media_root):
    source_id = raw_identity(raw, index["source_index"])
    require(index.get("source_id") == source_id, "Index source_id differs from the raw row")
    prompt_source = index.get("prompt_source", next((key for key in PROMPT_FIELDS if isinstance(raw.get(key), str) and raw[key].strip()), None))
    prompt = raw_prompt(raw, prompt_source)
    require(index.get("prompt") == prompt, f"Index prompt differs from raw source {source_id}")
    images = raw_images(raw, media_root)
    require(isinstance(index.get("reference_images"), list), "Index requires ordered reference_images")
    require([str(absolute(path, media_root)) for path in index["reference_images"]] == images, f"Index reference order/path differs from raw source {source_id}")
    count = len(images)
    require(type(index.get("ref_image_count")) is int and index["ref_image_count"] == count, "Index reference count differs from raw source")
    directory = absolute(index.get("export_dir"), index_path.parent)
    manifest_path = nonempty_file(directory / "manifest.json")
    if "manifest_path" in index:
        require(absolute(index["manifest_path"], index_path.parent) == manifest_path, "Index manifest_path differs from export_dir")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "Export manifest must be an object")
    schema = manifest.get("schema")
    require(schema in (SCHEMA, LEGACY_SCHEMA), f"Invalid export schema: {schema!r}")
    final_only = schema == SCHEMA
    export_mode = "final" if final_only else "trajectory"
    require(manifest.get("export_mode", None if final_only else "trajectory") == export_mode, f"Invalid export_mode for {schema}: expected {export_mode!r}")
    saved_states = 1 if final_only else 30
    if final_only:
        require(type(manifest.get("final_state_index")) is int and manifest["final_state_index"] == 29, "Final-only export requires final_state_index=29")
        for key in ("final_video_sigma", "final_audio_sigma"):
            require(type(manifest.get(key)) in (int, float) and manifest[key] == 0.0, f"Final-only export requires {key}=0.0")
    for key, expected in {
        "task": "ref2av",
        "normalized": True,
        "dtype": "float32",
        "num_steps": 29,
        "num_states": saved_states,
        "num_frames": 124,
        "fps": 24,
        "scheduler_type": "reference_blend",
        "transformer_partition": "transformer_ref",
    }.items():
        require(manifest.get(key) == expected and (type(manifest[key]) is type(expected) or key == "fps"), f"Invalid export {key}: expected {expected!r}")
    for key in ("source_id", "cache_fingerprint", "seed", "prompt"):
        require(manifest.get(key) == index.get(key), f"Export {key} differs from index for {source_id}")
    require(isinstance(manifest["cache_fingerprint"], str) and bool(manifest["cache_fingerprint"]), "Export cache_fingerprint is required")
    require(type(manifest["seed"]) is int and manifest["seed"] >= 0, "Export seed must be a nonnegative integer")
    manifest_images = manifest.get("image_paths")
    require(
        isinstance(manifest_images, list) and [str(absolute(path, media_root)) for path in manifest_images] == images and not manifest.get("video_paths") and not manifest.get("audio_paths"),
        "Export ordered reference paths differ or contain video/audio",
    )
    require(manifest.get("ref_image_count", manifest.get("reference_image_count")) == count, "Export reference count differs")
    for key in ("reference_video_count", "reference_audio_count", "reference_audio_rows"):
        require(manifest.get(key) == 0, f"Image-only export requires {key}=0")
    height, width = manifest.get("height"), manifest.get("width")
    require(all(type(value) is int and value > 0 and value % 32 == 0 for value in (height, width)), "Export geometry must be positive multiples of 32")
    for key, expected in {"target_height": height, "target_width": width, "num_frames": 124, "fps": 24}.items():
        require(index.get(key) == expected, f"Index {key} differs from export")
    require(manifest.get("video_shape") == [24, 37, height // 16, width // 16] and manifest.get("audio_shape") == [2, 32, 207], "Export must contain full normalized AV target shapes")
    expected_states = [] if final_only else [f"state_{number:03d}.pt" for number in range(30)]
    require(manifest.get("state_paths") == expected_states, "Final-only export requires state_paths=[]" if final_only else "Export must list all 30 ordered trajectory states")
    for name in expected_states:
        path = absolute(name, directory)
        require(path.parent == directory, "Trajectory file escapes its sample directory")
        nonempty_file(path)
    for key in ("video_sigmas", "audio_sigmas"):
        schedule(manifest.get(key), key)
    for key, value in (("video_flow_shift", 12), ("audio_flow_shift", 3)):
        require(type(manifest.get(key)) in (int, float) and manifest[key] == value, f"Export requires {key}={value}")
    for key in ("model_path", "transformer_path", "model_variant"):
        require(isinstance(manifest.get(key), str) and bool(manifest[key].strip()), f"Export model provenance requires {key}")
    prompt_tokens, reference_rows = manifest.get("prompt_token_count"), manifest.get("reference_video_rows")
    require(all(type(value) is int and value > 0 for value in (prompt_tokens, reference_rows)), "Export requires measured prompt and reference token counts")
    descriptors = manifest.get("references")
    require(isinstance(descriptors, list) and len(descriptors) == count, "Export requires one descriptor for every ordered image reference")
    measured_rows = 0
    for reference, expected_path in zip(descriptors, images):
        require(isinstance(reference, dict) and reference.get("kind") == "image" and reference.get("normalized") is True, "Export reference descriptor must be a normalized image")
        require(str(absolute(reference.get("path"), media_root)) == expected_path, "Export reference descriptor path/order differs from raw source")
        require(reference.get("num_latent_frames") == 1 and reference.get("num_audio_latents") == 0, "Export reference descriptor must describe a single image")
        reference_height, reference_width = reference.get("latent_height"), reference.get("latent_width")
        require(all(type(value) is int and value > 0 and value % 2 == 0 for value in (reference_height, reference_width)), "Invalid exported reference latent geometry")
        measured_rows += reference_height * reference_width // 4
    require(measured_rows == reference_rows, "Export reference rows differ from measured descriptor geometry")
    condition = export_file(directory, manifest, "condition_path", "condition.pt")
    video = export_file(directory, manifest, "video_latents_path", "video_latents.pt")
    audio = export_file(directory, manifest, "audio_latents_path", "audio_latents.pt")
    for key, path in (("condition_path", condition), ("video_latents_path", video), ("audio_latents_path", audio)):
        if key in index:
            require(absolute(index[key], index_path.parent) == path, f"Index {key} differs from finalized export")
    # Only settings that must remain constant across this dataset. Prompt, seed,
    # reference geometry and generated target geometry may vary per example.
    settings_keys = (
        "schema",
        "export_mode",
        "task",
        "dtype",
        "num_steps",
        "num_states",
        "final_state_index",
        "final_video_sigma",
        "final_audio_sigma",
        "num_frames",
        "fps",
        "scheduler_type",
        "video_sigmas",
        "audio_sigmas",
        "video_flow_shift",
        "audio_flow_shift",
        "model_path",
        "transformer_partition",
        "transformer_path",
        "model_variant",
        "guidance_scale",
        "enable_cfg",
        "lora_configs",
        "reference_image_resize_mode",
        "model_identity",
        "transformer_identity",
        "decfg_merged_identity",
        "model_provenance",
        "config_sha256",
        "settings",
    )
    settings = {key: manifest[key] for key in settings_keys if key in manifest}
    row = dict(
        task="ref2av",
        source_id=source_id,
        source_index=index["source_index"],
        caption=prompt,
        prompt=prompt,
        prompt_source=prompt_source,
        seed=manifest["seed"],
        condition_path=str(condition),
        video_latent_path=str(video),
        audio_latent_path=str(audio),
        cache_fingerprint=manifest["cache_fingerprint"],
        latent_fingerprint=manifest.get("latent_fingerprint", digest({"schema": schema, "manifest": manifest})),
        video_latent_shape=manifest["video_shape"],
        audio_latent_shape=manifest["audio_shape"],
        vae_normalized=True,
        cache_storage_dtype="float32",
        source_kind="teacher_generated",
        target_num_frames=124,
        num_frames=124,
        target_height=height,
        target_width=width,
        fps=24,
        target_orientation="landscape" if width >= height else "portrait",
        reference_image_count=count,
        ref_image_count=count,
        reference_video_count=0,
        reference_audio_count=0,
        references=[{"kind": "image", "path": path} for path in images],
        prompt_token_count=prompt_tokens,
        reference_video_rows=reference_rows,
        reference_audio_rows=0,
        reference_compute_cost=prompt_tokens + reference_rows,
        packed_sequence_tokens_124=prompt_tokens + reference_rows + 37 * (height // 32) * (width // 32) + 414,
        teacher_trajectory_manifest=str(manifest_path),
        teacher_trajectory_states=saved_states,
        teacher_export_schema=schema,
        teacher_export_mode=export_mode,
        teacher_saved_states=saved_states,
        teacher_final_state_index=29,
        teacher_final_video_sigma=manifest["video_sigmas"][-1],
        teacher_final_audio_sigma=manifest["audio_sigmas"][-1],
        teacher_scheduler_type="reference_blend",
        teacher_model_path=manifest["model_path"],
        teacher_inference_config_sha256=manifest.get("inference_config_sha256"),
    )
    require(isinstance(row["latent_fingerprint"], str) and bool(row["latent_fingerprint"]), "Export latent_fingerprint must be nonempty")
    for key in ("request_fingerprint", "selection_sha256", "sample_key", "node_index"):
        if key in index:
            row[key] = index[key]
    return row, settings


def validate_tensors(row):
    """Read only condition + final AV (never the potentially 2 TB trajectory)."""
    import torch

    def tensor(value, shape, name, dtype=None):
        require(torch.is_tensor(value) and list(value.shape) == shape, f"Invalid {name} shape")
        require(dtype is None or value.dtype == dtype, f"Invalid {name} dtype")
        require(bool(torch.isfinite(value).all()), f"Nonfinite {name}")
        return value

    video = torch.load(row["video_latent_path"], map_location="cpu", weights_only=True)
    require(isinstance(video, dict) and video.get("normalized") is True, "Final video must explicitly declare normalized=True")
    tensor(video.get("latents"), row["video_latent_shape"], "final video", torch.float32)
    if "latent_fingerprint" in video:
        require(video["latent_fingerprint"] == row["latent_fingerprint"], "Final video latent_fingerprint differs")
    audio = torch.load(row["audio_latent_path"], map_location="cpu", weights_only=True)
    tensor(audio, row["audio_latent_shape"], "final audio", torch.float32)
    item = torch.load(row["condition_path"], map_location="cpu", weights_only=True)
    require(isinstance(item, dict), "Condition must be a mapping")
    for key, expected in (("cache_schema_version", 1), ("cache_stage", "complete"), ("source_id", row["source_id"]), ("cache_fingerprint", row["cache_fingerprint"])):
        require(item.get(key) == expected, f"Condition {key} differs from its paired target")
    require(isinstance(item.get("conditioning"), dict), "Condition requires conditioning.positive")
    positive = item["conditioning"].get("positive")
    require(isinstance(positive, dict), "Condition requires conditioning.positive")
    for container in (item, item["conditioning"], positive):
        for key in ("negative", "negative_conditions", "keyframe_anchors", "keyframe_latents", "condition_audio_latents"):
            require(container.get(key) is None or isinstance(container[key], (list, dict, tuple)) and not container[key], f"Unexpected condition {key}")
    for key in ("task", "target_num_frames", "target_height", "target_width"):
        require(positive.get(key) == row[key], f"Condition {key} differs from its paired target")
    require(item.get("prompt", item["conditioning"].get("prompt")) == row["prompt"], "Condition prompt differs from its paired raw caption")
    count = row["prompt_token_count"]
    embeds = positive.get("prompt_embeds")
    require(torch.is_tensor(embeds) and embeds.ndim in (2, 3), "Invalid prompt embeddings")
    if embeds.ndim == 3:
        require(embeds.shape[0] == 1, "Prompt embedding batch must be one")
        embeds = embeds[0]
    tensor(embeds, [count, 5120], "prompt embeddings")
    tags = positive.get("text_token_tags")
    if torch.is_tensor(tags) and tags.ndim == 2 and tags.shape[0] == 1:
        tags = tags[0]
    tensor(tags, [count], "text token tags")
    require(bool((tags == 0).any()) and bool(((tags == 0) | (tags == 1)).all()), "Condition requires multimodal prompt/vision tags")
    references = positive.get("references")
    require(isinstance(references, list) and len(references) == row["reference_image_count"], "Condition reference count differs")
    reference_rows = 0
    for reference, expected in zip(references, row["references"]):
        require(isinstance(reference, dict), "Condition reference must be a mapping")
        require(reference.get("kind") == "image" and reference.get("normalized") is True, "Condition reference must be normalized image latents")
        require(str(absolute(reference.get("path"), Path("/"))) == expected["path"], "Condition reference path/order differs from raw source")
        height, width = reference.get("latent_height"), reference.get("latent_width")
        require(all(type(value) is int and value > 0 and value % 2 == 0 for value in (height, width)), "Invalid condition reference geometry")
        require(
            reference.get("num_latent_frames") == 1 and reference.get("num_audio_latents", 0) == 0 and reference.get("audio_latents") is None,
            "Condition must contain only single-frame image references",
        )
        rows = height * width // 4
        tensor(reference.get("video_latents"), [rows, 96], "packed reference video", torch.float32)
        reference_rows += rows
    require(reference_rows == row["reference_video_rows"], "Condition measured reference cost differs from manifest")


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def import_exports(raw_jsonl, export_indexes, output_dir, *, media_root=Path(DEFAULT_MEDIA_ROOT), expected_count=None, check_tensors=False):
    raw_jsonl, output_dir, media_root = (Path(path).expanduser().resolve() for path in (raw_jsonl, output_dir, media_root))
    export_indexes = [Path(path).expanduser().resolve() for path in export_indexes]
    require(expected_count is None or type(expected_count) is int and expected_count > 0, "expected_count must be a positive integer")
    require(bool(export_indexes) and len(set(export_indexes)) == len(export_indexes), "Provide distinct export index files")
    output_paths = {output_dir / f"{name}.{suffix}" for name in ("metadata", "condition_metadata") for suffix in ("jsonl", "complete.json")}
    require(not output_paths.intersection({raw_jsonl, *export_indexes}), "Output publication would overwrite an input manifest")
    indexed = [(path, ordinal, row) for path in export_indexes for ordinal, row in jsonl(path)]
    require(all(type(row.get("source_index")) is int and row["source_index"] >= 0 for _, _, row in indexed), "Every index source_index must be a nonnegative integer")
    selected_indices = {row["source_index"] for _, _, row in indexed}
    raw_rows, raw_count = {}, 0
    for ordinal, raw in jsonl(raw_jsonl):
        raw_count += 1
        if ordinal in selected_indices:
            raw_rows[ordinal] = raw
    rows, seen, shared_settings = [], {key: set() for key in ("source_id", "source_index", "condition_path", "video_latent_path", "audio_latent_path", "cache_fingerprint")}, None
    for index_path, line, index in indexed:
        location = f"{index_path}:record {line}"
        try:
            source_index = index["source_index"]
            require(source_index in raw_rows, "Index source_index does not identify a raw JSONL record")
            if "source_jsonl" in index:
                require(absolute(index["source_jsonl"], index_path.parent) == raw_jsonl, "Index source_jsonl differs from --raw-jsonl")
            row, settings = validate_export(raw_rows[source_index], index, index_path, media_root)
            for key, values in seen.items():
                require(row[key] not in values, f"Duplicate {key}: {row[key]}")
                values.add(row[key])
            require(shared_settings is None or settings == shared_settings, "Inconsistent teacher model or generation settings across exports")
            shared_settings = settings
            if check_tensors:
                validate_tensors(row)
            rows.append(row)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            raise ValueError(f"{location}: {error}") from error
    require(bool(rows), "No finalized exports were supplied")
    require(expected_count is None or len(rows) == expected_count, f"Expected {expected_count} finalized unique exports, found {len(rows)}; no dataset published")
    rows.sort(key=lambda row: row["source_index"])
    # Exercise the existing manifest-only GAN contract before publishing anything.
    gan = _leaf("lightx2v_train/trainers/dmd/dmd2_gan_data.py", "_teacher_import_gan_contract")
    for row in rows:
        gan._check_ref_metadata(dict(row), row["teacher_trajectory_manifest"])
    input_identity = {
        "schema": "h3_teacher_import_v1",
        "raw_jsonl": str(raw_jsonl),
        "raw_sha256": file_digest(raw_jsonl),
        "indexes": [{"path": str(path), "sha256": file_digest(path)} for path in export_indexes],
        "settings": shared_settings,
        "rows": rows,
    }
    receipt = dict(
        schema="h3_teacher_import_v1",
        completed_count=len(rows),
        failed_count=0,
        input_total_rows=len(rows),
        raw_source_rows=raw_count,
        raw_unselected_rows=raw_count - len(rows),
        expected_count=expected_count,
        preprocess_fingerprint=digest(input_identity),
        raw_jsonl=str(raw_jsonl),
        export_indexes=[str(path) for path in export_indexes],
        check_tensors=check_tensors,
        tensor_validation_scope="condition_and_final_av" if check_tensors else "manifest_and_file_presence_only",
        trajectory_tensors_loaded=False,
        teacher_settings=shared_settings,
        reference_image_count_histogram=dict(sorted(Counter(row["reference_image_count"] for row in rows).items())),
    )
    # A failed validation above leaves any existing valid publication intact.
    # Invalidate receipts before replacing manifests; publish receipts last.
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metadata", "condition_metadata"):
        (output_dir / f"{name}.complete.json").unlink(missing_ok=True)
    condition_rows = [{key: value for key, value in row.items() if key not in {"video_latent_path", "audio_latent_path", "video_latent_shape", "audio_latent_shape"}} for row in rows]
    publications = []
    for name, records in (("metadata", rows), ("condition_metadata", condition_rows)):
        content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in records).encode()
        atomic_write(output_dir / f"{name}.jsonl", content)
        publications.append((name, {**receipt, "manifest_sha256": hashlib.sha256(content).hexdigest()}))
    for name, record in publications:
        atomic_write(output_dir / f"{name}.complete.json", (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-jsonl", type=Path, required=True, help="Original JSONL; source_index is its zero-based nonblank record ordinal")
    parser.add_argument("--export-index", "--index", type=Path, nargs="+", required=True, help="Completed node indexes, e.g. index.node0.jsonl index.node1.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, default=Path(DEFAULT_MEDIA_ROOT))
    parser.add_argument("--expected-count", type=int, help="Require exactly this many unique finalized exports before publishing (production: 5000)")
    parser.add_argument("--check-tensors", action="store_true", help="Also validate condition and clean final AV tensors on CPU; does not deserialize any trajectory states")
    args = parser.parse_args(argv)
    receipt = import_exports(args.raw_jsonl, args.export_index, args.output_dir, media_root=args.media_root, expected_count=args.expected_count, check_tensors=args.check_tensors)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "completed_count": receipt["completed_count"],
                "expected_count": receipt["expected_count"],
                "tensor_validation_scope": receipt["tensor_validation_scope"],
                "preprocess_fingerprint": receipt["preprocess_fingerprint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
