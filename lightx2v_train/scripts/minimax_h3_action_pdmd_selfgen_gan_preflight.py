#!/usr/bin/env python3
"""CPU recipe checks and immutable dataset views for action PDMD -> selfgen GAN.

Each view is a complete canonical H3 cache tree published by one directory
rename. Concurrent ACP workers may build the same bytes, but cannot replace a
published view. Its receipt binds the source manifest, selection version and
resolved paths; changing any of them requires a fresh view/output directory.
"""

import errno
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_h3_staged_legacy_preflight", Path(__file__).with_name("minimax_h3_pdmd_gan_preflight.py"))
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
ROOT, load_module = _legacy.ROOT, _legacy.load_module
own_stage_resume = _legacy.own_stage_resume
validate_cache, validate_prompts = _legacy.validate_cache, _legacy.validate_prompts
_validate_recipe = _legacy.validate_recipe


SCHEMA = "h3_action_pdmd_selfgen_gan_view_v1"
MANIFEST = Path("t2av/landscape/metadata.jsonl")
RECEIPT = "view.complete.json"
MIN_ROWS = 64  # DP32 stratified sampling needs two complete global batches.


def load_config(path):
    runtime = load_module("_h3_staged_gan_runtime", ROOT / "lightx2v_train/runtime/config.py")
    return runtime.load_config(path)


def validate_recipe(config, gan_enabled=None):
    source, adapter = _validate_recipe(config, gan_enabled=gan_enabled)
    training, data = config["training"], config["data"]["train"]
    if data.get("defer_latent_loading") is not True:
        raise ValueError("Both stages require defer_latent_loading=true")
    if data.get("name") != "minimax_h3_cache_dataset" or data.get("max_samples") is not None or data.get("dataset_repeat", 1) != 1 or data.get("prompt_column", "caption") != "caption":
        raise ValueError("Both stages must consume the complete canonical caption-condition view")
    if (training["student"]["optimizer"].get("learning_rate"), training["fake"]["optimizer"].get("learning_rate")) != (2e-5, 4e-6):
        raise ValueError("Both stages require student/fake learning rates 2e-5/4e-6")
    if training.get("save_every_iters") != 100 or training.get("save_total_limit") != 2:
        raise ValueError("Both stages require save_every_iters=100 and save_total_limit=2")
    if training.get("gradient_accumulation_iters") != 1:
        raise ValueError("Both stages require gradient_accumulation_iters=1")
    if config["distributed"]["fsdp2"].get("enabled") is not True:
        raise ValueError("Both stages require enabled FSDP32")
    if training["dmd"].get("generation_shapes") != [{"value": [124, 768, 1344]}]:
        raise ValueError("Both stages require generation shape 124x768x1344")
    enabled = bool(training["dmd"].get("gan", {}).get("enabled", False))
    if not enabled and int(training["max_train_iters"]) != 600:
        raise ValueError("Action phase must finish exactly 100 student updates / 600 role iterations")
    if enabled:
        expected = Path(data["data_path"]).expanduser().resolve() / MANIFEST
        actual = Path(training["dmd"]["gan"]["real_cache"]).expanduser().resolve()
        if actual != expected:
            raise ValueError("Selfgen generator conditions and GAN positives must use the same filtered manifest")
    return source, adapter


def validate_model(config):
    model = Path(config["model"]["pretrained_model_name_or_path"])
    for component in ("transformer/config.json", "text_encoder/config.json", "vae/config.json"):
        if not (model / component).is_file():
            raise FileNotFoundError(model / component)
    return str(model)


def validate_warm_start(config, source=None):
    source = Path(source or config["training"]["student"]["checkpoint_path"]).expanduser().resolve()
    for relative in ("official_pdmd.complete", "dist_state/.metadata"):
        if not (source / relative).is_file():
            raise FileNotFoundError(f"Source must be a complete official-PDMD checkpoint: {source / relative}")
    adapter = load_module("_h3_staged_gan_adapter", ROOT / "scripts/minimax_h3_pdmd_from_turbo_preflight.py")
    weights, count = adapter.validate_source(source, config["training"]["student"]["lora"])
    return {"source": str(source), "student_weights": str(weights), "lora_tensors": count}


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _rows(content):
    return [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]


def _check_rows(rows, kind):
    if len(rows) < MIN_ROWS:
        raise ValueError(f"{kind} view requires at least {MIN_ROWS} eligible rows for DP32 stratified sampling; found {len(rows)}")
    conditions, fingerprints = set(), set()
    for index, row in enumerate(rows):
        if row.get("task") != "t2av" or tuple(row.get(k) for k in ("target_num_frames", "target_height", "target_width")) != (124, 768, 1344) or row.get("fps") != 24:
            raise ValueError(f"View row {index} must be T2AV 124x768x1344 at 24fps")
        if any(str(row.get(key) or "").strip().lower() not in {"", "landscape"} for key in ("aspect_bucket", "target_orientation")):
            raise ValueError(f"View row {index} has conflicting landscape orientation metadata")
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise ValueError(f"View row {index} requires a nonempty caption")
        keys = ("condition_path", "video_latent_path", "audio_latent_path") if kind == "selfgen" else ("condition_path",)
        for key in keys:
            path = Path(row.get(key, ""))
            if not path.is_absolute() or not path.is_file():
                raise FileNotFoundError(f"View row {index} has missing/nonabsolute {key}: {path}")
        if kind == "action" and any(row.get(key) for key in ("video_latent_path", "audio_latent_path")):
            raise ValueError("Action view must contain prompt conditions only")
        condition = row["condition_path"]
        if condition in conditions:
            raise ValueError(f"Duplicate condition_path in view: {condition}")
        conditions.add(condition)
        fingerprint = str(row.get("cache_fingerprint", "")).strip()
        if fingerprint and fingerprint in fingerprints:
            raise ValueError(f"Duplicate cache_fingerprint in view: {fingerprint}")
        fingerprints.add(fingerprint)


def validate_view(output, kind=None):
    output = Path(output).expanduser().resolve()
    receipt = json.loads((output / RECEIPT).read_text())
    actual_kind = receipt.get("kind")
    if receipt.get("schema") != SCHEMA or actual_kind not in {"action", "selfgen"} or (kind is not None and actual_kind != kind):
        raise ValueError(f"Unexpected view schema/kind: {output}")
    manifests = list(output.rglob("metadata.jsonl"))
    if manifests != [output / MANIFEST]:
        raise ValueError(f"View must contain only its canonical T2AV manifest: {output}")
    if {path.relative_to(output) for path in output.rglob("*") if path.is_file()} != {MANIFEST, Path(RECEIPT)}:
        raise ValueError(f"View contains unexpected files (including possible negative conditioning): {output}")
    content = (output / MANIFEST).read_bytes()
    rows = _rows(content)
    identity = {key: value for key, value in receipt.items() if key != "fingerprint"}
    if receipt.get("manifest_sha256") != _sha(content) or receipt.get("rows") != len(rows) or receipt.get("fingerprint") != _sha(_json(identity)):
        raise ValueError(f"View manifest/receipt changed: {output}; use a fresh output directory")
    _check_rows(rows, actual_kind)
    return {**receipt, "root": str(output), "manifest": str(output / MANIFEST)}


def _publish(output, kind, rows, source_identity, dry_run):
    output = Path(output).expanduser().resolve()
    _check_rows(rows, kind)
    content = b"".join(_json(row) + b"\n" for row in rows)
    receipt = {"schema": SCHEMA, "kind": kind, "selection": "t2av_124x768x1344_24fps", "source": source_identity, "rows": len(rows), "manifest_sha256": _sha(content)}
    receipt["fingerprint"] = _sha(_json(receipt))
    summary = {**receipt, "root": str(output), "manifest": str(output / MANIFEST)}

    def check_existing():
        existing = validate_view(output, kind)
        if existing != summary:
            raise ValueError(f"Existing view uses different source data or selection: {output}; use a fresh output directory")
        return existing

    if output.exists():
        return check_existing()
    if dry_run:
        return summary
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    try:
        (temporary / MANIFEST).parent.mkdir(parents=True)
        for path, payload in ((temporary / MANIFEST, content), (temporary / RECEIPT, _json(receipt) + b"\n")):
            with path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            # The destination is always a complete nonempty directory. POSIX
            # rename cannot overwrite another worker's published directory.
            os.rename(temporary, output)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            return check_existing()
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return check_existing()


def prepare_action_view(prompt_cache, output, prompts, dry_run=False):
    cache = Path(prompt_cache).expanduser().resolve()
    prompt_rows = validate_prompts(prompts)
    receipt = validate_cache(cache, prompt_rows)
    content = (cache / "metadata.jsonl").read_bytes()
    rows = _rows(content)
    for row in rows:
        geometry = tuple(row.get(key, row.get(fallback)) for key, fallback in (("target_num_frames", "num_frames"), ("target_height", "height"), ("target_width", "width")))
        if geometry != (124, 768, 1344) or row.get("fps", 24) != 24:
            raise ValueError("Action cache must use 124x768x1344 at 24fps")
        row["condition_path"] = str((cache / row["condition_path"]).resolve())
        row.update(task="t2av", target_num_frames=124, target_height=768, target_width=1344, fps=24)
    identity = {"manifest": str(cache / "metadata.jsonl"), "manifest_sha256": _sha(content), "prompt_sha256": _sha(Path(prompts).read_bytes()), "cache_receipt": receipt}
    if identity["manifest_sha256"] != receipt["manifest_sha256"]:
        raise ValueError("Prompt manifest changed while preparing its view")
    return _publish(output, "action", rows, identity, dry_run)


def prepare_selfgen_view(source, output, dry_run=False):
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        source /= "metadata.jsonl"
    before = _sha(source.read_bytes())
    helper = load_module("_h3_staged_gan_real", ROOT / "lightx2v_train/trainers/dmd/dmd2_gan_data.py")
    manifest, rows, fingerprint = helper.load_gan_real_records(source)
    if _sha(source.read_bytes()) != before:
        raise ValueError("Self-generated source manifest changed while preparing its view")
    identity = {"manifest": str(manifest), "manifest_sha256": before, "gan_selection_fingerprint": fingerprint}
    return _publish(output, "selfgen", rows, identity, dry_run)
