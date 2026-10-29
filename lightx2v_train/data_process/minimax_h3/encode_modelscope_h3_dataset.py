#!/usr/bin/env python3
"""Shard H3 self-generated MP4s into clean AV latents and cached conditions.

One process owns one GPU. Text encoders are released before the FP32 VAEs
are loaded; the diffusion transformer is never loaded. Source frame geometry
is preserved, including native aspect ratios outside the training 768p views.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SCHEMA_VERSION = 1
TRAIN_ROOT = Path(__file__).resolve().parents[2]


def _module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()).hexdigest()


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_rows(path, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    rows, identities = [], set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("source_id") if isinstance(row, dict) else None
            if not isinstance(identity, str) or not identity.strip() or identity in identities:
                raise ValueError(f"Missing or duplicate source_id at {path}:{line_number}")
            identities.add(identity)
            if row.get("task") not in {"t2av", "ref2av"}:
                raise ValueError(f"Unsupported task at {path}:{line_number}")
            prompt = row.get("caption", row.get("prompt"))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing prompt at {path}:{line_number}")
            height, width = int(row["target_height"]), int(row["target_width"])
            frames = int(row.get("target_num_frames", row.get("num_frames", 0)))
            if height <= 0 or width <= 0 or height % 32 or width % 32 or frames != 124:
                raise ValueError(f"Expected 124 frames and positive 32-aligned native dimensions at {path}:{line_number}")
            row = dict(row, caption=prompt, prompt=prompt, target_num_frames=frames, num_frames=frames)
            if (len(identities) - 1) % num_shards == shard_index:
                rows.append(row)
    return rows


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_identity(model_path, source_model_path, tasks, reference_helper):
    result = {"model_path": str(model_path), "source_model_path": str(source_model_path)}
    result["vae"] = [reference_helper._component_identity(model_path, name) for name in ("vae", "audio_vae")]
    if "t2av" in tasks:
        result["text"] = [reference_helper._component_identity(model_path, name) for name in ("text_encoder", "tokenizer", "processor")]
    if "ref2av" in tasks:
        result["reference"] = reference_helper.model_identity(source_model_path, model_path)
    return result


def initialize_namespace(path, descriptor):
    if path.is_file():
        if json.loads(path.read_text()) != descriptor:
            raise RuntimeError(f"Encoding namespace changed at {path}; use a new cache output directory.")
    else:
        atomic_json(path, descriptor)


def _silent_reference(path, output_dir):
    """Video 1 is visual-only even when its source MP4 contains an audio track."""
    import av

    source = Path(path).resolve()
    with av.open(str(source)) as container:
        if not container.streams.video:
            raise ValueError(f"Reference has no video stream: {source}")
        if not container.streams.audio:
            return source
    key = digest(_file_identity(source))
    destination = output_dir / "silent_references" / f"{key}.mp4"
    if destination.is_file():
        with av.open(str(destination)) as container:
            if not container.streams.video or container.streams.audio:
                raise RuntimeError(f"Invalid cached silent reference: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{key}.{os.getpid()}.tmp.mp4")
    try:
        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-c:v", "copy", "-an", "-movflags", "+faststart", str(temporary)],
                check=True,
                capture_output=True,
            )
        else:
            # PyAV already provides the decoder dependency. Remux encoded
            # packets directly so removing reference audio needs no extra OS
            # binary and preserves the original visual stream without a decode.
            with av.open(str(source)) as input_file, av.open(str(temporary), mode="w", format="mp4", options={"movflags": "+faststart"}) as output_file:
                input_stream = input_file.streams.video[0]
                if hasattr(output_file, "add_stream_from_template"):
                    output_stream = output_file.add_stream_from_template(input_stream)
                else:
                    output_stream = output_file.add_stream(template=input_stream)
                for packet in input_file.demux(input_stream):
                    if packet.dts is None:
                        continue
                    packet.stream = output_stream
                    output_file.mux(packet)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_jobs(rows, output_dir, descriptor, reference_helper, dtype_name):
    jobs = []
    for row in rows:
        identity = row["source_id"]
        key = hashlib.sha256(identity.encode()).hexdigest()
        source = Path(row["video_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing target video for {identity}: {source}")
        job = {
            "row": row,
            "key": key,
            "source_video": source,
            "condition_path": output_dir / "conditions" / f"{key}.pt",
            "video_latent_path": output_dir / "video_latents" / f"{key}.pt",
            "audio_latent_path": output_dir / "audio_latents" / f"{key}.pt",
            "item_receipt": output_dir / "items" / f"{key}.json",
            "expected_video": (24, 37, row["target_height"] // 16, row["target_width"] // 16),
            "expected_audio": (2, 32, 207),
        }
        if row["task"] == "ref2av":
            ref_row = copy.deepcopy(row)
            ref_row["sample_id"] = identity
            ref_row["enhanced_prompt"] = row["caption"]
            for reference in ref_row.get("references", []):
                if reference.get("kind", reference.get("type")) == "video" and reference.get("include_embedded_audio") is False:
                    reference["path"] = str(_silent_reference(reference["path"], output_dir))
            sample = reference_helper.normalize_sample(
                ref_row,
                int(row["source_index"]),
                output_dir,
                dtype_name,
                descriptor,
                reference_resize_mode="match",
                prompt_policy="enhanced-only",
                reference_latent_dtype=dtype_name,
            )
            job["reference_sample"] = sample
            job["condition_fingerprint"] = sample.fingerprint
        else:
            job["condition_fingerprint"] = digest(
                {
                    "schema_version": SCHEMA_VERSION,
                    "source_id": identity,
                    "prompt": row["caption"],
                    "height": row["target_height"],
                    "width": row["target_width"],
                    "frames": 124,
                    "task": "t2av",
                    "model": descriptor,
                    "dtype": dtype_name,
                }
            )
        job["latent_fingerprint"] = digest(
            {
                "condition_fingerprint": job["condition_fingerprint"],
                "source": _file_identity(source),
                "expected_video": job["expected_video"],
                "expected_audio": job["expected_audio"],
                "posterior": "mode",
                "vae_compute_dtype": "fp32",
                "storage_dtype": dtype_name,
            }
        )
        jobs.append(job)
    return jobs


def finite(value, label):
    import torch

    if torch.is_tensor(value):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite cached tensor: {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            finite(item, f"{label}[{index}]")


def validate_condition(job, dtype, reference_helper, require_complete=True):
    import torch

    payload = torch.load(job["condition_path"], map_location="cpu", weights_only=True)
    if payload.get("source_id") != job["row"]["source_id"] or payload.get("cache_fingerprint") != job["condition_fingerprint"]:
        raise RuntimeError(f"Condition cache identity changed: {job['condition_path']}; use a new cache output directory.")
    if job["row"]["task"] == "ref2av":
        stage = reference_helper.validate_cache_payload(payload, job["reference_sample"], job["condition_path"], dtype, require_complete)
    else:
        positive = payload.get("conditioning", {}).get("positive", {})
        embeds, tags = positive.get("prompt_embeds"), positive.get("text_token_tags")
        if not torch.is_tensor(embeds) or embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[2] != 5120 or embeds.dtype != dtype:
            raise ValueError(f"Invalid T2AV text embeddings: {job['condition_path']}")
        if not torch.is_tensor(tags) or tags.shape != (embeds.shape[1],) or tags.dtype != torch.long or not bool((tags == 1).all()):
            raise ValueError(f"Invalid T2AV text-only tags: {job['condition_path']}")
        scalars = {"task": "t2av", "target_height": job["row"]["target_height"], "target_width": job["row"]["target_width"], "target_num_frames": 124}
        if any(positive.get(key) != value for key, value in scalars.items()) or payload.get("cache_stage") != "complete":
            raise ValueError(f"Invalid T2AV condition geometry/stage: {job['condition_path']}")
        stage = "complete"
    finite(payload, str(job["condition_path"]))
    return payload, stage


def validate_targets(job, dtype):
    import torch

    video = torch.load(job["video_latent_path"], map_location="cpu", weights_only=True)
    audio = torch.load(job["audio_latent_path"], map_location="cpu", weights_only=True)
    if not isinstance(video, dict) or video.get("latent_fingerprint") != job["latent_fingerprint"] or video.get("normalized") is not True:
        raise RuntimeError(f"Video latent source/normalization mismatch: {job['video_latent_path']}")
    tensor = video.get("latents")
    if not torch.is_tensor(tensor) or tuple(tensor.shape) != job["expected_video"] or tensor.dtype != dtype:
        raise ValueError(f"Invalid video latent shape/dtype: {job['video_latent_path']}")
    if not torch.is_tensor(audio) or tuple(audio.shape) != job["expected_audio"] or audio.dtype != dtype:
        raise ValueError(f"Invalid audio latent shape/dtype: {job['audio_latent_path']}")
    finite((tensor, audio), job["row"]["source_id"])


def manifest_row(job, payload, reference_helper):
    row = dict(job["row"])
    if row["task"] == "ref2av":
        row.update(reference_helper.manifest_row(job["reference_sample"], job["condition_path"], payload))
    row.update(
        {
            "source_id": job["row"]["source_id"],
            "cache_fingerprint": job["condition_fingerprint"],
            "condition_path": str(job["condition_path"]),
            "video_latent_path": str(job["video_latent_path"]),
            "audio_latent_path": str(job["audio_latent_path"]),
            "video_path": str(job["source_video"]),
            "video_latent_shape": list(job["expected_video"]),
            "audio_latent_shape": list(job["expected_audio"]),
            "latent_fingerprint": job["latent_fingerprint"],
            "vae_normalized": True,
            "target_num_frames": 124,
            "num_frames": 124,
            "fps": 24,
            "posterior_mode": "mode",
            "source_kind": "teacher_generated",
            "cache_storage_dtype": str(payload["conditioning"]["positive"]["prompt_embeds"].dtype).removeprefix("torch."),
        }
    )
    positive = payload["conditioning"]["positive"]
    row["prompt_token_count"] = int(positive["prompt_embeds"].shape[1])
    row["target_orientation"] = "landscape" if row["target_width"] > row["target_height"] else "portrait"
    row["packed_sequence_tokens_124"] = (
        row["prompt_token_count"] + row.get("reference_video_rows", 0) + row.get("reference_audio_rows", 0) + 37 * (row["target_height"] // 32) * (row["target_width"] // 32) + 414
    )
    return row


def condition_only_row(row):
    row = dict(row)
    for key in ("video_latent_path", "audio_latent_path", "video_latent_shape", "audio_latent_shape"):
        row.pop(key, None)
    return row


def adapt_reference_runtime(runtime):
    """Bridge the released helper API to capability-detected modular blocks.

    Older Diffusers exposes static preparation/encoding methods. Newer builds
    expose in-memory modality-specific references and PipelineState blocks.
    Keep the adaptation local to this worker; shared/legacy cache builders are
    unchanged.
    """
    required = (("MiniMaxH3Ref2VASetupStep", "prepare_references"), ("MiniMaxH3Ref2VATextEncoderStep", "encode_prompt"), ("MiniMaxH3Ref2VAReferenceEncoderStep", "encode_references"))
    if all(hasattr(runtime[key], method) for key, method in required):
        return runtime
    import torch
    from PIL import Image
    from diffusers.modular_pipelines.minimax_h3 import references as reference_types
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    setup = runtime["MiniMaxH3Ref2VASetupStep"]
    text_step = runtime["MiniMaxH3Ref2VATextEncoderStep"]
    reference_step = runtime["MiniMaxH3Ref2VAReferenceEncoderStep"]
    classes = {kind: getattr(reference_types, name) for kind, name in (("image", "MiniMaxH3ImageReference"), ("video", "MiniMaxH3VideoReference"), ("audio", "MiniMaxH3AudioReference"))}
    if not all(callable(getattr(cls, "from_file", None)) for cls in classes.values()) or not hasattr(setup, "_normalize_video_condition") or not hasattr(setup, "_normalize_audio_condition"):
        raise RuntimeError("Unsupported MiniMax-H3 Ref2VA modular API; need static encoders or in-memory reference/PipelineState blocks.")

    def reference_factory(**kwargs):
        if len(kwargs) != 1 or next(iter(kwargs)) not in classes:
            raise ValueError(f"Expected one image/video/audio reference path, got {kwargs}")
        kind, path = next(iter(kwargs.items()))
        return classes[kind].from_file(path)

    resolver_module = SimpleNamespace(resolve_reference_image_size=lambda width, height: (height, width))

    class SetupAdapter:
        @staticmethod
        def prepare_references(components, references, num_frames):
            prepared = []
            for entry in references:
                audio = None
                if entry.has_audio:
                    audio = setup._normalize_audio_condition(entry.audio, entry.sample_rate or components.audio_sampling_rate, components.audio_sampling_rate, num_frames / 24.0)
                if entry.kind == "image":
                    image = entry.image.convert("RGB")
                    height, width = resolver_module.resolve_reference_image_size(image.width, image.height)
                    prepared.append(classes["image"](image=image.resize((width, height), Image.Resampling.LANCZOS)))
                elif entry.kind == "video":
                    frames = setup._normalize_video_condition(entry.frames, entry.fps, num_frames, 32, 768, 768 * 1344, 24.0)
                    prepared.append(classes["video"](frames=frames, fps=24.0, audio=audio, sample_rate=components.audio_sampling_rate))
                else:
                    prepared.append(classes["audio"](audio=audio, sample_rate=components.audio_sampling_rate))
            return prepared, num_frames

    class TextAdapter:
        @staticmethod
        def encode_prompt(components, prompt, references, device, dtype):
            components.fps, components.text_tag, components.video_tag, components.text_encoder_layer = 24.0, 1, 0, 50
            state = PipelineState({"prompt": prompt, "normalized_references": references})
            _, state = text_step(video_sample_fps=2.0)(components, state)
            return state.get("prompt_embeds").to(dtype=dtype), state.get("text_token_tags")

    class ReferenceAdapter:
        @staticmethod
        def encode_references(components, references, device):
            from lightx2v_train.model_zoo.native.minimax_h3.packing import patchify_video_latents

            components.pixel_mean = (0.485, 0.456, 0.406)
            components.pixel_std = (0.229, 0.224, 0.225)
            components.keyframe_encode_seed = 42
            components.vae_frames_per_chunk, components.vae_latents_per_chunk = 17, 5
            state = PipelineState({"normalized_references": references})
            _, state = reference_step()(components, state)
            videos, audios = state.get("condition_latents"), state.get("audio_condition_latents")
            video_rows = audio_rows = None
            if videos:
                if len(videos) != 1:
                    raise ValueError("Reference cache adapter encodes exactly one reference at a time")
                latent = videos[0]
                references[0].num_latent_frames = int(latent.shape[2])
                references[0].latent_height = int(latent.shape[3])
                references[0].latent_width = int(latent.shape[4])
                video_rows = patchify_video_latents(latent, components.patch_size)
            if audios:
                audio_rows = torch.cat(audios, dim=0)
                references[0].num_audio_latents = int(audio_rows.shape[0]) // 2
            return video_rows, audio_rows

    return {
        **runtime,
        "before_encoder_module": resolver_module,
        "MiniMaxH3Reference": reference_factory,
        "MiniMaxH3Ref2VASetupStep": SetupAdapter,
        "MiniMaxH3Ref2VATextEncoderStep": TextAdapter,
        "MiniMaxH3Ref2VAReferenceEncoderStep": ReferenceAdapter,
    }


def release_models():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args):
    import torch

    sys.path.insert(0, str(TRAIN_ROOT))
    helper = _module("encode_minimax_h3_teacher_av_latents.py", "_modelscope_h3_av")
    reference = _module("build_minimax_h3_ref2av_condition_caches.py", "_modelscope_h3_ref")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    rows = read_rows(args.metadata, args.shard_index, args.num_shards)
    descriptor = model_identity(args.model_path, args.source_model_path, {row["task"] for row in rows}, reference)
    descriptor["preprocessing"] = {"vae_tiling": args.vae_tiling, "tile_size": 512, "tile_overlap": 64, "reference_resize_mode": "match", "posterior": "mode"}
    shard = f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f".{shard}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace = {
            "schema_version": SCHEMA_VERSION,
            "prepared_sha256": sha256_file(args.metadata),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "dtype": args.dtype,
            "model": descriptor,
        }
        initialize_namespace(args.output_dir / f"config.{shard}.json", namespace)
        complete_path = args.output_dir / f"receipt.{shard}.json"
        complete_path.unlink(missing_ok=True)
        jobs = build_jobs(rows, args.output_dir, descriptor, reference, args.dtype)
        failures = []

        def fail(job, stage, error):
            job["failed"] = True
            failures.append({"source_id": job["row"]["source_id"], "source_index": job["row"]["source_index"], "stage": stage, "error": str(error)})
            print(f"[failed] {stage} {job['row']['source_id']}: {error}", flush=True)

        for job in jobs:
            if job["condition_path"].is_file():
                payload, stage = validate_condition(job, dtype, reference, require_complete=False)
                job["condition_stage"] = stage
                del payload
            if job["item_receipt"].is_file():
                saved = json.loads(job["item_receipt"].read_text())
                if saved.get("latent_fingerprint") != job["latent_fingerprint"]:
                    raise RuntimeError(f"Latent source contract changed for {job['row']['source_id']}; use a new cache output directory.")
                validate_targets(job, dtype)
                for field in ("condition_path", "video_latent_path", "audio_latent_path"):
                    if saved["files"].get(field) != sha256_file(job[field]):
                        raise RuntimeError(f"Cached file content changed for {job[field]}; use a new cache output directory.")
                job["targets_complete"] = True

        t2av_jobs = [job for job in jobs if job["row"]["task"] == "t2av" and not job.get("condition_stage")]
        if t2av_jobs:
            from lightx2v_train.model_zoo.minimax_h3.condition_encoder import MiniMaxH3ConditionEncoder

            print(f"[{shard}] T2AV Qwen text encoding samples={len(t2av_jobs)} dtype={args.dtype}", flush=True)
            encoder = MiniMaxH3ConditionEncoder(str(args.model_path), device=args.device, dtype=dtype)
            for index, job in enumerate(t2av_jobs, 1):
                try:
                    positive = {key: value.detach().cpu().contiguous() if torch.is_tensor(value) else value for key, value in encoder.encode(job["row"]["caption"]).items()}
                    positive.update(task="t2av", target_height=job["row"]["target_height"], target_width=job["row"]["target_width"], target_num_frames=124)
                    payload = {
                        "cache_schema_version": 1,
                        "cache_stage": "complete",
                        "cache_fingerprint": job["condition_fingerprint"],
                        "source_id": job["row"]["source_id"],
                        "source_index": job["row"]["source_index"],
                        "task": "t2av",
                        "conditioning": {"positive": positive},
                    }
                    finite(payload, job["row"]["source_id"])
                    helper.atomic_torch_save(payload, job["condition_path"])
                    validate_condition(job, dtype, reference)
                    job["condition_stage"] = "complete"
                    print(f"[{shard} text {index}/{len(t2av_jobs)}] {job['row']['source_id']}", flush=True)
                except Exception as error:
                    fail(job, "text", error)
            del encoder
            release_models()

        ref_jobs = [job for job in jobs if job["row"]["task"] == "ref2av" and not job.get("failed")]
        pending_text = [job for job in ref_jobs if not job.get("condition_stage")]
        runtime = adapt_reference_runtime(reference.import_official_runtime()) if ref_jobs else None
        if pending_text:
            print(f"[{shard}] Ref2AV Qwen multimodal encoding samples={len(pending_text)} dtype={args.dtype}", flush=True)
            conditioner = reference.load_conditioner(runtime, args.source_model_path, args.device, dtype)
            for index, job in enumerate(pending_text, 1):
                try:
                    payload = reference.encode_text_stage(job["reference_sample"], runtime, conditioner, dtype)
                    finite(payload, job["row"]["source_id"])
                    helper.atomic_torch_save(payload, job["condition_path"])
                    job["condition_stage"] = "text"
                    print(f"[{shard} ref-text {index}/{len(pending_text)}] {job['row']['source_id']}", flush=True)
                except Exception as error:
                    fail(job, "reference_text", error)
            del conditioner
            release_models()

        need_vae = [job for job in jobs if not job.get("failed") and (not job.get("targets_complete") or job.get("condition_stage") == "text")]
        if need_vae:
            from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

            print(f"[{shard}] FP32 video/audio VAE encoding samples={len(need_vae)} storage={args.dtype}; no DiT is loaded", flush=True)
            device = torch.device(args.device)
            vae = AutoencoderKLMiniMaxH3.from_pretrained(str(args.model_path / "vae"), torch_dtype=torch.float32, local_files_only=True).requires_grad_(False).eval().to(device)
            audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(args.model_path / "audio_vae"), torch_dtype=torch.float32, local_files_only=True).requires_grad_(False).eval().to(device)
            if args.vae_tiling:
                vae.enable_tiling(tile_sample_min_height=512, tile_sample_min_width=512, tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)
            components = SimpleNamespace(vae=vae, audio_vae=audio_vae, _execution_device=device, patch_size=(1, 2, 2), audio_latent_channels=32)
            for index, job in enumerate(need_vae, 1):
                try:
                    if job.get("condition_stage") == "text":
                        payload, _ = validate_condition(job, dtype, reference, require_complete=False)
                        payload = reference.encode_reference_stage(job["reference_sample"], payload, runtime, components, dtype)
                        finite(payload, job["row"]["source_id"])
                        helper.atomic_torch_save(payload, job["condition_path"])
                        job["condition_stage"] = "complete"
                    if not job.get("targets_complete"):
                        waveform = helper.decode_audio(job["source_video"], 207 * 800)
                        pixels = helper.decode_video(job["source_video"], job["row"]["target_height"], job["row"]["target_width"], 124, 24.0)
                        video = helper.encode_video_latent(vae, pixels, device, "mode", 0).to(dtype=dtype)
                        del pixels
                        audio = helper.encode_audio_latent(audio_vae, waveform, device).to(dtype=dtype)
                        del waveform
                        if tuple(video.shape) != job["expected_video"] or tuple(audio.shape) != job["expected_audio"]:
                            raise ValueError(f"Unexpected AV latent shapes: video={tuple(video.shape)}, audio={tuple(audio.shape)}")
                        finite((video, audio), job["row"]["source_id"])
                        helper.atomic_torch_save(
                            {
                                "latents": video.contiguous(),
                                "normalized": True,
                                "posterior_mode": "mode",
                                "latent_fingerprint": job["latent_fingerprint"],
                                "pixel_num_frames": 124,
                                "pixel_height": job["row"]["target_height"],
                                "pixel_width": job["row"]["target_width"],
                            },
                            job["video_latent_path"],
                        )
                        helper.atomic_torch_save(audio.contiguous(), job["audio_latent_path"])
                        validate_targets(job, dtype)
                        validate_condition(job, dtype, reference)
                        atomic_json(
                            job["item_receipt"],
                            {
                                "schema_version": 1,
                                "source_id": job["row"]["source_id"],
                                "latent_fingerprint": job["latent_fingerprint"],
                                "files": {field: sha256_file(job[field]) for field in ("condition_path", "video_latent_path", "audio_latent_path")},
                            },
                        )
                        job["targets_complete"] = True
                        del video, audio
                    print(f"[{shard} AV {index}/{len(need_vae)}] {job['row']['source_id']}", flush=True)
                except Exception as error:
                    fail(job, "reference_or_target_vae", error)
            del vae, audio_vae, components
            release_models()

        completed = []
        for job in jobs:
            if job.get("failed") or not job.get("targets_complete") or job.get("condition_stage") != "complete":
                continue
            payload, _ = validate_condition(job, dtype, reference)
            validate_targets(job, dtype)
            completed.append(manifest_row(job, payload, reference))
        atomic_jsonl(args.output_dir / f"metadata.{shard}.jsonl", completed)
        atomic_jsonl(args.output_dir / f"condition_metadata.{shard}.jsonl", [condition_only_row(row) for row in completed])
        atomic_jsonl(args.output_dir / f"failed.{shard}.jsonl", failures)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "prepared_sha256": namespace["prepared_sha256"],
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "expected_count": len(rows),
            "completed_count": len(completed),
            "failed_count": len(failures),
            "model": descriptor,
            "storage_dtype": args.dtype,
            "vae_compute_dtype": "fp32",
        }
        if receipt["completed_count"] + receipt["failed_count"] != receipt["expected_count"]:
            raise RuntimeError(f"Shard accounting mismatch: {receipt}")
        atomic_json(complete_path, receipt)
        print(json.dumps({key: value for key, value in receipt.items() if key != "model"}, ensure_ascii=False), flush=True)
        return 1 if failures and not args.allow_failures else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-model-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--no-vae-tiling", action="store_false", dest="vae_tiling", help="Disable default 512px/64px-overlap tiling; requires a new cache directory.")
    parser.set_defaults(vae_tiling=True)
    parser.add_argument("--stage", choices=("all",), default="all")
    args = parser.parse_args(argv)
    for name in ("metadata", "output_dir", "model_path"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.source_model_path = (args.source_model_path or args.model_path / "Ref2VA").expanduser().resolve()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require num_shards > 0 and 0 <= shard_index < num_shards")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
