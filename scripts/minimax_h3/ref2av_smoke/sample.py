#!/usr/bin/env python3
"""Select 10 combined Seedance/H3 + 10 Omni successful IR rows; copy targets.

No model imports or API calls. Scan one input snapshot, assign seeded random
priorities, then validate candidates until each quota is filled. This avoids
probing every video in a 130k-row manifest. Source JSONL/media are never edited.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KINDS = ("image", "video", "audio")
LABELS = {"image": "Picture", "video": "Video", "audio": "Audio"}
LIMITS = {"image": 9, "video": 3, "audio": 3}
DEFAULT_INPUT = "/mnt/lm_data_afs/gushiqiao/datasets/h3_ref2av_ir_simple/ref2av.minimaxi.com.jsonl"
DEFAULT_DATA_ROOT = Path("/mnt/lm_data_afs/gushiqiao/datasets")


def classify(row):
    """Classify provenance fields, never words inside the prompt."""
    source = str(row.get("source_name") or "").lower()
    sample_id = str(row.get("sample_id") or "").lower()
    seko = row.get("seko") or {}
    if not isinstance(seko, dict):
        seko = {}
    provider = str(seko.get("provider") or "").lower()
    model = " ".join(str(value or "").lower() for value in (row.get("subtask"), seko.get("trans_group")))
    if source in {"omni", "omni_r2v", "omni-r2v", "omini-r2v"} or sample_id.startswith("r2v_"):
        return "omni", "omni"
    is_seko = source.startswith("seko") or sample_id.startswith("seko_")
    if is_seko and ("h3" in model or sample_id.startswith("seko_minimax_")):
        return "seko", "h3"
    if is_seko and ("seedance" in model or "volc_2_5" in model or sample_id.startswith("seko_volc_")):
        return "seko", "seedance"
    # These aliases are used by split-source exports with no sample prefix.
    if source == "seko_minimax" and provider in {"", "minimax"}:
        return "seko", "h3"
    if source == "seko_volc" and provider in {"", "volc"}:
        return "seko", "seedance"
    return None


def local_file(value, root, repairs):
    if isinstance(value, dict):
        value = value.get("path") or value.get("rel_path")
    if not isinstance(value, str) or not value.strip() or "://" in value or value.startswith("data:"):
        raise ValueError("invalid_local_path: a local filename is required")
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError("invalid_local_path: parent traversal is unsupported")
    path = path if path.is_absolute() else root / path
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        # Never rewrite the media root or choose a replacement outside it.
        try:
            relative = path.relative_to(root)
        except ValueError:
            raise FileNotFoundError(f"missing_file: {path}") from None
        candidates = set()
        for i, part in enumerate(relative.parts[:-1]):
            if part == "downloads":
                candidate = root.joinpath(*(relative.parts[:i] + relative.parts[i + 1 :]))
                if candidate.is_file():
                    candidates.add(candidate)
        if len(candidates) != 1:
            raise FileNotFoundError(f"missing_file: {path}") from None
        replacement = candidates.pop()
        repairs.append({"old_path": str(path), "new_path": str(replacement)})
        path, size = replacement, replacement.stat().st_size
    if not path.is_file() or size <= 0:
        raise ValueError(f"empty_or_nonfile: {path}")
    return path.resolve()


def ffprobe_video(path, executable, timeout):
    result = subprocess.run(
        [executable, "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not any(item.get("codec_type") == "video" for item in streams):
        raise ValueError("invalid_video: no video stream")
    return any(item.get("codec_type") == "audio" for item in streams)


def validate_candidate(row, group, args):
    if row.get("ir_status") != "succeeded" or not isinstance(row.get("enhanced_prompt"), str) or not row["enhanced_prompt"].strip():
        raise ValueError("not_enhanced: requires successful nonempty enhanced_prompt")
    if row.get("task", "ref2av") not in {"ref2av", "ref2va"}:
        raise ValueError("wrong_task: requires Ref2AV")
    root = (args.seko_media_root if group == "seko" else args.omni_media_root).expanduser().resolve()
    values = row.get("references")
    if not isinstance(values, list) or not values or len(values) > 12:
        raise ValueError("reference_count: requires 1..12 ordered references")
    orders = [item.get("order") for item in values if isinstance(item, dict)]
    if any(value is not None for value in orders) and orders not in (list(range(len(values))), list(range(1, len(values) + 1))):
        raise ValueError("reference_order: order must match the existing list")
    references, repairs, counts, video_audio = [], [], Counter(), 0
    last_kind = -1
    for item in values:
        if not isinstance(item, dict) or item.get("kind") not in KINDS:
            raise ValueError("reference_kind: expected image/video/audio")
        kind = item["kind"]
        if KINDS.index(kind) < last_kind:
            raise ValueError("reference_order: expected image then video then audio")
        last_kind = KINDS.index(kind)
        counts[kind] += 1
        if counts[kind] > LIMITS[kind]:
            raise ValueError("reference_count: exceeds 9 images / 3 videos / 3 audios")
        label = f"<{LABELS[kind]} {counts[kind]}>"
        if item.get("canonical_label") is not None and item["canonical_label"] != label:
            raise ValueError("reference_label: inconsistent within-modality numbering")
        path = local_file(item.get("path") or item.get("rel_path"), root, repairs)
        if "," in str(path) or str(path) != str(path).strip():
            raise ValueError("unsupported_filename: comma/edge whitespace cannot be passed through this API")
        reference = {**item, "kind": kind, "path": str(path), "canonical_label": label}
        if kind == "video":
            has_audio = ffprobe_video(path, args.ffprobe, args.probe_timeout)
            video_audio += int(has_audio)
            reference["has_audio"] = has_audio
        references.append(reference)
    if not counts["image"] + counts["video"]:
        raise ValueError("audio_only: a visual reference is required")
    if video_audio and counts["audio"]:
        raise ValueError("audio_numbering: video soundtrack plus standalone audio is ambiguous for IR labels")
    if video_audio + counts["audio"] > 3:
        raise ValueError("audio_count: video soundtracks count toward the 3-audio limit")
    target = local_file(row.get("target") or row.get("target_video_path") or row.get("target_path"), root, repairs)
    if any(Path(reference["path"]) == target for reference in references):
        raise ValueError("target_conditioning: target video must not also be a reference")
    if target.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        raise ValueError("target_format: expected a local target video")
    return references, target, repairs


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_target(source, destination):
    """Real copy, not symlink; never overwrite a different previous target."""
    if destination.exists():
        if file_sha256(source) != file_sha256(destination):
            raise FileExistsError(f"Target copy differs: {destination}; choose a new output directory")
        return
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".copy-", delete=False) as out:
        temporary = Path(out.name)
        try:
            with source.open("rb") as src:
                shutil.copyfileobj(src, out, 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
            # Link creation is atomic and refuses an existing destination.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def write_json_atomic(path, value, jsonl=False):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
        temporary = Path(out.name)
        try:
            if jsonl:
                for row in value:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(value, out, ensure_ascii=False, indent=2)
                out.write("\n")
            out.flush()
            os.fsync(out.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def run(args):
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".sample.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "samples.jsonl").exists():
            raise FileExistsError(f"{out / 'samples.jsonl'} exists; use it for post.py or select a new --output-dir")
        return select_rows(args, out)


def select_rows(args, out):
    candidates = {"seko": [], "omni": []}
    rejected, eligible, seen = Counter(), Counter(), set()
    selected = []
    path = args.input.expanduser().resolve()
    with path.open("rb") as handle:
        snapshot_bytes = os.fstat(handle.fileno()).st_size
        line_number = 0
        while handle.tell() < snapshot_bytes:
            offset = handle.tell()
            raw = handle.readline(snapshot_bytes - offset)
            line_number += 1
            if line_number % 10000 == 0:
                print(json.dumps({"stage": "scan", "lines": line_number, "eligible_before_media_check": eligible}), flush=True)
            if not raw.strip():
                continue
            if not raw.endswith(b"\n"):
                # The enhancing process may still be appending its final row.
                rejected["incomplete_final_line"] += 1
                break
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                rejected["invalid_json"] += 1
                continue
            if not isinstance(row, dict) or row.get("ir_status") != "succeeded" or not isinstance(row.get("enhanced_prompt"), str) or not row["enhanced_prompt"].strip():
                rejected["not_enhanced"] += 1
                continue
            origin = classify(row)
            if origin is None:
                rejected["other_source"] += 1
                continue
            group, provider = origin
            identity = str(row.get("sample_id") or row.get("metadata_id", f"line-{line_number}"))
            if (group, identity) in seen:
                rejected["duplicate_sample"] += 1
                continue
            seen.add((group, identity))
            priority = hashlib.sha256(f"{args.seed}:{group}:{identity}".encode()).hexdigest()
            candidates[group].append((priority, offset, len(raw), line_number, provider, hashlib.sha256(raw).hexdigest()))
            eligible[group] += 1
        print(json.dumps({"stage": "scan_complete", "lines": line_number, "eligible_before_media_check": eligible, "excluded": rejected}, ensure_ascii=False), flush=True)
        for group, quota in (("seko", args.seko_count), ("omni", args.omni_count)):
            count = 0
            for _, offset, length, source_line, provider, digest in sorted(candidates[group]):
                if count == quota:
                    break
                handle.seek(offset)
                raw = handle.read(length)
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise RuntimeError("Input changed during sampling; retry against a stable snapshot")
                row = json.loads(raw)
                try:
                    refs, target, repairs = validate_candidate(row, group, args)
                except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
                    reason = str(error).split(":", 1)[0] if isinstance(error, ValueError) else type(error).__name__
                    rejected[f"{group}:{reason}"] += 1
                    continue
                index = len(selected)
                slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("sample_id") or row.get("metadata_id", source_line)))[:80].strip("._") or "sample"
                sample_id = f"{index:02d}_{group}_{slug}"
                target_copy = out / "targets" / f"{sample_id}{target.suffix.lower()}"
                record = {
                    **row,
                    "smoke": {
                        "id": sample_id,
                        "index": index,
                        "group": group,
                        "provider": provider,
                        "input_manifest": str(path),
                        "input_line": source_line,
                        "source_row_sha256": digest,
                        "references": refs,
                        "path_repairs": repairs,
                        "target_source": str(target),
                        "target_copy": str(target_copy),
                        "generated_path": str(out / "generated" / f"{sample_id}.mp4"),
                        "seed": args.seed + index,
                        "num_frames": 124,
                        "size": [768, 1344],
                        "fps": 24,
                    },
                }
                selected.append(record)
                count += 1
                print(f"selected {group} {count}/{quota}: {sample_id}", flush=True)
            if count != quota:
                raise ValueError(f"Not enough valid {group} rows: need {quota}, found {count}; exclusions={dict(rejected)}. No samples.jsonl was written.")
    (out / "targets").mkdir(exist_ok=True)
    (out / "generated").mkdir(exist_ok=True)
    for row in selected:
        smoke = row["smoke"]
        copy_target(Path(smoke["target_source"]), Path(smoke["target_copy"]))
    summary = {
        "input": str(path),
        "snapshot_bytes": snapshot_bytes,
        "seed": args.seed,
        "selected": dict(Counter(row["smoke"]["group"] for row in selected)),
        "providers": dict(Counter(row["smoke"]["provider"] for row in selected)),
        "excluded": dict(rejected),
        "num_frames": 124,
        "fps": 24,
        "size": [768, 1344],
        "target_policy": "byte-for-byte copy; original duration and resolution retained",
        "prompt_policy": "enhanced_prompt unchanged; output geometry overrides source timing",
    }
    write_json_atomic(out / "selection.json", summary)
    write_json_atomic(out / "samples.jsonl", selected, jsonl=True)
    print(json.dumps({"manifest": str(out / "samples.jsonl"), **summary}, ensure_ascii=False), flush=True)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=Path, default=REPO / "save_results/ref2av_smoke20")
    parser.add_argument("--seko-count", type=int, default=10, help="Combined Seedance/H3 count, not ten each")
    parser.add_argument("--omni-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seko-media-root", type=Path, default=DEFAULT_DATA_ROOT / "seko_data")
    parser.add_argument("--omni-media-root", type=Path, default=DEFAULT_DATA_ROOT / "omini-r2v")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--probe-timeout", type=float, default=15)
    args = parser.parse_args(argv)
    if args.seko_count < 0 or args.omni_count < 0 or args.seko_count + args.omni_count == 0 or args.seed < 0 or not math.isfinite(args.probe_timeout) or args.probe_timeout <= 0:
        parser.error("counts/seed must be nonnegative, total count positive, and probe timeout positive")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}")
