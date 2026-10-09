"""Strict, stdlib-only join of Ref2AV conditions and paired DMAD targets.

Run this file directly; no torch, CUDA, model, or target generation is needed::

    python minimax_h3_dmad_manifest.py --conditions conditions.jsonl \
        --real real.jsonl --teacher teacher.jsonl --output paired.jsonl

Every input row contains condition_path, cache_fingerprint, target_height,
target_width, and target_num_frames (num_frames is accepted as an alias).
Target rows also contain normalized=true and real_latent_path or
teacher_latent_path respectively (latent_path is accepted as an alias).
Paths are relative to their own manifest. The fingerprint identifies the exact
prompt, ordered references, and target assignment, not merely a source video.
Source IDs, if present on a condition row, must also match both target rows.

The output retains all condition metadata (including reference-cost sampler
fields), adds absolute target paths and dmad_schema_version=1. Existing output
files are never overwritten. Tensor content is validated by the dataset, not
by this metadata-only utility.
"""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
GEOMETRY_KEYS = ("target_height", "target_width", "target_num_frames")
SOURCE_ID_KEYS = ("source_id", "source_row_uid", "sample_id")


def _integer(value, key):
    if isinstance(value, bool) or value is None:
        raise ValueError(f"DMAD {key} must be an integer, got {value!r}.")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"DMAD {key} must be an integer, got {value!r}.") from error
    if isinstance(value, float) and value != result:
        raise ValueError(f"DMAD {key} must be an integer, got {value!r}.")
    return result


def target_geometry(row):
    """Return validated H3 output geometry, without importing torch."""
    height = _integer(row.get("target_height"), "target_height")
    width = _integer(row.get("target_width"), "target_width")
    frames = _integer(row.get("target_num_frames", row.get("num_frames")), "target_num_frames")
    if min(height, width) <= 0 or height % 32 or width % 32:
        raise ValueError(f"DMAD target height/width must be positive multiples of 32, got {height}x{width}.")
    if not 107 <= frames <= 362 or frames % 17 != 5:
        raise ValueError(f"DMAD target_num_frames must be 17*n+5 in [107,362], got {frames}.")
    if "num_frames" in row and _integer(row["num_frames"], "num_frames") != frames:
        raise ValueError("DMAD num_frames and target_num_frames disagree.")
    orientation = "landscape" if width > height else "portrait" if height > width else "square"
    for key in ("target_orientation", "aspect_bucket"):
        if row.get(key) not in (None, "", orientation):
            raise ValueError(f"DMAD {key}={row[key]!r} disagrees with {height}x{width} geometry.")
    return height, width, frames


def condition_fingerprint(row):
    value = row.get("cache_fingerprint")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("DMAD requires a non-empty cache_fingerprint identifying the exact condition; a source ID alone is insufficient.")
    return value


def require_matching_identity(expected, actual, label, *, require_source_ids=False):
    """Require exact condition fingerprint and reject conflicting source IDs."""
    if condition_fingerprint(actual) != condition_fingerprint(expected):
        raise ValueError(f"DMAD {label} cache_fingerprint does not match its condition.")
    for key in SOURCE_ID_KEYS:
        expected_value, actual_value = expected.get(key), actual.get(key)
        if expected_value not in (None, "") and (require_source_ids or actual_value not in (None, "")):
            if str(actual_value) != str(expected_value):
                raise ValueError(f"DMAD {label} {key} does not match its condition.")


def resolve_manifest_path(value, base_dir, key):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"DMAD requires {key}; condition-only caches are insufficient for DMAD.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DMAD {key} points to a missing file: {path}")
    return str(path)


def read_manifest(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "metadata.jsonl"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        elif path.suffix.lower() == ".json":
            rows = json.load(handle)
            if isinstance(rows, dict):
                rows = rows.get("samples", rows.get("items", rows.get("data", [rows])))
        elif path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(handle))
        else:
            raise ValueError(f"DMAD manifests must be JSONL, JSON, or CSV: {path}")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"DMAD manifest must contain non-empty mapping rows: {path}")
    return path, rows


def validate_joined_row(row, base_dir):
    row = dict(row)
    version = _integer(row.get("dmad_schema_version", SCHEMA_VERSION), "dmad_schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported DMAD manifest schema version: {version}")
    condition_fingerprint(row)
    row.update(zip(GEOMETRY_KEYS, target_geometry(row)))
    for key in ("condition_path", "real_latent_path", "teacher_latent_path"):
        row[key] = resolve_manifest_path(row.get(key), base_dir, key)
    if row["real_latent_path"] == row["teacher_latent_path"]:
        raise ValueError("DMAD real and teacher targets must be distinct files.")
    row["dmad_schema_version"] = SCHEMA_VERSION
    return row


def validate_manifest(path):
    """Validate a joined manifest using only metadata and file-existence checks."""
    path, records = read_manifest(path)
    rows = [validate_joined_row(row, path.parent) for row in records]
    _unique_rows(rows, str(path))
    return rows


def _unique_rows(rows, label):
    seen_paths, seen_fingerprints = set(), set()
    for row in rows:
        fingerprint = condition_fingerprint(row)
        condition_path = row["condition_path"]
        if condition_path in seen_paths or fingerprint in seen_fingerprints:
            raise ValueError(f"Duplicate DMAD condition_path/cache_fingerprint in {label}: {condition_path}")
        seen_paths.add(condition_path)
        seen_fingerprints.add(fingerprint)


def _normalized(value):
    return value is True or isinstance(value, str) and value.lower() == "true"


def _load_inputs(paths, role):
    result = []
    for path in paths:
        path, rows = read_manifest(path)
        for original in rows:
            row = dict(original)
            condition_fingerprint(row)
            row.update(zip(GEOMETRY_KEYS, target_geometry(row)))
            row["condition_path"] = resolve_manifest_path(row.get("condition_path"), path.parent, "condition_path")
            if role != "condition":
                key = f"{role}_latent_path"
                row[key] = resolve_manifest_path(row.get(key, row.get("latent_path")), path.parent, key)
                if not _normalized(row.get("normalized")):
                    raise ValueError(f"DMAD {role} manifest rows must declare normalized=true.")
            else:
                # Preserve paths consumed by the existing conditioning loader
                # after moving the joined manifest to a different directory.
                for key in ("negative_condition_path", "video_latent_path", "audio_latent_path"):
                    if row.get(key) not in (None, ""):
                        row[key] = resolve_manifest_path(row[key], path.parent, key)
                negative = path.parent / "negative_condition.pt"
                if not row.get("negative_condition_path") and negative.is_file():
                    row["negative_condition_path"] = str(negative.resolve())
            result.append(row)
    _unique_rows(result, f"{role} manifests")
    return result


def build_joined_rows(condition_manifests, real_manifests, teacher_manifests):
    conditions = _load_inputs(condition_manifests, "condition")
    real = {row["condition_path"]: row for row in _load_inputs(real_manifests, "real")}
    teacher = {row["condition_path"]: row for row in _load_inputs(teacher_manifests, "teacher")}
    condition_paths = {row["condition_path"] for row in conditions}
    for role, targets in (("real", real), ("teacher", teacher)):
        if condition_paths != targets.keys():
            missing = sorted(condition_paths - targets.keys())
            extra = sorted(targets.keys() - condition_paths)
            raise ValueError(f"DMAD {role} target manifest does not cover exactly the conditions: missing={missing[:5]}, extra={extra[:5]}.")
    joined = []
    for condition in conditions:
        row = dict(condition)
        for role, targets in (("real", real), ("teacher", teacher)):
            target = targets[condition["condition_path"]]
            require_matching_identity(condition, target, role, require_source_ids=True)
            if target_geometry(condition) != target_geometry(target):
                raise ValueError(f"DMAD {role} target geometry does not match its condition: {condition['condition_path']}")
            row[f"{role}_latent_path"] = target[f"{role}_latent_path"]
        joined.append(validate_joined_row(row, Path.cwd()))
    return joined


def write_manifest_atomic(rows, output):
    """Publish a complete manifest atomically, never replacing another file."""
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite DMAD manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A same-filesystem hard link is atomic and fails if output appeared
        # between the initial existence check and publication (unlike replace).
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", action="append", required=True, help="Condition metadata; repeat to join shards.")
    parser.add_argument("--real", action="append", required=True, help="Real-target metadata; repeat to join shards.")
    parser.add_argument("--teacher", action="append", required=True, help="Teacher-target metadata; repeat to join shards.")
    parser.add_argument("--output", required=True, help="New joined JSONL file; must not already exist.")
    args = parser.parse_args(argv)
    try:
        rows = build_joined_rows(args.conditions, args.real, args.teacher)
        output = write_manifest_atomic(rows, args.output)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"DMAD manifest error: {error}\n")
    print(f"Joined {len(rows)} DMAD condition/real/teacher records: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
