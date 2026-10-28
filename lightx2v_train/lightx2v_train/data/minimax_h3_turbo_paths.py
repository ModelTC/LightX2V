"""Standard-library-only path/manifest handling shared by Turbo CM preflight.

Only full target AV paths and the condition path are remapped. Original media
and provenance stay untouched. Relative cache paths remain relative to their
metadata directory, independent of any old-to-new absolute-prefix mapping.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

CM_CACHE_PATH_KEYS = ("condition_path", "video_latent_path", "audio_latent_path")


def normalize_path_mapping(mapping=None):
    """Return validated absolute prefix pairs, longest-prefix-first."""
    if mapping is None:
        return ()
    if isinstance(mapping, Mapping):
        items = mapping.items()
    elif isinstance(mapping, (tuple, list)):
        items = mapping
    else:
        raise TypeError("CM path_mapping must map absolute old prefixes to absolute new prefixes.")
    result = []
    for old, new in items:
        if not isinstance(old, (str, Path)) or not isinstance(new, (str, Path)) or not str(old).strip() or not str(new).strip():
            raise ValueError("CM path_mapping prefixes must be non-empty absolute paths.")
        old_path, new_path = Path(old).expanduser(), Path(new).expanduser()
        if not old_path.is_absolute() or not new_path.is_absolute():
            raise ValueError("CM path_mapping prefixes must be absolute paths.")
        result.append((str(old_path), str(new_path)))
    return tuple(sorted(result, key=lambda pair: (-len(Path(pair[0]).parts), pair)))


def resolve_cache_path(value, metadata_path, path_mapping=None):
    """Resolve a cache path, remapping only an absolute path's prefix."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("CM cache paths must be non-empty strings.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        return (Path(metadata_path).expanduser().resolve().parent / path).resolve()
    for old, new in normalize_path_mapping(path_mapping):
        try:
            relative = path.relative_to(old)
        except ValueError:
            continue
        path = Path(new) / relative
        break
    return path.resolve()


def load_cm_records(metadata_path, path_mapping=None, tasks=None, check_files=True):
    """Validate one T2AV full-target manifest and return resolved records.

    Returns ``(manifest_path, rows, fingerprint)``. The SHA256 binds metadata
    bytes, resolved target/condition paths and path mapping; file stat times do
    not enter the deterministic sampling/resume identity.
    """
    manifest = Path(metadata_path).expanduser().resolve()
    if manifest.is_dir():
        manifest = manifest / "metadata.jsonl"
    if manifest.suffix.lower() != ".jsonl" or not manifest.is_file():
        raise FileNotFoundError(f"CM requires one metadata.jsonl file: {manifest}")
    selected_tasks = ("t2av",) if tasks is None else tuple(str(task).lower() for task in tasks)
    if selected_tasks != ("t2av",):
        raise ValueError("The independent Turbo CM cache currently supports tasks=[t2av] only.")
    mapping = normalize_path_mapping(path_mapping)
    raw = manifest.read_bytes()
    rows, source_ids, fingerprints, conditions = [], set(), set(), set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        location = f"{manifest}:{line_number}"
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"CM metadata must contain JSON objects: {location}")
        row = dict(row)
        if str(row.get("task", "")).lower() != "t2av":
            raise ValueError(f"CM metadata requires task=t2av: {location}")
        source_id = row.get("source_id")
        if isinstance(source_id, bool) or not isinstance(source_id, (str, int)) or not str(source_id).strip():
            raise ValueError(f"CM metadata requires an explicit source_id: {location}")
        source_id = str(source_id)
        fingerprint = row.get("cache_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"CM metadata requires cache_fingerprint: {location}")
        if source_id in source_ids or fingerprint in fingerprints:
            raise ValueError(f"CM metadata has duplicate source_id/cache_fingerprint: {location}")
        source_ids.add(source_id)
        fingerprints.add(fingerprint)
        height, width = int(row.get("target_height", 0)), int(row.get("target_width", 0))
        frames = int(row.get("target_num_frames", row.get("num_frames", 0)))
        if frames != 124 or (height, width) not in {(768, 1344), (1344, 768)}:
            raise ValueError(f"CM requires 124 frames and 768x1344 or 1344x768 geometry: {location}")
        expected = {"video_latent_shape": [24, 37, height // 16, width // 16], "audio_latent_shape": [2, 32, 207]}
        for key, shape in expected.items():
            if key in row and row[key] != shape:
                raise ValueError(f"CM {key} does not match full target geometry at {location}: {row[key]}")
        if "vae_normalized" in row and row["vae_normalized"] is not True:
            raise ValueError(f"CM full target AV latents must be normalized: {location}")
        for key in CM_CACHE_PATH_KEYS:
            path = resolve_cache_path(row.get(key), manifest, mapping)
            if check_files and not path.is_file():
                raise FileNotFoundError(f"CM {key} points to a missing file at {location}: {path}")
            row[key] = str(path)
        if row["condition_path"] in conditions:
            raise ValueError(f"CM metadata repeats one condition_path: {location}")
        conditions.add(row["condition_path"])
        rows.append(row)
    if not rows:
        raise ValueError(f"CM metadata has no complete T2AV rows: {manifest}")
    identity = {
        "schema": "h3_turbo_cm_full_av_v1",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "path_mapping": mapping,
        "resolved_cache_paths": [[row[key] for key in CM_CACHE_PATH_KEYS] for row in rows],
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return manifest, rows, fingerprint
