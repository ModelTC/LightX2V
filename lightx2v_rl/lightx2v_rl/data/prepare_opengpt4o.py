"""Prepare portable OpenGPT-4o-Image manifests without extracting target images."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import tarfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


def normalized_prompt(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError(f"Unsafe dataset image path: {value!r}")
    return path.as_posix()


def read_annotations(source: Path) -> list[dict]:
    rows = []
    for task, filename in (("t2i", "gen.json"), ("i2i", "editing.json")):
        annotations = json.loads((source / filename).read_text())
        for index, item in enumerate(annotations):
            prompt = item["input_prompt"]
            if not prompt.strip():
                raise ValueError(f"Empty prompt: {filename} row {index}")
            images = [safe_relative_path(p) for p in item.get("input_image", [])]
            if task == "i2i" and not images:
                raise ValueError(f"Missing reference images: {filename} row {index}")
            rows.append({
                "id": f"opengpt4o-{task}-{index:06d}",
                "task_type": task,
                "prompt": prompt,
                "source_images": images,
                "metadata": {
                    "annotation_file": filename,
                    "annotation_index": index,
                    "category": item.get("meta-task", ""),
                    "target_image": safe_relative_path(item["output_image"]),
                    "num_references": len(images),
                },
            })
    return rows


class ConcatenatedReader(io.RawIOBase):
    """Read split archives sequentially without creating a merged 88 GB file."""

    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded archive reads are not supported")
        chunks = []
        remaining = size
        while remaining:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = Path(path).open("rb")
            chunk = self.current.read(remaining)
            if not chunk:
                self.current.close()
                self.current = None
                continue
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def extract_sources(source: Path, rows: list[dict], index_path: Path) -> dict:
    wanted = {image for row in rows for image in row["source_images"]}
    cached = json.loads(index_path.read_text()) if index_path.exists() else {}
    hashes = {}
    for name in sorted(wanted):
        destination = source / name
        if destination.is_file():
            size = destination.stat().st_size
            entry = cached.get(name, {})
            digest = entry.get("sha256") if entry.get("size") == size else None
            hashes[name] = {"sha256": digest or file_digest(destination), "size": size}
    missing = wanted - hashes.keys()
    if not missing:
        write_json(index_path, hashes)
        return hashes
    parts = sorted(source.glob("editing.tar.gz.[0-9][0-9]"))
    archives = [parts] if parts else []
    if (source / "editing.tar.gz").is_file():
        archives.append([source / "editing.tar.gz"])
    # Some releases ship the additional two-reference insertion subset here.
    if (source / "temp/ref_editing.tar.gz").is_file():
        archives.append([source / "temp/ref_editing.tar.gz"])
    if not archives:
        raise FileNotFoundError(f"Missing {len(missing)} source images and no editing archive in {source}")
    last_progress = time.monotonic()
    for archive_parts in archives:
        print(f"Scanning {archive_parts[0].name}: {len(missing)} references still needed", flush=True)
        with ConcatenatedReader(archive_parts) as stream, tarfile.open(fileobj=stream, mode="r|gz", bufsize=4 * 1024 * 1024) as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if name.startswith("OpenGPT-4o-Image/"):
                    name = name.removeprefix("OpenGPT-4o-Image/")
                if name not in missing:
                    continue
                safe_relative_path(name)
                if not member.isfile():
                    raise ValueError(f"Reference must be a regular archive file: {name}")
                destination = source / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".rl-extract.tmp")
                digest = hashlib.sha256()
                with archive.extractfile(member) as data, temporary.open("wb") as output:
                    for block in iter(lambda: data.read(1024 * 1024), b""):
                        digest.update(block)
                        output.write(block)
                if temporary.stat().st_size != member.size:
                    raise IOError(f"Incomplete extracted reference: {name}")
                temporary.replace(destination)
                hashes[name] = {"sha256": digest.hexdigest(), "size": member.size}
                missing.remove(name)
                if time.monotonic() - last_progress >= 30:
                    write_json(index_path, hashes)
                    print(f"Extracted {len(hashes)}/{len(wanted)} source images", flush=True)
                    last_progress = time.monotonic()
                if not missing:
                    break
        write_json(index_path, hashes)
        if not missing:
            break
    if missing:
        raise FileNotFoundError(f"{len(missing)} reference images absent from archives, e.g. {sorted(missing)[:5]}")
    return hashes


def split_rows(rows: list[dict], source_hashes: dict, *, validation_size: int = 256, seed: int = 42):
    """Group shared prompts, source paths and source bytes before splitting.

    Grouping is global across tasks. Validation counts are lower bounds where
    practical; an entire connected component is retained to prevent leakage.
    """
    if validation_size < 1:
        raise ValueError("validation_size must be positive")
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = [("prompt", normalized_prompt(row["prompt"]))]
        keys += [("source", p) for p in row["source_images"]]
        keys += [("sha256", source_hashes[p]["sha256"]) for p in row["source_images"] if p in source_hashes]
        # A target reused as another row's source must not cross the split.
        keys.append(("source", row["metadata"]["target_image"]))
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    components = list(groups.values())
    # Stable ordering makes the split independent of annotation traversal order.
    components.sort(key=lambda group: min(row["id"] for row in group))
    random.Random(seed).shuffle(components)
    def stratum(row):
        return "i2i_multi_reference" if len(row["source_images"]) > 1 else row["task_type"]

    totals = Counter(stratum(row) for row in rows)
    limits = {task: min(validation_size, max(1, total // 10)) for task, total in totals.items()}
    counts = Counter()
    val = []
    deferred = []
    for group in components:
        group_counts = Counter(stratum(row) for row in group)
        fits = all(counts[task] + amount <= limits[task] for task, amount in group_counts.items())
        leaves_training = all(counts[task] + amount < totals[task] for task, amount in group_counts.items())
        if fits and leaves_training:
            val.extend(group)
            counts.update(group_counts)
        else:
            deferred.append(group)
    # Prefer a small overshoot to admitting a large connected duplicate group.
    # This also keeps the default single-reference validation set near 256 even
    # when separate multi-reference manifests are prepared in the same run.
    train = []
    for group in sorted(deferred, key=len):
        group_counts = Counter(stratum(row) for row in group)
        need = any(counts[task] < limits[task] for task in group_counts)
        leaves_training = all(counts[task] + amount < totals[task] for task, amount in group_counts.items())
        if need and leaves_training:
            val.extend(group)
            counts.update(group_counts)
        else:
            train.extend(group)
    missing_tasks = set(totals) - set(counts)
    if missing_tasks:
        raise ValueError(f"Cannot create leakage-free train and validation for tasks: {sorted(missing_tasks)}")
    return sorted(train, key=lambda r: r["id"]), sorted(val, key=lambda r: r["id"])


def write_manifest(path: Path, rows: list[dict]):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            value = {**row, "source_images": [f"source/{p}" for p in row["source_images"]]}
            value["metadata"] = {**row["metadata"], "target_image": f"source/{row['metadata']['target_image']}"}
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(source: Path, output: Path, *, validation_size=256, seed=42, include_multi_reference=False, extract=True):
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    link = output / "source"
    relative_source = os.path.relpath(source, output)
    if link.is_symlink():
        if os.readlink(link) != relative_source:
            raise ValueError(f"Existing source symlink has another target: {link}")
    elif link.exists():
        raise ValueError(f"Expected a source symlink, found existing path: {link}")
    else:
        link.symlink_to(relative_source, target_is_directory=True)
    rows = read_annotations(source)
    hash_path = output / "source_hashes.json"
    if extract:
        hashes = extract_sources(source, rows, hash_path)
    else:
        hashes = {}
        for name in sorted({p for row in rows for p in row["source_images"]}):
            path = source / name
            if not path.is_file():
                raise FileNotFoundError(f"Reference missing; rerun without --skip-extract: {path}")
            hashes[name] = {"sha256": file_digest(path), "size": path.stat().st_size}
        write_json(hash_path, hashes)
    train, val = split_rows(rows, hashes, validation_size=validation_size, seed=seed)
    summary = {"schema_version": 1, "seed": seed, "requested_validation_per_task": validation_size,
               "include_multi_reference": include_multi_reference, "source": relative_source,
               "annotation_counts": dict(Counter(row["task_type"] for row in rows)),
               "targets_extracted": False,
               "reference_files": len(hashes), "reference_bytes": sum(x["size"] for x in hashes.values()),
               "duplicate_reference_files": len(hashes) - len({entry["sha256"] for entry in hashes.values()}),
               "split_grouping": ["normalized_prompt", "source_path", "source_sha256", "target_source_path_overlap"]}
    for split, selected in (("train", train), ("val", val)):
        enabled = [r for r in selected if include_multi_reference or len(r["source_images"]) <= 1]
        write_manifest(output / f"{split}.jsonl", enabled)
        for task in ("t2i", "i2i"):
            write_manifest(output / f"{split}_{task}.jsonl", [r for r in enabled if r["task_type"] == task])
        write_manifest(output / f"{split}_i2i_multi_reference.jsonl", [r for r in selected if len(r["source_images"]) > 1])
        summary[split] = dict(Counter(r["task_type"] for r in enabled))
        summary[f"{split}_multi_reference"] = sum(len(r["source_images"]) > 1 for r in selected)
    write_json(output / "manifest_info.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project / "data/OpenGPT-4o-Image")
    parser.add_argument("--output", type=Path, default=project / "data/opengpt4o")
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-multi-reference", action="store_true", help="Include two-reference rows in train/val (requires a suitable reward)")
    parser.add_argument("--skip-extract", action="store_true", help="Use already extracted images, checking hashes and existence")
    args = parser.parse_args(argv)
    prepare(args.source, args.output, validation_size=args.validation_size, seed=args.seed,
            include_multi_reference=args.include_multi_reference, extract=not args.skip_extract)


if __name__ == "__main__":
    main()
