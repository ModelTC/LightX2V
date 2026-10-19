#!/usr/bin/env python3
"""Copy FewStepRL reward assets into a self-contained, relocatable local directory.

Only local files are read. Hugging Face snapshot symlinks are dereferenced;
destination files are independent copies (copy-on-write reflinks when supported),
never hardlinks to the source cache. No model is loaded and no GPU is required.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DESTINATION = Path(__file__).resolve().parents[2] / "models" / "rewards" / "fewstep"
DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[4] / "FewStepRL"
MANIFEST_NAME = "reward-assets.json"
MODEL_REPOS = {
    "hpsv3": "MizzenAI/HPSv3",
    "hpsv3_base": "Qwen/Qwen2-VL-7B-Instruct",
    "videoalign_base": "Qwen/Qwen2-VL-2B-Instruct",
}
ASSET_PATHS = {
    "hpsv3": {"checkpoint": "hpsv3/HPSv3.safetensors", "base_model": "hpsv3/base_model"},
    "videoalign": {"checkpoint_dir": "videoalign", "base_model": "videoalign/base_model"},
}
BASE_REQUIRED = ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json")


@dataclass(frozen=True)
class AssetFile:
    source: Path
    destination: str
    origin: str


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative asset path: {value!r}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_roots(source_root: Path, explicit: list[Path]) -> list[Path]:
    candidates = list(explicit)
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(key):
            candidates.append(Path(os.environ[key]))
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]) / "hub")
    candidates += [source_root.parent / ".cache/huggingface/hub", Path.home() / ".cache/huggingface/hub"]
    return list(dict.fromkeys(path.expanduser().resolve() for path in candidates))


def _snapshot(repo: str, cache_roots: list[Path], required: tuple[str, ...]) -> Path:
    for cache in cache_roots:
        model = cache / ("models--" + repo.replace("/", "--"))
        refs = model / "refs/main"
        preferred = [model / "snapshots" / refs.read_text().strip()] if refs.is_file() else []
        candidates = preferred + sorted((model / "snapshots").glob("*"))
        for candidate in dict.fromkeys(candidates):
            if candidate.is_dir() and all((candidate / name).is_file() for name in required):
                return candidate
    raise FileNotFoundError(f"No complete local snapshot for {repo}; searched {', '.join(map(str, cache_roots))}. Provide the cache location with --hf-cache. This script never downloads weights.")


def _base_files(base: Path, destination: str, origin: str) -> list[AssetFile]:
    for name in BASE_REQUIRED:
        if not (base / name).is_file():
            raise FileNotFoundError(f"Missing base model dependency: {base / name}")
    index = base / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if not shards:
            raise ValueError(f"Empty model index: {index}")
        for name in shards:
            if not (base / _safe_relative(name)).is_file():
                raise FileNotFoundError(f"Missing base model shard: {base / name}")
    elif not (base / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing model.safetensors or sharded index: {base}")
    return _tree_files(base, destination, origin)


def _tree_files(root: Path, destination: str, origin: str) -> list[AssetFile]:
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_file():
            files.append(AssetFile(path, f"{destination}/{relative.as_posix()}", origin))
        elif path.is_symlink() and not path.exists():
            raise FileNotFoundError(f"Broken source asset symlink: {path}")
    return files


def discover_assets(source_root: Path, cache_roots: list[Path]) -> list[AssetFile]:
    """Resolve all weights, tokenizer files and model configuration dependencies."""
    hps = _snapshot(MODEL_REPOS["hpsv3"], cache_roots, ("HPSv3.safetensors",))
    hps_base = _snapshot(MODEL_REPOS["hpsv3_base"], cache_roots, BASE_REQUIRED)
    video_base = _snapshot(MODEL_REPOS["videoalign_base"], cache_roots, BASE_REQUIRED)
    video = source_root / "genrl/reward/VideoAlign/checkpoints"
    required_video = ("model_config.json", "checkpoint-11352/model.pth", "checkpoint-11352/tokenizer/tokenizer.json")
    for name in required_video:
        if not (video / name).is_file():
            raise FileNotFoundError(f"Missing VideoAlign asset: {video / name}")
    files = [AssetFile(hps / "HPSv3.safetensors", ASSET_PATHS["hpsv3"]["checkpoint"], f"{MODEL_REPOS['hpsv3']}@{hps.name}")]
    files += _base_files(hps_base, "hpsv3/base_model", f"{MODEL_REPOS['hpsv3_base']}@{hps_base.name}")
    files += _tree_files(video, "videoalign", "FewStepRL/genrl/reward/VideoAlign/checkpoints")
    files += _base_files(video_base, "videoalign/base_model", f"{MODEL_REPOS['videoalign_base']}@{video_base.name}")
    # Preserve the code licenses alongside the local checkpoints and model card.
    for code_name, destination in (("HPSv3", "hpsv3"), ("VideoAlign", "videoalign")):
        license_path = source_root / "genrl/reward" / code_name / "LICENSE"
        if license_path.is_file():
            files.append(AssetFile(license_path, f"{destination}/SOURCE_CODE_LICENSE", f"FewStepRL/{code_name}"))
    if len({file.destination for file in files}) != len(files):
        raise ValueError("Asset discovery produced conflicting destination paths")
    return sorted(files, key=lambda file: file.destination)


def _copy_file(source: Path, destination: Path) -> tuple[str, str]:
    """Create an independent destination inode and compute its digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    method = "copy"
    try:
        import fcntl

        with source.open("rb") as src, destination.open("xb") as dst:
            try:
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE
                method = "reflink"
            except OSError as exc:
                if exc.errno not in (errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV, errno.EINVAL, errno.ENOSYS):
                    raise
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    except ImportError:
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    shutil.copystat(source, destination)
    digest = _sha256(destination)
    resolved_source = source.resolve()
    # HF LFS cache blobs are named by their SHA-256; verify known source hashes
    # without another full read of the cached multi-gigabyte checkpoint.
    if resolved_source.parent.name == "blobs" and len(resolved_source.name) == 64 and all(character in "0123456789abcdef" for character in resolved_source.name) and digest != resolved_source.name:
        raise ValueError(f"Copied asset does not match its Hugging Face blob digest: {source}")
    return method, digest


def check_assets(destination: Path, *, verify_hashes: bool = False) -> dict:
    """Validate a migrated bundle without requiring source paths or network access."""
    manifest_path = destination / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing asset manifest: {manifest_path}; run prepare_reward_assets.py first")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1 or manifest.get("assets") != ASSET_PATHS or not manifest.get("files"):
        raise ValueError(f"Unsupported or incomplete asset manifest: {manifest_path}")
    listed = set()
    for entry in manifest["files"]:
        relative = _safe_relative(entry["path"])
        if entry["path"] in listed:
            raise ValueError(f"Duplicate manifest entry: {entry['path']}")
        listed.add(entry["path"])
        path = destination / relative
        if path.is_symlink() or destination.resolve() not in path.resolve().parents:
            raise ValueError(f"Migrated assets must be independent local files: {path}")
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Missing or size-mismatched reward asset: {path}")
        if verify_hashes and _sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
    for relative in ("hpsv3/HPSv3.safetensors", "videoalign/model_config.json", "videoalign/checkpoint-11352/model.pth"):
        if relative not in listed:
            raise ValueError(f"Required asset absent from manifest: {relative}")
    for relative in ("hpsv3/base_model", "videoalign/base_model"):
        dependencies = _base_files(destination / relative, relative, "local")
        if any(file.destination not in listed for file in dependencies):
            raise ValueError(f"Untracked base model dependency under {relative}")
    return manifest


def prepare_assets(files: list[AssetFile], destination: Path) -> dict:
    """Atomically publish a bundle; never overwrite a partial or conflicting one."""
    if destination.exists():
        manifest = check_assets(destination)
        expected = {file.destination: file.source.stat().st_size for file in files}
        actual = {entry["path"]: entry["size"] for entry in manifest["files"]}
        if expected != actual:
            raise FileExistsError(f"Existing bundle differs from source inventory: {destination}")
        indexed = {entry["path"]: entry for entry in manifest["files"]}
        for file in files:
            entry = indexed[file.destination]
            if entry["origin"] != file.origin:
                raise FileExistsError(f"Source revision changed for existing asset: {file.destination}")
            if entry.get("source_mtime_ns") != file.source.stat().st_mtime_ns and _sha256(file.source) != entry["sha256"]:
                raise FileExistsError(f"Source contents changed for existing asset: {file.destination}")
        return manifest
    total_bytes = sum(file.source.stat().st_size for file in files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < total_bytes:
        raise OSError(f"Need up to {total_bytes / 2**30:.2f} GiB free to copy assets to {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-staging-", dir=destination.parent))
    entries = []
    try:
        for index, file in enumerate(files, start=1):
            before = file.source.stat()
            print(f"[{index}/{len(files)}] {file.destination} ({before.st_size / 2**20:.1f} MiB)", flush=True)
            method, digest = _copy_file(file.source, staging / _safe_relative(file.destination))
            after = file.source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed during migration: {file.source}")
            entries.append({"path": file.destination, "size": after.st_size, "sha256": digest, "origin": file.origin, "source_mtime_ns": after.st_mtime_ns, "copy_method": method})
        manifest = {"version": 1, "assets": ASSET_PATHS, "total_bytes": total_bytes, "files": entries}
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        check_assets(staging)
        if destination.exists():
            raise FileExistsError(f"Destination appeared during migration: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="Local FewStepRL checkout")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--hf-cache", type=Path, action="append", default=[], help="HF hub cache directory (repeatable)")
    parser.add_argument("--check", action="store_true", help="Check only; source checkout/cache are not needed")
    parser.add_argument("--verify-hashes", action="store_true", help="Read every migrated file and verify SHA-256")
    parser.add_argument("--dry-run", action="store_true", help="Print source inventory without copying")
    args = parser.parse_args(argv)
    destination = args.destination.expanduser().resolve()
    try:
        if args.check:
            manifest = check_assets(destination, verify_hashes=args.verify_hashes)
        else:
            source_root = args.source_root.expanduser().resolve()
            files = discover_assets(source_root, _cache_roots(source_root, args.hf_cache))
            if args.dry_run:
                for file in files:
                    print(f"{file.destination}: {file.source.stat().st_size} bytes <- {file.source}")
                print(f"Total: {sum(file.source.stat().st_size for file in files) / 2**30:.2f} GiB in {len(files)} files")
                return 0
            manifest = prepare_assets(files, destination)
            if args.verify_hashes:
                check_assets(destination, verify_hashes=True)
        print(f"Reward assets ready: {destination} ({manifest['total_bytes'] / 2**30:.2f} GiB, {len(manifest['files'])} files)")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Reward asset preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
