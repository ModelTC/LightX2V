#!/usr/bin/env python3
"""Expose prepared reward weights through portable links; never copy model files.

Select only the rewards you prepared with --models. Existing bundles and
downloaded snapshots remain intact for other consumers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from uuid import uuid4

DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def link_weights(
    repository_root: Path = DEFAULT_REPOSITORY_ROOT,
    *,
    storage_dir: Path | None = None,
    unifiedreward_dir: Path | None = None,
    models: tuple[str, ...] = ("hpsv3", "videoalign", "unifiedreward"),
    check: bool = False,
) -> list[tuple[Path, str]]:
    repository_root = repository_root.resolve()
    if not models or set(models) - {"hpsv3", "videoalign", "unifiedreward"}:
        raise ValueError("Select hpsv3, videoalign and/or unifiedreward")

    def source_path(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else repository_root / path).resolve()

    model_store = repository_root / "lightx2v_rl/models"
    default_storage = (model_store if model_store.is_dir() else repository_root.parent / "models") / "rewards"
    storage = source_path(storage_dir if storage_dir is not None else default_storage)
    weights = repository_root / "lightx2v_rl/weights"
    links = [(storage / name, storage / "fewstep" / name) for name in ("hpsv3", "videoalign") if name in models]
    if set(models) & {"hpsv3", "videoalign"}:
        links.append((storage / "reward-assets.json", storage / "fewstep/reward-assets.json"))
    if "unifiedreward" in models:
        unified = source_path(unifiedreward_dir) if unifiedreward_dir is not None else storage / "UnifiedReward-Edit-qwen35-9b"
        links.append((storage / "unifiedreward", unified))
    links.append((weights, storage))
    updates = []
    result = []
    # Check every target and existing entry before changing any links. In
    # particular, never replace a real directory or an unrelated user symlink.
    for link, target in links:
        if not target.exists():
            raise FileNotFoundError(f"Reward assets missing: {target}; prepare/download the weights first")
        relative = os.path.relpath(target, start=link.parent)
        result.append((link, relative))
        if link.is_symlink():
            current = link.resolve()
            allowed = {target.resolve()}
            if link == weights:
                allowed.add((storage / "fewstep").resolve())
            if current not in allowed:
                raise FileExistsError(f"Refusing to replace unrelated symlink: {link} -> {os.readlink(link)}")
            if os.readlink(link) == relative:
                continue
        elif link.exists():
            raise FileExistsError(f"Refusing to replace existing file or directory: {link}")
        if check:
            raise ValueError(f"Missing or outdated link: {link}; run link_reward_weights.py without --check")
        updates.append((link, relative))
    for link, relative in updates:
        link.parent.mkdir(parents=True, exist_ok=True)
        temporary = link.with_name(f".{link.name}-{uuid4().hex}.tmp")
        try:
            temporary.symlink_to(relative)
            # Atomic replacement also keeps the old weights entry usable until
            # all child aliases have been created.
            temporary.replace(link)
        finally:
            if temporary.is_symlink():
                temporary.unlink()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("hpsv3", "videoalign", "unifiedreward"), default=("hpsv3", "videoalign", "unifiedreward"))
    parser.add_argument("--storage-dir", type=Path, help="Reward storage; defaults to models/rewards under the RL resource entry")
    parser.add_argument("--unifiedreward-dir", type=Path, help="UnifiedReward snapshot; defaults to the 9B snapshot in reward storage")
    parser.add_argument("--check", action="store_true", help="Check the links without changing them")
    args = parser.parse_args(argv)
    try:
        links = link_weights(storage_dir=args.storage_dir, unifiedreward_dir=args.unifiedreward_dir, models=tuple(args.models), check=args.check)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Reward weight links: {error}\n")
    for link, target in links:
        print(f"{os.path.relpath(link, DEFAULT_REPOSITORY_ROOT)} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
