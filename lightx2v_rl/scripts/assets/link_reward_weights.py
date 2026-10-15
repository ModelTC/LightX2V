#!/usr/bin/env python3
"""Expose prepared reward weights through portable links; never copy model files.

Run after preparing HPSv3/VideoAlign and downloading UnifiedReward. The old
FewStep bundle and downloaded snapshot remain intact for existing consumers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from uuid import uuid4

DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def link_weights(repository_root: Path = DEFAULT_REPOSITORY_ROOT, *, check: bool = False) -> list[tuple[Path, str]]:
    repository_root = repository_root.resolve()
    storage = repository_root.parent / "models/rewards"
    weights = repository_root / "lightx2v_rl/weights"
    links = [
        (storage / "hpsv3", storage / "fewstep/hpsv3"),
        (storage / "videoalign", storage / "fewstep/videoalign"),
        (storage / "reward-assets.json", storage / "fewstep/reward-assets.json"),
        (storage / "unifiedreward", storage / "UnifiedReward-Edit-qwen35-9b"),
        (weights, storage),
    ]
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
    parser.add_argument("--check", action="store_true", help="Check the links without changing them")
    args = parser.parse_args(argv)
    try:
        links = link_weights(check=args.check)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Reward weight links: {error}\n")
    for link, target in links:
        print(f"{link.relative_to(DEFAULT_REPOSITORY_ROOT.parent)} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
