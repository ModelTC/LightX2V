#!/usr/bin/env python3
"""Expose shared models and the source dataset through portable local links.

The source directories stay next to the repository. This only creates links;
it does not download models, copy data, or regenerate prepared data manifests.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from uuid import uuid4

DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def link_resources(repository_root: Path = DEFAULT_REPOSITORY_ROOT, *, check: bool = False) -> list[tuple[Path, str]]:
    repository_root = Path(repository_root).resolve()
    module_root = repository_root / "lightx2v_rl"
    links = [
        (module_root / "models", repository_root.parent / "models"),
        (module_root / "data/OpenGPT-4o-Image", repository_root.parent / "OpenGPT-4o-Image"),
    ]
    updates = []
    result = []
    # Validate both resources before making any change, so a missing dataset or
    # conflicting entry cannot leave a partially updated set of links.
    for link, target in links:
        if not target.is_dir():
            raise FileNotFoundError(f"Resource directory missing: {target}; prepare this directory before linking")
        for parent in link.parents:
            if parent.exists() or parent.is_symlink():
                if not parent.is_dir():
                    raise FileExistsError(f"Link parent is not a directory: {parent}")
                break
        relative = os.path.relpath(target, start=link.parent)
        result.append((link, relative))
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise FileExistsError(f"Refusing to replace unrelated symlink: {link} -> {os.readlink(link)}")
            if os.readlink(link) == relative:
                continue
        elif link.exists():
            raise FileExistsError(f"Refusing to replace existing file or directory: {link}")
        if check:
            raise ValueError(f"Missing or outdated link: {link}; run link_resources.py without --check")
        updates.append((link, relative))

    for link, relative in updates:
        link.parent.mkdir(parents=True, exist_ok=True)
        temporary = link.with_name(f".{link.name}-{uuid4().hex}.tmp")
        try:
            temporary.symlink_to(relative, target_is_directory=True)
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
        links = link_resources(check=args.check)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Resource links: {error}\n")
    for link, target in links:
        print(f"{link.relative_to(DEFAULT_REPOSITORY_ROOT.parent)} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
