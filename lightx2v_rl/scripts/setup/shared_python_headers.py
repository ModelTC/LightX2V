#!/usr/bin/env python3
"""Cache system Python headers on shared storage, then link them in fresh containers."""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def environment():
    include = Path(sysconfig.get_path("include"))
    if sys.platform != "linux" or sys.base_prefix not in ("/usr", "/usr/local"):
        raise RuntimeError("Only Linux system Python (/usr or /usr/local) is supported.")
    if include != Path(sys.base_prefix) / "include" / f"python{sys.version_info.major}.{sys.version_info.minor}":
        raise RuntimeError(f"Unsupported system Python include layout: {include}")
    distro = platform.freedesktop_os_release()
    identity = dict(
        python=list(sys.version_info[:3]),
        soabi=sysconfig.get_config_var("SOABI"),
        machine=platform.machine(),
        distro=distro["ID"],
        distro_version=distro["VERSION_ID"],
        multiarch=sysconfig.get_config_var("MULTIARCH"),
    )
    if not all(identity.values()):
        raise RuntimeError("Cannot determine complete Python/platform identity.")
    return identity, include.parent


def cache_path(cache_base, identity):
    version = ".".join(map(str, identity["python"]))
    parts = (f"{identity['distro']}-{identity['distro_version']}", f"python-{version}-{identity['machine']}-{identity['soabi']}")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        raise RuntimeError("Unsafe Python/platform cache identity.")
    return Path(cache_base).joinpath(*parts)


def probe(include_root, directories):
    compiler = shlex.split(os.environ.get("CC", "")) or [shutil.which("cc") or shutil.which("gcc") or ""]
    if not compiler[0]:
        raise RuntimeError("A C compiler (cc/gcc, or CC) is required to verify Python headers.")
    command = compiler + ["-E", "-x", "c", "-", f"-I{include_root / directories[0]}", f"-I{include_root}"]
    result = subprocess.run(command, input="#include <Python.h>\n", text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError(f"Python headers failed C preprocessing:\n{result.stderr[-4000:]}")


def _files(path):
    result = {}
    for item in sorted((path / "include").rglob("*")):
        if item.is_symlink():
            raise RuntimeError(f"Cached headers must be self-contained, found symlink: {item}")
        if item.is_file():
            result[item.relative_to(path).as_posix()] = hashlib.sha256(item.read_bytes()).hexdigest()
    return result


def validate_cache(path, identity):
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        main = f"python{identity['python'][0]}.{identity['python'][1]}"
        arch = f"{identity['multiarch']}/{main}"
        if path.is_symlink() or manifest["schema"] != 1 or manifest["identity"] != identity:
            raise RuntimeError("Cached Python/platform identity is incompatible.")
        if manifest["directories"] not in ([main], [main, arch]):
            raise RuntimeError("Invalid cached header directories.")
        if not all((path / "include" / item / "pyconfig.h").is_file() for item in manifest["directories"]):
            raise RuntimeError("Cached Python configuration headers are incomplete.")
        files = _files(path)
        if files != manifest["files"] or f"include/{main}/Python.h" not in files:
            raise RuntimeError("Cached headers are incomplete or their checksums changed.")
        config = path / "include" / main / "pyconfig.h"
        if arch in config.read_text() and f"include/{arch}/pyconfig.h" not in files:
            raise RuntimeError("Cached pyconfig.h references missing multiarch headers.")
        probe(path / "include", manifest["directories"])
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Invalid Python header cache at {path}: {error}") from error


def cache_headers(cache_base, include_root, identity):
    path = cache_path(cache_base, identity)
    if os.path.lexists(path):
        validate_cache(path, identity)
        return path
    main = f"python{identity['python'][0]}.{identity['python'][1]}"
    arch = f"{identity['multiarch']}/{main}"
    directories = [main] + ([arch] if (include_root / arch).is_dir() else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".headers-", dir=path.parent) as temporary:
        staged = Path(temporary) / "cache"
        for directory in directories:
            shutil.copytree(include_root / directory, staged / "include" / directory)
        manifest = dict(schema=1, identity=identity, directories=directories, files=_files(staged))
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        validate_cache(staged, identity)
        if os.path.lexists(path):
            validate_cache(path, identity)
        else:
            staged.rename(path)
    return path


def link_headers(cache_base, include_root, identity):
    path = cache_path(cache_base, identity)
    manifest = validate_cache(path, identity)
    missing = []
    for directory in manifest["directories"]:
        target = include_root / directory
        if not os.path.lexists(target):
            missing.append((target, path / "include" / directory))
            continue
        prefix = f"include/{directory}/"
        files = [name[len(prefix) :] for name in manifest["files"] if name.startswith(prefix)]
        if not target.is_dir() or not all((target / name).is_file() for name in files):
            raise RuntimeError(f"Refusing to replace existing incomplete header path: {target}")
    created = []
    try:
        for target, source in missing:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
            created.append(target)
        probe(include_root, manifest["directories"])
    except BaseException:
        for target in reversed(created):
            target.unlink()
        raise
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cache", "link"))
    args = parser.parse_args()
    try:
        identity, include_root = environment()
        cache_base = Path(__file__).resolve().parents[2] / "models" / ".system-deps" / "python-headers"
        path = {"cache": cache_headers, "link": link_headers}[args.action](cache_base, include_root, identity)
        print(f"Python headers {args.action} verified: {path}")
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Python headers: {error}\n")


if __name__ == "__main__":
    main()
