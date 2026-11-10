"""Local paths, JSON editing and the ComfyUI file browser."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
from pathlib import Path

MAX_CONFIG_BYTES = 1024 * 1024


def validate_config(data: bytes) -> dict:
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError("Config JSON must be at most 1 MiB")
    try:
        value = json.loads(data.decode("utf-8-sig"))
        json.dumps(value, allow_nan=False)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Config must be valid UTF-8 JSON with finite numbers") from exc
    if not isinstance(value, dict):
        raise ValueError("Config JSON must contain an object")  # noqa: TRY004 - invalid input value
    return value


def local_path(value: str, base: Path | None = None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Select a local path")
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    if "${" in expanded:
        raise ValueError("Unresolved environment variable in path: " + value)
    path = Path(expanded)
    return Path(os.path.abspath(path if path.is_absolute() else (base or Path.cwd()) / path))


def config_path(value: str, base: Path | None = None) -> Path:
    path = local_path(value, base)
    if not path.is_file():
        raise ValueError("config_json is not a file visible to ComfyUI: " + str(path))
    with path.open("rb") as handle:
        validate_config(handle.read(MAX_CONFIG_BYTES + 1))
    return path


def directory_path(value: str, base: Path | None = None) -> Path:
    path = local_path(value, base)
    if not path.is_dir():
        raise ValueError("Not a directory visible to ComfyUI: " + str(path))
    return path


def lora_path(value: str, base: Path | None = None) -> Path:
    path = local_path(value, base)
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        raise ValueError("Select a .safetensors weight file visible to ComfyUI: " + str(path))
    # Check readability without loading or deserializing model weights.
    with path.open("rb"):
        pass
    return path


def directory_listing(value: str = "", base: Path | None = None, *, directories_only: bool = False, file_type: str = "config") -> dict:
    suffixes = {"config": ".json", "lora": ".safetensors"}
    if file_type not in suffixes:
        raise ValueError("Unknown file type: " + file_type)
    suffix = suffixes[file_type]
    directory = local_path(value or str(base or Path.cwd()), base)
    selected = ""
    if not directories_only and directory.is_file():
        if directory.suffix.lower() == suffix:
            selected = str(directory)
        directory = directory.parent
    if not directory.is_dir():
        raise FileNotFoundError("Directory not found: " + str(directory))
    if directories_only:
        selected = str(directory)
    entries = []
    truncated = False
    with os.scandir(directory) as scan:
        for index, entry in enumerate(scan):
            if index >= 5000:
                truncated = True
                break
            try:
                is_directory = entry.is_dir()
                if not is_directory and (directories_only or not (entry.name.lower().endswith(suffix) and entry.is_file())):
                    continue
                entries.append({"name": entry.name, "path": str(directory / entry.name), "directory": is_directory})
            except OSError:
                continue
    entries.sort(key=lambda item: (not item["directory"], item["name"].casefold()))
    return {"directory": str(directory), "parent": str(directory.parent), "entries": entries, "selected": selected, "truncated": truncated}


_edit_lock = threading.Lock()


class ConfigConflict(ValueError):
    pass


def _target(value, base=None):
    path = local_path(value, local_path(base) if base else None).resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Only .json config files can be edited")
    return path


def read_config(value, base=None):
    path = _target(value, base)
    with path.open("rb") as handle:
        data = handle.read(MAX_CONFIG_BYTES + 1)
    validate_config(data)
    return {"path": str(path), "content": data.decode("utf-8-sig"), "sha256": hashlib.sha256(data).hexdigest()}


def save_config(value, content, expected_sha256=None, *, create_new=False, base=None):
    if not isinstance(content, str):
        raise ValueError("Config content must be text")  # noqa: TRY004 - invalid input value
    data = content.encode("utf-8")
    validate_config(data)
    path = _target(value, base)
    backup = None
    # Serializes this process's editors; the hash also detects external edits since opening.
    with _edit_lock:
        if create_new:
            try:
                with path.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as error:
                raise ConfigConflict("Save as requires a new filename; the destination already exists") from error
        else:
            with path.open("rb") as handle:
                old = handle.read(MAX_CONFIG_BYTES + 1)
            validate_config(old)
            if hashlib.sha256(old).hexdigest() != expected_sha256:
                raise ConfigConflict("Config changed since it was opened. Reload it or save to a new file")
            if data != old:
                backups = path.parent / ".lightx2v-config-backups"
                backups.mkdir(exist_ok=True)
                backup = backups / (path.stem + "." + expected_sha256 + ".json")
                try:
                    with backup.open("xb") as handle:
                        handle.write(old)
                except FileExistsError:
                    if backup.read_bytes() != old:
                        raise ConfigConflict("Existing config backup does not match the original")
                fd, temporary = tempfile.mkstemp(prefix=".lightx2v-config-", dir=path.parent)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
                    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
                        raise ConfigConflict("Config changed while saving. Reload before retrying")
                    os.replace(temporary, path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
        result = read_config(str(path))
    return dict(result, backup=str(backup) if backup else None)
