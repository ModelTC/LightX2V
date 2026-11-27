"""Publish a complete checkpoint before retiring older optimizer snapshots."""

import json
import shutil
from pathlib import Path


def load_model_checkpoint(model, directory):
    """Load model tensors only, retaining newly unfrozen parameters and fresh Adam state."""
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(directory)
    keys = sorted(name.removeprefix("model.") for name in reader.read_metadata().state_dict_metadata if name.startswith("model."))
    parameters = dict(model.named_parameters())
    state = {"model": {name: parameters[name].detach() for name in keys}}
    dcp.load(state, storage_reader=reader)
    return len(keys)


def publish_checkpoint(root, temporary, name, progress, limit):
    root, temporary = Path(root), Path(temporary)
    (temporary / "progress.json").write_text(json.dumps(progress, indent=2))
    temporary.rename(root / name)
    pointer = root / "latest.tmp"
    pointer.write_text(json.dumps({"directory": name, "stage": progress["stage"]}))
    pointer.replace(root / "latest.json")
    completed = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".") and (path / "progress.json").exists()]
    completed.sort(key=lambda path: (path.name == name, (path / "progress.json").stat().st_mtime_ns), reverse=True)
    for path in completed[limit:]:
        shutil.rmtree(path)


def latest_checkpoint(root, stage):
    root = Path(root)
    pointer = root / "latest.json"
    if not pointer.exists():
        return None
    latest = json.loads(pointer.read_text())
    return root / latest["directory"] if latest["stage"] == stage else None
