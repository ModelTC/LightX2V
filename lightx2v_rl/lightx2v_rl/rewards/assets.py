"""Resolve prepared reward assets without an online model lookup."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DEFAULT_ASSETS_ROOT = Path(__file__).resolve().parents[2] / "models" / "rewards"
EXPECTED_ASSETS = {
    "hpsv3": {"checkpoint": "hpsv3/HPSv3.safetensors", "base_model": "hpsv3/base_model"},
    "videoalign": {"checkpoint_dir": "videoalign", "base_model": "videoalign/base_model"},
}


def resolve_reward_assets(root: str | Path | None = None, *, verify_hashes: bool = False) -> dict[str, dict[str, Path]]:
    """Return absolute local model paths; relocating the whole bundle is supported.

    The default uses the checkout's ``models/rewards`` entry, and can be overridden
    with ``LIGHTX2V_RL_REWARD_ASSETS`` or an explicit ``root``. The manifest stores
    only relative paths, so moving the workspace from /mnt/... to /data does not
    require rewriting it. Relative overrides are resolved from the repository
    root. Hash verification is optional because it reads ~40 GiB.
    """
    root = Path(root or os.environ.get("LIGHTX2V_RL_REWARD_ASSETS") or DEFAULT_ASSETS_ROOT).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[3] / root
    root = root.resolve()
    manifest_path = root / "reward-assets.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing local reward assets: {manifest_path}. "
            "Prepare HPSv3/VideoAlign with lightx2v_rl/scripts/assets/prepare_reward_assets.py, "
            "then run lightx2v_rl/scripts/assets/link_resources.py and link_reward_weights.py."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1 or manifest.get("assets") != EXPECTED_ASSETS or not manifest.get("files"):
        raise ValueError(f"Unsupported or incomplete reward asset manifest: {manifest_path}")
    listed = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["path"] in listed:
            raise ValueError(f"Invalid reward asset manifest path: {entry['path']!r}")
        listed.add(entry["path"])
        path = root / relative
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError(f"Reward asset is not an independent local file: {path}")
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Missing or size-mismatched reward asset: {path}")
        if verify_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"SHA-256 mismatch for reward asset: {path}")
    required = ["hpsv3/HPSv3.safetensors", "videoalign/model_config.json", "videoalign/checkpoint-11352/model.pth"]
    for base in ("hpsv3/base_model", "videoalign/base_model"):
        required += [f"{base}/{name}" for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json")]
        index = root / base / "model.safetensors.index.json"
        if index.is_file():
            required.append(f"{base}/model.safetensors.index.json")
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards:
                raise ValueError(f"Empty reward base model index: {index}")
            required += [f"{base}/{shard}" for shard in shards]
        else:
            required.append(f"{base}/model.safetensors")
    if missing := set(required) - listed:
        raise ValueError(f"Reward asset manifest omits required dependencies: {sorted(missing)}")
    return {name: {key: root / relative for key, relative in paths.items()} for name, paths in EXPECTED_ASSETS.items()}
