"""Load MiniMax-H3 diffusers checkpoints by requested tensor or block."""

import re
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

_H3_BLOCK_KEY_RE = re.compile(r"^transformer_blocks\.(\d+)\.")


class MiniMaxH3ShardCheckpoint:
    """Index safetensors headers for selective loading, using upstream file discovery."""

    def __init__(self, checkpoint_dir):
        checkpoint = Path(checkpoint_dir)
        files = sorted(checkpoint.glob("*.safetensors")) if checkpoint.is_dir() else [checkpoint]
        if not files or any(not path.is_file() for path in files):
            raise FileNotFoundError(f"MiniMax-H3 safetensors checkpoint not found: {checkpoint}")
        self.checkpoint_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
        self.weight_map = {}
        # Match the upstream model loader's directory/single-file discovery.
        # Read only headers here; tensor data is loaded when a block requests it.
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as source:
                self.weight_map.update(dict.fromkeys(source.keys(), path.name))

    @property
    def tensor_names(self):
        return tuple(sorted(self.weight_map))

    @property
    def block_indices(self):
        pattern = _H3_BLOCK_KEY_RE
        return tuple(sorted({int(match.group(1)) for name in self.weight_map if (match := pattern.match(name)) is not None}))

    def tensor_names_for_block(self, block_index):
        block_prefix = f"transformer_blocks.{int(block_index)}."
        return tuple(sorted(name for name in self.weight_map if name.startswith(block_prefix)))

    def non_block_tensor_names(self):
        pattern = _H3_BLOCK_KEY_RE
        return tuple(sorted(name for name in self.weight_map if pattern.match(name) is None))

    def shard_for_tensor(self, name):
        try:
            return self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"MiniMax-H3 checkpoint is missing requested tensor: {name}") from error

    def load_tensors(self, names, device="cpu"):
        missing = sorted(name for name in names if name not in self.weight_map)
        if missing:
            raise KeyError(f"MiniMax-H3 checkpoint is missing requested tensors: {missing}")

        by_shard = defaultdict(list)
        for name in names:
            by_shard[self.weight_map[name]].append(name)

        tensors = {}
        for shard_name in sorted(by_shard):
            shard_path = self.checkpoint_dir / shard_name
            with safe_open(shard_path, framework="pt", device=device) as shard:
                for name in sorted(by_shard[shard_name]):
                    tensors[name] = shard.get_tensor(name)
        return tensors


__all__ = ["MiniMaxH3ShardCheckpoint"]
