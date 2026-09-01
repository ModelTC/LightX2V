import json
import re
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

_H3_BLOCK_KEY_RE = re.compile(r"^transformer_blocks\.(\d+)\.")


class MiniMaxH3ShardCheckpoint:
    """Synchronous reader for official MiniMax-H3 sharded safetensors."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.index_path = self.checkpoint_dir / "model.safetensors.index.json"
        if not self.index_path.is_file():
            raise FileNotFoundError(f"MiniMax-H3 safetensors index not found: {self.index_path}")

        with self.index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"MiniMax-H3 safetensors index must contain a dict weight_map: {self.index_path}")
        if not weight_map:
            raise ValueError(f"MiniMax-H3 safetensors index weight_map is empty: {self.index_path}")

        invalid_shard_names = sorted(
            name
            for name, shard_name in weight_map.items()
            if not isinstance(shard_name, str) or not shard_name
        )
        if invalid_shard_names:
            raise ValueError(
                f"MiniMax-H3 safetensors index contains invalid shard file names for tensors: {invalid_shard_names}"
            )

        self.weight_map = dict(weight_map)
        shard_names = set(self.weight_map.values())
        missing_shards = sorted(
            shard_name
            for shard_name in shard_names
            if not (self.checkpoint_dir / shard_name).is_file()
        )
        if missing_shards:
            raise FileNotFoundError(
                f"MiniMax-H3 safetensors index references missing shard files: {missing_shards}"
            )

    @property
    def tensor_names(self):
        return tuple(sorted(self.weight_map))

    @property
    def block_indices(self):
        return tuple(
            sorted(
                {
                    int(match.group(1))
                    for name in self.weight_map
                    if (match := _H3_BLOCK_KEY_RE.match(name)) is not None
                }
            )
        )

    def tensor_names_for_block(self, block_index):
        block_prefix = f"transformer_blocks.{int(block_index)}."
        return tuple(sorted(name for name in self.weight_map if name.startswith(block_prefix)))

    def non_block_tensor_names(self):
        return tuple(sorted(name for name in self.weight_map if _H3_BLOCK_KEY_RE.match(name) is None))

    def shard_for_tensor(self, name):
        try:
            return self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"MiniMax-H3 checkpoint is missing requested tensor: {name}") from error

    def block_names(self, block_index):
        return list(self.tensor_names_for_block(block_index))

    def non_block_names(self):
        return list(self.non_block_tensor_names())

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
