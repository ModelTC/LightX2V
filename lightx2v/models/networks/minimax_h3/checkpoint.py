"""Load MiniMax-H3 diffusers checkpoints by requested tensor or block."""

import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import torch
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
        self._shard_headers = {}
        # Match the upstream model loader's directory/single-file discovery.
        # Read only headers here; tensor data is loaded when a block requests it.
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as source:
                self.weight_map.update(dict.fromkeys(source.keys(), path.name))

    def tensor_metadata(self, name):
        """Return dtype, shape and absolute file range without loading payloads."""
        shard_name = self.shard_for_tensor(name)
        if shard_name not in self._shard_headers:
            with (self.checkpoint_dir / shard_name).open("rb") as source:
                header_size = struct.unpack("<Q", source.read(8))[0]
                header = json.loads(source.read(header_size))
                self._shard_headers[shard_name] = (8 + header_size, header)
        data_start, header = self._shard_headers[shard_name]
        entry = header[name]
        dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
        if entry["dtype"] not in dtypes:
            raise ValueError(f"Shared H3 weight loading does not support dtype {entry['dtype']}: {name}")
        dtype, shape = dtypes[entry["dtype"]], tuple(entry["shape"])
        begin, end = entry["data_offsets"]
        nbytes = math.prod(shape) * dtype.itemsize
        if begin < 0 or end - begin != nbytes:
            raise ValueError(f"Invalid safetensors byte range for {name}")
        return dtype, shape, data_start + begin, nbytes

    def load_tensors_into(self, destinations):
        """Read file payloads directly into contiguous CPU tensors or MPS aliases.

        The caller must finish GPU accesses before host writes and synchronize
        again before using the written storage on the GPU.
        """
        by_shard = defaultdict(list)
        for name, tensor in destinations.items():
            dtype, shape, offset, nbytes = self.tensor_metadata(name)
            if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(f"Direct-load target must be contiguous CPU {dtype} with shape {shape}: {name}")
            by_shard[self.shard_for_tensor(name)].append((offset, nbytes, name, tensor))
        for shard_name, entries in sorted(by_shard.items()):
            with (self.checkpoint_dir / shard_name).open("rb", buffering=0) as source:
                if sys.platform == "darwin":
                    import fcntl

                    # Streaming weights should not compete with MPS allocations
                    # for a second copy in the filesystem cache.
                    fcntl.fcntl(source.fileno(), fcntl.F_NOCACHE, 1)
                for offset, nbytes, name, tensor in sorted(entries):
                    source.seek(offset)
                    target = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")
                    completed = 0
                    while completed < nbytes:
                        count = source.readinto(target[completed:])
                        if not count:
                            raise EOFError(f"Incomplete safetensors payload for {name} in {shard_name}")
                        completed += count

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
