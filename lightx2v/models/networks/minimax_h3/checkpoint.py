import json
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors import safe_open

_H3_BLOCK_KEY_RE = re.compile(r"^transformer_blocks\.(\d+)\.")


class TargetSpec(NamedTuple):
    name: str
    shape: tuple
    dtype: str


class SourcePlan(NamedTuple):
    source_name: str
    source_shard: str
    transform: str
    shape: tuple
    dtype: str
    targets: tuple


def validate_mapping(entries, source_names, target_names):
    """Require an exact source classification and exactly one producer per target."""
    sources, targets = [], []
    for entry in entries:
        sources.append(entry.source_name)
        targets.extend(target.name for target in entry.targets)
    for label, actual, expected in (("source", sources, set(source_names)), ("target", targets, set(target_names))):
        if len(actual) != len(set(actual)):
            raise ValueError(f"MiniMax-H3 duplicate {label} coverage")
        missing, unknown = sorted(expected - set(actual)), sorted(set(actual) - expected)
        if missing or unknown:
            raise ValueError(f"MiniMax-H3 {label} mapping mismatch: missing={missing}, unknown={unknown}")


def _official_schema(config):
    """Logical checkpoint shapes only; runtime MM transposes belong to the writer."""
    h = config["hidden_size"]
    if config.get("adaln_out_features", 18 * h) != 18 * h or config.get("final_adaln_out_features", 2 * h) != 2 * h:
        raise ValueError("MiniMax-H3 incompatible AdaLN output dimensions")
    inner = config["num_attention_heads"] * config["attention_head_dim"]
    ffn, time = config["ffn_dim"], config["time_embed_dim"]
    patch = config["in_channels"]
    for axis in config["patch_size"]:
        patch *= axis
    schema = {}

    def add(source, target, shape, dtype="BF16", transform="rename", target_shape=None):
        names = (target,) if isinstance(target, str) else target
        schema[source] = (transform, tuple(shape), dtype, tuple(TargetSpec(n, tuple(target_shape or shape), dtype) for n in names))

    for source, target, shape, dtype in (
        ("video_patch_proj", "proj_in", (h, patch), "F32"),
        ("audio_patch_proj", "audio_proj_in", (h, config["audio_in_channels"]), "F32"),
        ("condition_proj", "context_embedder", (h, config["text_dim"]), "BF16"),
        ("time_embedder.proj_in", "time_embedder.linear_1", (config["time_embed_hidden_dim"], config["freq_dim"]), "F32"),
        ("time_embedder.proj_out", "time_embedder.linear_2", (time, config["time_embed_hidden_dim"]), "F32"),
        ("final_layer.adaln_proj.linear", "norm_out.linear", (2 * h, time), "BF16"),
        ("final_layer.video_out", "proj_out", (patch, h), "F32"),
        ("final_layer.audio_out", "audio_proj_out", (config["audio_in_channels"], h), "F32"),
    ):
        add(source + ".weight", target + ".weight", shape, dtype)
        add(source + ".bias", target + ".bias", shape[:1], dtype)
    add("token_refiner.final_norm.weight", "token_refiner.final_norm.weight", (h,), transform="identity")
    add("final_layer.norm.weight", "norm_out.norm.weight", (h,))
    add("rope.inv_freq", (), (config["rope_freq_dim"],), "F32", "validate_rope")
    for source_prefix, target_prefix, layers, adaln in (
        ("blocks", "transformer_blocks", config["num_layers"], True),
        ("token_refiner.blocks", "token_refiner.refiner_blocks", config["num_refiner_layers"], False),
    ):
        for i in range(layers):
            source, target = f"{source_prefix}.{i}", f"{target_prefix}.{i}"
            add(source + ".attn.qkv_proj.weight", tuple(target + f".attn.to_{q}.weight" for q in "qkv"), (3 * inner, h), transform="qkv_head_interleaved", target_shape=(inner, h))
            for q in "qk":
                add(source + f".attn.{q}_norm.weight", target + f".attn.norm_{q}.weight", (config["attention_head_dim"],))
            add(source + ".attn.out_proj.weight", target + ".attn.to_out.0.weight", (h, inner))
            add(source + ".mlp.fc1.weight", target + ".ff.net.0.proj.weight", (2 * ffn, h), transform="swap_gate_value")
            add(source + ".mlp.fc2.weight", target + ".ff.net.2.weight", (h, ffn))
            for n in (1, 2):
                add(source + f".norm{n}.weight", target + f".norm{n}.weight", (h,))
            if adaln:
                add(source + ".adaln_proj.linear.weight", target + ".adaln_proj.linear.weight", (18 * h, time))
                add(source + ".adaln_proj.linear.bias", target + ".adaln_proj.linear.bias", (18 * h,))
    return schema


_CONFIG_ALIASES = {
    "token_refiner_num_layers": "num_refiner_layers",
    "ffn_hidden_size": "ffn_dim",
    "latents_dim": "in_channels",
    "audio_latents_dim": "audio_in_channels",
    "timestep_input_dim": "freq_dim",
    "time_embed_hidden_size": "time_embed_hidden_dim",
    "rope_inv_freq_len": "rope_freq_dim",
}


def _native_config(config):
    normalized = dict(config)
    if "patch_size" in normalized:
        normalized["patch_size"] = tuple(normalized["patch_size"])
    for source, target in _CONFIG_ALIASES.items():
        if source in config:
            if target in config and config[source] != config[target]:
                raise ValueError(f"MiniMax-H3 conflicting config fields: {source}, {target}")
            normalized[target] = config[source]
    normalized.setdefault("rope_theta", 10000.0)
    return normalized


class MiniMaxH3CheckpointPlan:
    """Index/config-only planning. No shard existence checks or tensor reads."""

    def __init__(self, checkpoint_dir, config=None):
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
        raw = any(n.startswith(("blocks.", "video_patch_proj.", "audio_patch_proj.", "condition_proj.", "final_layer.", "token_refiner.blocks.")) for n in weight_map)
        native = any(n.startswith(("transformer_blocks.", "proj_in.", "audio_proj_in.", "context_embedder.", "norm_out.", "token_refiner.refiner_blocks.")) for n in weight_map)
        if raw and native:
            raise ValueError("MiniMax-H3 mixed official raw and native checkpoint keys")
        self.format = "official_raw" if raw else "native"
        self.entries = {}
        self.targets = {}
        if raw:
            signature = {"blocks.0.attn.qkv_proj.weight", "blocks.0.mlp.fc1.weight", "video_patch_proj.weight", "final_layer.video_out.weight"}
            if not signature.issubset(weight_map):
                raise ValueError(f"MiniMax-H3 incomplete official signature: missing {sorted(signature - weight_map.keys())}")
            config_path = self.checkpoint_dir / "config.json"
            self.config = _native_config(json.loads(config_path.read_text()) if config_path.is_file() else (config or {}))
            schema = _official_schema(self.config)
            if config is not None:
                runtime = _native_config(config)
                keys = (
                    "hidden_size", "num_layers", "num_attention_heads", "attention_head_dim", "ffn_dim", "time_embed_dim",
                    "num_refiner_layers", "freq_dim", "rope_freq_dim", "rope_theta", "time_embed_hidden_dim", "in_channels",
                    "audio_in_channels", "text_dim", "patch_size",
                )
                for key in keys:
                    if key == "rope_theta" and key not in config:
                        continue
                    if key in runtime and runtime[key] != self.config[key]:
                        raise ValueError(f"MiniMax-H3 checkpoint/runtime config mismatch: {key}")
            missing, unknown = sorted(schema.keys() - weight_map.keys()), sorted(weight_map.keys() - schema.keys())
            if missing or unknown:
                raise ValueError(f"MiniMax-H3 source/target mapping mismatch: missing={missing}, unknown={unknown}")
            self.entries = {name: SourcePlan(name, weight_map[name], *schema[name]) for name in sorted(schema)}
            expected = [target.name for _, _, _, targets in schema.values() for target in targets]
            validate_mapping(self.entries.values(), weight_map, expected)
            self.targets = {target.name: (entry, target) for entry in self.entries.values() for target in entry.targets}

    @property
    def tensor_names(self):
        return tuple(sorted(self.weight_map))

    @property
    def block_indices(self):
        pattern = re.compile(r"^blocks\.(\d+)\.") if self.format == "official_raw" else _H3_BLOCK_KEY_RE
        return tuple(
            sorted(
                {
                    int(match.group(1))
                    for name in self.weight_map
                    if (match := pattern.match(name)) is not None
                }
            )
        )

    def tensor_names_for_block(self, block_index):
        prefix = "blocks" if self.format == "official_raw" else "transformer_blocks"
        block_prefix = f"{prefix}.{int(block_index)}."
        return tuple(sorted(name for name in self.weight_map if name.startswith(block_prefix)))

    def non_block_tensor_names(self):
        pattern = re.compile(r"^blocks\.(\d+)\.") if self.format == "official_raw" else _H3_BLOCK_KEY_RE
        return tuple(sorted(name for name in self.weight_map if pattern.match(name) is None))

    def shard_for_tensor(self, name):
        try:
            return self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"MiniMax-H3 checkpoint is missing requested tensor: {name}") from error

    def block_names(self, block_index):
        return list(self.tensor_names_for_block(block_index))

    def non_block_names(self):
        return list(self.non_block_tensor_names())

    def shards_for_sources(self, names):
        return tuple(sorted({self.shard_for_tensor(name) for name in names}))


class MiniMaxH3SelectedSourceReader:
    """Bounded CPU slice staging, with no large mmap views retained across sources."""

    def __init__(self, plan, row_chunk_size=128):
        if plan.format != "official_raw":
            raise ValueError("MiniMax-H3 selected adapter requires official raw format")
        if row_chunk_size < 1:
            raise ValueError("row_chunk_size must be positive")
        self.plan = plan
        self.row_chunk_size = row_chunk_size

    @contextmanager
    def _source(self, name):
        entry = self.plan.entries[name]
        path = self.plan.checkpoint_dir / entry.source_shard
        if not path.is_file():
            raise FileNotFoundError(f"MiniMax-H3 requested shard missing for {name}: {path}")
        with safe_open(path, framework="pt", device="cpu") as reader:
            source = reader.get_slice(name)
            if tuple(source.get_shape()) != entry.shape:
                raise ValueError(f"MiniMax-H3 shape mismatch for {name}: {source.get_shape()} != {entry.shape}")
            if source.get_dtype() != entry.dtype:
                raise ValueError(f"MiniMax-H3 dtype mismatch for {name}: {source.get_dtype()} != {entry.dtype}")
            yield source

    def read_source_slice(self, name, row_start, row_end):
        entry = self.plan.entries[name]
        if not 0 <= row_start < row_end <= entry.shape[0]:
            raise ValueError(f"MiniMax-H3 invalid row slice for {name}: {row_start}:{row_end}")
        with self._source(name) as source:
            # get_slice's PyTorch result can retain the whole source mmap storage.
            # Copy only the requested rows so callers cannot retain that mapping.
            return source[row_start:row_end].clone()

    def validate_sources(self, names):
        for name in names:
            with self._source(name) as source:
                if name == "rope.inv_freq":
                    n = self.plan.config["rope_freq_dim"]
                    expected = 1.0 / (self.plan.config["rope_theta"] ** (torch.arange(0, 2 * n, 2, dtype=torch.float32, device="cpu") / (2 * n)))
                    actual = source[:].clone()
                    if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
                        raise ValueError("MiniMax-H3 rope.inv_freq differs from native reconstruction")

    def _ranges(self, entry, requested):
        if entry.transform == "qkv_head_interleaved":
            dim = self.plan.config["attention_head_dim"]
            for head in range(self.plan.config["num_attention_heads"]):
                for component, target in enumerate(entry.targets):
                    if target.name in requested:
                        yield target.name, (head * 3 + component) * dim, head * dim, dim
        elif entry.transform == "swap_gate_value":
            half = entry.shape[0] // 2
            for source_start, target_start in ((half, 0), (0, half)):
                for row in range(0, half, self.row_chunk_size):
                    yield entry.targets[0].name, source_start + row, target_start + row, min(self.row_chunk_size, half - row)
        else:
            size = entry.shape[0]
            step = size if len(entry.shape) == 1 else self.row_chunk_size
            for row in range(0, size, step):
                yield entry.targets[0].name, row, row, min(step, size - row)

    def write_targets(self, destinations):
        """Write {logical_name: (runtime_tensor, transpose)} in place, never reallocating."""
        requested = set(destinations)
        unknown = requested - self.plan.targets.keys()
        if unknown:
            raise KeyError(f"MiniMax-H3 missing target plan: {sorted(unknown)}")
        sources = sorted({self.plan.targets[name][0].source_name for name in requested})
        self.validate_sources(sources)
        for name, (tensor, transpose) in destinations.items():
            spec = self.plan.targets[name][1]
            shape = tuple(reversed(spec.shape)) if transpose else spec.shape
            dtype = torch.float32 if spec.dtype == "F32" else torch.bfloat16
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"MiniMax-H3 destination shape/dtype mismatch for {name}: expected {shape}, {dtype}")
        for name in sources:
            entry = self.plan.entries[name]
            with self._source(name) as source:
                for target, start, out_start, rows in self._ranges(entry, requested):
                    tile = source[start : start + rows].clone()
                    destination, transpose = destinations[target]
                    if transpose:
                        destination[:, out_start : out_start + rows].copy_(tile.t())
                    else:
                        destination[out_start : out_start + rows].copy_(tile)
                    del tile

    def load_modules(self, roots, device="cpu", block_index=None, reusable=False):
        """Bind official slices to native base_attrs; transpose exactly at this boundary.

        Reusable leaves retain both their *_cuda_buffer and active tensor identity.
        Pre/post CPU leaves use pin_* attributes, matching native offload semantics.
        No attention, RoPE, or padding computation is changed here.
        """
        destinations, bindings, visited = {}, [], set()
        stack = list(roots)
        while stack:
            module = stack.pop()
            if id(module) in visited:
                continue
            visited.add(id(module))
            for name, attr, transpose in getattr(module, "base_attrs", ()):
                if block_index is not None:
                    name = _H3_BLOCK_KEY_RE.sub(f"transformer_blocks.{int(block_index)}.", name)
                if name in destinations:
                    raise ValueError(f"MiniMax-H3 duplicate target binding: {name}")
                if name not in self.plan.targets:
                    raise KeyError(f"MiniMax-H3 missing target plan: {name}")
                spec = self.plan.targets[name][1]
                dtype = torch.float32 if spec.dtype == "F32" else torch.bfloat16
                storage_attr = f"{attr}_cuda_buffer" if reusable else f"pin_{attr}" if torch.device(device).type == "cpu" else attr
                tensor = getattr(module, storage_attr, None)
                if tensor is None:
                    tensor = torch.empty(spec.shape, dtype=dtype, device=device)
                    if transpose:
                        tensor = tensor.t()
                    setattr(module, storage_attr, tensor)
                elif tensor.device.type != torch.device(device).type or (torch.device(device).index is not None and tensor.device.index != torch.device(device).index):
                    raise ValueError(f"MiniMax-H3 destination device mismatch: {name}")
                destinations[name] = (tensor, transpose)
                bindings.append((module, attr, tensor))
            stack.extend(child for child in getattr(module, "_modules", {}).values() if child is not None)
            stack.extend(child for child in getattr(module, "_parameters", {}).values() if child is not None)
        self.write_targets(destinations)
        for module, attr, tensor in bindings:
            setattr(module, attr, tensor if reusable or tensor.device.type != "cpu" else None)
            if hasattr(module, "bias_name") and module.bias_name is None:
                module.bias = None
                module.pin_bias = None


class MiniMaxH3ShardCheckpoint(MiniMaxH3CheckpointPlan):
    """Production entry: retain fail-fast validation of the complete shard set."""

    def __init__(self, checkpoint_dir, config=None):
        super().__init__(checkpoint_dir, config=config)
        missing = sorted(name for name in set(self.weight_map.values()) if not (self.checkpoint_dir / name).is_file())
        if missing:
            raise FileNotFoundError(f"MiniMax-H3 safetensors index references missing shard files: {missing}")
        self.selected_reader = MiniMaxH3SelectedSourceReader(self) if self.format == "official_raw" else None
        if self.selected_reader is not None:
            self.selected_reader.validate_sources(["rope.inv_freq"])

    def load_tensors(self, names, device="cpu"):
        if self.selected_reader is not None:
            raise ValueError("MiniMax-H3 official raw tensors must use the selected slice adapter")
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


__all__ = ["MiniMaxH3CheckpointPlan", "MiniMaxH3SelectedSourceReader", "MiniMaxH3ShardCheckpoint"]
