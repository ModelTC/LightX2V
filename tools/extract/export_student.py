#!/usr/bin/env python3
"""Export full train/RL student weights on CPU, without importing either trainer."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def torch_dtype(name):
    types = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
    if name not in types:
        raise ValueError(f"Unsupported dtype: {name}")
    return types[name]


def normalize_key(key):
    """Remove training/inference wrappers, preserving architecture module names."""
    prefixes = ("module.", "_orig_mod.", "_fsdp_wrapped_module.", "diffusion_model.", "transformer.", "model.")
    while key.startswith(prefixes):
        key = next(key[len(prefix) :] for prefix in prefixes if key.startswith(prefix))
    if any(part in key for part in ("lora_", ".lora.", ".base_layer.", "base_model.model.")):
        raise ValueError("Expected full student weights, found an adapter checkpoint. Merge the trained adapter into its base before exporting/extracting.")
    return key


def _load_pt(path, distributed=False):
    # RL rank files serialize DTensors; plain model_state.pt uses tensor-only loading.
    kwargs = {"map_location": "cpu", "weights_only": not distributed}
    if "mmap" in inspect.signature(torch.load).parameters:
        kwargs["mmap"] = True
    state = torch.load(path, **kwargs)
    while isinstance(state, Mapping):
        nested = [key for key in ("student_model", "student", "state_dict", "model") if isinstance(state.get(key), Mapping)]
        if not nested:
            break
        state = state[nested[0]]
    if not isinstance(state, Mapping) or not state or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError(f"Expected a model tensor state dict: {path}")
    return state


def resolve_model_path(path):
    """Choose a denoiser/student only; never glob recursively into fake/optimizer/VAE."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() or (path / ".metadata").is_file() or (path / "fsdp_config.json").is_file():
        return path
    for relative in ("student_consolidated", "transformer", "dist_state/student", "dist_state", "actor", "model_state.pt", "consolidated.safetensors"):
        candidate = path / relative
        if candidate.exists():
            return resolve_model_path(candidate)
    return path


class ModelWeights(Mapping):
    """Lazy safetensors/DCP reader; RL rank states are CPU memory-mapped on modern torch."""

    def __init__(self, path):
        self.path = resolve_model_path(path)
        self.entries = {}
        self.shapes = {}
        self.states = []
        self.reader = None
        print(f"Loading model inventory: {self.path}", flush=True)
        if self.path.is_dir() and (self.path / ".metadata").is_file():
            self.kind = "dcp"
            self._index_dcp()
        elif self.path.is_dir() and (self.path / "fsdp_config.json").is_file():
            self.kind = "rl_fsdp"
            self._index_rl()
        elif self.path.is_file() and self.path.suffix in (".pt", ".pth", ".bin"):
            self.kind = "pytorch"
            self.states = [_load_pt(self.path)]
            for key, tensor in self.states[0].items():
                self._add(key, key, tensor.shape)
        else:
            self.kind = "safetensors"
            self._index_safetensors()
        if not self.entries:
            raise ValueError(f"No full model tensors found: {self.path}")
        print(f"Successfully indexed {len(self)} tensors ({self.kind})", flush=True)

    def _add(self, key, entry, shape):
        name = normalize_key(key)
        if name in self.entries:
            raise ValueError(f"Duplicate normalized weight: {name}")
        self.entries[name] = entry
        self.shapes[name] = tuple(shape)

    def _index_safetensors(self):
        if self.path.is_file():
            files = [self.path]
            weight_map = None
        else:
            indices = sorted(self.path.glob("*.safetensors.index.json"))
            if len(indices) > 1:
                raise ValueError(f"Multiple weight indices; specify a transformer directory: {self.path}")
            weight_map = json.loads(indices[0].read_text())["weight_map"] if indices else None
            files = sorted({self.path / name for name in weight_map.values()}) if weight_map else sorted(self.path.glob("*.safetensors"))
        if not files:
            raise ValueError(f"No supported model weights in {self.path}")
        seen = set()
        for file in tqdm(files, desc="Indexing weight files", unit="file"):
            with safe_open(str(file), framework="pt", device="cpu") as archive:
                keys = archive.keys()
                for key in keys:
                    if weight_map is not None and weight_map.get(key) != file.name:
                        raise ValueError(f"Weight index disagrees with shard: {key} in {file}")
                    self._add(key, (file, key), archive.get_slice(key).get_shape())
                    seen.add(key)
        if weight_map is not None and seen != set(weight_map):
            raise ValueError(f"Missing indexed weights: {sorted(set(weight_map) - seen)[:8]}")

    def _index_dcp(self):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata

        self.reader = dcp.FileSystemReader(str(self.path))
        metadata = self.reader.read_metadata().state_dict_metadata
        prefix = next((prefix for prefix in ("student_model.", "model.") if any(key.startswith(prefix) for key in metadata)), None)
        if prefix is None:
            raise ValueError(f"No model/student_model namespace in DCP: {self.path}")
        for key, value in metadata.items():
            if key.startswith(prefix) and isinstance(value, TensorStorageMetadata):
                self._add(key[len(prefix) :], (key, value.properties.dtype), value.size)

    def _index_rl(self):
        config = json.loads((self.path / "fsdp_config.json").read_text())
        count = config.get("world_size")
        if config.get("FSDP_version") != 2 or type(count) is not int or count < 1:
            raise ValueError("RL export requires an FSDP2 checkpoint with a positive world_size")
        files = [self.path / f"model_world_size_{count}_rank_{rank}.pt" for rank in range(count)]
        if set(files) != set(self.path.glob("model_world_size_*_rank_*.pt")):
            raise ValueError("RL checkpoint has missing or unexpected model rank files")
        for file in tqdm(files, desc="Opening RL model ranks", unit="rank"):
            state = _load_pt(file, distributed=True)
            if self.states and state.keys() != self.states[0].keys():
                raise ValueError(f"RL rank tensor keys disagree: {file}")
            self.states.append(state)
        for key, tensor in self.states[0].items():
            self._add(key, key, tensor.shape)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __contains__(self, key):
        return key in self.entries

    def __getitem__(self, key):
        if self.kind == "dcp":
            return self.read_many([key])[key]
        if self.kind == "safetensors":
            file, original = self.entries[key]
            with safe_open(str(file), framework="pt", device="cpu") as archive:
                return archive.get_tensor(original)
        if self.kind == "pytorch":
            return self.states[0][self.entries[key]]
        return _reconstruct_ranks([state[self.entries[key]] for state in self.states])

    def read_many(self, keys):
        if self.kind != "dcp":
            return {key: self[key] for key in keys}
        import torch.distributed.checkpoint as dcp

        state = {self.entries[key][0]: torch.empty(self.shapes[key], dtype=self.entries[key][1]) for key in keys}
        # A subset of model tensors is deliberate: optimizer and fake roles are never loaded.
        dcp.load_state_dict(state, storage_reader=self.reader, no_dist=True)
        return {key: state[self.entries[key][0]] for key in keys}


def _reconstruct_ranks(values):
    """Reassemble FSDP2's one-dimensional Shard/Replicate layouts without a process group."""
    try:
        from torch.distributed.tensor import DTensor, Replicate, Shard
    except ImportError:
        from torch.distributed._tensor import DTensor, Replicate, Shard

    first = values[0]
    if not isinstance(first, DTensor):
        if any(type(value) is not torch.Tensor or not torch.equal(first, value) for value in values):
            raise ValueError("Plain RL tensors must be identical full replicas, not ambiguous shards")
        return first
    mesh = first.device_mesh.mesh
    if mesh.ndim != 1 or sorted(mesh.tolist()) != list(range(len(values))) or len(first.placements) != 1:
        raise ValueError("Only one-dimensional FSDP2 meshes covering all saved ranks are supported")
    placement = first.placements[0]
    if not isinstance(placement, (Shard, Replicate)):
        raise TypeError(f"Unsupported FSDP placement: {placement}")
    if any(
        not isinstance(value, DTensor) or value.shape != first.shape or value.dtype != first.dtype or value.placements != first.placements or not torch.equal(value.device_mesh.mesh, mesh)
        for value in values
    ):
        raise ValueError("RL tensor shapes, dtypes, meshes or placements disagree")
    local = [values[rank].to_local() for rank in mesh.tolist()]
    if isinstance(placement, Replicate):
        if any(value.shape != first.shape or not torch.equal(value, local[0]) for value in local):
            raise ValueError("RL replicated tensors disagree")
        return local[0]
    dim = placement.dim
    chunk = (first.shape[dim] + len(values) - 1) // len(values)
    for rank, tensor in enumerate(local):
        expected = list(first.shape)
        expected[dim] = max(0, min(chunk, first.shape[dim] - rank * chunk))
        if list(tensor.shape) != expected:
            raise ValueError("Unexpected RL local shard shape")
    return torch.cat(local, dim=dim).contiguous()


def export_student(checkpoint, output, base_model=None, dtype="preserve", max_shard_size=2 * 1024**3, fill_missing_from_base=False):
    """Write a standalone transformer directory; model names/config are preserved."""
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(output)
    if max_shard_size <= 0:
        raise ValueError("max_shard_size must be positive")
    output_dtype = None if dtype == "preserve" else torch_dtype(dtype)
    student = ModelWeights(checkpoint)
    base = ModelWeights(base_model) if base_model else None
    missing = set(base) - set(student) if base is not None else set()
    if base is not None:
        extra = set(student) - set(base)
        mismatches = [key for key in set(student) & set(base) if student.shapes[key] != base.shapes[key]]
        if extra or mismatches or (missing and not fill_missing_from_base):
            raise ValueError(
                f"Student/base schema mismatch: missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}, shapes={mismatches[:8]}. Use --fill-missing-from-base only for intentionally frozen parameters omitted by DCP."
            )
    elif fill_missing_from_base:
        raise ValueError("--fill-missing-from-base requires --base-model")
    if missing:
        print(f"Filling {len(missing)} missing frozen tensors from base: {sorted(missing)}", flush=True)
    config_candidates = [student.path / "config.json", student.path / "huggingface/config.json"]
    if base is not None:
        config_candidates.append((base.path if base.path.is_dir() else base.path.parent) / "config.json")
    config_path = next((path for path in config_candidates if path.is_file()), None)
    if config_path is None:
        raise ValueError("No transformer config.json found; supply --base-model with the matching model directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=str(output.parent)))
    try:
        pending, size, total, shards = {}, 0, 0, []

        def flush():
            if pending:
                file = f"part-{len(shards) + 1:05d}.safetensors"
                print(f"Writing {file}: {len(pending)} tensors", flush=True)
                save_file(pending, str(staging / file), metadata={"format": "pt"})
                shards.append((file, list(pending)))
                pending.clear()

        for key in tqdm(sorted(set(student) | missing), desc="Exporting student weights", unit="tensor"):
            tensor = (base[key] if key in missing else student[key]).detach().cpu()
            if tensor.is_floating_point():
                tensor = tensor.to(output_dtype or tensor.dtype)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite exported weight: {key}")
            # Own the storage; safetensors cannot save aliased checkpoint tensors.
            tensor = tensor.clone().contiguous()
            nbytes = tensor.numel() * tensor.element_size()
            if pending and size + nbytes > max_shard_size:
                flush()
                size = 0
            pending[key] = tensor
            size += nbytes
            total += nbytes
        flush()
        weight_map = {}
        for i, (file, keys) in enumerate(shards, 1):
            name = "diffusion_pytorch_model.safetensors" if len(shards) == 1 else f"diffusion_pytorch_model-{i:05d}-of-{len(shards):05d}.safetensors"
            (staging / file).rename(staging / name)
            weight_map.update({key: name for key in keys})
        (staging / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")
        config = json.loads(config_path.read_text())
        if output_dtype is not None:
            config["torch_dtype"] = str(output_dtype).split(".")[-1]
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "export_manifest.json").write_text(
            json.dumps(
                {"checkpoint": str(student.path), "base_model": str(base.path) if base is not None else None, "tensor_count": len(weight_map), "dtype": dtype, "filled_from_base": sorted(missing)},
                indent=2,
            )
            + "\n"
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f"Student export completed: {output} ({len(weight_map)} tensors, {len(shards)} shards, {total / 1024**3:.3f} GiB)", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Train checkpoint, RL global_step/actor, or full model weights")
    parser.add_argument("--base-model", help="Matching base transformer/pipeline, for config and schema validation")
    parser.add_argument("--output", required=True, help="New standalone transformer directory")
    parser.add_argument("--dtype", default="preserve", choices=["preserve", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--max-shard-size-gb", type=float, default=2.0)
    parser.add_argument("--fill-missing-from-base", action="store_true", help="Explicitly restore frozen parameters omitted by DCP from --base-model")
    args = parser.parse_args()
    export_student(args.checkpoint, args.output, args.base_model, args.dtype, int(args.max_shard_size_gb * 1024**3), args.fill_missing_from_base)


if __name__ == "__main__":
    main()
