#!/usr/bin/env python3
"""Materialize score-only H3 DeCFG checkpoints without loading a full model.

The input is the converted Diffusers H3 root used by training. Adapted weights
are written in FP32 so a full-FP32 fake preserves the update, while loading the
same files into a BF16 teacher reproduces the online merge. Shared encoders and
VAEs are symlinked to the input root. The original base must remain available.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

SCRIPT_ROOT = Path(__file__).resolve().parent
MANIFEST_NAME = "decfg_merged.json"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
_TASKS = {"fl2av": ("transformer", "minimax_h3_t2av", "model_for_comfy_dit.safetensors"), "ref2av": ("transformer_ref", "minimax_h3_ref2av", "model_ref2va_for_comfy_dit.safetensors")}


def _adapter_module():
    location = SCRIPT_ROOT.parent / "lightx2v_train/model_zoo/minimax_h3/training_adapter.py"
    spec = importlib.util.spec_from_file_location("_offline_h3_decfg_adapter", location)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    with Path(path).open("w") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _weight_map(directory):
    for name in (INDEX_NAME, "model.safetensors.index.json"):
        path = directory / name
        if path.is_file():
            mapping = json.loads(path.read_text()).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"Invalid checkpoint index: {path}")
            break
    else:
        for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
            if (directory / name).is_file():
                with safe_open(directory / name, framework="pt", device="cpu") as handle:
                    mapping = {key: name for key in handle.keys()}
                break
        else:
            raise FileNotFoundError(f"No Diffusers safetensors checkpoint in {directory}")
    if not mapping or not all(isinstance(key, str) and isinstance(value, str) and Path(value).name == value for key, value in mapping.items()):
        raise ValueError(f"Unsafe or empty safetensors index in {directory}")
    return mapping


def _plans(config, pairs):
    result = {}
    expected = set()
    groups = (("blocks", "transformer_blocks", config["num_layers"]), ("token_refiner.blocks", "token_refiner.refiner_blocks", config["num_refiner_layers"]))
    suffixes = {"attn.out_proj": "attn.to_out.0", "mlp.fc1": "ff.net.0.proj", "mlp.fc2": "ff.net.2"}
    for source, target, count in groups:
        for index in range(int(count)):
            for suffix in (*suffixes, "attn.qkv_proj"):
                key = f"{source}.{index}.{suffix}"
                expected.add(key)
                if suffix == "attn.qkv_proj":
                    for role_index, role in enumerate(("q", "k", "v")):
                        result[f"{target}.{index}.attn.to_{role}.weight"] = (key, role_index, False)
                else:
                    result[f"{target}.{index}.{suffixes[suffix]}.weight"] = (key, None, suffix == "mlp.fc1")
    if set(pairs) != expected:
        raise ValueError(f"TrainingAdapter coverage differs: missing={sorted(expected - pairs.keys())[:8]}, unexpected={sorted(pairs.keys() - expected)[:8]}")
    return result


def _inspect(source, adapter, component, adapter_module, scale, layout):
    directory = source / component
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("_class_name") != "MiniMaxH3Transformer3DModel" or "num_refiner_layers" not in config:
        raise ValueError(f"Expected converted trainable Diffusers H3 config: {config_path}")
    if (source / MANIFEST_NAME).exists() or (directory / MANIFEST_NAME).exists():
        raise ValueError(f"Source already contains merged DeCFG; refusing a second merge: {directory}")
    options = adapter_module.AdapterOptions.from_mapping({"enabled": True, "path": str(adapter), "scale": scale, "roles": ["fake", "teacher"], "qkv_layout": layout})
    weight_map = _weight_map(directory)
    with safe_open(options.path, framework="pt", device="cpu") as factors:
        pairs = adapter_module._factor_names(factors)
        plans = _plans(config, pairs)
        if set(plans) - weight_map.keys():
            raise ValueError(f"Base lacks adapted weights: {sorted(set(plans) - weight_map.keys())[:8]}")
        for pair in pairs.values():
            for key in pair.values():
                value = factors.get_tensor(key)
                if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Nonfinite or nonfloating adapter factor: {key}")
        headers = {}
        for shard in dict.fromkeys(weight_map.values()):
            with safe_open(directory / shard, framework="pt", device="cpu") as base:
                if set(base.keys()) != {key for key, name in weight_map.items() if name == shard}:
                    raise ValueError(f"Checkpoint index does not match shard: {directory / shard}")
                headers[shard] = {}
                for key in base.keys():
                    value = base.get_slice(key)
                    shape, dtype = value.get_shape(), value.get_dtype()
                    if dtype not in _DTYPE_BYTES:
                        raise ValueError(f"Unsupported base dtype {dtype}: {key}")
                    if key in plans:
                        pair_key, qkv_role, swap_halves = plans[key]
                        a = factors.get_slice(pairs[pair_key]["A"]).get_shape()
                        b = factors.get_slice(pairs[pair_key]["B"]).get_shape()
                        if (
                            len(shape) != 2
                            or len(a) != 2
                            or len(b) != 2
                            or a[0] <= 0
                            or a[1] != shape[1]
                            or b != [shape[0] * (3 if qkv_role is not None else 1), a[0]]
                            or (swap_halves and shape[0] % 2)
                        ):
                            raise ValueError(f"Adapter/base shape mismatch: {key}: base={shape}, A={a}, B={b}")
                        # Online teacher first casts its base to BF16, then adds
                        # the FP32 delta. Requiring BF16 source targets makes
                        # this identical to casting our merged FP32 on load.
                        if dtype != "BF16":
                            raise ValueError(f"Adapted source must be BF16 to preserve both fake and teacher semantics: {key} ({dtype})")
                        dtype = "F32"
                    headers[shard][key] = {"dtype": dtype, "shape": shape}
    return {"directory": directory, "config": config, "options": options, "weight_map": weight_map, "pairs": pairs, "plans": plans, "headers": headers}


def _write_shard(path, source, header, inspection, factors, adapter_module, *, device, tile_rows):
    """Write valid safetensors incrementally, with only one row tile on device."""
    offset = 0
    encoded_header = {"__metadata__": {"format": "pt", "lightx2v_score_decfg": "1"}}
    for key, value in header.items():
        size = math.prod(value["shape"]) * _DTYPE_BYTES[value["dtype"]]
        encoded_header[key] = {**value, "data_offsets": [offset, offset + size]}
        offset += size
    raw_header = json.dumps(encoded_header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    digest = hashlib.sha256()
    with Path(path).open("xb") as destination, safe_open(source, framework="pt", device="cpu") as base:

        def write(raw):
            destination.write(raw)
            digest.update(raw)

        write(struct.pack("<Q", len(raw_header)))
        write(raw_header)
        cached_pair, a, b = None, None, None
        for key, value in header.items():
            plan = inspection["plans"].get(key)
            shape = value["shape"]
            if not shape:
                tensor = base.get_tensor(key)
                write(memoryview(tensor.reshape(-1).contiguous().view(torch.uint8).numpy()))
                continue
            if plan is not None:
                pair_key, qkv_role, swap_halves = plan
                if pair_key != cached_pair:
                    pair = inspection["pairs"][pair_key]
                    a = factors.get_tensor(pair["A"]).float().to(device)
                    b = factors.get_tensor(pair["B"]).float().to(device)
                    cached_pair = pair_key
            # A row can be very wide (AdaLN); cap passthrough reads at 32 MiB.
            row_bytes = max(1, math.prod(shape[1:]) * _DTYPE_BYTES[value["dtype"]])
            rows = min(tile_rows, max(1, 32 * 1024 * 1024 // row_bytes))
            for start in range(0, shape[0], rows):
                stop = min(start + rows, shape[0])
                tensor = base.get_slice(key)[start:stop]
                if plan is not None:
                    selected = adapter_module._source_rows(
                        torch.arange(start, stop),
                        output_rows=shape[0],
                        qkv_role=qkv_role,
                        swap_halves=swap_halves,
                        layout=inspection["options"].qkv_layout,
                        head_dim=int(inspection["config"].get("attention_head_dim", 128)),
                    )
                    delta = b.index_select(0, selected.to(device)) @ a
                    tensor = tensor.to(device=device, dtype=torch.float32).add_(delta, alpha=inspection["options"].scale).cpu()
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError(f"Merged weight is nonfinite: {key}, rows {start}:{stop}")
                write(memoryview(tensor.contiguous().view(torch.uint8).reshape(-1).numpy()))
        destination.flush()
        os.fsync(destination.fileno())
    return {"path": Path(path).name, "size_bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}, offset


def merge(model_path, adapter_dir, output_dir, *, tasks=("fl2av", "ref2av"), scale=1.0, qkv_layout="comfy", device="cpu", tile_rows=256):
    source, adapters, output = (Path(path).expanduser().resolve() for path in (model_path, adapter_dir, output_dir))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}; reuse the completed checkpoint or choose a new output directory")
    if source in output.parents:
        raise ValueError("Output must be outside the source model directory")
    if not source.is_dir() or not tasks or len(set(tasks)) != len(tasks) or set(tasks) - _TASKS.keys() or tile_rows <= 0:
        raise ValueError("Require a source model directory, unique fl2av/ref2av tasks, and positive tile rows")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Score DeCFG scale must be finite and positive")
    if qkv_layout not in {"comfy", "official"}:
        raise ValueError("qkv_layout must be comfy or official")
    module = _adapter_module()
    inspections = {}
    for task in tasks:
        component, model_name, filename = _TASKS[task]
        if qkv_layout == "official":
            filename = filename.replace("_for_comfy_dit", "")
        inspection = _inspect(source, adapters / filename, component, module, scale, qkv_layout)
        inspections[component] = (model_name, inspection)
    estimated_bytes = sum(math.prod(value["shape"]) * _DTYPE_BYTES[value["dtype"]] for _, inspection in inspections.values() for header in inspection["headers"].values() for value in header.values())
    print(f"[merge] tasks={','.join(tasks)} weight_bytes={estimated_bytes} device={device} source={source}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.merge-", dir=output.parent))
    staging.chmod(0o755)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        manifest = {
            "schema_version": 1,
            "format": "lightx2v_h3_score_decfg",
            "complete": True,
            "source_model_path": str(source),
            "precision": "fp32_merged_targets",
            "student_uses_decfg": False,
            "components": {},
        }
        # Other transformer partitions are intentionally omitted, so requesting
        # an unmerged task cannot silently fall back to the original checkpoint.
        for item in source.iterdir():
            if item.name not in {"transformer", "transformer_ref", MANIFEST_NAME}:
                (staging / item.name).symlink_to(item, target_is_directory=item.is_dir())
        for component, (model_name, inspection) in inspections.items():
            directory = staging / component
            directory.mkdir()
            shutil.copyfile(inspection["directory"] / "config.json", directory / "config.json")
            shards, total_size = [], 0
            with safe_open(inspection["options"].path, framework="pt", device="cpu") as factors:
                for index, (shard, header) in enumerate(inspection["headers"].items(), 1):
                    print(f"[merge] component={component} shard={index}/{len(inspection['headers'])} file={shard}", flush=True)
                    metadata, size = _write_shard(directory / shard, inspection["directory"] / shard, header, inspection, factors, module, device=device, tile_rows=tile_rows)
                    shards.append(metadata)
                    total_size += size
            _json(directory / INDEX_NAME, {"metadata": {"total_size": total_size}, "weight_map": inspection["weight_map"]})
            manifest["components"][component] = {
                "model_name": model_name,
                "source_transformer_dir": str(inspection["directory"]),
                "training_adapter": inspection["options"].metadata(),
                "config_sha256": _digest(directory / "config.json"),
                "index_sha256": _digest(directory / INDEX_NAME),
                "shards": shards,
                "adapted_weights": len(inspection["plans"]),
            }
        # A completion manifest is published only after every tensor is durable.
        _json(staging / MANIFEST_NAME, manifest)
        for component in inspections:
            _json(staging / component / MANIFEST_NAME, manifest)
        # Refuse a concurrent publication rather than replacing its contents.
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Another merge published {output}")
        staging.rename(output)
        print(f"[merge] complete output={output} manifest={output / MANIFEST_NAME}", flush=True)
        return manifest
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Converted H3 root containing transformer/ and transformer_ref/")
    parser.add_argument("--adapter-dir", required=True, help="Downloaded MiniMax-H3-TrainingAdapter directory")
    parser.add_argument("--output-dir", required=True, help="New score-only merged model root; existing paths are never overwritten")
    parser.add_argument("--tasks", nargs="+", choices=tuple(_TASKS), default=list(_TASKS))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--qkv-layout", choices=("comfy", "official"), default="comfy")
    parser.add_argument("--device", default="cpu", help="cpu or one CUDA device; only row tiles and low-rank factors occupy GPU memory")
    parser.add_argument("--tile-rows", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    merge(args.model_path, args.adapter_dir, args.output_dir, tasks=tuple(args.tasks), scale=args.scale, qkv_layout=args.qkv_layout, device=args.device, tile_rows=args.tile_rows)


if __name__ == "__main__":
    main()
