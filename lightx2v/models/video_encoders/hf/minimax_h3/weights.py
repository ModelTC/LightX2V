# Copyright 2026 The LightX2V Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small, dependency-free loader for MiniMax-H3 safetensors components.

The released H3 component directories may contain either one safetensors file
or several indexed shards.  The VAE inference wrappers are decode-only, so
loading an entire shard into a temporary state dict would waste many gigabytes
on encoder weights.  This helper scans the files and assigns only the exact
parameters/buffers present in the target module, one tensor at a time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open


@dataclass(frozen=True)
class SafetensorsSubsetReport:
    component_dir: Path
    files: tuple[Path, ...]
    loaded_keys: tuple[str, ...]
    ignored_keys: int


def _component_files(component_dir: str | Path) -> tuple[Path, ...]:
    component_dir = Path(component_dir)
    if component_dir.is_file():
        return (component_dir,)
    if not component_dir.is_dir():
        raise FileNotFoundError(f"MiniMax-H3 checkpoint does not exist: {component_dir}")
    files = tuple(sorted(component_dir.glob("*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found in MiniMax-H3 component directory: {component_dir}")
    return files


def _get_parent(module: nn.Module, key: str) -> tuple[nn.Module, str]:
    parts = key.split(".")
    parent: nn.Module = module
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _assign_tensor(module: nn.Module, key: str, tensor: torch.Tensor) -> None:
    parent, name = _get_parent(module, key)
    if name in parent._parameters:
        old_parameter = parent._parameters[name]
        requires_grad = False if old_parameter is None else old_parameter.requires_grad
        parent._parameters[name] = nn.Parameter(tensor, requires_grad=requires_grad)
    elif name in parent._buffers:
        parent._buffers[name] = tensor
    else:
        raise KeyError(f"Cannot assign checkpoint tensor {key!r}: it is not a parameter or buffer")


_SAFETENSORS_DTYPES = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.bool: "BOOL",
}


def _expected_specs(module: nn.Module) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {key: (tuple(value.shape), value.dtype) for key, value in module.state_dict().items()}


def validate_safetensors_subset(module: nn.Module, component_dir: str | Path) -> SafetensorsSubsetReport:
    """Validate exact key and shape parity without materializing checkpoint tensors."""

    files = _component_files(component_dir)
    expected = _expected_specs(module)
    expected_roots = {key.partition(".")[0] for key in expected}
    found: set[str] = set()
    unexpected: list[str] = []
    ignored = 0

    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key not in expected:
                    if key.partition(".")[0] in expected_roots:
                        unexpected.append(key)
                    else:
                        ignored += 1
                    continue
                tensor_slice = checkpoint.get_slice(key)
                shape = tuple(tensor_slice.get_shape())
                expected_shape, expected_dtype = expected[key]
                if shape != expected_shape:
                    raise ValueError(f"Shape mismatch for {key!r}: model expects {expected_shape}, checkpoint contains {shape}")
                checkpoint_dtype = str(tensor_slice.get_dtype())
                if checkpoint_dtype != _SAFETENSORS_DTYPES.get(expected_dtype):
                    raise TypeError(f"Dtype mismatch for {key!r}: model expects {expected_dtype}, checkpoint contains {checkpoint_dtype}")
                if key in found:
                    unexpected.append(f"duplicate:{key}")
                found.add(key)

    missing = sorted(set(expected) - found)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:20]}{' ...' if len(missing) > 20 else ''}")
        if unexpected:
            details.append(f"unexpected={unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
        raise RuntimeError("MiniMax-H3 checkpoint does not match the native module: " + ", ".join(details))

    return SafetensorsSubsetReport(Path(component_dir), files, tuple(sorted(found)), ignored)


def load_safetensors_subset(module: nn.Module, component_dir: str | Path) -> SafetensorsSubsetReport:
    """Load exactly ``module.state_dict()`` from one or more original H3 shards.

    This also supports a module constructed on the ``meta`` device: each meta
    parameter is replaced directly by its CPU checkpoint tensor, avoiding a
    second full-size initialized copy of the VAE.
    """

    report = validate_safetensors_subset(module, component_dir)
    expected = set(module.state_dict())
    loaded: set[str] = set()

    for filename in report.files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key not in expected:
                    continue
                _assign_tensor(module, key, checkpoint.get_tensor(key))
                loaded.add(key)

    # validate_safetensors_subset already checked this, but keep the invariant
    # local to the mutating operation as well.
    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(f"Failed to load MiniMax-H3 tensors: {missing[:20]}")
    return SafetensorsSubsetReport(report.component_dir, report.files, tuple(sorted(loaded)), report.ignored_keys)


_OFFICIAL_VIDEO_VAE_SIGNATURE = {
    "decoder.x_embedder.weight",
    "decoder.transformer_blocks.0.attn.to_qkv.weight",
    "decoder.transformer_blocks.0.ff.w1.weight",
}


def _is_official_video_vae_checkpoint(component_dir: str | Path) -> bool:
    """Detect the released Video VAE schema from safetensors keys."""
    keys: set[str] = set()
    for filename in _component_files(component_dir):
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            keys.update(checkpoint.keys())
    return _OFFICIAL_VIDEO_VAE_SIGNATURE <= keys and "decoder.proj_in.weight" not in keys


def _official_video_vae_targets(source_key: str) -> tuple[str, ...] | None:
    if source_key == "decoder.mask_token":
        return ()

    qkv = re.fullmatch(r"(decoder\.transformer_blocks\.\d+\.attn)\.to_qkv\.(weight|bias)", source_key)
    if qkv:
        prefix, suffix = qkv.groups()
        return tuple(f"{prefix}.to_{name}.{suffix}" for name in ("q", "k", "v"))

    w1 = re.fullmatch(r"(decoder\.transformer_blocks\.\d+\.ff)\.w1\.(weight|bias)", source_key)
    if w1:
        prefix, suffix = w1.groups()
        return (f"{prefix}.net.0.proj.{suffix}",)

    target = source_key
    down_block = re.fullmatch(r"encoder\.down\.(\d+)\.block\.(\d+)\.(.+)", target)
    if down_block:
        stage, block, suffix = down_block.groups()
        suffix = suffix.replace("nin_shortcut", "conv_shortcut")
        target = f"encoder.down_blocks.{stage}.resnets.{block}.{suffix}"
    else:
        downsample = re.fullmatch(r"encoder\.down\.(\d+)\.downsample\.conv\.(weight|bias)", target)
        if downsample:
            stage, suffix = downsample.groups()
            target = f"encoder.down_blocks.{stage}.downsamplers.0.conv.{suffix}"

    target = target.replace("decoder.x_embedder.", "decoder.proj_in.")
    target = re.sub(r"(decoder\.transformer_blocks\.\d+\.attn)\.to_out\.", r"\1.to_out.0.", target)
    target = re.sub(r"(decoder\.transformer_blocks\.\d+\.ff)\.w2\.", r"\1.net.2.", target)
    return (target,)


def _official_validation_error(
    *, unknown: list[str], missing: list[str], duplicates: list[str], shape_mismatches: list[str], dtype_mismatches: list[str]
) -> RuntimeError:
    details = []
    for label, values in (
        ("unknown", unknown),
        ("missing", missing),
        ("duplicate", duplicates),
        ("shape_mismatch", shape_mismatches),
        ("dtype_mismatch", dtype_mismatches),
    ):
        if values:
            details.append(f"{label}={values[:20]}{' ...' if len(values) > 20 else ''}")
    return RuntimeError("Official MiniMax-H3 Video VAE checkpoint validation failed: " + ", ".join(details))


def validate_minimax_h3_video_vae_checkpoint(module: nn.Module, component_dir: str | Path) -> SafetensorsSubsetReport:
    """Validate official-to-native coverage using safetensors metadata only."""
    files = _component_files(component_dir)
    expected = _expected_specs(module)
    assigned: set[str] = set()
    unknown: list[str] = []
    duplicates: list[str] = []
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    ignored = 0

    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for source_key in checkpoint.keys():
                tensor_slice = checkpoint.get_slice(source_key)
                source_shape = tuple(tensor_slice.get_shape())
                source_dtype = str(tensor_slice.get_dtype())
                targets = _official_video_vae_targets(source_key)
                if targets == ():
                    decoder_dim = expected.get("decoder.proj_in.weight", ((0,), torch.float32))[0][0]
                    if source_shape != (1, 1, decoder_dim):
                        shape_mismatches.append(f"{source_key}:{source_shape}!={(1, 1, decoder_dim)}")
                    elif source_dtype != _SAFETENSORS_DTYPES.get(expected["decoder.proj_in.weight"][1]):
                        dtype_mismatches.append(source_key)
                    ignored += 1
                    continue
                if targets is None or any(target not in expected for target in targets):
                    unknown.append(source_key)
                    continue

                if len(targets) == 3:
                    if not source_shape or source_shape[0] % 3:
                        shape_mismatches.append(source_key)
                        continue
                    mapped_shape = (source_shape[0] // 3, *source_shape[1:])
                else:
                    mapped_shape = source_shape
                for target in targets:
                    expected_shape, expected_dtype = expected[target]
                    if mapped_shape != expected_shape:
                        shape_mismatches.append(f"{source_key}->{target}:{mapped_shape}!={expected_shape}")
                    if source_dtype != _SAFETENSORS_DTYPES.get(expected_dtype):
                        dtype_mismatches.append(f"{source_key}->{target}")
                    if target in assigned:
                        duplicates.append(target)
                    assigned.add(target)

    missing = sorted(set(expected) - assigned)
    if unknown or missing or duplicates or shape_mismatches or dtype_mismatches or ignored != 1:
        if ignored != 1:
            unknown.append(f"ignored_count:{ignored} (expected decoder.mask_token exactly once)")
        raise _official_validation_error(
            unknown=unknown,
            missing=missing,
            duplicates=duplicates,
            shape_mismatches=shape_mismatches,
            dtype_mismatches=dtype_mismatches,
        )
    return SafetensorsSubsetReport(Path(component_dir), files, tuple(sorted(assigned)), ignored)


def load_minimax_h3_video_vae_checkpoint(module: nn.Module, component_dir: str | Path) -> SafetensorsSubsetReport:
    """Stream the released official Video VAE schema into the native module."""
    report = validate_minimax_h3_video_vae_checkpoint(module, component_dir)
    loaded: set[str] = set()
    for filename in report.files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for source_key in checkpoint.keys():
                targets = _official_video_vae_targets(source_key)
                if targets == ():
                    continue
                tensor_slice = checkpoint.get_slice(source_key)
                if len(targets) == 3:
                    chunk_size = tensor_slice.get_shape()[0] // 3
                    for index, target in enumerate(targets):
                        _assign_tensor(module, target, tensor_slice[index * chunk_size : (index + 1) * chunk_size])
                        loaded.add(target)
                elif ".ff.w1." in source_key:
                    target = targets[0]
                    rows = tensor_slice.get_shape()[0]
                    half = rows // 2
                    gate = tensor_slice[:half]
                    reordered = torch.empty(tuple(tensor_slice.get_shape()), dtype=gate.dtype)
                    reordered[half:].copy_(gate)
                    del gate
                    value = tensor_slice[half:]
                    reordered[:half].copy_(value)
                    del value
                    _assign_tensor(module, target, reordered)
                    loaded.add(target)
                else:
                    target = targets[0]
                    _assign_tensor(module, target, checkpoint.get_tensor(source_key))
                    loaded.add(target)
    if loaded != set(report.loaded_keys):
        raise RuntimeError("Official MiniMax-H3 Video VAE load did not reproduce validated target coverage")
    return SafetensorsSubsetReport(report.component_dir, report.files, tuple(sorted(loaded)), report.ignored_keys)
