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
