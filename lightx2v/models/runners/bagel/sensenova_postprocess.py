# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
import importlib.util
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


def resolve_pose_string(pose_str):
    def extract(tag, count):
        rows = []
        for value in re.findall(rf"<{tag}>(.*?)</{tag}>", pose_str, flags=re.DOTALL):
            numbers = re.findall(r"-?\d+", value)
            if len(numbers) == count:
                rows.append([int(number) / 1000 for number in numbers])
        return np.asarray(rows, dtype=np.float32)

    quaternions = extract("quat", 4)
    offsets = extract("offset", 3)
    scale_values = []
    for value in re.findall(r"<scale>(.*?)</scale>", pose_str, flags=re.DOTALL):
        numbers = re.findall(r"-?\d+", value)
        if numbers:
            scale_values.append(int(numbers[0]))
    scales = np.asarray(scale_values, dtype=np.float32)

    if offsets.size == 0 or scales.size == 0:
        return None
    if len(offsets) != len(scales) or len(quaternions) != len(offsets):
        return None
    valid = np.isfinite(offsets).all(axis=1) & np.isfinite(scales)
    offsets = offsets[valid]
    scales = scales[valid]
    quaternions = quaternions[valid]
    if offsets.size == 0:
        return None
    return {
        "rotation": quaternions.tolist(),
        "translation": (offsets * scales[:, None] / 100.0).tolist(),
    }


@lru_cache(maxsize=None)
def load_official_example_constant(source_path, constant_name):
    """Read one literal prompt constant without importing the official model code."""
    source_root = Path(source_path).expanduser().resolve()
    example_path = source_root / "inference" / "example_visualize.py"
    if not example_path.is_file():
        raise FileNotFoundError(f"SenseNova-Vision official example is missing: {example_path}")

    tree = ast.parse(example_path.read_text(encoding="utf-8"), filename=str(example_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == constant_name for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"SenseNova-Vision official constant {constant_name!r} must be a non-empty string.")
        return value
    raise KeyError(f"SenseNova-Vision official constant {constant_name!r} was not found in {example_path}.")


def load_official_postprocess(source_path):
    """Load the official GLB postprocessor lazily; normal tasks do not need Open3D."""

    source_root = Path(source_path).expanduser().resolve()
    if not (source_root / "inference" / "utils_3d").is_dir():
        raise FileNotFoundError(f"SenseNova-Vision source directory is invalid: {source_root}. Expected inference/utils_3d for GLB postprocessing.")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        module = importlib.import_module("inference.utils_3d")
    except ModuleNotFoundError as exc:
        raise RuntimeError("SenseNova recon3d GLB postprocessing requires the official source and optional packages open3d/trimesh/scipy. Raw .npy output is still available.") from exc
    return module.postprocess_reconstruction


def load_official_visualizers(source_path):
    """Load upstream visualization helpers without claiming the global ``utils`` package."""

    source_root = Path(source_path).expanduser().resolve()
    utils_root = source_root / "utils"
    init_path = utils_root / "__init__.py"
    visualize_path = utils_root / "visualize.py"
    if not init_path.is_file() or not visualize_path.is_file():
        raise FileNotFoundError(f"SenseNova-Vision source directory is invalid: {source_root}. Expected utils/visualize.py for official result visualization.")

    package_name = "_lightx2v_sensenova_official_utils"
    if package_name not in sys.modules:
        package_spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(utils_root)],
        )
        if package_spec is None or package_spec.loader is None:
            raise RuntimeError(f"Failed to load SenseNova visualization package from {utils_root}")
        package = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = package
        package_spec.loader.exec_module(package)

    return importlib.import_module(f"{package_name}.visualize")
