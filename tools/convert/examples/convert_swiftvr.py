#!/usr/bin/env python3
"""Convert a SwiftVR Diffusers checkpoint to LightX2V's Wan key layout.

Example:
    python tools/convert/examples/convert_swiftvr.py \
        --source /path/to/SwiftVR \
        --output /path/to/SwiftVR_lightx2v
"""

import argparse
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from safetensors import safe_open

CONVERTER = Path(__file__).resolve().parents[1] / "converter.py"
TRANSFORMER_PATH = Path("transformer/diffusion_pytorch_model.safetensors")
REQUIRED_KEYS = {
    "blocks.0.self_attn.q.weight",
    "blocks.0.cross_attn.q.weight",
    "blocks.0.ffn.0.weight",
    "blocks.0.norm3.weight",
    "blocks.0.modulation",
    "head.head.weight",
    "head.modulation",
}


def checkpoint_signature(path: Path):
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        tensors = Counter()
        for key in keys:
            tensor = checkpoint.get_slice(key)
            tensors[(tensor.get_dtype(), tuple(tensor.get_shape()))] += 1
    return keys, tensors


def validate_conversion(source: Path, converted: Path):
    source_keys, source_tensors = checkpoint_signature(source)
    converted_keys, converted_tensors = checkpoint_signature(converted)
    if len(converted_keys) != len(source_keys):
        raise RuntimeError(f"Converted checkpoint has {len(converted_keys)} tensors; expected {len(source_keys)}")
    if converted_tensors != source_tensors:
        raise RuntimeError("Converted checkpoint changed tensor shapes or dtypes")
    missing = REQUIRED_KEYS - converted_keys
    if missing:
        raise RuntimeError(f"Converted checkpoint is missing LightX2V keys: {sorted(missing)}")


def convert_swiftvr(source: Path, output: Path):
    source = source.resolve()
    output = output.resolve()
    source_transformer = source / TRANSFORMER_PATH
    if not source_transformer.is_file():
        raise FileNotFoundError(f"SwiftVR transformer checkpoint not found: {source_transformer}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as workspace:
        converted_model = Path(workspace) / output.name
        shutil.copytree(
            source,
            converted_model,
            ignore=shutil.ignore_patterns(TRANSFORMER_PATH.name),
        )
        converted_transformer = converted_model / TRANSFORMER_PATH
        subprocess.run(
            [
                sys.executable,
                str(CONVERTER),
                "--source",
                str(source_transformer),
                "--output",
                str(converted_transformer.parent),
                "--output_name",
                converted_transformer.stem,
                "--direction",
                "backward",
                "--model_type",
                "wan_dit",
                "--device",
                "cpu",
                "--single_file",
            ],
            check=True,
        )
        validate_conversion(source_transformer, converted_transformer)
        converted_model.rename(output)

    print(f"Converted SwiftVR checkpoint: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official SwiftVR model directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination LightX2V model directory")
    args = parser.parse_args()
    convert_swiftvr(args.source, args.output)


if __name__ == "__main__":
    main()
