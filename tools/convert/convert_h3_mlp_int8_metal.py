"""Backward-compatible MLP-only entry point for the H3 INT8 converter."""

import argparse
import json
from pathlib import Path

from tools.convert.convert_h3_int8_metal import convert as convert_profile
from tools.convert.convert_h3_int8_metal import file_hash, quantize_weight  # noqa: F401


def convert(source, output, device="cpu"):
    return convert_profile(source, output, device, scope="mlp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, args.device), indent=2))
