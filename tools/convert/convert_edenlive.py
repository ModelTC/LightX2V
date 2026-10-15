"""Convert EdenLive DiT keys with the existing Wan mapping, preserving widened inputs."""

import argparse
import json
import re
from pathlib import Path

import torch
from converter import get_key_mapping_rules
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rules = [(re.compile(pattern), replacement) for pattern, replacement in get_key_mapping_rules("backward", "wan_dit")]
    weights = {}
    for shard in sorted(source.glob("*.safetensors")):
        for key, tensor in load_file(str(shard)).items():
            for pattern, replacement in rules:
                key = pattern.sub(replacement, key)
            if key in weights:
                raise ValueError(f"Duplicate converted weight: {key}")
            weights[key] = tensor.to(torch.bfloat16)
    # Do not let an empty source become an apparently valid checkpoint.
    patch = weights["patch_embedding.weight"]
    config = json.loads((source / "config.json").read_text())
    if patch.shape[1] != config["in_channels"] + 1024:
        raise ValueError("EdenLive checkpoint must include the trained 1024 proxy channels")
    destination = output / "diffusion_pytorch_model.safetensors"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(destination)
    save_file(weights, str(destination))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
