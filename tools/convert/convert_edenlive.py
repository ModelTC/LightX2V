"""Convert one EdenLive DiT or a complete release to native LightX2V weights."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.convert.key_mapping import get_key_mapping_rules  # noqa: E402


def skip_checkpoint(destination, required_key, overwrite):
    if not destination.exists() or overwrite:
        return False
    from safetensors import safe_open

    with safe_open(str(destination), framework="pt", device="cpu") as handle:
        if required_key not in handle.keys():
            raise ValueError(f"Existing file is not a native checkpoint: {destination}")
    print(f"Keep existing: {destination}", flush=True)
    return True


def save_checkpoint(weights, destination):
    from safetensors.torch import save_file

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    save_file(weights, str(temporary))
    temporary.replace(destination)
    print(f"Converted: {destination}", flush=True)


def copy_tree(source, output, overwrite, exclude=()):
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if relative in exclude:
            continue
        destination = output / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif not destination.exists() or overwrite:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Immutable weights can share storage while the output remains independent of source paths.
            if not destination.exists() and path.suffix in (".safetensors", ".pth", ".bin") and path.stat().st_dev == destination.parent.stat().st_dev:
                os.link(path, destination)
            else:
                temporary = destination.with_name(destination.name + ".tmp")
                shutil.copy2(path, temporary)
                temporary.replace(destination)
    print(f"Copied native assets: {output}", flush=True)


def convert_dit(source, output, overwrite):
    import torch
    from safetensors.torch import load_file

    destination = output / "diffusion_pytorch_model.safetensors"
    if skip_checkpoint(destination, "head.head.weight", overwrite):
        return
    rules = [(re.compile(pattern), replacement) for pattern, replacement in get_key_mapping_rules("backward", "wan_dit")]
    weights = {}
    for shard in sorted(source.glob("*.safetensors")):
        for key, tensor in load_file(str(shard)).items():
            for pattern, replacement in rules:
                key = pattern.sub(replacement, key)
            if key in weights:
                raise ValueError(f"Duplicate converted weight: {key}")
            weights[key] = tensor.to(torch.bfloat16)
    patch = weights["patch_embedding.weight"]
    config = json.loads((source / "config.json").read_text())
    if patch.shape[1] != config["in_channels"] + 1024:
        raise ValueError("EdenLive checkpoint must include the trained 1024 proxy channels")
    save_checkpoint(weights, destination)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def convert_encoder(source, output, kind, overwrite):
    import torch
    from safetensors.torch import load_file

    from tools.convert.edenlive_weights import remap_hf_clip_vision_weights, remap_hf_umt5_weights

    destination = output / "model.safetensors"
    key = "token_embedding.weight" if kind == "t5" else "visual.patch_embedding.weight"
    if skip_checkpoint(destination, key, overwrite):
        return
    state = load_file(str(source / "model.safetensors"))
    config = json.loads((source / "config.json").read_text())
    if kind == "t5":
        weights = remap_hf_umt5_weights(state)
    else:
        weights = {"visual." + name: tensor for name, tensor in remap_hf_clip_vision_weights(state, config["num_hidden_layers"]).items()}
        # The vision-only HF checkpoint has no contrastive text/image temperature.
        weights["log_scale"] = torch.tensor(math.log(1 / 0.07))
    weights = {name: tensor.to(torch.bfloat16).contiguous() for name, tensor in weights.items()}
    save_checkpoint(weights, destination)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def convert_release(source, output, overwrite):
    for name in ("dit-5b", "dit-5b-bidir", "dit-14b", "dit-14b-bidir"):
        convert_dit(source / name, output / name, overwrite)
    convert_encoder(source / "umt5/text_encoder", output / "umt5/text_encoder", "t5", overwrite)
    convert_encoder(source / "clip/image_encoder", output / "clip/image_encoder", "clip", overwrite)
    converted_files = set()
    for name in ("dit-5b", "dit-5b-bidir", "dit-14b", "dit-14b-bidir", "umt5/text_encoder", "clip/image_encoder"):
        converted_files.add(Path(name) / "config.json")
        converted_files.update(path.relative_to(source) for path in (source / name).glob("*.safetensors"))
    # Preserve all other assets, including DINO, VAE, tokenizers and auxiliary model files.
    copy_tree(source, output, overwrite, exclude=converted_files)
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    profiles = REPO_ROOT / "configs/edenlive"
    for profile in sorted(profiles.rglob("*.json")):
        destination = config_dir / profile.relative_to(profiles)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(profile, destination)
    print(f"Ready: model_path={output}, config_json={config_dir}/<profile>.json", flush=True)


def quantize_dit(source, output, linear_type, output_name, device, overwrite):
    from safetensors import safe_open

    checkpoint = source / "diffusion_pytorch_model.safetensors" if source.is_dir() else source
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        if "head.head.weight" not in handle.keys():
            raise ValueError("Quantization requires a converted native EdenLive DiT checkpoint")
    name = output_name or f"{source.stem}-{linear_type}"
    if skip_checkpoint(output / f"{name}.safetensors", "head.head.weight", overwrite):
        return
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/convert/converter.py"),
            "--source",
            str(checkpoint),
            "--output",
            str(output),
            "--output_name",
            name,
            "--model_type",
            "wan_dit",
            "--quantized",
            "--linear_type",
            linear_type,
            "--non_linear_dtype",
            "torch.bfloat16",
            "--ignore-keys",
            "none",
            "--device",
            device,
            "--single_file",
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Convert a complete EdenLive release, including encoders and VAE assets")
    mode.add_argument("--quantize", choices=["fp8", "int8-convrot"], help="Quantize a converted native DiT into one safetensors file")
    parser.add_argument("--output_name", help="Quantized output filename, without .safetensors")
    parser.add_argument("--device", default="cuda", help="Device for quantization, e.g. cuda:0 or cpu")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs; otherwise keep native checkpoints already present")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the source directory")
    output.mkdir(parents=True, exist_ok=True)
    if args.quantize:
        quantize_dit(source, output, args.quantize, args.output_name, args.device, args.overwrite)
    elif args.all:
        convert_release(source, output, args.overwrite)
    else:
        convert_dit(source, output, args.overwrite)


if __name__ == "__main__":
    main()
