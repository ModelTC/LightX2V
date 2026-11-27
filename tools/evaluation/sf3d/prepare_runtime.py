"""Build offline weight overlays without changing native SF3D computation."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.paths.read_text())
    original = Path(paths["weights"]).resolve()
    config = original / "config.upstream.yaml"
    for name in ("lightx2v", "native"):
        destination = args.output.resolve() / name
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "model.safetensors"
        if not target.exists():
            target.symlink_to(original / "model.safetensors")
        (destination / "config.upstream.yaml").write_bytes(config.read_bytes())
        cfg = OmegaConf.load(config)
        if name == "native":
            cfg.image_tokenizer.pretrained_model_name_or_path = paths["dino"]
            cfg.image_estimator.pretrain = str(Path(paths["openclip"]) / "open_clip_pytorch_model.bin")
            OmegaConf.save(cfg, destination / "config.yaml")
        else:
            (destination / "config.yaml").write_bytes(config.read_bytes())
    paths["weights"] = str(args.output.resolve() / "lightx2v")
    paths["native_weights"] = str(args.output.resolve() / "native")
    output = args.output.resolve() / "paths.local.json"
    output.write_text(json.dumps(paths, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
