"""Select prepared inputs and startup arguments for the numbered Seed-VC examples."""

import argparse
import sys
from pathlib import Path

PRESETS = {
    1: ("svc", []),
    2: ("svc", ["--semitone_shift", "2"]),
    3: ("svc", ["--auto_f0_adjust"]),
    4: ("svc", ["--length_adjust", "1.2"]),
    5: ("vc", []),
    6: ("svc_lora", []),
}


def build_sample_args(model_path, data_dir, function_id):
    variant, options = PRESETS[function_id]
    model_variant = variant.removesuffix("_lora")
    return [
        "--model_path",
        str(Path(model_path).expanduser().resolve() / model_variant),
        "--model-variant",
        variant,
        "--sample_path",
        str(Path(data_dir).expanduser().resolve() / f"{variant}.json"),
        *options,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Weight root containing svc/ and vc/.")
    parser.add_argument("data_dir", help="Shared prepared sample directory.")
    parser.add_argument("function_id", type=int, choices=PRESETS)
    args = parser.parse_args()
    values = build_sample_args(args.model_path, args.data_dir, args.function_id)
    sys.stdout.buffer.write(("\0".join(values) + "\0").encode())


if __name__ == "__main__":
    main()
