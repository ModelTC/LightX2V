"""Export lightweight dummy-meta safetensors files from full model safetensors.

The output files contain ONLY tensor metadata (key, shape, dtype) in the
``__metadata__`` JSON header, with zero tensor data.  They are typically a few
KB regardless of the original model size, and can be used as a drop-in
replacement when ``dummy_model: true`` is set in the config.

Usage examples
--------------
# Export a single file (output next to input with _dummy_meta suffix):
python tools/convert/export_dummy_meta.py /data/model/model.safetensors

# Export a single file to a specific output path:
python tools/convert/export_dummy_meta.py /data/model/model.safetensors -o /data/dummy/model.safetensors

# Export all *.safetensors in a directory (output to a separate directory):
python tools/convert/export_dummy_meta.py /data/model/ -o /data/model_dummy_meta/
"""

import argparse
import glob
import json
import os
import struct
import sys


def read_tensor_metadata(file_path: str) -> dict:
    """Read tensor metadata from a full safetensors file header."""
    with open(file_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
    header = json.loads(header_json)

    tensor_meta = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        tensor_meta[key] = {"shape": info["shape"], "dtype": info["dtype"]}
    return tensor_meta


def write_dummy_meta_safetensors(tensor_meta: dict, output_path: str, source_filename: str = ""):
    """Write a lightweight safetensors file that stores only tensor metadata."""
    header = {
        "__metadata__": {
            "_is_dummy_meta": "true",
            "_tensor_meta": json.dumps(tensor_meta, separators=(",", ":")),
            "_source_file": source_filename,
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)


def export_single_file(input_path: str, output_path: str):
    """Export one safetensors file to its dummy-meta counterpart."""
    tensor_meta = read_tensor_metadata(input_path)
    write_dummy_meta_safetensors(tensor_meta, output_path, source_filename=os.path.basename(input_path))

    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    print(f"  {os.path.basename(input_path)}: {input_size / 1024 / 1024:.1f} MB -> {output_size / 1024:.1f} KB  ({len(tensor_meta)} tensors)")


def main():
    parser = argparse.ArgumentParser(description="Export lightweight dummy-meta safetensors files from full model safetensors.")
    parser.add_argument(
        "input",
        help="Path to a safetensors file or a directory containing *.safetensors files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=("Output path. For single-file input: output file path (default: input_dummy_meta.safetensors next to input). For directory input: output directory (default: {input}_dummy_meta/)."),
    )
    args = parser.parse_args()

    if os.path.isdir(args.input):
        safetensors_files = sorted(glob.glob(os.path.join(args.input, "*.safetensors")))
        if not safetensors_files:
            print(f"No *.safetensors files found in {args.input}", file=sys.stderr)
            sys.exit(1)

        output_dir = args.output or (args.input.rstrip("/") + "_dummy_meta")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Exporting {len(safetensors_files)} files from {args.input} -> {output_dir}")

        for sf in safetensors_files:
            out_path = os.path.join(output_dir, os.path.basename(sf))
            export_single_file(sf, out_path)
    else:
        if not args.input.endswith(".safetensors"):
            print(f"Input file must be a .safetensors file: {args.input}", file=sys.stderr)
            sys.exit(1)

        if args.output:
            output_path = args.output
        else:
            stem = args.input.rsplit(".safetensors", 1)[0]
            output_path = stem + "_dummy_meta.safetensors"

        print(f"Exporting {args.input} -> {output_path}")
        export_single_file(args.input, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
