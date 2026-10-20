"""Compile the INT8 Metal kernels for selected (K, N) shapes."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_shape(value):
    try:
        k, n = map(int, value.lower().split("x"))
        if k > 0 and n > 0 and k % 1024 == 0 and n % 128 == 0:
            return k, n
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Expected KxN; K must be a positive multiple of 1024 and N a positive multiple of 128")


def main():
    parser = argparse.ArgumentParser(description="Compile INT8 Metal kernels offline; at least one (K,N) pair is required.")
    parser.add_argument("shapes", nargs="+", type=parse_shape, metavar="KxN", help="Required shapes, e.g. 4096x4096 4096x12288")
    parser.add_argument("--output-dir", type=Path, default=Path("build/metallib"), help="Output directory; defaults to build/metallib under the current directory")
    args = parser.parse_args()

    try:
        subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"], check=True, stdout=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "Metal compiler is unavailable. Install and configure Xcode; if the Metal Toolchain is missing, run:\n  xcodebuild -downloadComponent MetalToolchain",
            file=sys.stderr,
        )
        return 1

    source = (ROOT / "int8_matmul.metal").read_text()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for k, n in args.shapes:
        stem = output / f"int8_k{k}_n{n}"
        metal = stem.with_suffix(".metal")
        air = stem.with_suffix(".air")
        library = stem.with_suffix(".metallib")
        metal.write_text(source.replace("{{K}}", str(k)).replace("{{N}}", str(n)))
        print(f"Compiling K={k}, N={n} ...", flush=True)
        try:
            subprocess.run(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metal",
                    "-std=metal4.0",
                    "-mmacos-version-min=26.4",
                    "-fno-fast-math",
                    "-c",
                    str(metal),
                    "-o",
                    str(air),
                ],
                check=True,
            )
            subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(library)], check=True)
        except subprocess.CalledProcessError as error:
            print(f"Compilation failed (exit code {error.returncode}); see the toolchain errors above.", file=sys.stderr)
            return 1
        print(f"Generated: {library}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
