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
    raise argparse.ArgumentTypeError("尺寸应为 KxN；K 必须为 1024 的正整数倍，N 为 128 的正整数倍")


def main():
    parser = argparse.ArgumentParser(description="离线编译 INT8 Metal 算子，必须指定至少一个 (K,N) 组合。")
    parser.add_argument("shapes", nargs="+", type=parse_shape, metavar="KxN", help="必填，例如 4096x4096 4096x12288")
    parser.add_argument("--output-dir", type=Path, default=Path("build/metallib"), help="产物目录，默认当前目录下的 build/metallib")
    args = parser.parse_args()

    try:
        subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"], check=True, stdout=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "Metal 编译器不可用。请先安装并配置 Xcode；缺少 Metal Toolchain 时执行：\n  xcodebuild -downloadComponent MetalToolchain",
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
        print(f"编译 K={k}, N={n} ...", flush=True)
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
            print(f"编译失败（退出码 {error.returncode}），请查看上方工具链错误。", file=sys.stderr)
            return 1
        print(f"生成：{library}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
