import argparse
import subprocess

import imageio_ffmpeg as ffmpeg


def build_scale_filter(args):
    if args.width is not None and args.height is not None:
        return f"scale={args.width}:{args.height}:flags={args.interp}"
    if args.scale is None:
        raise ValueError("Either --scale or --width/--height must be provided.")
    # Keep dimensions even for video codecs
    return (
        f"scale=trunc(iw*{args.scale}/2)*2:"
        f"trunc(ih*{args.scale}/2)*2:flags={args.interp}"
    )


def main():
    parser = argparse.ArgumentParser(description="Upscale video with simple interpolation.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--scale", type=float, default=None, help="Scale factor (e.g., 2.0)")
    parser.add_argument("--width", type=int, default=None, help="Target width")
    parser.add_argument("--height", type=int, default=None, help="Target height")
    parser.add_argument(
        "--interp",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="bicubic",
        help="Interpolation method",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Copy audio stream if present",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF for libx264 (lower is higher quality)",
    )
    parser.add_argument(
        "--preset",
        default="fast",
        help="x264 preset (e.g., veryfast, fast, medium)",
    )
    args = parser.parse_args()

    scale_filter = build_scale_filter(args)
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        args.input,
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-crf",
        str(args.crf),
        "-preset",
        args.preset,
        "-pix_fmt",
        "yuv420p",
    ]

    if args.keep_audio:
        cmd += ["-map", "0:v:0", "-map", "0:a?", "-c:a", "copy"]
    else:
        cmd += ["-an"]

    cmd.append(args.output)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
