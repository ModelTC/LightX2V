"""Make an RGB-only Wan VAE manifest without modifying the source manifest."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def prepare_metadata(source, output, video_root, *, check_files=False):
    source, output, video_root = Path(source), Path(output), Path(video_root)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not video_root.is_absolute():
        raise ValueError("video_root must be an absolute path on the training machine.")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with TemporaryDirectory(prefix=".wan-vae-meta-", dir=output.parent) as directory:
        temporary = Path(directory) / "metadata.jsonl"
        with source.open(encoding="utf-8") as reader, temporary.open("w", encoding="utf-8") as writer:
            for line in reader:
                if not line.strip():
                    continue
                record = json.loads(line)
                relative = Path(record["video"]).relative_to("videos")
                if ".." in relative.parts:
                    raise ValueError(f"Video path escapes videos/: {record['video']}")
                video = video_root / relative
                if check_files and not video.is_file():
                    raise FileNotFoundError(video)
                record["video"] = str(video)
                record.pop("prompt_path", None)
                record.pop("text_path", None)
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        if count == 0:
            raise ValueError("Source manifest has no records.")
        temporary.replace(output)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--check-files", action="store_true", help="Check video existence on the training machine.")
    args = parser.parse_args()
    count = prepare_metadata(args.input, args.output, args.video_root, check_files=args.check_files)
    print(f"Wrote rows={count} checked_files={args.check_files} output={args.output}")


if __name__ == "__main__":
    main()
