"""Provision GenEval's non-code vocabulary from its pinned source revision."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def prepare(root, check_only=False):
    source = json.loads((HERE / "configs/nvidia/source-lock.json").read_text())["geneval"]
    assets = json.loads((HERE / "configs/common/evaluator-assets.json").read_text())["files"]
    verified = []
    for entry in assets:
        destination = root / entry["path"]
        if not destination.exists() and not check_only:
            url = f"https://raw.githubusercontent.com/{source['repo']}/{source['revision']}/{entry['path']}"
            content = urllib.request.urlopen(url, timeout=30).read()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError("Downloaded evaluator vocabulary differs from its lock")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Missing or modified evaluator vocabulary: {destination}")
        verified.append(entry)
    return {"source": source["repo"], "revision": source["revision"], "verified": verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = prepare(args.source_root.resolve(), args.check_only)
    content = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
