#!/usr/bin/env python3
"""Download immutable model snapshots directly under models (no HF cache).

Run from any directory. Defaults resolve relative to this checkout. Downloads
resume .incomplete files and verify repository hashes before atomic publication.
The JSON manifest contains only repository-relative paths, sizes and hashes.
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import quote

REPOSITORIES = {
    "qwen": ("Qwen/Qwen-Image-2.1-Turbo", "Qwen-Image-2.1-Turbo", "fc13ba1986fcb905854dc8de2f52eee395f49b5b"),
    "unifiedreward": ("CodeGoat24/UnifiedReward-Edit-qwen35-9b", "rewards/UnifiedReward-Edit-qwen35-9b", "31acd924c34fa3afdd31e787803a589a0f40c384"),
}
PRINT_LOCK = threading.Lock()
CURL_RESOLVE = []
RANGE_CHUNK_SIZE = 64 * 1024 * 1024


def log(message):
    with PRINT_LOCK:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def run_transfer(command):
    # Handle curl 18 (truncated response) too; --retry-all-errors needs newer curl.
    for attempt in range(6):
        result = subprocess.run(command)
        if result.returncode == 0:
            return
        if result.returncode == 2 or attempt == 5:
            raise subprocess.CalledProcessError(result.returncode, command)
        time.sleep(min(attempt + 1, 5))


def curl(url, output, resume=False):
    command = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "8",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "20",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "--silent",
        "--show-error",
        "--output",
        str(output),
    ]
    if resume:
        command += ["--continue-at", "-"]
    for value in CURL_RESOLVE:
        command += ["--resolve", value]
    run_transfer(command + [url])


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    if entry.get("lfs"):
        digest = hashlib.sha256()
        expected = entry["lfs"]["sha256"]
    else:
        digest = hashlib.sha1()
        digest.update(("blob %d\0" % entry["size"]).encode())
        expected = entry["blobId"]
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def write_manifest(path, metadata):
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(str(partial), str(path))


def ranged_download(url, partial, size, connections):
    """Resume independent byte ranges, retaining already downloaded prefixes."""
    chunk_size = RANGE_CHUNK_SIZE
    state_path = partial.with_name(partial.name + ".ranges.json")
    count = (size + chunk_size - 1) // chunk_size
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["size"] != size or state["chunk_size"] != chunk_size:
            raise ValueError("Incompatible partial range state")
    else:
        prefix = partial.stat().st_size if partial.exists() else 0
        state = {"size": size, "chunk_size": chunk_size, "completed": list(range(min(prefix // chunk_size, count)))}
        write_manifest(state_path, state)
    with partial.open("ab") as stream:
        stream.truncate(size)
    lock = threading.Lock()
    done = set(state["completed"])

    def fetch_chunk(index):
        start = index * chunk_size
        end = min(start + chunk_size, size) - 1
        segment = partial.with_name(partial.name + ".part%d" % index)
        headers = segment.with_name(segment.name + ".headers")
        command = [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "4",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "15",
            "--speed-time",
            "60",
            "--speed-limit",
            "1024",
            "--silent",
            "--show-error",
            "--range",
            "%d-%d" % (start, end),
            "--dump-header",
            str(headers),
            "--output",
            str(segment),
        ]
        for value in CURL_RESOLVE:
            command += ["--resolve", value]
        run_transfer(command + [url])
        expected_range = "content-range: bytes %d-%d/%d" % (start, end, size)
        if not segment.is_file() or segment.stat().st_size != end - start + 1:
            raise RuntimeError("Incorrect byte-range length: %s" % segment.name)
        if expected_range not in headers.read_text().lower():
            raise RuntimeError("Server did not honor the requested byte range")
        with partial.open("r+b") as output, segment.open("rb") as source:
            output.seek(start)
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                output.write(chunk)
            output.flush()
        with lock:
            done.add(index)
            state["completed"] = sorted(done)
            write_manifest(state_path, state)
            if len(done) % 8 == 0 or len(done) == count:
                log("range progress %s: %d/%d (%d bytes)" % (partial.name, len(done), count, sum(min(chunk_size, size - i * chunk_size) for i in done)))
        segment.unlink()
        headers.unlink()

    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as executor:
        list(executor.map(fetch_chunk, (i for i in range(count) if i not in done)))


def snapshot(key, models_root, endpoint, workers, connections):
    repo, relative, revision = REPOSITORIES[key]
    target = models_root / relative
    target.mkdir(parents=True, exist_ok=True)
    lock_stream = (target / ".download.lock").open("a")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another downloader is active for %s" % repo) from exc
    metadata_path = target / "download-manifest.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["repository"] != repo or metadata["revision"] != revision:
            raise ValueError("Existing manifest belongs to a different repository/revision")
    else:
        api_file = target / ".repository-info.json.incomplete"
        curl(endpoint + "/api/models/" + repo + "/revision/" + revision + "?blobs=true", api_file)
        info = json.loads(api_file.read_text())
        if info["sha"] != revision:
            raise ValueError("Repository API returned an unexpected revision")
        metadata = {"version": 1, "repository": repo, "revision": info["sha"], "complete": False, "files": info["siblings"]}
        if not metadata["files"] or any("size" not in item for item in metadata["files"]):
            raise ValueError("Repository API did not return complete file metadata")
        write_manifest(metadata_path, metadata)
        api_file.unlink()
    metadata["complete"] = False
    write_manifest(metadata_path, metadata)
    total = sum(item["size"] for item in metadata["files"])
    log("%s revision=%s files=%d bytes=%d" % (repo, metadata["revision"], len(metadata["files"]), total))

    def fetch(entry):
        relative_path = Path(entry["rfilename"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Invalid repository path")
        path = target / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if verified(path, entry):
            log("verified existing %s/%s" % (relative, relative_path))
            return
        partial = path.with_name(path.name + ".incomplete")
        if partial.is_file() and partial.stat().st_size >= entry["size"] and not partial.with_name(partial.name + ".ranges.json").exists():
            if verified(partial, entry):
                os.replace(str(partial), str(path))
                log("verified completed partial %s/%s" % (relative, relative_path))
                return
            partial.unlink()  # Discard only downloader-owned, corrupt temporary data.
        url = endpoint + "/" + repo + "/resolve/" + metadata["revision"] + "/" + quote(str(relative_path), safe="/")
        log("downloading %s/%s (%d bytes)" % (relative, relative_path, entry["size"]))
        if connections > 1 and entry["size"] >= 256 * 1024 * 1024:
            ranged_download(url, partial, entry["size"], connections)
        else:
            curl(url, partial, resume=True)
        if not verified(partial, entry):
            raise RuntimeError("Downloaded file failed size/hash verification: %s" % relative_path)
        os.replace(str(partial), str(path))
        range_state = partial.with_name(partial.name + ".ranges.json")
        if range_state.exists():
            range_state.unlink()
        log("verified %s/%s" % (relative, relative_path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(fetch, metadata["files"]))
    metadata["complete"] = True
    write_manifest(metadata_path, metadata)
    log("COMPLETE %s (%d verified bytes)" % (repo, total))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path(__file__).resolve().parents[2] / "models")
    parser.add_argument("--models", nargs="+", choices=tuple(REPOSITORIES), default=list(REPOSITORIES))
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--connections", type=int, default=1, help="Parallel ranges per large file")
    parser.add_argument("--resolve", action="append", default=[], help="Optional curl HOST:PORT:IP DNS override")
    args = parser.parse_args()
    CURL_RESOLVE.extend(args.resolve)
    if args.workers < 1 or args.connections < 1:
        parser.error("--workers must be positive")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.models)) as executor:
        futures = [executor.submit(snapshot, key, args.models_root, args.endpoint.rstrip("/"), args.workers, args.connections) for key in args.models]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
