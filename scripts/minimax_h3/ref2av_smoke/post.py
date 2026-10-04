#!/usr/bin/env python3
"""Serial, restart-safe HTTP caller for the local MiniMax-H3 smoke manifest.

Only ``smoke.references`` condition generation. Target videos are validation /
comparison artifacts, never request inputs. No model libraries are imported.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import error, parse, request

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "save_results/ref2av_smoke20/samples.jsonl"
TERMINAL = {"completed", "failed", "cancelled"}


class ClientError(RuntimeError):
    pass


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_digest(payload):
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def stable_task_id(manifest_path, sample_id):
    identity = f"{Path(manifest_path).resolve()}\n{sample_id}"
    return "h3-ref2av-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def local_file(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ClientError(f"{label} must be an absolute server-local path")
    path = Path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise ClientError(f"{label} is missing, empty, or not a regular file: {path}")
    return path


def probe_media(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        data = json.loads(result.stdout)
        return data["streams"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        raise ClientError(f"Cannot inspect local media {path}: {exc}") from exc


def build_payload(row, manifest_path=DEFAULT_MANIFEST):
    smoke = row["smoke"]
    references = smoke["references"]
    return {
        "task_id": stable_task_id(manifest_path, smoke["id"]),
        "task": "ref2av",
        "prompt": row["enhanced_prompt"],
        "image_path": ",".join(ref["path"] for ref in references if ref["kind"] == "image"),
        "video_path": ",".join(ref["path"] for ref in references if ref["kind"] == "video"),
        "audio_path": ",".join(ref["path"] for ref in references if ref["kind"] == "audio"),
        "seed": smoke["seed"],
        "num_frames": 124,
        "size": [768, 1344],
        "save_result_path": smoke["generated_path"],
    }


def load_manifest(manifest_path, expected_count=20):
    rows = []
    with Path(manifest_path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ClientError(f"Manifest line {number} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ClientError(f"Manifest line {number} must be an object")
            rows.append(row)
    if len(rows) != expected_count:
        raise ClientError(f"Expected {expected_count} samples, found {len(rows)}")

    ids, outputs = set(), set()
    for number, row in enumerate(rows, 1):
        smoke = row.get("smoke")
        if not isinstance(smoke, dict):
            raise ClientError(f"Manifest sample {number} lacks a smoke object")
        sample_id = smoke.get("id")
        if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample_id):
            raise ClientError(f"Invalid sample id at manifest sample {number}: {sample_id!r}")
        if sample_id in ids:
            raise ClientError(f"Duplicate sample id: {sample_id}")
        ids.add(sample_id)
        if not isinstance(row.get("enhanced_prompt"), str) or not row["enhanced_prompt"].strip():
            raise ClientError(f"{sample_id}: enhanced_prompt must be a nonempty string")
        if smoke.get("group") not in {"seko", "omni"} or smoke.get("provider") not in {"h3", "seedance", "omni"}:
            raise ClientError(f"{sample_id}: invalid group/provider")
        if smoke.get("num_frames") != 124 or smoke.get("size") != [768, 1344] or smoke.get("fps") != 24:
            raise ClientError(f"{sample_id}: smoke geometry must be 124 frames, 24 fps, [768, 1344]")
        if type(smoke.get("seed")) is not int or smoke["seed"] < 0:
            raise ClientError(f"{sample_id}: seed must be a non-negative integer")
        target_source = local_file(smoke.get("target_source"), f"{sample_id}: target_source")
        target_copy = local_file(smoke.get("target_copy"), f"{sample_id}: target_copy")
        generated = smoke.get("generated_path")
        if not isinstance(generated, str) or not Path(generated).is_absolute() or Path(generated).suffix.lower() != ".mp4":
            raise ClientError(f"{sample_id}: generated_path must be an absolute .mp4 path")
        output = Path(generated).resolve()
        if output in outputs or output in {target_source.resolve(), target_copy.resolve()}:
            raise ClientError(f"{sample_id}: output path overlaps another output or a target")
        outputs.add(output)

        references = smoke.get("references")
        if not isinstance(references, list) or not 1 <= len(references) <= 12:
            raise ClientError(f"{sample_id}: expected 1 to 12 references")
        kinds = []
        video_audio_count = 0
        for ref in references:
            if not isinstance(ref, dict) or ref.get("kind") not in {"image", "video", "audio"}:
                raise ClientError(f"{sample_id}: invalid reference object")
            kind = ref["kind"]
            kinds.append(kind)
            path = local_file(ref.get("path"), f"{sample_id}: {kind} reference")
            if "," in str(path):
                raise ClientError(f"{sample_id}: comma in reference path is not supported: {path}")
            if path.resolve() in {output, target_source.resolve(), target_copy.resolve()}:
                raise ClientError(f"{sample_id}: reference overlaps a target/output; refusing target conditioning")
            if kind == "video":
                streams = probe_media(path)
                if not any(stream.get("codec_type") == "video" for stream in streams):
                    raise ClientError(f"{sample_id}: video reference has no video stream: {path}")
                video_audio_count += any(stream.get("codec_type") == "audio" for stream in streams)
        if kinds != sorted(kinds, key={"image": 0, "video": 1, "audio": 2}.get):
            raise ClientError(f"{sample_id}: references must be ordered images, videos, audios; refusing silent reorder")
        if kinds.count("image") > 9 or kinds.count("video") > 3 or kinds.count("audio") + video_audio_count > 3:
            raise ClientError(f"{sample_id}: H3 reference count limit exceeded")
        if all(kind == "audio" for kind in kinds):
            raise ClientError(f"{sample_id}: H3 forbids audio-only references")
        if "audio" in kinds and video_audio_count:
            raise ClientError(f"{sample_id}: video soundtrack would shift standalone Audio N labels; choose an unambiguous sample without changing its prompt or media")
    # Catch an output colliding with an input belonging to any other sample.
    for row in rows:
        smoke = row["smoke"]
        inputs = [smoke["target_source"], smoke["target_copy"]] + [ref["path"] for ref in smoke["references"]]
        if any(Path(path).resolve() in outputs for path in inputs):
            raise ClientError(f"{smoke['id']}: an output path overlaps a manifest input")
    return rows


def atomic_write(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def persist(state, state_path, results_path, rows):
    atomic_write(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    results = []
    for row in rows:
        smoke = row["smoke"]
        entry = state["samples"].get(smoke["id"])
        if entry is None:
            continue
        results.append(
            {
                "id": smoke["id"],
                "group": smoke["group"],
                "provider": smoke["provider"],
                "task_id": entry["task_id"],
                "status": entry["status"],
                "error": entry.get("error"),
                "error_type": entry.get("error_type", ""),
                "target_source": smoke["target_source"],
                "target_copy": smoke["target_copy"],
                "generated_path": smoke["generated_path"],
                "save_result_path": entry.get("save_result_path"),
                "payload_sha256": entry["payload_sha256"],
            }
        )
    atomic_write(results_path, "".join(canonical_json(result) + "\n" for result in results))


class HttpClient:
    def __init__(self, base_url, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Server and inputs are local; do not send requests via ambient proxies.
        self.opener = request.build_opener(request.ProxyHandler({}))

    def json_request(self, endpoint, payload=None):
        body = None if payload is None else canonical_json(payload).encode("utf-8")
        req = request.Request(self.base_url + endpoint, data=body, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                result = json.load(response)
        except error.HTTPError:
            raise
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise ClientError(f"HTTP request to {endpoint} failed: {exc}") from exc
        if not isinstance(result, dict):
            raise ClientError(f"Unexpected non-object HTTP response from {endpoint}")
        return result

    def task_status(self, task_id):
        try:
            return self.json_request(f"/v1/tasks/{parse.quote(task_id, safe='')}/status")
        except error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise


def wait_healthy(client, timeout, interval):
    deadline = time.monotonic() + timeout
    last_error = None
    while True:
        try:
            if client.json_request("/health").get("status") == "ok":
                return
            last_error = "health response status was not ok"
        except (OSError, ValueError, ClientError) as exc:
            last_error = str(exc)
        if time.monotonic() >= deadline:
            raise ClientError(f"Service did not become healthy within {timeout}s: {last_error}")
        time.sleep(min(interval, max(0, deadline - time.monotonic())))


def verify_output(row, status):
    expected = Path(row["smoke"]["generated_path"]).resolve()
    reported = status.get("save_result_path")
    if not reported or Path(reported).resolve() != expected:
        raise ClientError(f"Completed task output path mismatch: expected {expected}, got {reported!r}")
    local_file(str(expected), "completed output")
    streams = probe_media(expected)
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1 or not any(stream.get("codec_type") == "audio" for stream in streams):
        raise ClientError(f"Generated MP4 must contain one video stream and audio: {expected}")
    video = videos[0]
    if (video.get("height"), video.get("width")) != (768, 1344):
        raise ClientError(f"Generated output has incorrect dimensions: {expected}")
    rate = video.get("avg_frame_rate", "0/1")
    try:
        numerator, denominator = map(int, rate.split("/"))
        if not denominator or numerator != 24 * denominator:
            raise ValueError(rate)
    except (AttributeError, ValueError):
        raise ClientError(f"Generated output has incorrect frame rate {rate!r}: {expected}")
    if str(video.get("nb_frames")) != "124":
        raise ClientError(f"Generated output does not report 124 frames: {expected}")


def run(args, rows):
    manifest = Path(args.manifest).resolve()
    state_path = manifest.parent / ".post.state.json"
    results_path = manifest.parent / "results.jsonl"
    lock_path = manifest.parent / ".post.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ClientError(f"Another caller holds {lock_path}") from exc
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("version") != 1 or state.get("manifest") != str(manifest) or state.get("base_url") != args.base_url.rstrip("/"):
                raise ClientError("Existing caller state belongs to a different manifest, service, or version")
        else:
            if results_path.exists() and results_path.stat().st_size:
                raise ClientError(f"Refusing to replace existing results without matching state: {results_path}")
            state = {"version": 1, "manifest": str(manifest), "base_url": args.base_url.rstrip("/"), "samples": {}}
        for row in rows:
            payload = build_payload(row, manifest)
            smoke = row["smoke"]
            entry = state["samples"].get(smoke["id"])
            if entry is not None and (entry.get("payload_sha256") != payload_digest(payload) or entry.get("task_id") != payload["task_id"]):
                raise ClientError(f"{smoke['id']}: payload changed since prior submission; refusing resume")
            output = Path(smoke["generated_path"])
            if entry is None and output.exists() and output.stat().st_size:
                raise ClientError(f"{smoke['id']}: refusing to overwrite existing output without matching completed state: {output}")

        if all(state["samples"].get(row["smoke"]["id"], {}).get("status") in TERMINAL for row in rows):
            # Completed local runs remain inspectable/resumable after the service
            # is shut down, and never need to send another generation request.
            for row in rows:
                entry = state["samples"][row["smoke"]["id"]]
                if entry["status"] == "completed":
                    verify_output(row, entry)
            persist(state, state_path, results_path, rows)
            failed = [row["smoke"]["id"] for row in rows if state["samples"][row["smoke"]["id"]]["status"] != "completed"]
            print(canonical_json({"completed": len(rows) - len(failed), "failed": failed, "results": str(results_path), "resumed": True}), flush=True)
            return 1 if failed else 0

        client = HttpClient(args.base_url, args.http_timeout)
        wait_healthy(client, args.health_timeout, args.poll_interval)
        metadata = client.json_request("/v1/service/metadata")
        print(canonical_json({"service_metadata": metadata}), flush=True)
        if metadata.get("model_cls") != "minimax_h3" or metadata.get("nproc_per_node") != 8:
            raise ClientError(f"Expected an 8-worker minimax_h3 service, got {metadata}")
        for row in rows:
            smoke = row["smoke"]
            sample_id = smoke["id"]
            payload = build_payload(row, manifest)
            entry = state["samples"].get(sample_id)
            if entry and entry["status"] in TERMINAL:
                if entry["status"] == "completed":
                    verify_output(row, entry)
                print(canonical_json({"id": sample_id, "status": entry["status"], "resumed": True}), flush=True)
                continue
            # Look up the client-assigned ID before ever attempting submission.
            status = client.task_status(payload["task_id"])
            if entry is None:
                if status is not None:
                    raise ClientError(f"{sample_id}: task ID already exists on server without matching local payload state; refusing to adopt an unverified task")
                entry = {
                    "task_id": payload["task_id"],
                    "payload_sha256": payload_digest(payload),
                    "status": "prepared",
                    "error": None,
                }
                state["samples"][sample_id] = entry
                persist(state, state_path, results_path, rows)
            elif status is None and entry["status"] != "prepared":
                raise ClientError(f"{sample_id}: task {entry['task_id']} is absent from service after a prior submission attempt; the server may have restarted. Refusing automatic resubmission.")
            if status is None:
                output = Path(smoke["generated_path"])
                if output.exists() and output.stat().st_size:
                    raise ClientError(f"{sample_id}: refusing to submit over an existing nonempty output: {output}")
                # Persist ambiguity BEFORE the HTTP write. A timeout / process death
                # can never cause an unattended duplicate submission on restart.
                entry["status"] = "submission_uncertain"
                persist(state, state_path, results_path, rows)
                try:
                    submitted = client.json_request("/v1/tasks/video/", payload)
                    if submitted.get("task_id") != payload["task_id"]:
                        raise ClientError("Server returned a different task_id")
                    entry["status"] = submitted.get("task_status", "pending")
                    persist(state, state_path, results_path, rows)
                except (OSError, ValueError, ClientError) as exc:
                    entry["error"] = f"Submission result uncertain: {exc}"
                    persist(state, state_path, results_path, rows)
                    try:
                        status = client.task_status(payload["task_id"])
                    except (OSError, ValueError, ClientError):
                        status = None
                    if status is None:
                        raise ClientError(f"{sample_id}: {entry['error']}; refusing automatic resubmission") from exc
            deadline = time.monotonic() + args.task_timeout
            previous = None
            while True:
                try:
                    status = status or client.task_status(payload["task_id"])
                except (OSError, ValueError, ClientError) as exc:
                    if time.monotonic() >= deadline:
                        raise ClientError(f"{sample_id}: polling timed out: {exc}") from exc
                    time.sleep(args.poll_interval)
                    continue
                if status is None:
                    raise ClientError(f"{sample_id}: accepted task vanished; refusing automatic resubmission")
                if status.get("task_id") != payload["task_id"]:
                    raise ClientError(f"{sample_id}: task status response has a mismatched task_id")
                current = status.get("status")
                if current not in {"pending", "processing"} | TERMINAL:
                    raise ClientError(f"{sample_id}: unknown task status {current!r}")
                if current == "completed":
                    try:
                        verify_output(row, status)
                    except ClientError as exc:
                        status = dict(status, status="failed", error=f"Local output validation failed: {exc}", error_type="OutputValidationError")
                        current = "failed"
                entry.update({key: status.get(key) for key in ("status", "error", "error_type", "save_result_path")})
                persist(state, state_path, results_path, rows)
                if current != previous:
                    print(canonical_json({"id": sample_id, "task_id": entry["task_id"], "status": current, "error": entry.get("error")}), flush=True)
                    previous = current
                if current in TERMINAL:
                    break
                if time.monotonic() >= deadline:
                    raise ClientError(f"{sample_id}: task polling timed out; state preserved for resume")
                status = None
                time.sleep(args.poll_interval)
        persist(state, state_path, results_path, rows)
        failed = [row["smoke"]["id"] for row in rows if state["samples"][row["smoke"]["id"]]["status"] != "completed"]
        print(canonical_json({"completed": len(rows) - len(failed), "failed": failed, "results": str(results_path)}), flush=True)
        return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dry-run", action="store_true", help="Validate local files and print payload summaries without HTTP or writes")
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument("--health-timeout", type=float, default=3600)
    parser.add_argument("--task-timeout", type=float, default=7200)
    parser.add_argument("--http-timeout", type=float, default=30)
    parser.add_argument("--poll-interval", type=float, default=5)
    args = parser.parse_args(argv)
    if args.expected_count < 1 or any(not math.isfinite(value) or value <= 0 for value in (args.health_timeout, args.task_timeout, args.http_timeout, args.poll_interval)):
        parser.error("Counts and timeout/interval arguments must be positive")
    url = parse.urlsplit(args.base_url)
    if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
        parser.error("--base-url must be an HTTP(S) service URL without query or fragment")
    try:
        rows = load_manifest(args.manifest, args.expected_count)
        if args.dry_run:
            for row in rows:
                payload = build_payload(row, args.manifest)
                print(
                    canonical_json(
                        {
                            "id": row["smoke"]["id"],
                            "task_id": payload["task_id"],
                            "task": payload["task"],
                            "references": {kind: sum(ref["kind"] == kind for ref in row["smoke"]["references"]) for kind in ("image", "video", "audio")},
                            "seed": payload["seed"],
                            "num_frames": payload["num_frames"],
                            "size": payload["size"],
                            "save_result_path": payload["save_result_path"],
                            "payload_sha256": payload_digest(payload),
                        }
                    )
                )
            return 0
        return run(args, rows)
    except (ClientError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
