#!/usr/bin/env python3
"""Submit one node's deterministic shard to its local 8-GPU service."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import artifact_path, atomic_json, atomic_jsonl, digest, file_digest, file_identity, records


def api(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(url, timeout=1800, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}; inspect server log")
        try:
            if api(url + "/health").get("status") == "ok":
                return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError("Local service did not become ready")


def model_identity(model_path, config):
    root = Path(model_path).resolve()
    transformer = Path(config.get("dit_original_ckpt") or root / "transformer_ref").resolve()
    files = sorted(transformer.glob("*.safetensors")) if transformer.is_dir() else [transformer]
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"Merged ref transformer weights not found: {transformer}")
    components = [file_identity(path) for path in files]
    # Detect replacing encoders/VAEs or their configs as well as transformer shards.
    for component in ("text_encoder", "vae", "audio_vae", "tokenizer", "processor", "transformer_ref"):
        directory = root / component
        for path in sorted(directory.rglob("*")) if directory.is_dir() else []:
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".pt", ".model"}:
                components.append(file_identity(path))
    merged_manifests = [{**file_identity(path), "sha256": file_digest(path)} for path in (root / "decfg_merged.json", transformer / "decfg_merged.json") if path.is_file()]
    return {"model_path": str(root), "transformer_path": str(transformer), "files": components, "decfg_manifests": merged_manifests}


def request_for(row, export_dir, latents_only=True):
    return {
        "task": "ref2av",
        "prompt": row["prompt"],
        "image_path": ",".join(row["reference_images"]),
        "size": [row["target_height"], row["target_width"]],
        "num_frames": 124,
        "seed": row["seed"],
        "teacher_latents_dir": str(export_dir),
        "teacher_latents_only": latents_only,
        "teacher_sample_id": row["source_id"],
    }


def request_fingerprint(row, config, model, selection_sha256):
    return digest(
        {"schema": "h3_teacher_request_v1", "sample": row, "config": config, "model": model, "selection_sha256": selection_sha256, "images": [file_identity(path) for path in row["reference_images"]]}
    )


def validate_export(export_dir, row):
    export_dir = Path(export_dir)
    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema": "h3_teacher_trajectory_v1",
        "task": "ref2av",
        "dtype": "float32",
        "normalized": True,
        "num_steps": 29,
        "num_states": 30,
        "num_frames": 124,
        "height": row["target_height"],
        "width": row["target_width"],
        "fps": 24,
        "seed": row["seed"],
        "prompt": row["prompt"],
        "source_id": row["source_id"],
        "image_paths": row["reference_images"],
        "scheduler_type": "reference_blend",
        "model_variant": "ref2av",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Export {manifest_path} has wrong {key}: {manifest.get(key)!r}")
    if not manifest.get("cache_fingerprint"):
        raise ValueError("Export is missing runtime cache_fingerprint")
    states = manifest.get("state_paths", [])
    if len(states) != 30 or [Path(path).name for path in states] != [f"state_{index:03d}.pt" for index in range(30)]:
        raise ValueError("Export must contain ordered state_000.pt through state_029.pt")
    for value in states + [manifest[key] for key in ("condition_path", "video_latents_path", "audio_latents_path")]:
        path = artifact_path(export_dir, value)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty export artifact: {path}")
    return manifest


def completed(export_dir, row, fingerprint):
    try:
        receipt = json.loads((Path(export_dir) / "completion.json").read_text())
        if receipt.get("request_fingerprint") != fingerprint or receipt.get("manifest_sha256") != file_digest(Path(export_dir) / "manifest.json"):
            return None
        manifest = validate_export(export_dir, row)
        return manifest if receipt.get("cache_fingerprint") == manifest["cache_fingerprint"] else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def index_row(row, export_dir, manifest, fingerprint, selection_hash):
    result = {
        **row,
        "export_dir": str(export_dir),
        "manifest_path": str(export_dir / "manifest.json"),
        "request_fingerprint": fingerprint,
        "selection_sha256": selection_hash,
        "cache_fingerprint": manifest["cache_fingerprint"],
    }
    for key in ("condition_path", "video_latents_path", "audio_latents_path"):
        result[key] = str(artifact_path(export_dir, manifest[key]))
    return result


def collect(output_dir, node_index, config_path, model_path, url="http://127.0.0.1:8000", task_timeout=7200):
    root = Path(output_dir).resolve()
    selection_dir = root / "selection" / f"node{node_index}"
    selection = json.loads((selection_dir / "manifest.json").read_text())
    selection_hash = selection["selection_sha256"]
    all_rows = [row for _, row in records(selection_dir / "selected.jsonl")]
    if digest(all_rows) != selection_hash:
        raise ValueError("Selection file does not match its manifest")
    peer = root / "selection" / f"node{1 - node_index}" / "manifest.json"
    if peer.exists() and json.loads(peer.read_text())["selection_sha256"] != selection_hash:
        raise ValueError("Node selections differ")
    rows = [row for row in all_rows if row["node_index"] == node_index]
    config = json.loads(Path(config_path).read_text())
    model = model_identity(model_path, config)
    successes, failures = [], []
    index_path, failed_path = root / f"index.node{node_index}.jsonl", root / f"failures.node{node_index}.jsonl"
    failure_history = root / f"failure_history.node{node_index}.jsonl"
    wait_ready(url)
    (root / f"complete.node{node_index}.json").unlink(missing_ok=True)
    for position, row in enumerate(rows):
        fingerprint = request_fingerprint(row, config, model, selection_hash)
        export_dir = root / "samples" / f"images{row['ref_image_count']}" / row["sample_key"] / fingerprint[:16]
        manifest = completed(export_dir, row, fingerprint)
        try:
            if manifest is None:
                # Preserve incomplete/corrupt output; only a validated matching receipt is resumable.
                if export_dir.exists():
                    export_dir.rename(export_dir.with_name(export_dir.name + f".incomplete.{time.time_ns()}"))
                response = api(url + "/v1/tasks/video/", request_for(row, export_dir))
                task_id = response["task_id"]
                deadline = time.monotonic() + task_timeout
                while True:
                    status = api(url + f"/v1/tasks/{task_id}/status")
                    if status["status"] == "completed":
                        break
                    if status["status"] in {"failed", "cancelled"}:
                        raise RuntimeError(f"Task {task_id}: {status.get('error') or status}")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Task {task_id} exceeded {task_timeout}s; stop this service before retrying")
                    time.sleep(2)
                manifest = validate_export(export_dir, row)
                atomic_json(
                    export_dir / "completion.json",
                    {
                        "schema": "h3_teacher_completion_v1",
                        "request_fingerprint": fingerprint,
                        "cache_fingerprint": manifest["cache_fingerprint"],
                        "manifest_sha256": file_digest(export_dir / "manifest.json"),
                        "source_index": row["source_index"],
                        "source_id": row["source_id"],
                        "selection_sha256": selection_hash,
                    },
                )
            successes.append(index_row(row, export_dir, manifest, fingerprint, selection_hash))
            print(f"node={node_index} {position + 1}/{len(rows)} complete source={row['source_id']}", flush=True)
        except Exception as error:
            failure = {
                "source_index": row["source_index"],
                "source_id": row["source_id"],
                "export_dir": str(export_dir),
                "request_fingerprint": fingerprint,
                "error": str(error),
                "time_ns": time.time_ns(),
            }
            failures.append(failure)
            with failure_history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"FAILED node={node_index} source={row['source_id']}: {error}", flush=True)
            if isinstance(error, (TimeoutError, urllib.error.URLError, ConnectionError)):
                atomic_jsonl(index_path, successes)
                atomic_jsonl(failed_path, failures)
                raise
        atomic_jsonl(index_path, successes)
        atomic_jsonl(failed_path, failures)
    atomic_json(
        root / f"complete.node{node_index}.json",
        {
            "selection_sha256": selection_hash,
            "expected": len(rows),
            "completed": len(successes),
            "failed": len(failures),
            "index_sha256": file_digest(index_path),
            "failures_sha256": file_digest(failed_path),
        },
    )
    if failures:
        raise RuntimeError(f"{len(failures)} samples failed; inspect {failed_path}, rerun to retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    collect(args.output_dir, args.node_index, args.config_json, args.model_path, args.url)


if __name__ == "__main__":
    main()
