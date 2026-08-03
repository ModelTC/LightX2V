#!/usr/bin/env python3
import argparse
import base64
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

TASKS = (
    "understanding",
    "binary_segmentation",
    "depth",
    "normal",
    "gcg_segmentation",
    "object_detection",
    "point_detection",
    "keypoint",
    "ocr",
    "recon3d",
    "panoptic_segmentation",
    "interactive_segmentation",
    "vgd_segmentation",
    "camera_pose",
)


def image_source_for_request(source: str) -> str:
    """Encode an existing client-local file; preserve URL/base64/server paths."""
    if source.startswith(("http://", "https://", "data:image/")):
        return source
    try:
        path = Path(source).expanduser()
        if path.is_file():
            return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        pass
    return source


def raise_for_response(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    raise RuntimeError(f"Request failed ({response.status_code}): {detail}")


def submit_async(base_url: str, payload: dict, timeout: int, poll_interval: float) -> dict:
    response = requests.post(
        f"{base_url}/v1/tasks/sensenova-vision/",
        json=payload,
        timeout=30,
    )
    raise_for_response(response)
    task_id = response.json()["task_id"]
    print(f"Submitted task: {task_id}")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = requests.get(f"{base_url}/v1/tasks/{task_id}/status", timeout=30)
        raise_for_response(response)
        status_data = response.json()
        status = status_data["status"]
        print(f"Task {task_id}: {status}", flush=True)
        if status == "completed":
            response = requests.get(
                f"{base_url}/v1/tasks/sensenova-vision/{task_id}/result",
                timeout=30,
            )
            raise_for_response(response)
            return response.json()
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"SenseNova-Vision task {status}: {status_data}")
        time.sleep(poll_interval)
    raise TimeoutError(f"SenseNova-Vision task {task_id} exceeded {timeout} seconds")


def submit_sync(base_url: str, payload: dict, timeout: int, poll_interval: float) -> dict:
    response = requests.post(
        f"{base_url}/v1/tasks/sensenova-vision/sync",
        params={
            "timeout_seconds": timeout,
            "poll_interval_seconds": poll_interval,
        },
        json=payload,
        timeout=timeout + 30,
    )
    raise_for_response(response)
    return response.json()


def save_result(base_url: str, result: dict, output_root: Path, download: bool) -> Path:
    task_dir = output_root / result["task_id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_dir / "result.json"
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if result.get("text"):
        print(f"SenseNova-Vision text output:\n{result['text']}")
    for warning in result.get("warnings", []):
        print(f"Warning: {warning}")

    if download:
        for artifact in result.get("artifacts", []):
            response = requests.get(urljoin(f"{base_url}/", artifact["url"]), timeout=120)
            raise_for_response(response)
            destination = task_dir / Path(artifact["filename"]).name
            destination.write_bytes(response.content)
            print(f"Saved {artifact['kind']}: {destination}")
    print(f"Saved manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one task to the resident LightX2V SenseNova-Vision service.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Input image; repeat for multi-view tasks. Local files are sent as base64.",
    )
    parser.add_argument("--prompt", default="", help="Task query/instruction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--no-visualize", action="store_true", help="Skip official-style visualization")
    parser.add_argument("--postprocess-3d", action="store_true", help="Also create GLB for recon3d")
    parser.add_argument("--sync", action="store_true", help="Use the blocking /sync endpoint")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="save_results/sensenova_vision_client",
        help="Client-side artifact directory",
    )
    parser.add_argument("--no-download", action="store_true", help="Only save the JSON manifest")
    args = parser.parse_args()

    payload = {
        "task": args.task,
        "prompt": args.prompt,
        "images": [image_source_for_request(source) for source in args.image],
        "seed": args.seed,
        "target_shape": list(args.target_shape or []),
        "visualize": not args.no_visualize,
        "postprocess_3d": args.postprocess_3d,
    }
    base_url = args.url.rstrip("/")
    if args.sync:
        result = submit_sync(base_url, payload, args.timeout_seconds, args.poll_interval_seconds)
    else:
        result = submit_async(base_url, payload, args.timeout_seconds, args.poll_interval_seconds)
    save_result(base_url, result, Path(args.output_dir), not args.no_download)


if __name__ == "__main__":
    main()
