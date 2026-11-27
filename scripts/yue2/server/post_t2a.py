"""Submit a prepared song, wait for completion and download the final audio."""

import argparse
import json
import time
from pathlib import Path

import requests
from sample_utils import read_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function-id", type=int, choices=range(1, 14), required=True)
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[3] / "assets" / "inputs" / "yue2"))
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    # This deployment fixes one task at startup. Its HTTP schema omits the
    # discriminator; the worker supplies t2a from the runner configuration.
    payload = read_sample(args.data_dir, args.function_id)
    response = requests.post(base + "/v1/tasks/", json=payload, timeout=60)
    if not response.ok:
        raise RuntimeError(f"Submission failed ({response.status_code}): {response.text}")
    task = response.json()["task_id"]
    print(json.dumps({"task_id": task, "submitted": True}), flush=True)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        response = requests.get(f"{base}/v1/tasks/{task}/status", timeout=30)
        response.raise_for_status()
        state = response.json()
        if state["status"] == "completed":
            break
        if state["status"] in ("failed", "cancelled"):
            raise RuntimeError(json.dumps(state, ensure_ascii=False))
        time.sleep(1)
    else:
        raise TimeoutError(f"Task {task} has not completed; query its status before resubmitting")
    response = requests.get(f"{base}/v1/tasks/{task}/results", timeout=30)
    response.raise_for_status()
    audio = next(item for item in response.json()["results"] if item["type"] == "audio")
    response = requests.get(base + audio["url"], timeout=120)
    response.raise_for_status()
    path = Path(args.output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    print(json.dumps({"task_id": task, "audio": str(path), "bytes": len(response.content)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
