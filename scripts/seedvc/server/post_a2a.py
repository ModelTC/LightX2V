"""Submit a prepared Seed-VC sample, wait for completion and download the final audio."""

import argparse
import json
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--semitone-shift", type=int, default=0)
    parser.add_argument("--auto-f0-adjust", action="store_true")
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    payload = {"sample_path": args.sample_path, "semitone_shift": args.semitone_shift, "auto_f0_adjust": args.auto_f0_adjust, "diffusion_steps": args.diffusion_steps, "seed": 2026}
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
