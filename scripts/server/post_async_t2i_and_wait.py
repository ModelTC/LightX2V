import argparse
import time
from pathlib import Path
from typing import List, Optional

import requests


def submit_t2i_task(
    base_url: str,
    prompt: str,
    negative_prompt: str,
    infer_steps: int,
    seed: int,
    aspect_ratio: str,
    target_shape: Optional[List[int]],
    save_result_path: str,
    use_prompt_enhancer: bool,
) -> str:
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "infer_steps": infer_steps,
        "seed": seed,
        "aspect_ratio": aspect_ratio,
        "save_result_path": save_result_path,
        "use_prompt_enhancer": use_prompt_enhancer,
    }
    if target_shape:
        payload["target_shape"] = target_shape

    submit_url = f"{base_url.rstrip('/')}/v1/tasks/image/"
    response = requests.post(submit_url, json=payload, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Submit task failed ({response.status_code}): {response.text}")

    data = response.json()
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Submit task succeeded but no task_id found: {data}")
    return task_id


def wait_task_done(base_url: str, task_id: str, timeout_seconds: int, poll_interval: float) -> dict:
    status_url = f"{base_url.rstrip('/')}/v1/tasks/{task_id}/status"
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        response = requests.get(status_url, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"Get task status failed ({response.status_code}): {response.text}")

        status = response.json()
        task_status = status.get("status")
        print(f"[poll] task_id={task_id}, status={task_status}")

        if task_status == "completed":
            return status
        if task_status in ("failed", "cancelled"):
            raise RuntimeError(f"Task ended with status={task_status}, detail={status.get('error')}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Task {task_id} timeout after {timeout_seconds}s")


def download_result(base_url: str, task_id: str, output: str) -> Path:
    result_url = f"{base_url.rstrip('/')}/v1/tasks/{task_id}/result"
    response = requests.get(result_url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Download result failed ({response.status_code}): {response.text}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Submit T2I task to /v1/tasks/image/ and wait for final result.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Server base url")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt text")
    parser.add_argument("--infer_steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--aspect_ratio", type=str, default="16:9", help="Aspect ratio for image task")
    parser.add_argument(
        "--target_shape",
        type=int,
        nargs="+",
        default=None,
        help="Target output shape, e.g. --target_shape 1536 2752",
    )
    parser.add_argument("--save_result_path", type=str, default="", help="Server-side save_result_path")
    parser.add_argument("--use_prompt_enhancer", action="store_true", help="Enable prompt enhancer")
    parser.add_argument("--timeout_seconds", type=int, default=600, help="Polling timeout in seconds")
    parser.add_argument("--poll_interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--output", type=str, default="save_results/t2i_result.png", help="Local output image path")

    args = parser.parse_args()

    task_id = submit_t2i_task(
        base_url=args.url,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        infer_steps=args.infer_steps,
        seed=args.seed,
        aspect_ratio=args.aspect_ratio,
        target_shape=args.target_shape,
        save_result_path=args.save_result_path,
        use_prompt_enhancer=args.use_prompt_enhancer,
    )
    print(f"Task submitted successfully, task_id={task_id}")

    final_status = wait_task_done(
        base_url=args.url,
        task_id=task_id,
        timeout_seconds=args.timeout_seconds,
        poll_interval=args.poll_interval,
    )
    print(f"Task completed: {final_status}")

    output_path = download_result(args.url, task_id, args.output)
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
