import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

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


def format_duration(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    total_seconds, milliseconds = divmod(total_ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def ensure_local_no_proxy(base_url: str) -> None:
    hostname = urlparse(base_url).hostname
    if hostname not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        return

    local_hosts = ["127.0.0.1", "localhost", "0.0.0.0", "::1"]
    for env_name in ("NO_PROXY", "no_proxy"):
        existing = [item.strip() for item in os.environ.get(env_name, "").split(",") if item.strip()]
        merged = local_hosts + [item for item in existing if item not in local_hosts]
        os.environ[env_name] = ",".join(merged)


def load_prompts(prompt_json: str) -> List[str]:
    path = Path(prompt_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")

    prompts = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            prompt = item
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompt = item["prompt"]
        else:
            raise ValueError(f"{path}[{index}] must be a string or an object with a string prompt")
        prompts.append(prompt)
    return prompts


def write_summary_line(summary_file: Optional[Path], line: str) -> None:
    if summary_file is None:
        return
    with summary_file.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def run_one_task(args, prompt: str, save_result_path: str, output: str) -> dict:
    case_start = time.perf_counter()
    task_id = submit_t2i_task(
        base_url=args.url,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        infer_steps=args.infer_steps,
        seed=args.seed,
        aspect_ratio=args.aspect_ratio,
        target_shape=args.target_shape,
        save_result_path=save_result_path,
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

    output_path = download_result(args.url, task_id, output)
    elapsed = time.perf_counter() - case_start
    print(f"Result saved to: {output_path}")
    print(f"Task elapsed: {format_duration(elapsed)} ({elapsed:.3f}s)")

    return {
        "task_id": task_id,
        "status": "success",
        "elapsed": elapsed,
        "output": str(output_path),
        "save_result_path": final_status.get("save_result_path", save_result_path),
    }


def run_batch(args) -> int:
    prompts = load_prompts(args.prompt_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(args.summary_file) if args.summary_file else output_dir / f"{args.output_prefix}_summary_{run_stamp}.log"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(prompts)
    completed = 0
    failed = 0
    batch_start = time.perf_counter()

    summary_file.write_text(
        "\n".join(
            [
                f"Run started at: {started_at}",
                f"Prompt JSON: {args.prompt_json}",
                f"Server URL: {args.url}",
                f"Output directory: {output_dir}",
                f"Total prompts: {total}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Posting {total} prompts from {args.prompt_json} to {args.url}")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_file}")

    for index, prompt in enumerate(prompts, 1):
        number = f"{index:03d}"
        server_save_result_path = f"{args.output_prefix}_{number}.png"
        output = output_dir / server_save_result_path
        case_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        case_start = time.perf_counter()

        print(f"[{index}/{total}] submitting {server_save_result_path}")
        write_summary_line(summary_file, f"Case {number} started at: {case_started_at}")

        try:
            result = run_one_task(args, prompt, server_save_result_path, str(output))
        except Exception as e:
            failed += 1
            elapsed = time.perf_counter() - case_start
            print(f"[{index}/{total}] failed: {server_save_result_path}: {e}")
            write_summary_line(summary_file, f"Case {number} status: failed, elapsed: {format_duration(elapsed)} ({elapsed:.3f}s), error: {e}")
            if args.stop_on_error:
                break
        else:
            completed += 1
            write_summary_line(
                summary_file,
                f"Case {number} status: success, task_id: {result['task_id']}, elapsed: {format_duration(result['elapsed'])} ({result['elapsed']:.3f}s), output: {result['output']}",
            )
            print(f"[{index}/{total}] saved {output}")

    total_elapsed = time.perf_counter() - batch_start
    ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_summary_line(summary_file, "")
    write_summary_line(summary_file, f"Run ended at: {ended_at}")
    write_summary_line(summary_file, f"Elapsed seconds: {total_elapsed:.3f}")
    write_summary_line(summary_file, f"Elapsed time: {format_duration(total_elapsed)}")
    write_summary_line(summary_file, f"Completed prompts: {completed}")
    write_summary_line(summary_file, f"Failed prompts: {failed}")

    print(f"Finished: completed={completed}/{total}, failed={failed}/{total}")
    print(f"Total elapsed: {format_duration(total_elapsed)} ({total_elapsed:.3f}s)")
    print(f"Summary written to: {summary_file}")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="Submit T2I task to /v1/tasks/image/ and wait for final result.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Server base url")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text")
    parser.add_argument("--prompt_json", "--json", dest="prompt_json", type=str, default=None, help="JSON file containing prompts")
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
    parser.add_argument("--output_dir", type=str, default="save_results/hidream_o1_image_test_json", help="Batch output directory")
    parser.add_argument("--output_prefix", type=str, default="hidream_o1_image_test_json", help="Batch output filename prefix")
    parser.add_argument("--summary_file", type=str, default=None, help="Batch timing summary file")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop batch mode after the first failed prompt")

    args = parser.parse_args()
    ensure_local_no_proxy(args.url)

    if args.prompt_json:
        raise SystemExit(run_batch(args))

    if not args.prompt:
        parser.error("--prompt is required unless --prompt_json is set")

    run_one_task(args, args.prompt, args.save_result_path, args.output)


if __name__ == "__main__":
    main()
