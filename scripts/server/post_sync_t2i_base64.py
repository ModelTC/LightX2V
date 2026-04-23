import argparse
from pathlib import Path
from typing import List, Optional

import requests


def build_payload(args: argparse.Namespace) -> dict:
    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "infer_steps": args.infer_steps,
        "seed": args.seed,
        "aspect_ratio": args.aspect_ratio,
        "save_result_path": args.save_result_path,
        "use_prompt_enhancer": args.use_prompt_enhancer,
    }
    if args.target_shape:
        payload["target_shape"] = args.target_shape
    return payload


def request_t2i_sync(
    base_url: str,
    payload: dict,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> bytes:
    endpoint = f"{base_url.rstrip('/')}/v1/tasks/image/sync?timeout_seconds={timeout_seconds}&poll_interval_seconds={poll_interval_seconds}"
    response = requests.post(endpoint, json=payload, timeout=timeout_seconds + 30)
    if response.status_code != 200:
        raise RuntimeError(f"Sync t2i failed ({response.status_code}): {response.text}")
    return response.content


def save_result(content: bytes, output_path: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Call /v1/tasks/image/sync for T2I and save final image.")
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
    parser.add_argument("--timeout_seconds", type=int, default=600, help="Sync API timeout_seconds")
    parser.add_argument("--poll_interval_seconds", type=float, default=0.5, help="Sync API poll_interval_seconds")
    parser.add_argument("--output", type=str, default="save_results/t2i_sync_result.png", help="Local output image path")
    args = parser.parse_args()

    target_shape: Optional[List[int]] = args.target_shape
    if target_shape is not None and len(target_shape) < 2:
        raise ValueError("--target_shape must provide at least 2 integers, e.g. --target_shape 1536 2752")

    payload = build_payload(args)
    content = request_t2i_sync(
        base_url=args.url,
        payload=payload,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    output = save_result(content, args.output)
    print(f"Saved sync t2i result to: {output}")


if __name__ == "__main__":
    main()
