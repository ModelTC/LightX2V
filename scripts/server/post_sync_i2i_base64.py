import argparse
import base64
from pathlib import Path

import requests


def image_file_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Call /v1/tasks/image/sync with base64 image inputs.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Server base url")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt text")
    parser.add_argument("--infer_steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--aspect_ratio", type=str, default="16:9", help="Aspect ratio for image task")
    parser.add_argument("--timeout_seconds", type=int, default=600, help="Sync API timeout_seconds")
    parser.add_argument("--poll_interval_seconds", type=float, default=0.5, help="Sync API poll_interval_seconds")
    parser.add_argument("--save_result_path", type=str, default="", help="Server-side save_result_path")
    parser.add_argument("--output", type=str, default="sync_result.png", help="Local output image path")

    # Base64 inputs (preferred)
    parser.add_argument("--image_base64", type=str, default="", help="Base64 content for image_path")
    parser.add_argument("--image_mask_base64", type=str, default="", help="Base64 content for image_mask_path")

    # Optional local files to convert to base64
    parser.add_argument("--image_path", type=str, default="", help="Local image file path; encoded if image_base64 is empty")
    parser.add_argument("--image_mask_path", type=str, default="", help="Local mask file path; encoded if image_mask_base64 is empty")

    args = parser.parse_args()

    image_base64 = args.image_base64
    if not image_base64:
        if not args.image_path:
            raise ValueError("Either --image_base64 or --image_path must be provided")
        image_base64 = image_file_to_base64(args.image_path)

    image_mask_base64 = args.image_mask_base64
    if not image_mask_base64 and args.image_mask_path:
        image_mask_base64 = image_file_to_base64(args.image_mask_path)

    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "image_path": image_base64,
        "infer_steps": args.infer_steps,
        "seed": args.seed,
        "aspect_ratio": args.aspect_ratio,
        "save_result_path": args.save_result_path,
    }
    if image_mask_base64:
        payload["image_mask_path"] = image_mask_base64

    endpoint = f"{args.url.rstrip('/')}/v1/tasks/image/sync?timeout_seconds={args.timeout_seconds}&poll_interval_seconds={args.poll_interval_seconds}"
    resp = requests.post(endpoint, json=payload, timeout=args.timeout_seconds + 30)

    if resp.status_code != 200:
        raise RuntimeError(f"Request failed ({resp.status_code}): {resp.text}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(resp.content)
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
