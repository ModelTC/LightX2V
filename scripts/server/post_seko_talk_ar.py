import argparse
import time
from pathlib import Path

import requests
from loguru import logger

DEFAULT_PROMPT = (
    "In a high-fidelity realistic lifestyle aesthetic, a young woman is captured "
    "lounging comfortably on a plush beige sectional sofa within a bright, minimalist "
    "interior. She is actively speaking with natural facial animation and subtle hand "
    "gestures. The camera is fixed in a static medium shot."
)

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, pixelated, low resolution, noise, artifacts, poor lighting, "
    "overexposed, underexposed, distorted, unnatural, deformed, watermark, logo, text, "
    "bad hands, malformed hands, missing fingers, static"
)


def submit_task(args) -> str:
    url = f"{args.url.rstrip('/')}/v1/tasks/video/"
    message = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "image_path": args.image_path,
        "audio_path": args.audio_path,
        "save_result_path": args.save_result_path,
        "seed": args.seed,
        "infer_steps": args.infer_steps,
        "video_duration": args.video_duration,
        "target_fps": args.target_fps,
        "resize_mode": args.resize_mode,
    }
    if args.target_shape:
        message["target_shape"] = args.target_shape

    logger.info(f"submit url: {url}")
    logger.info(f"message: {message}")
    response = requests.post(url, json=message, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"submit failed ({response.status_code}): {response.text}")

    data = response.json()
    logger.info(f"submit response: {data}")
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"submit response has no task_id: {data}")
    return task_id


def wait_task_done(base_url: str, task_id: str, timeout_seconds: int, poll_interval: float) -> dict:
    status_url = f"{base_url.rstrip('/')}/v1/tasks/{task_id}/status"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(status_url, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"status failed ({response.status_code}): {response.text}")
        status = response.json()
        task_status = status.get("status")
        logger.info(f"task_id={task_id}, status={task_status}")
        if task_status == "completed":
            return status
        if task_status in ("failed", "cancelled"):
            raise RuntimeError(f"task ended with status={task_status}, detail={status}")
        time.sleep(poll_interval)
    raise TimeoutError(f"task {task_id} timeout after {timeout_seconds}s")


def download_result(base_url: str, task_id: str, output: str) -> Path:
    result_url = f"{base_url.rstrip('/')}/v1/tasks/{task_id}/result"
    response = requests.get(result_url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"download failed ({response.status_code}): {response.text}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Submit a Seko Talk AR rs2v task to LightX2V server.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--image_path", type=str, default="/data/nvme4/gushiqiao/new/example/1_素材图.png", help="Reference image path, URL, or base64")
    parser.add_argument("--audio_path", type=str, default="/data/nvme4/gushiqiao/new/example/1_素材图.mp3", help="Driving audio path, URL, or base64")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--save_result_path", type=str, default="seko_talk_ar_server_test.mp4", help="Server-side output filename/path")
    parser.add_argument("--output", type=str, default="save_results/seko_talk_ar_server_test.mp4", help="Downloaded result path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--infer_steps", type=int, default=4)
    parser.add_argument("--video_duration", type=int, default=5)
    parser.add_argument("--target_fps", type=int, default=16)
    parser.add_argument("--resize_mode", type=str, default="fixed_shape")
    parser.add_argument("--target_shape", type=int, nargs=2, default=None, help="Optional target shape: H W")
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    parser.add_argument("--poll_interval", type=float, default=2.0)
    parser.add_argument("--no_download", action="store_true", help="Only submit and poll status")
    return parser.parse_args()


def main():
    args = parse_args()
    task_id = submit_task(args)
    final_status = wait_task_done(args.url, task_id, args.timeout_seconds, args.poll_interval)
    logger.info(f"final status: {final_status}")
    if not args.no_download:
        output_path = download_result(args.url, task_id, args.output)
        logger.info(f"result saved to: {output_path}")


if __name__ == "__main__":
    main()
