"""
Qwen Image I2I 3-way Disagg request script.

Request order (same as Wan 3-way):
  1. Send request to Decoder first (starts Phase2 receiver, blocking)
  2. Send request to Transformer (Phase1 receiver + Phase2 sender)
  3. Send request to Encoder (text + image encoding + Phase1 send)
  4. Poll Decoder for completion (image saved on Decoder node)
"""

import base64
import time

import requests
from loguru import logger

ENCODER_URL = "http://localhost:8012"
TRANSFORMER_URL = "http://localhost:8013"
DECODER_URL = "http://localhost:8014"
ENDPOINT = "/v1/tasks/image/"

IMAGE_PATH = "/path/to/test.png"


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def poll_task(url, task_id, timeout=300, interval=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{url}/v1/tasks/{task_id}/status", timeout=10)
        data = r.json()
        status = data.get("status")
        logger.info(f"Task {task_id} status: {status}")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"Task failed: {data.get('error')}")
        time.sleep(interval)
    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")


if __name__ == "__main__":
    assert IMAGE_PATH, "Set IMAGE_PATH to your input image for I2I."

    payload = {
        "prompt": "Change the person to a standing position, bending over to hold the dog's front paws.",
        "negative_prompt": "",
        "image_path": image_to_base64(IMAGE_PATH),
        "seed": 42,
        "save_result_path": "save_results/qwen_i2i_disagg_3way.png",
        "aspect_ratio": "16:9",
    }

    # Step 1: Send to Decoder first (sets up Phase2 receiver)
    logger.info("Step 1: Sending request to Decoder...")
    resp_d = requests.post(f"{DECODER_URL}{ENDPOINT}", json=payload, timeout=30)
    decoder_task_id = resp_d.json().get("task_id")
    logger.info(f"Decoder task_id: {decoder_task_id}")

    # Step 2: Send to Transformer (Phase1 receiver + Phase2 sender)
    logger.info("Step 2: Sending request to Transformer...")
    resp_t = requests.post(f"{TRANSFORMER_URL}{ENDPOINT}", json=payload, timeout=30)
    transformer_task_id = resp_t.json().get("task_id")
    logger.info(f"Transformer task_id: {transformer_task_id}")

    # Step 3: Send to Encoder (text + image encoding + Phase1 send)
    logger.info("Step 3: Sending request to Encoder...")
    resp_e = requests.post(f"{ENCODER_URL}{ENDPOINT}", json=payload, timeout=30)
    logger.info(f"Encoder response: {resp_e.json()}")

    # Step 4: Poll Decoder for completion (image saved on Decoder node)
    logger.info("Step 4: Polling Decoder for completion...")
    t_start = time.time()
    result = poll_task(DECODER_URL, decoder_task_id)
    elapsed = time.time() - t_start
    logger.info(f"I2I 3-way Disagg completed in {elapsed:.2f}s: {result}")
