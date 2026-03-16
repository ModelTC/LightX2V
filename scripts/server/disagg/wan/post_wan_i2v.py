"""
Wan2.1 I2V 三段式 Disagg request script.
三段式请求流程：
  1. 先发请求到 Decoder (启动 Phase2 接收等待)
  2. 再发请求到 Transformer (启动 Phase1 接收 + Phase2 发送)
  3. 最后发请求到 Encoder (运行 T5/CLIP/VAE 编码 + Phase1 发送)
  4. Poll Decoder 等待最终结果（视频由 Decoder 节点保存）
"""
import base64
import time

import requests
from loguru import logger

ENCODER_URL = "http://localhost:8004"
TRANSFORMER_URL = "http://localhost:8005"
DECODER_URL = "http://localhost:8007"
ENDPOINT = "/v1/tasks/video/"

IMAGE_PATH = "/path/to/img_0.jpg"  # 必须填写本地图片路径


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def poll_task(url, task_id, timeout=600, interval=5):
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
    assert IMAGE_PATH, "请设置 IMAGE_PATH 为本地图片路径"

    payload = {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds.",
        "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "image_path": image_to_base64(IMAGE_PATH),
        "seed": 42,
        "save_result_path": "save_results/wan_i2v_disagg.mp4",
    }

    # Step 1: Send to Decoder first (sets up Phase2 receiver, starts blocking)
    logger.info("Step 1: Sending request to Decoder...")
    resp_d = requests.post(f"{DECODER_URL}{ENDPOINT}", json=payload, timeout=30)
    decoder_task_id = resp_d.json().get("task_id")
    logger.info(f"Decoder task_id: {decoder_task_id}")

    # Step 2: Send to Transformer (sets up Phase1 receiver + Phase2 sender)
    logger.info("Step 2: Sending request to Transformer...")
    resp_t = requests.post(f"{TRANSFORMER_URL}{ENDPOINT}", json=payload, timeout=30)
    transformer_task_id = resp_t.json().get("task_id")
    logger.info(f"Transformer task_id: {transformer_task_id}")

    # Step 3: Send to Encoder (triggers T5/CLIP/VAE encoding + Phase1 Mooncake send)
    logger.info("Step 3: Sending request to Encoder...")
    resp_e = requests.post(f"{ENCODER_URL}{ENDPOINT}", json=payload, timeout=30)
    logger.info(f"Encoder response: {resp_e.json()}")

    # Step 4: Poll Decoder for final completion (video is saved on Decoder node)
    logger.info("Step 4: Polling Decoder for completion...")
    t_start = time.time()
    result = poll_task(DECODER_URL, decoder_task_id)
    elapsed = time.time() - t_start
    logger.info(f"I2V 3-way Disagg completed in {elapsed:.2f}s: {result}")
