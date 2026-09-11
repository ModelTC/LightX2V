import base64
from pathlib import Path

import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"
    lightx2v_path = Path(__file__).resolve().parents[3]
    image_path = lightx2v_path / "assets/inputs/imgs/flf2v_input_first_frame-fs8.png"
    last_frame_path = lightx2v_path / "assets/inputs/imgs/flf2v_input_last_frame-fs8.png"

    message = {
        "prompt": "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective.",
        "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "image_path": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
        "last_frame_path": base64.b64encode(last_frame_path.read_bytes()).decode("utf-8"),
        "seed": 42,
        "num_frames": 81,
        "size": [720, 1280],
        "save_result_path": "./output_lightx2v_wan_flf2v.mp4",
    }

    logger.info(f"image_path: {image_path}")
    logger.info(f"last_frame_path: {last_frame_path}")
    response = requests.post(url, json=message)
    logger.info(f"response: {response.json()}")
