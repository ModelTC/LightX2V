import base64

import requests
from loguru import logger


def image_to_base64(image_path):
    """Convert an image file to base64 string"""
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"

    message = {
        "task": "i2av",
        "prompt": "Animate this image with natural synchronized sound.",
        "image_path": image_to_base64("assets/inputs/imgs/img_0.jpg"),
        "seed": 42,
        "num_frames": 124,
        "size": [544, 960],
        "save_result_path": "./minimax_h3_i2av.mp4",
    }

    logger.info(f"message: {message}")
    response = requests.post(url, json=message)
    response.raise_for_status()
    logger.info(f"response: {response.json()}")
