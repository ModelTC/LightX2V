import base64

import requests
from loguru import logger


def image_to_base64(image_path):
    """Convert an image file to base64 string"""
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/image/"

    message = {
        "prompt": "turn the style of the photo to anime style",
        "image_path": image_to_base64("assets/inputs/imgs/snake.png"),
        "lora_name": "example.safetensors",
        "lora_strength": 1.0,
    }

    logger.info(f"message: {message}")

    response = requests.post(url, json=message)

    logger.info(f"response: {response.json()}")
