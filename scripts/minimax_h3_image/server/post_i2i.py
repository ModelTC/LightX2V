import base64
from pathlib import Path

import requests
from loguru import logger


def image_to_base64(image_path):
    """Encode a client-local image file for the HTTP request."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/image/"
    image_path = "/path/to/reference.png"

    message = {
        "task": "i2i",
        "prompt": "Keep the scene in <Picture 1> and change the lighting to warm sunset light.",
        # For multiple references, use a comma-separated server-local path string here.
        "image_path": image_to_base64(image_path),
        "seed": 1234,
        "size": [2048, 2048],
        "save_result_path": "./minimax_h3_image_i2i.png",
    }

    logger.info(f"Submitting {message['task']} request with reference image: {image_path}")
    response = requests.post(url, json=message, timeout=60)
    response.raise_for_status()
    logger.info(f"response: {response.json()}")
