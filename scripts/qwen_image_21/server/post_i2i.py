import base64
from pathlib import Path

import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/image/sync"
    image_path = Path("assets/inputs/imgs/img_1.jpg")
    output_path = Path("save_results/qwen_image_21_i2i.png")

    message = {
        "task": "i2i",
        "prompt": "Keep the main subject unchanged and replace the background with a sunlit garden.",
        "image_path": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
        "seed": 42,
        "size": [1024, 1024],  # [height, width]
    }

    logger.info(f"image_path: {image_path}, prompt: {message['prompt']}")
    response = requests.post(url, json=message, timeout=(10, 610))
    response.raise_for_status()
    # The sync endpoint returns PNG bytes, not JSON.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    logger.info(f"Image saved to: {output_path.resolve()}")
