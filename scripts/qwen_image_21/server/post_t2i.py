from pathlib import Path

import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/image/sync"
    output_path = Path("save_results/qwen_image_21_t2i.png")

    message = {
        "task": "t2i",
        "prompt": "A capybara wearing a wizard hat, oil painting",
        "seed": 42,
        "size": [1024, 1024],  # [height, width]
    }

    logger.info(f"message: {message}")
    response = requests.post(url, json=message, timeout=(10, 610))
    response.raise_for_status()
    # The sync endpoint returns PNG bytes, not JSON.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    logger.info(f"Image saved to: {output_path.resolve()}")
