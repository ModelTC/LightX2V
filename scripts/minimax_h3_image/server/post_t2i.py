import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/image/"

    message = {
        "task": "t2i",
        "prompt": "A photorealistic photograph of a cat sitting by a sunlit window.",
        "seed": 42,
        "size": [2048, 2048],
        "save_result_path": "./minimax_h3_image_t2i.png",
    }

    logger.info(f"message: {message}")
    response = requests.post(url, json=message, timeout=60)
    response.raise_for_status()
    logger.info(f"response: {response.json()}")
