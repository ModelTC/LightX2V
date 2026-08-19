import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"

    message = {
        "video_path": "path/to/test.mp4",
        # Choose one output-size option: target_shape is [height, width]; sr_ratio scales both input dimensions.
        "target_shape": [1440, 2520],
        # "sr_ratio": 1.875,
        "seed": 42,
        "save_result_path": "path/to/output.mp4",
    }

    logger.info(f"message: {message}")

    response = requests.post(url, json=message)

    logger.info(f"response: {response.json()}")
