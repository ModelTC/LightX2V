import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/image/"

    message = {
        "image_path": "path/to/test.png",
        # Choose one output-size option: sr_ratio scales both dimensions; target_shape sets [height, width].
        # "target_shape": [1440, 2520],
        "sr_ratio": 2,
        "save_result_path": "path/to/output.png",
    }

    logger.info(f"message: {message}")

    response = requests.post(url, json=message)

    logger.info(f"response: {response.json()}")
