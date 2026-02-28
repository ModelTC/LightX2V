import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"

    message = {"video_path": "input.mp4", "seed": 42, "save_result_path": "./output_lightx2v_seedvr2_sr.mp4"}

    logger.info(f"message: {message}")

    response = requests.post(url, json=message)

    logger.info(f"response: {response.json()}")
