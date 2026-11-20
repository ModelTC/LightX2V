import base64
from pathlib import Path

import requests
from loguru import logger


def file_to_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"
    assets = Path(__file__).resolve().parents[3] / "assets/inputs/audio/multi_person"
    message = {
        "prompt": "Two people talking to each other.",
        "image_path": file_to_base64(assets / "seko_input.png"),
        "audio_path": [file_to_base64(assets / "p1.mp3"), file_to_base64(assets / "p2.mp3")],
    }

    logger.info(f"Submitting s2v request to {url}")
    response = requests.post(url, json=message, timeout=60)
    response.raise_for_status()
    print(response.json()["task_id"])
