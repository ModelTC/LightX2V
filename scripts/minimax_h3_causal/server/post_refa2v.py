import base64
from pathlib import Path

import requests
from loguru import logger


def file_to_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"
    lightx2v_path = Path(__file__).resolve().parents[3]

    message = {
        "task": "refa2v",
        "prompt": "The man in the reference image speaks naturally into the microphone, with lip movements synchronized to the audio, subtle facial expressions and natural hand gestures. The camera remains stationary.",
        # Encode client-local media so the server does not need these paths.
        "image_path": file_to_base64(lightx2v_path / "assets/inputs/audio/seko_input.png"),
        "audio_path": file_to_base64(lightx2v_path / "assets/inputs/audio/seko_input.mp3"),
        "seed": 0,
        "num_frames": 124,
        "size": [544, 960],
        "save_result_path": "./minimax_h3_causal_refa2v.mp4",
    }

    logger.info(f"Submitting refa2v request to {url}")
    response = requests.post(url, json=message)
    response.raise_for_status()
    logger.info(f"response: {response.json()}")
