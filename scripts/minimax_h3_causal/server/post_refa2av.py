import base64
from pathlib import Path

import requests
import yaml
from loguru import logger


def file_to_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"

    global_file = "/data/zhangbilang/handoff/assets/global.txt"
    motion_file = "/data/zhangbilang/handoff/assets/motions.yaml"
    image_path = "/data/zhangbilang/handoff/assets/meinv_01.png"
    audio_path = "/data/zhangbilang/handoff/assets/meinv_01.wav"

    # Submit the replay script once. The runner encodes only the first motion
    # initially, then each later motion at the next 17-frame boundary after its frame cue.
    prompt = Path(global_file).read_text(encoding="utf-8").strip()
    motions = yaml.safe_load(Path(motion_file).read_text(encoding="utf-8"))

    message = {
        "task": "refa2av",
        "prompt": prompt,
        "image_path": file_to_base64(image_path),
        "audio_path": file_to_base64(audio_path),
        "motions": motions,
        "seed": 42,
        "num_frames": 957,
        "size": [1376, 768],
        "save_result_path": "./minimax_h3_causal_refa2av.mp4",
    }

    logger.info(f"Submitting refa2av request to {url}")
    response = requests.post(url, json=message)
    response.raise_for_status()
    logger.info(f"response: {response.json()}")
