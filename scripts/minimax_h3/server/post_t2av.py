import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"

    message = {
        "task": "t2av",
        "prompt": "integrated_multimodal_description: A cinematic fox walks through a snowy pine forest at dawn. overall_soundscape: Soft wind, crunching snow, and distant birds. non_diegetic_music: Quiet warm strings.",
        "seed": 42,
        "num_frames": 124,
        "target_shape": [544, 960],
        "save_result_path": "./minimax_h3_t2av.mp4",
    }

    logger.info(f"message: {message}")
    response = requests.post(url, json=message)
    response.raise_for_status()
    logger.info(f"response: {response.json()}")
