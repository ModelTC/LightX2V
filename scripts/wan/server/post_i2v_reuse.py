import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"

    # Start the service with start_server_i2v_reuse.sh.
    # Run with reuse=False first; after successful completion, set reuse=True.
    # Keep the image file, prompts, frame count and target shape unchanged; seed may change.
    message = {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression.",
        "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        # Use an absolute path readable by the server for every request.
        "image_path": "/path/to/LightX2V/assets/inputs/imgs/img_0.jpg",
        "seed": 42,
        "num_frames": 81,
        "size": [480, 832],
        "reuse": False,
        "save_result_path": "./output_lightx2v_wan_i2v_reuse.mp4",
    }

    logger.info(f"message: {message}")
    response = requests.post(url, json=message)
    response.raise_for_status()
    logger.info(f"response: {response.json()}")
