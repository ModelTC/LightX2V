import requests
from loguru import logger


url = "http://localhost:8000/v1/local/video/generate"

message = {
    "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "use_prompt_enhancer": True,
    "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "image_path": "",
    "num_fragments": 1,
    "save_video_path": "./output_lightx2v_wan_t2v_ap4.mp4",  # It is best to set it to an absolute path.
}

logger.info(f"message: {message}")

response = requests.post(url, json=message)

logger.info(f"response: {response.json()}")
