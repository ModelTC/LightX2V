import base64
from pathlib import Path

import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"
    lightx2v_path = Path(__file__).resolve().parents[3]
    reference_image_paths = [
        lightx2v_path / "assets/inputs/imgs/girl.png",
        lightx2v_path / "assets/inputs/imgs/snake.png",
    ]

    message = {
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。",
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "src_ref_images": [base64.b64encode(path.read_bytes()).decode("utf-8") for path in reference_image_paths],
        "seed": 42,
        "num_frames": 81,
        "target_shape": [480, 832],
        "save_result_path": "./output_lightx2v_wan_vace.mp4",
    }

    logger.info(f"reference_image_paths: {reference_image_paths}")
    response = requests.post(url, json=message)
    logger.info(f"response: {response.json()}")
