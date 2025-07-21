import requests
from loguru import logger
import random
import string
import time
from datetime import datetime


# same as lightx2v/utils/generate_task_id.py
# from lightx2v.utils.generate_task_id import generate_task_id
def generate_task_id():
    """
    Generate a random task ID in the format XXXX-XXXX-XXXX-XXXX-XXXX.
    Features:
    1. Does not modify the global random state.
    2. Each X is an uppercase letter or digit (0-9).
    3. Combines time factors to ensure high randomness.
    For example: N1PQ-PRM5-N1BN-Z3S1-BGBJ
    """
    # Save the current random state (does not affect external randomness)
    original_state = random.getstate()

    try:
        # Define character set (uppercase letters + digits)
        characters = string.ascii_uppercase + string.digits

        # Create an independent random instance
        local_random = random.Random(time.perf_counter_ns())

        # Generate 5 groups of 4-character random strings
        groups = []
        for _ in range(5):
            # Mix new time factor for each group
            time_mix = int(datetime.now().timestamp())
            local_random.seed(time_mix + local_random.getstate()[1][0] + time.perf_counter_ns())

            groups.append("".join(local_random.choices(characters, k=4)))

        return "-".join(groups)

    finally:
        # Restore the original random state
        random.setstate(original_state)


def post_all_tasks(urls, messages):
    msg_num = len(messages)
    msg_index = 0
    available_urls = []
    for url in urls:
        try:
            _ = requests.get(f"{url}/v1/service/status").json()
        except Exception as e:
            continue
        available_urls.append(url)

    if not available_urls:
        logger.error("No available urls.")
        return

    logger.info(f"available_urls: {available_urls}")

    while True:
        for url in available_urls:
            response = requests.get(f"{url}/v1/service/status").json()
            if response["service_status"] == "idle":
                logger.info(f"{url} service is idle, start task...")
                response = requests.post(f"{url}/v1/tasks/", json=messages[msg_index])
                logger.info(f"response: {response.json()}")
                msg_index += 1
                if msg_index == msg_num:
                    logger.info("All tasks have been sent.")
                    return
        time.sleep(5)


if __name__ == "__main__":
    urls = ["http://localhost:8000", "http://localhost:8001"]

    messages = [
        {
            "task_id": generate_task_id(),  # task_id also can be string you like, such as "test_task_001"
            "task_id_must_unique": True,  # If True, the task_id must be unique, otherwise, it will raise an error. Default is False.
            "prompt": "A cat walks on the grass, realistic style.",
            "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "image_path": "",
            "save_video_path": "./output_lightx2v_wan_t2v_t01.mp4",  # It is best to set it to an absolute path.
        },
        {
            "task_id": generate_task_id(),  # task_id also can be string you like, such as "test_task_001"
            "task_id_must_unique": True,  # If True, the task_id must be unique, otherwise, it will raise an error. Default is False.
            "prompt": "A person is riding a bike. Realistic, Natural lighting, Casual.",
            "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "image_path": "",
            "save_video_path": "./output_lightx2v_wan_t2v_t02.mp4",  # It is best to set it to an absolute path.
        },
        {
            "task_id": generate_task_id(),  # task_id also can be string you like, such as "test_task_001"
            "task_id_must_unique": True,  # If True, the task_id must be unique, otherwise, it will raise an error. Default is False.
            "prompt": "A car turns a corner. Realistic, Natural lighting, Casual.",
            "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "image_path": "",
            "save_video_path": "./output_lightx2v_wan_t2v_t03.mp4",  # It is best to set it to an absolute path.
        },
        {
            "task_id": generate_task_id(),  # task_id also can be string you like, such as "test_task_001"
            "task_id_must_unique": True,  # If True, the task_id must be unique, otherwise, it will raise an error. Default is False.
            "prompt": "An astronaut is flying in space, Van Gogh style. Dark, Mysterious.",
            "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "image_path": "",
            "save_video_path": "./output_lightx2v_wan_t2v_t04.mp4",  # It is best to set it to an absolute path.
        },
        {
            "task_id": generate_task_id(),  # task_id also can be string you like, such as "test_task_001"
            "task_id_must_unique": True,  # If True, the task_id must be unique, otherwise, it will raise an error. Default is False.
            "prompt": "A beautiful coastal beach in spring, waves gently lapping on the sand, the camera movement is Zoom In. Realistic, Natural lighting, Peaceful.",
            "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "image_path": "",
            "save_video_path": "./output_lightx2v_wan_t2v_t05.mp4",  # It is best to set it to an absolute path.
        },
        {
            "task_id": generate_task_id(),  # task_id also can be string you like, such as "test_task_001"
            "task_id_must_unique": True,  # If True, the task_id must be unique, otherwise, it will raise an error. Default is False.
            "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
            "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "image_path": "",
            "save_video_path": "./output_lightx2v_wan_t2v_t06.mp4",  # It is best to set it to an absolute path.
        },
    ]

    logger.info(f"urls: {urls}")
    logger.info(f"message: {messages}")

    post_all_tasks(urls, messages)
