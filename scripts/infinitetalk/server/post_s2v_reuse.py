import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"

    # Start the service with start_server_reuse.sh.
    # First request: reuse=False, reuse_prefix_segments=0. Wait for successful completion.
    # Then use the values below; seed may change. Keep conditioning inputs unchanged
    # and retain the previous output file so its prefix can be copied.
    message = {
        "prompt": "A man and a woman stand by a bench in a sunny park, talking and gesturing naturally as they respond to each other.",
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        # Fixed server-local paths; the audio directory contains config.json, audio and masks.
        "image_path": "/path/to/LightX2V/assets/inputs/audio/multi_person/seko_input.png",
        "audio_path": "/path/to/LightX2V/assets/inputs/audio/multi_person",
        "seed": 42,
        "num_frames": 81,  # Frames per segment, including overlapping motion frames.
        "video_duration": 10,
        "reuse": True,
        # 0: reuse input encoding only; 1: retain the first segment and regenerate the rest.
        # The prefix segment count must be smaller than the total segment count.
        "reuse_prefix_segments": 1,
        "save_result_path": "./output_lightx2v_infinitetalk_reuse.mp4",
    }

    logger.info(f"message: {message}")
    response = requests.post(url, json=message)
    response.raise_for_status()
    logger.info(f"response: {response.json()}")
