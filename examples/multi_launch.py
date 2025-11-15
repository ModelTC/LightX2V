import os

from lightx2v import LightGenerator
from lightx2v.utils.utils import get_configs_dir

config_json = os.path.join(get_configs_dir(), "dist_infer/wan_i2v_dist_cfg_ulysses.json")

generator = LightGenerator(
    model_path="/path/to/Wan2.1-I2V-14B-480P",
    model_cls="wan2.1",
    task="i2v",
    config_json=config_json,
)

video_path = generator.generate(
    prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    image_path="/path/to/img.jpg",
    seed=42,
    save_result_path="output.mp4",
)
