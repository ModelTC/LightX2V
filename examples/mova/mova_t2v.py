import os
import sys
sys.path.append('/devsft_AFS/mova_code/LightX2V')

from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(
    model_path="/devsft_AFS/mova_weights",   # 权重路径
    model_cls="mova",
    task="i2av",
)
# 使用绝对路径指定配置文件（请确保该路径真实存在）
config_path = "/devsft_AFS/mova_code/LightX2V/configs/mova/mova_t2v.json"
# 加载配置文件（可选，如果不指定，会使用默认配置）
pipe.create_generator(config_json=config_path)
# 启用 offload
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",
    text_encoder_offload=True,
    vae_offload=True,
)

seed = 42
prompt = "A close-up shot of two anthropomorphic cats in colorful boxing gear. One cat is orange with white gloves, the other is gray with blue gloves. They stand on a spotlighted boxing ring, fists raised. The orange cat swings a right hook, connecting with the gray cat's cheek with a solid \"thwack\". The gray cat recoils, shakes its head, and growls \"You'll pay for that!\". They circle each other, their feet shuffling on the canvas. The audience roars in the background. The camera follows the action closely."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path = "/devsft_AFS/mova_output/cats.mp4"
image_path = "/devsft_AFS/input_cats.jpg"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    image_path=image_path,
    save_result_path=save_result_path,
)