"""
Wan2.1 VACE (Video Animate Character Exchange) generation example.
This example demonstrates how to use LightX2V with Wan2.1 VACE model for character exchange in videos.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for VACE task
pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.1-VACE-1.3B",
    src_ref_images="../assets/inputs/imgs/girl.png,../assets/inputs/imgs/snake.png",
    model_cls="wan2.1_vace",
    task="vace",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/wan/wan_vace.json"
# )

# Optional: enable offloading to significantly reduce VRAM usage
# Suitable for RTX 30/40/50 consumer GPUs
# pipe.enable_offload(
#     cpu_offload=True,
#     offload_granularity="block",
#     text_encoder_offload=True,
#     image_encoder_offload=False,
#     vae_offload=False,
# )

# Create generator with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=40,
    height=480,  # Can be set to 720 for higher resolution
    width=832,  # Can be set to 1280 for higher resolution
    num_frames=81,
    guidance_scale=5,
    sample_shift=16,
)

seed = 42
prompt = "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path = "/path/to/save_results/output.mp4"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
