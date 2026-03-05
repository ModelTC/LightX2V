"""
Wan2.1 text-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.1 model for T2V generation.
"""

from lightx2v import LightX2VPipeline
import time
s= time.time()

# Initialize pipeline for Wan2.1 T2V task
pipe = LightX2VPipeline(
    model_path=r"D:\valencia\models\Wan2.1-T2V-1.3B",
    model_cls="wan2.1",
    task="t2v",
)
pipe.create_generator(
    config_json="../../configs/platforms/intel_xpu/wan_t2v_1_3_t5_quant.json"
)




seed = 42
prompt = "a cat"
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path = "./output.mp4"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
e=time.time()
print("e2e_perf",e-s)
