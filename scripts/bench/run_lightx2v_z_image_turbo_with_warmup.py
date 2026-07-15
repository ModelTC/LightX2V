import sys
from datetime import datetime

lightx2v_path = "/path/to/LightX2V"
model_path = "/path/to/Tongyi-MAI/Z-Image-Turbo"

sys.path.append(lightx2v_path)

from lightx2v import LightX2VPipeline  # noqa: E402

ts = datetime.now().strftime("%y%m%d%H%M")

model_cls = "z_image"

task = "t2i"


pipe = LightX2VPipeline(
    model_path=model_path,
    model_cls=model_cls,
    task=task,
)

pipe.create_generator(config_json=f"{lightx2v_path}/configs/z_image/z_image_turbo_t2i.json")


# Generation parameters
seed = 42
prompt = "A fantasy landscape with mountains and a river, detailed, vibrant colors"

negative_prompt = " "

target_shape = [1024, 1024]

save_result_path = f"{lightx2v_path}/save_results/{model_cls}_{task}_{ts}.png"

# warmup
pipe.generate(
    seed=seed,
    prompt=prompt,
    target_shape=target_shape,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)

# Generate video
pipe.generate(
    seed=seed,
    prompt=prompt,
    target_shape=target_shape,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
    return_result_tensor=True,
)
