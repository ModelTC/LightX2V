import os
import sys
from datetime import datetime

lightx2v_path = "/path/to/LightX2V"
model_path = "/path/to/Qwen/Qwen-Image-Edit-2511"

sys.path.append(lightx2v_path)
os.environ["PROFILING_DEBUG_LEVEL"] = "2"

from lightx2v import LightX2VPipeline # noqa: E402

ts = datetime.now().strftime("%y%m%d%H%M")

model_cls = "qwen_image"

task = "i2i"


pipe = LightX2VPipeline(
    model_path=model_path,
    model_cls=model_cls,
    task=task,
)

pipe.create_generator(config_json=f"{lightx2v_path}/configs/qwen_image/qwen_image_i2i_2511.json")


# Generation parameters
seed = 42
prompt = "Transform into anime style"

negative_prompt = " "

target_shape = [1024, 1024]

# https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
image_path = "/path/to/cat.png"

save_result_path = f"{lightx2v_path}/save_results/{model_cls}_{task}_{ts}.png"

# warmup
pipe.generate(
    seed=seed,
    prompt=prompt,
    target_shape=target_shape,
    negative_prompt=negative_prompt,
    image_path=image_path,
    save_result_path=save_result_path,
)

# Generate video
pipe.generate(
    seed=seed,
    prompt=prompt,
    target_shape=target_shape,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
    image_path=image_path,
    return_result_tensor=True,
)
