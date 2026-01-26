# torchrun --nproc_per_node=8 run_lightx2v_wan22_t2v_8gpu_with_warmup.py

import os
import sys
from datetime import datetime

lightx2v_path = "/path/to/LightX2V"
model_path = "/path/to/Wan-AI/Wan2.2-T2V-A14B"

sys.path.append(lightx2v_path)
os.environ["PROFILING_DEBUG_LEVEL"] = "2"

from lightx2v import LightX2VPipeline  # noqa: E402

ts = datetime.now().strftime("%y%m%d%H%M")

model_cls = "wan2.2_moe"

task = "t2v"


pipe = LightX2VPipeline(
    model_path=model_path,
    model_cls=model_cls,
    task=task,
)

pipe.enable_parallel(cfg_p_size=2, seq_p_size=2, seq_p_attn_type="ulysses")
# pipe.enable_parallel(seq_p_size=8)
pipe.create_generator(config_json=f"{lightx2v_path}/configs/dist_infer/wan22_moe_t2v_cfg_ulysses.json")


# Generation parameters
seed = 42
prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

negative_prompt = " "

target_shape = [720, 1280]

save_result_path = f"{lightx2v_path}/save_results/{model_cls}_{task}_{ts}.mp4"

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
