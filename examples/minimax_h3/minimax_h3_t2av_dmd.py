"""MiniMax-H3 4-step 768p T2AV inference with the released DMD LoRA.

Before running, set ``model_path`` to a local MiniMax-H3 model directory and
ensure the LoRA path in the selected config resolves to a local checkpoint.
"""

import os

# MiniMax-H3 retains sensitive projection layers in FP32.
os.environ["SENSITIVE_LAYER_DTYPE"] = "FP32"

from lightx2v import LightX2VPipeline

MODEL_PATH = "/data/nvme6/gushiqiao/models/MiniMax-H3"
CONFIG_PATH = "configs/minimax_h3/dmd/minimax_h3_bf16_4step_single_gpu_offload.json"
OUTPUT_PATH = "save_results/minimax_h3_t2av_dmd_768p.mp4"


pipe = LightX2VPipeline(
    model_path=MODEL_PATH,
    model_cls="minimax_h3",
    task="t2av",
)
pipe.create_generator(config_json=CONFIG_PATH)

pipe.generate(
    seed=42,
    prompt=(
        "integrated_multimodal_description: [Shot 1] A cinematic red fox walks "
        "through a snowy pine forest at dawn. The camera tracks beside the fox "
        "as its paws leave crisp prints in fresh snow and warm sunlight breaks "
        "through the trees. overall_soundscape: Soft wind moves through pine "
        "branches, snow crunches underfoot, and distant birds call gently. "
        "non_diegetic_music: A quiet orchestral score with warm strings slowly rises."
    ),
    save_result_path=OUTPUT_PATH,
)
