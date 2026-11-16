"""
Wan2.2 animate video generation example.
This example demonstrates how to use LightX2V with Wan2.2 model for animate video generation.

First, run preprocessing:
1. Set up environment: pip install -r /path/to/LightX2V/requirements_animate.txt
2. For animate mode:
   python /path/to/LightX2V/tools/preprocess/preprocess_data.py \
       --ckpt_path /path/to/Wan2.2-Animate-14B/process_checkpoint \
       --video_path /path/to/video \
       --refer_path /path/to/ref_img \
       --save_path /path/to/LightX2V/save_results/animate/process_results \
       --resolution_area 1280 720 \
       --retarget_flag
3. For replace mode:
   python /path/to/LightX2V/tools/preprocess/preprocess_data.py \
       --ckpt_path /path/to/Wan2.2-Animate-14B/process_checkpoint \
       --video_path /path/to/video \
       --refer_path /path/to/ref_img \
       --save_path /path/to/LightX2V/save_results/replace/process_results \
       --resolution_area 1280 720 \
       --iterations 3 \
       --k 7 \
       --w_len 1 \
       --h_len 1 \
       --replace_flag
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[3] / "configs/wan22/wan_animate.json").read_text())
config.update(
    {
        "model_path": "/path/to/Wan2.2-Animate-14B",
        "model_cls": "wan2.2_animate",
        "task": "animate",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = "视频中的人在做动作"
    pose_video_path = "/path/to/LightX2V/save_results/animate/process_results/src_pose.mp4"
    face_video_path = "/path/to/LightX2V/save_results/animate/process_results/src_face.mp4"
    ref_image_paths = ["/path/to/LightX2V/save_results/animate/process_results/src_ref.png"]
    save_result_path = "/path/to/save_results/output.mp4"

    pipe.generate(
        seed=seed,
        pose_video_path=pose_video_path,
        face_video_path=face_video_path,
        ref_image_paths=ref_image_paths,
        prompt=prompt,
        save_result_path=save_result_path,
    )
