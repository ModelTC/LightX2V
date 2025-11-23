"""
Wan2.2 animate video generation example.
This example demonstrates how to use LightX2V with Wan2.2 model for animate video generation.

First, run preprocessing:
1. Set up environment: pip install -r ../requirements_animate.txt
2. For animate mode:
   python ../tools/preprocess/preprocess_data.py \
       --ckpt_path /path/to/Wan2.1-FLF2V-14B-720P/process_checkpoint \
       --video_path /path/to/video \
       --refer_path /path/to/ref_img \
       --save_path ../save_results/animate/process_results \
       --resolution_area 1280 720 \
       --retarget_flag
3. For replace mode:
   python ../tools/preprocess/preprocess_data.py \
       --ckpt_path /path/to/Wan2.1-FLF2V-14B-720P/process_checkpoint \
       --video_path /path/to/video \
       --refer_path /path/to/ref_img \
       --save_path ../save_results/replace/process_results \
       --resolution_area 1280 720 \
       --iterations 3 \
       --k 7 \
       --w_len 1 \
       --h_len 1 \
       --replace_flag
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for animate task
pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.1-FLF2V-14B-720P",
    src_pose_path="../save_results/animate/process_results/src_pose.mp4",
    src_face_path="../save_results/animate/process_results/src_face.mp4",
    src_ref_images="../save_results/animate/process_results/src_ref.png",
    model_cls="wan2.2_animate",
    task="animate",
)
pipe.replace_flag = True  # Set to True for replace mode, False for animate mode

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/wan/wan_animate_replace.json"
# )

# Create generator with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=20,
    height=480,  # Can be set to 720 for higher resolution
    width=832,  # Can be set to 1280 for higher resolution
    num_frames=77,
    guidance_scale=1,
    sample_shift=5.0,
    fps=30,
)

seed = 42
prompt = "视频中的人在做动作"
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path = "/path/to/save_results/output.mp4"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
