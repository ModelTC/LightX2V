"""
Shot Stream Inference Pipeline for SekoTalk Model
"""

import os

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DTYPE"] = "BF16"
os.environ["SENSITIVE_LAYER_DTYPE"] = "None"
os.environ["PROFILING_DEBUG_LEVEL"] = "2"


from lightx2v.shot_stream_infer import ClipConfig, ShotConfig, ShotStreamPipeline

seed = 42
prompt = "The video features a male speaking to the camera with arms spread out, a slightly furrowed brow, and a focused gaze."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
lightx2v_path = "/mnt/afs1/wangshankun/LightX2V/"
image_path = lightx2v_path + "/assets/inputs/audio/seko_input.png"
save_result_path = lightx2v_path + "/save_results/output_lightx2v_seko_talk.mp4"
audio_path = lightx2v_path + "/assets/inputs/audio/seko_input.mp3"

# Generate video
shot_cfg = ShotConfig(
    seed=seed,
    image_path=image_path,
    audio_path=audio_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
    clip_configs=[
        ClipConfig(name="s2v_clip", config_json=lightx2v_path + "configs/seko_talk/shot_stream/s2v.json"),
        ClipConfig(name="f2v_clip", config_json=lightx2v_path + "configs/seko_talk/shot_stream/f2v.json"),
    ],
)

shot_stream_pipe = ShotStreamPipeline(shot_cfg)

shot_stream_pipe.generate()
