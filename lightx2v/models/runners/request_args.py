"""Reusable argument definitions; model-specific fields can be declared locally."""

import argparse
import json

from lightx2v.utils.request_spec import Argument

ARGUMENTS = {
    "seed": Argument(type=int, help="The seed for random generator"),
    "prompt": Argument(help="Text prompt."),
    "action_prompts": Argument(type=json.loads, help="MiniMax-H3 causal timed actions as a JSON object or list"),
    "ref_video_prompt": Argument(help="Reference/driving-video prompt for Wan-Animate-2."),
    "negative_prompt": Argument(help="Negative text prompt."),
    "bot_task": Argument(help="HunyuanImage3 text generation mode."),
    "max_new_tokens": Argument(type=int, help="Maximum number of generated text tokens."),
    "system_prompt": Argument(help="System prompt for text generation."),
    "text_do_sample": Argument(action=argparse.BooleanOptionalAction, help="Enable sampling during text generation."),
    "text_temperature": Argument(type=float, help="Text sampling temperature."),
    "text_top_k": Argument(type=int, help="Top-k text sampling limit."),
    "text_top_p": Argument(type=float, help="Top-p text sampling threshold."),
    "image_path": Argument(
        help="The path to input image file(s), including HunyuanImage3 ti2t/ti2i and MiniMax-H3 ref2av reference images. Multiple paths should be comma-separated. Example: 'path1.jpg,path2.jpg'",
    ),
    "state_path": Argument(help="The path to input robot state file for robot i2v/i2va inference."),
    "last_frame_path": Argument(help="The path to last frame file for first-last-frame-to-video (flf2v) task"),
    "audio_path": Argument(help="Input audio path: driving audio for s2v / rs2v / ltx2_s2v / refa2v, or reference audio for MiniMax-H3 ref2av."),
    "video_path": Argument(help="Input source video path. Its role is determined by the selected task."),
    "video_duration": Argument(type=float, help="Requested output duration in seconds for audio-driven video generation."),
    "draft_steps": Argument(
        type=int, help="MiniMax-H3: only a positive K smaller than configured infer_steps enables a draft prefix and final velocity projection. Omit or use K >= infer_steps for normal sampling."
    ),
    "resample_source_video": Argument(help="MiniMax-H3 source video with audio; RGB frames are resized to the target canvas before VAE encoding and regeneration."),
    "resample_start_step": Argument(
        type=int, help="MiniMax-H3 zero-based regeneration start index in [0,infer_steps]. Defaults to the startup config or 1; 0 runs all steps, infer_steps only decodes the VAE-encoded source."
    ),
    "resize_mode": Argument(help="seko_talk s2v/rs2v image and mask resize mode; defaults to the startup config."),
    "fixed_area": Argument(help="seko_talk area/min-side resize tier: 480p, 720p, or 1080p; defaults to the startup config."),
    "image_strength": Argument(help="i2av: single float, or comma-separated floats (one per image, or one value broadcast). Example: 1.0 or 1.0,0.85,0.9"),
    "num_frames": Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
    "i2i_denoise_strength": Argument(
        type=float, help="(i2i) Single-image edit denoising strength in [0.0, 1.0]. 0.0 preserves the source image most; 1.0 redraws most. Omit to keep the model's existing behavior."
    ),
    "inpaint_blur_sigma": Argument(type=float, help="Flux2 inpainting mask blur sigma."),
    "inpaint_blur_size": Argument(type=int, help="Flux2 inpainting mask blur kernel size."),
    "image_frame_indices": Argument(help="i2av: comma-separated pixel frame indices (one per image). Omit or empty to evenly space frames in [0, num_frames-1]. Example: 0,40,80"),
    "ref_image_paths": Argument(help="The file list of the source reference images. Separated by ','. Default None."),
    "mask_path": Argument(help="Input mask path."),
    "pose_video_path": Argument(help="Pose driving video for Wan s2v / animate (e.g. /path/to/pose.mp4)."),
    "face_video_path": Argument(help="The file of the source face. Default None."),
    "background_video_path": Argument(help="The file of the source background. Default None."),
    "pose": Argument(help="Pose string (e.g., 'w-3, right-0.5') or JSON file path for WorldPlay models."),
    "action_path": Argument(help="Directory path for lingbot camera/action control files (poses.npy, intrinsics.npy, optional action.npy)."),
    "action_mode": Argument(choices=["forward_dynamics", "inverse_dynamics", "policy"], help="Cosmos3 action mode."),
    "domain_name": Argument(help="Cosmos3 action embodiment domain name."),
    "view_point": Argument(help="Cosmos3 action viewpoint label."),
    "input_path": Argument(help="(worldmirror/recon) Path to a directory of images, a video file, or a single image."),
    "strict_output_path": Argument(help="(worldmirror/recon) If set, write outputs directly here instead of under save_result_path/<subdir>/<timestamp>/."),
    "prior_cam_path": Argument(help="(worldmirror/recon) Optional camera prior JSON (extrinsics + intrinsics)."),
    "prior_depth_path": Argument(help="(worldmirror/recon) Optional depth prior directory (one .npy/.png per image)."),
    "save_rendered": Argument(action=argparse.BooleanOptionalAction, help="(worldmirror/recon) Render an interpolated fly-through video from Gaussian splats."),
    "render_interp_per_pair": Argument(type=int, help="(worldmirror/recon) Interpolated frames per camera pair for --save_rendered."),
    "render_depth": Argument(action=argparse.BooleanOptionalAction, help="(worldmirror/recon) Also render a depth video with --save_rendered."),
    "save_result_path": Argument(help="The path to save video path/file"),
    "return_result_tensor": Argument(action="store_true", help="Whether to return result tensor. (Useful for comfyui)"),
    "save_action_path": Argument(help="The path to save action predictions for Motus, LingBot-VA, or DreamZero."),
    "size": Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH"),
    "resolution": Argument(type=int, help="Qwen-Image-2.1 reference/automatic output resolution; defaults to the startup config"),
    "aspect_ratio": Argument(help="Output aspect ratio."),
    "align_image_size": Argument(action=argparse.BooleanOptionalAction, help="Align HunyuanImage3 reference image sizes during inference."),
    "keep_aspect_ratio": Argument(action=argparse.BooleanOptionalAction, help="(i2i) When exactly one reference image is provided, preserve its aspect ratio with max_size=2048."),
    "layout_bboxes": Argument(help="(i2i) Layout boxes as a JSON string or JSON file path for HiDream layout-conditioned editing."),
    "sr_ratio": Argument(type=float, help="super resolution ratio for sr task"),
    "match_target_size": Argument(action=argparse.BooleanOptionalAction, help="(SeedVR sr) Crop or resize decoded output to size; defaults to the startup config."),
    "reference_video_strength": Argument(type=float, help="(v2av) IC-LoRA reference-video conditioning strength in [0.0, 1.0]. 1.0 = full adherence to the control signal, 0.0 = ignore it."),
    "reference_video_frame_cap": Argument(type=int, help="(v2av) Maximum number of frames to read from the reference/control video. Defaults to the full clip."),
    "mux_audio_video_path": Argument(help="(v2av, optional) After saving, mux audio from this file into the output mp4 (ffmpeg). "),
    "task": Argument(cli=False),
    "stream_callback": Argument(cli=False),
    "policy_image": Argument(cli=False),
    "policy_state": Argument(cli=False),
}


def fields(*names):
    return {name: ARGUMENTS[name] for name in names}


def without(arguments, *names):
    return {name: argument for name, argument in arguments.items() if name not in names}


COMMON = fields("return_result_tensor", "save_result_path", "seed", "task")
PROMPT = fields("negative_prompt", "prompt")
VIDEO_OUTPUT = fields("size", "num_frames")
RESAMPLING = fields("draft_steps", "resample_source_video", "resample_start_step")
IMAGE = COMMON | PROMPT | fields("aspect_ratio", "size")
VIDEO = COMMON | PROMPT | VIDEO_OUTPUT
