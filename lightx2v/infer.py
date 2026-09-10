import argparse
import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.bagel.sensenova_tasks import OMNI_VISION_SUBTASK_CHOICES
from lightx2v.models.runners.runner_factory import RUNNER_MODULES, build_runner
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.set_config import build_cli_inputs, init_parallel, print_config, print_request
from lightx2v.utils.utils import validate_config_paths
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def distributed_barrier():
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return False

    from lightx2v_platform.base.global_var import AI_DEVICE

    if AI_DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()

    from loguru import logger

    logger.info(f"[Barrier] synchronized all ranks")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="The seed for random generator")
    parser.add_argument(
        "--model_cls",
        type=str,
        required=True,
        choices=RUNNER_MODULES,
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "t2v",
            "i2v",
            "t2t",
            "t2i",
            "ti2t",
            "ti2i",
            "i2i",
            "flf2v",
            "vace",
            "animate",
            "s2v",
            "rs2v",
            "t2av",
            "i2av",
            "l2av",
            "fl2av",
            "ref2av",
            "i2va",
            "v2av",
            "ltx2_s2v",
            "sr",
            "recon",
            "i23d",
            "omni_vision_task",
        ],
        required=True,
    )
    parser.add_argument(
        "--omni_vision_subtask",
        type=str,
        choices=OMNI_VISION_SUBTASK_CHOICES,
        default=None,
        help="Subtask used with --task omni_vision_task.",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None, help="The input prompt for text-to-video generation")
    parser.add_argument("--prompt_ref", type=str, default=None, help="Reference/driving-video prompt for Wan-Animate-2.")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--bot_task", type=str, default=None, help="HunyuanImage3 text generation mode.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Maximum number of generated text tokens.")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for text generation.")
    parser.add_argument("--text_do_sample", action=argparse.BooleanOptionalAction, default=None, help="Enable sampling during text generation.")
    parser.add_argument("--text_temperature", type=float, default=None, help="Text sampling temperature.")
    parser.add_argument("--text_top_k", type=int, default=None, help="Top-k text sampling limit.")
    parser.add_argument("--text_top_p", type=float, default=None, help="Top-p text sampling threshold.")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="The path to input image file(s), including HunyuanImage3 ti2t/ti2i and MiniMax-H3 ref2av reference images. Multiple paths should be comma-separated. Example: 'path1.jpg,path2.jpg'",
    )
    parser.add_argument("--state_path", type=str, default=None, help="The path to input robot state file for robot i2v/i2va inference.")
    parser.add_argument("--last_frame_path", type=str, default=None, help="The path to last frame file for first-last-frame-to-video (flf2v) task")
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Input audio path: Wan s2v / rs2v, LTX-2 ltx2_s2v, or MiniMax-H3 ref2av reference audio. H3 accepts comma-separated paths.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Input source video path. Its role is determined by the selected task.",
    )
    parser.add_argument("--video_duration", type=float, default=None, help="Requested output duration in seconds for audio-driven video generation.")
    parser.add_argument("--image_strength", type=str, default=None, help="i2av: single float, or comma-separated floats (one per image, or one value broadcast). Example: 1.0 or 1.0,0.85,0.9")
    parser.add_argument(
        "--num_frames",
        dest="target_video_length",
        type=int,
        default=None,
        help="Requested output frame count. Model-specific length constraints apply.",
    )
    parser.add_argument(
        "--i2i_denoise_strength",
        type=float,
        default=None,
        help="(i2i) Single-image edit denoising strength in [0.0, 1.0]. 0.0 preserves the source image most; 1.0 redraws most. Omit to keep the model's existing behavior.",
    )
    parser.add_argument("--inpaint_blur_sigma", type=float, default=None, help="Flux2 inpainting mask blur sigma.")
    parser.add_argument("--inpaint_blur_size", type=int, default=None, help="Flux2 inpainting mask blur kernel size.")
    parser.add_argument(
        "--image_frame_idx", type=str, default=None, help="i2av: comma-separated pixel frame indices (one per image). Omit or empty to evenly space frames in [0, num_frames-1]. Example: 0,40,80"
    )
    # [Warning] For vace task, need refactor.
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None.",
    )
    parser.add_argument("--mask_path", type=str, default=None, help="Input mask path.")
    parser.add_argument(
        "--src_pose_path",
        type=str,
        default=None,
        help="Pose driving video for Wan s2v / animate (e.g. examples/pose.mp4).",
    )
    parser.add_argument(
        "--src_face_path",
        type=str,
        default=None,
        help="The file of the source face. Default None.",
    )
    parser.add_argument(
        "--src_bg_path",
        type=str,
        default=None,
        help="The file of the source background. Default None.",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default=None,
        help="Pose string (e.g., 'w-3, right-0.5') or JSON file path for WorldPlay models.",
    )
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="Directory path for lingbot camera/action control files (poses.npy, intrinsics.npy, optional action.npy).",
    )
    parser.add_argument("--action_mode", type=str, default=None, choices=["forward_dynamics", "inverse_dynamics", "policy"], help="Cosmos3 action mode.")
    parser.add_argument("--domain_name", type=str, default=None, help="Cosmos3 action embodiment domain name.")
    parser.add_argument("--view_point", type=str, default=None, help="Cosmos3 action viewpoint label.")
    # WorldMirror (3D reconstruction) specific
    parser.add_argument("--input_path", type=str, default=None, help="(worldmirror/recon) Path to a directory of images, a video file, or a single image.")
    parser.add_argument("--strict_output_path", type=str, default=None, help="(worldmirror/recon) If set, write outputs directly here instead of under save_result_path/<subdir>/<timestamp>/.")
    parser.add_argument("--prior_cam_path", type=str, default=None, help="(worldmirror/recon) Optional camera prior JSON (extrinsics + intrinsics).")
    parser.add_argument("--prior_depth_path", type=str, default=None, help="(worldmirror/recon) Optional depth prior directory (one .npy/.png per image).")
    parser.add_argument("--save_rendered", action=argparse.BooleanOptionalAction, default=None, help="(worldmirror/recon) Render an interpolated fly-through video from Gaussian splats.")
    parser.add_argument("--render_interp_per_pair", type=int, default=None, help="(worldmirror/recon) Interpolated frames per camera pair for --save_rendered.")
    parser.add_argument("--render_depth", action=argparse.BooleanOptionalAction, default=None, help="(worldmirror/recon) Also render a depth video with --save_rendered.")

    parser.add_argument("--save_result_path", type=str, default=None, help="The path to save video path/file")
    parser.add_argument("--return_result_tensor", action="store_true", default=None, help="Whether to return result tensor. (Useful for comfyui)")
    parser.add_argument("--save_action_path", type=str, default=None, help="The path to save action predictions for Motus, LingBot-VA, or DreamZero.")
    parser.add_argument("--raw_output_path", type=str, default=None, help="Raw prediction output path for SenseNova-Vision.")
    parser.add_argument("--glb_output_path", type=str, default=None, help="GLB scene output path for SenseNova-Vision.")
    parser.add_argument("--postprocess_predictions", action=argparse.BooleanOptionalAction, default=None, help="Postprocess SenseNova-Vision predictions.")
    parser.add_argument("--target_shape", type=int, nargs="+", default=None, help="Set return video or image shape")
    parser.add_argument("--aspect_ratio", type=str, default=None)
    parser.add_argument("--infer_align_image_size", action=argparse.BooleanOptionalAction, default=None, help="Align HunyuanImage3 reference image sizes during inference.")
    parser.add_argument(
        "--keep_original_aspect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(i2i) When exactly one reference image is provided, preserve its aspect ratio with max_size=2048.",
    )
    parser.add_argument(
        "--layout_bboxes",
        type=str,
        default=None,
        help="(i2i) Layout boxes as a JSON string or JSON file path for HiDream layout-conditioned editing.",
    )
    parser.add_argument("--sr_ratio", type=float, default=None, help="super resolution ratio for sr task")
    parser.add_argument(
        "--reference_video_strength", type=float, default=None, help="(v2av) IC-LoRA reference-video conditioning strength in [0.0, 1.0]. 1.0 = full adherence to the control signal, 0.0 = ignore it."
    )
    parser.add_argument("--reference_video_frame_cap", type=int, default=None, help="(v2av) Maximum number of frames to read from the reference/control video. Defaults to the full clip.")
    parser.add_argument("--mux_audio_video_path", type=str, default=None, help="(v2av, optional) After saving, mux audio from this file into the output mp4 (ffmpeg). ")

    args = parser.parse_args()
    startup_config, request_data = build_cli_inputs(args)
    if startup_config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()
        init_parallel(startup_config)

    print_config(startup_config, title="Startup config")

    validate_config_paths(startup_config)

    with ProfilingContext4DebugL1("Total Cost"):
        runner = build_runner(startup_config)
        input_info = runner.prepare_request(request_data)
        print_request(input_info, runner.get_supported_request_fields(input_info.task))
        runner.run_request(input_info)

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
