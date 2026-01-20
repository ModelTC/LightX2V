import argparse
import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.common.ops import *
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_distill_runner import HunyuanVideo15DistillRunner  # noqa: F401
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner  # noqa: F401
from lightx2v.models.runners.longcat_image.longcat_image_runner import LongCatImageRunner  # noqa: F401
from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner  # noqa: F401
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_animate_runner import WanAnimateRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, WanAudioRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_matrix_game2_runner import WanSFMtxg2Runner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_vace_runner import WanVaceRunner  # noqa: F401
from lightx2v.models.runners.z_image.z_image_runner import ZImageRunner  # noqa: F401
from lightx2v.utils.envs import *
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all, validate_task_arguments
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def init_runner(config):
    torch.set_grad_enabled(False)
    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()
    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="The seed for random generator")
    parser.add_argument(
        "--model_cls",
        type=str,
        required=True,
        choices=[
            "wan2.1",
            "wan2.1_distill",
            "wan2.1_mean_flow_distill",
            "wan2.1_vace",
            "wan2.1_sf",
            "wan2.1_sf_mtxg2",
            "seko_talk",
            "wan2.2_moe",
            "wan2.2",
            "wan2.2_moe_audio",
            "wan2.2_audio",
            "wan2.2_moe_distill",
            "qwen_image",
            "longcat_image",
            "wan2.2_animate",
            "hunyuan_video_1.5",
            "hunyuan_video_1.5_distill",
            "z_image",
            "ltx2",
        ],
        default="wan2.1",
    )

    parser.add_argument("--task", type=str, choices=["t2v", "i2v", "t2i", "i2i", "flf2v", "vace", "animate", "s2v", "t2av", "i2av"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sf_model_path", type=str, required=False)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")

    parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument("--image_path", type=str, default="", help="The path to input image file for image-to-video (i2v) task")
    parser.add_argument("--last_frame_path", type=str, default="", help="The path to last frame file for first-last-frame-to-video (flf2v) task")
    parser.add_argument("--audio_path", type=str, default="", help="The path to input audio file or directory for audio-to-video (s2v) task")
    parser.add_argument(
        "--images",
        type=str,
        default="",
        help="Image conditioning for I2AV task. Format: 'path1:frame_idx1:strength1,path2:frame_idx2:strength2'. Example: 'cat.jpg:0:1.0,dog.jpg:60:0.8'",
    )
    # [Warning] For vace task, need refactor.
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None.",
    )
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.",
    )
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.",
    )
    parser.add_argument(
        "--src_pose_path",
        type=str,
        default=None,
        help="The file of the source pose. Default None.",
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
        "--src_mask_path",
        type=str,
        default=None,
        help="The file of the source mask. Default None.",
    )
    parser.add_argument("--save_result_path", type=str, default=None, help="The path to save video path/file")
    parser.add_argument("--return_result_tensor", action="store_true", help="Whether to return result tensor. (Useful for comfyui)")
    parser.add_argument("--target_shape", nargs="+", default=[], help="Set return video or image shape")
    parser.add_argument("--aspect_ratio", type=str, default="")

    args = parser.parse_args()
    validate_task_arguments(args)

    seed_all(args.seed)

    # set config
    config = set_config(args)
    # init input_info
    input_info = init_empty_input_info(args.task)

    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()
        set_parallel_config(config)

    print_config(config)

    with ProfilingContext4DebugL1("Total Cost"):
        # init runner
        runner = init_runner(config)
        # start to infer
        data = args.__dict__
        update_input_info_from_dict(input_info, data)
        runner.run_pipeline(input_info)

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
