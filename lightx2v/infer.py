import argparse

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroupNCCL
from loguru import logger

from lightx2v.common.ops import *
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner  # noqa: F401
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_animate_runner import WanAnimateRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, WanAudioRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_matrix_game2_runner import WanSFMtxg2Runner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_vace_runner import WanVaceRunner  # noqa: F401
from lightx2v.utils.envs import *
from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all


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
            "wan2.2_animate",
            "hunyuan_video_1.5",
        ],
        default="wan2.1",
    )

    parser.add_argument("--task", type=str, choices=["t2v", "i2v", "t2i", "i2i", "flf2v", "vace", "animate", "s2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sf_model_path", type=str, required=False)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")

    parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument("--image_path", type=str, default="", help="The path to input image file for image-to-video (i2v) task")
    parser.add_argument("--last_frame_path", type=str, default="", help="The path to last frame file for first-last-frame-to-video (flf2v) task")
    parser.add_argument("--audio_path", type=str, default="", help="The path to input audio file or directory for audio-to-video (s2v) task")

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

    parser.add_argument("--save_result_path", type=str, default=None, help="The path to save video path/file")
    parser.add_argument("--return_result_tensor", action="store_true", help="Whether to return result tensor. (Useful for comfyui)")
    args = parser.parse_args()

    seed_all(args.seed)

    # set config
    config = set_config(args)

    if config["parallel"]:
        run_device = config.get("run_device", "cuda")
        if "cuda" in run_device:
            pg_options = ProcessGroupNCCL.Options()
            pg_options.is_high_priority_stream = True
            dist.init_process_group(backend="nccl", pg_options=pg_options)
            torch.cuda.set_device(dist.get_rank())
        elif "mlu" in run_device:
            dist.init_process_group(backend="cncl")
            torch.mlu.set_device(dist.get_rank())
        set_parallel_config(config)

    print_config(config)

    with ProfilingContext4DebugL1("Total Cost"):
        runner = init_runner(config)
        input_info = set_input_info(args)
        runner.run_pipeline(input_info)

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
