import argparse

import torch.distributed as dist
from loguru import logger

from lightx2v.common.ops import *
from lightx2v.models.runners.cogvideox.cogvidex_runner import CogvideoxRunner  # noqa: F401
from lightx2v.models.runners.graph_runner import GraphRunner
from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_audio_runner import Wan22MoeAudioRunner, WanAudioRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_causvid_runner import WanCausVidRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_skyreels_v2_df_runner import WanSkyreelsV2DFRunner  # noqa: F401
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config
from lightx2v.utils.utils import seed_all


def init_runner(config):
    seed_all(config.seed)

    if CHECK_ENABLE_GRAPH_MODE():
        default_runner = RUNNER_REGISTER[config.model_cls](config)
        default_runner.init_modules()
        runner = GraphRunner(default_runner)
    else:
        runner = RUNNER_REGISTER[config.model_cls](config)
        runner.init_modules()
    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_cls",
        type=str,
        required=True,
        choices=["wan2.1", "hunyuan", "wan2.1_distill", "wan2.1_causvid", "wan2.1_skyreels_v2_df", "cogvideox", "wan2.1_audio", "wan2.2_moe", "wan2.2_moe_audio", "wan2.2"],
        default="wan2.1",
    )

    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")

    parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument("--image_path", type=str, default="", help="The path to input image file for image-to-video (i2v) task")
    parser.add_argument("--audio_path", type=str, default="", help="The path to input audio file for audio-to-video (a2v) task")

    parser.add_argument("--save_video_path", type=str, default="./output_lightx2v.mp4", help="The path to save video path/file")
    args = parser.parse_args()

    logger.info(f"args: {args}")

    with ProfilingContext("Total Cost"):
        config = set_config(args)
        print_config(config)
        runner = init_runner(config)

        runner.run_pipeline()

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
