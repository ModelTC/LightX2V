import asyncio
import argparse
import torch
import torch.distributed as dist
import json

from lightx2v.utils.envs import *
from lightx2v.utils.utils import seed_all
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.utils.registry_factory import RUNNER_REGISTER

from lightx2v.models.runners.magicdrive.mgcdr_runner import MagicDriverRunner
from lightx2v.models.runners.graph_runner import GraphRunner

from lightx2v.common.ops import *
from loguru import logger


def init_runner(config):
    seed_all(config.seed)

    if config.parallel_attn_type:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    if CHECK_ENABLE_GRAPH_MODE():
        default_runner = RUNNER_REGISTER[config.model_cls](config)
        runner = GraphRunner(default_runner)
        runner.runner.init_modules()
    else:
        runner = RUNNER_REGISTER[config.model_cls](config)
        runner.init_modules()
    return runner


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["mgcdr"], default="mgcdr")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--dataset_params_json", type=str, required=True)
    parser.add_argument("--camera_params_json", type=str, required=True)
    parser.add_argument("--raw_meta_files", type=str, required=True)
    # parser.add_argument("--prompt", type=str, required=True)
    # parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="./output_lightx2v.mp4", help="The path to save video path/file")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Total Cost"):
        config = set_config(args)
        config["mode"] = "infer"
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = init_runner(config)

        await runner.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
