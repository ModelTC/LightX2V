import os

from lightx2v.models.runners.registry import get_model_spec
from lightx2v.utils.cli import parse_cli_inputs


def main():
    args, startup_config, request_data = parse_cli_inputs()

    import torch.distributed as dist
    from loguru import logger

    from lightx2v.models.runners.runner_factory import build_runner
    from lightx2v.utils.profiler import ProfilingContext4DebugL1, no_sync_profiling
    from lightx2v.utils.set_config import configure_nccl_graph, init_parallel, print_config, print_request
    from lightx2v.utils.utils import validate_config_paths
    from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

    configure_nccl_graph(startup_config)
    if startup_config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER[os.getenv("PLATFORM", "cuda")]
        platform_device.init_parallel_env()
        init_parallel(startup_config)

    print_config(startup_config, title="Startup config")

    validate_config_paths(startup_config)

    runner = None
    try:
        with no_sync_profiling(startup_config["model_cls"] in ("taomate_h3", "streamrender_h3")), ProfilingContext4DebugL1("Total Cost"):
            runner = build_runner(startup_config)
            get_model_spec(startup_config["model_cls"]).prepare_runner(runner, args)
            input_info = runner.prepare_request(request_data)
            print_request(input_info, runner.get_supported_request_fields(input_info.task))
            runner.run_request(input_info)
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
