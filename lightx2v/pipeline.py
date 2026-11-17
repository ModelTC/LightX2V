import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.set_config import build_startup_config


class LightX2VPipeline:
    """Load a model from a config dictionary and reuse it across requests."""

    def __init__(self, config):
        from lightx2v.models.runners.runner_factory import build_runner
        from lightx2v.utils.config_logging import print_config
        from lightx2v.utils.parallel import init_parallel
        from lightx2v.utils.utils import validate_config_paths
        from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

        self.runner = None
        self._default_task = config.get("task")
        self._owns_process_group = False
        try:
            config = build_startup_config(config)
            validate_config_paths(config)
            if config.get("parallel", False):
                self._owns_process_group = not dist.is_initialized()
                PLATFORM_DEVICE_REGISTER[os.getenv("PLATFORM", "cuda")].init_parallel_env()
                init_parallel(config)
            print_config(config, title="Startup config")
            self.runner = build_runner(config)
        except BaseException:
            self.close()
            raise
        logger.info(f"Initialized {config['model_cls']} runner; supported tasks: {', '.join(self.runner.supported_tasks)}")

    @property
    def config(self):
        return self.runner.config

    @torch.no_grad()
    def generate(self, **inputs):
        """Validate request inputs in the runner and return its result."""
        if self.runner is None:
            raise RuntimeError("Pipeline is closed")
        if inputs.get("task") is None and self._default_task is not None:
            inputs["task"] = self._default_task
        input_info = self.runner.prepare_request(inputs)
        return self.runner.run_request(input_info)

    def close(self):
        runner, self.runner = self.runner, None
        try:
            if runner is not None:
                runner.close()
        finally:
            if self._owns_process_group:
                self._owns_process_group = False
                if dist.is_initialized():
                    dist.destroy_process_group()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
