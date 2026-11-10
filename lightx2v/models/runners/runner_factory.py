from importlib import import_module

import torch

from lightx2v.models.runners.registry import RUNNER_MODULES
from lightx2v.utils.registry_factory import RUNNER_REGISTER


def build_runner(config):
    """Instantiate a runner and initialize its model modules."""
    model_cls = config["model_cls"]
    if model_cls != "depth_anything3":
        import_module("lightx2v.common.ops")
    import_module(RUNNER_MODULES[model_cls])

    torch.set_grad_enabled(False)
    runner = RUNNER_REGISTER[model_cls](config)
    try:
        runner.init_modules()
    except BaseException as error:
        if config.get("cpu_offload_numa", False):
            try:
                runner.close()
            except BaseException as cleanup_error:
                error.add_note(f"NUMA runner cleanup also failed: {cleanup_error!r}")
        raise
    return runner
