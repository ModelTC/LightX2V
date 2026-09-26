import os

import torch
import torch.distributed as dist

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("biren_supa")
class BirenSupaDevice:
    name = "biren_supa"

    @staticmethod
    def init_device_env():
        # Keep SUPA runtime defaults in the plugin so callers only need to set
        # PLATFORM=biren_supa. Explicit environment values always win.
        runtime_defaults = {
            "BRTB_ENABLE_FORCE_UMA": "1",
            "BRTB_ENABLE_FORCE_SUDNN_CONV2d": "1",
            "BRTB_ENABLE_SUPA_FALLBACK": "1",
            "BRTB_ENABLE_NCDHW": "1",
            "BRTB_ENABLE_FORCE_EAGER_CONV2D": "1",
            "BRTB_DISABLE_L2_FLUSH": "1",
            "BRTB_ENABLE_SUBLAS_API": "1",
            "BRTB_ENABLE_WEIGHT_BYPASS": "1",
            "BRTB_ENABLE_REGISTER_BEFORE_D2H": "1",
            "BRTB_DISABLE_ZERO_WS": "1",
            "BRTB_DISABLE_ZERO_REORDER": "1",
            "BRTB_DISABLE_ZERO_OUTPUT_NUMA": "1",
            "BRTB_DISABLE_ZERO_OUTPUT_UMA": "1",
            "BRTB_ENABLE_EAGER_ADV_API": "1",
            "BRTB_DUMP_MEM_POOL_WHEN_OOM": "1",
            "BRTB_ENABLE_FORCE_CONV_BB": "1",
            "BRTB_ENABLE_SP_SPLIT_IN_STAGE": "1",
            "SHAPE_TRANSFORM_MIN_GRAN": "2048",
            "TORCHDYNAMO_DISABLE": "1",
            "BRTB_LOG_BACKEND": "empty",
            "PYTORCH_SUPA_ALLOC_CONF": "max_split_size_mb:512",
            "SUDNN_EAGER_ENABLE_ALPHA_BETA": "false",
            "SUDNN_KERNEL_CACHE_CAPACITY": "30000",
            "SUDNN_KERNEL_CACHE_EXCLUDE_UID": "1",
            "SUDNN_KERNEL_CACHE_DISK_LEVEL": "3",
            "SUDNN_KERNEL_CACHE_MAX_SIZE_MB": "102400",
        }
        cache_dir = os.environ.get(
            "BIREN_SUDNN_KERNEL_CACHE_FOLDER",
            os.path.expanduser("~/.cache/lightx2v/biren/sudnn"),
        )
        runtime_defaults["SUDNN_KERNEL_CACHE_FOLDER"] = cache_dir
        for key, value in runtime_defaults.items():
            os.environ.setdefault(key, value)

        # Keep device selection configurable without baking a card list into
        # the LightX2V scripts. This must be set before torch_br is activated.
        if os.environ.get("BIREN_VISIBLE_DEVICES") and not os.environ.get("SUPA_VISIBLE_DEVICES"):
            os.environ["SUPA_VISIBLE_DEVICES"] = os.environ["BIREN_VISIBLE_DEVICES"]

        import torch_br  # noqa: F401 — activates SUPA PrivateUse1 backend
        from torch_br.contrib import transfer_to_supa  # noqa: F401
        import torch_br.supa._debug as supa_debug

        supa_debug.set_enable_sublas_api(True)
        supa_debug.set_disable_zero_ws(True)
        supa_debug.set_disable_zero_output_uma(True)
        supa_debug.set_disable_zero_output_numa(True)
        supa_debug.set_disable_reorder_zero(True)

    @staticmethod
    def is_available() -> bool:
        try:
            return hasattr(torch, "supa") and torch.supa.is_available()
        except Exception:
            return False

    @staticmethod
    def get_device() -> str:
        return "supa"

    @staticmethod
    def init_parallel_env():
        dist.init_process_group(backend="sccl")
        if hasattr(torch, "supa") and hasattr(torch.supa, "set_device"):
            torch.supa.set_device(dist.get_rank())
        else:
            torch.cuda.set_device(dist.get_rank())
