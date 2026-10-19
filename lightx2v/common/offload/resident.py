"""Static, uniformly distributed resident blocks; no runtime calibration."""

import torch
from loguru import logger

from lightx2v_platform.base.global_var import AI_DEVICE


def uniform_resident_indices(num_blocks, count):
    if type(count) is not int or not 0 <= count <= num_blocks:
        raise ValueError(f"offload_resident_blocks must be an integer between 0 and {num_blocks}")
    return tuple(k * num_blocks // count for k in range(count))


def validate_residency_config(config):
    if not config.get("cpu_offload", False) or config.get("offload_granularity", "block") != "block":
        raise ValueError("Static residency requires CPU block offload")
    if AI_DEVICE not in ("cuda", "npu"):
        raise ValueError("Static residency requires CUDA or NPU stream events")
    # These paths change the block source, execution order, or model state.
    incompatible = (
        "lazy_load",
        "shared_cpu_weights",
        "parallel",
        "seq_parallel",
        "tensor_parallel",
        "cfg_parallel",
        "pipefusion_parallel",
        "lora_configs",
        "lora_dynamic_apply",
        "adapter_model_path",
        "changing_resolution",
        "use_compile",
        "enable_cuda_graph",
    )
    if any(config.get(key, False) for key in incompatible) or config.get("feature_caching", "NoCaching") != "NoCaching":
        raise ValueError("Static residency requires single-device eager execution with fixed weights, fixed resolution, and NoCaching")


class ResidentBlocks:
    def __init__(self, count):
        self.count = count
        self.indices = ()
        self.active = False
        self.originals = []

    def prepare(self, blocks):
        if self.active:
            return
        self.indices = uniform_resident_indices(len(blocks), self.count)
        self.active = True
        try:
            for index in self.indices:
                block = blocks[index]
                for _, leaf in block.named_weight_leaves():
                    # to_cuda creates active attributes from pin_* CPU masters.
                    names = {key[4:] for key in vars(leaf) if key.startswith("pin_")}
                    names.update(key for key, value in vars(leaf).items() if isinstance(value, torch.Tensor) and value.device.type == "cpu" and not key.startswith("pin_"))
                    self.originals.append((leaf, {key: getattr(leaf, key, None) for key in names}))
                block.to_cuda_async()
        except BaseException:
            self.release()
            raise
        logger.info("Static residency: resident={}, streamed={}, indices={}", self.count, len(blocks) - self.count, list(self.indices))

    def release(self):
        if not self.active:
            return
        getattr(torch, AI_DEVICE).synchronize()
        for leaf, values in self.originals:
            for key, value in values.items():
                setattr(leaf, key, value)
        self.originals.clear()
        self.active = False
