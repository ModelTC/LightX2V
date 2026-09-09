import torch.distributed as dist

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


def _linear(name, bias, force_fp32, config, tp_split=None):
    kind = "Default-ForceFp32" if force_fp32 else "Default"
    if config.get("tensor_parallel", False) and tp_split is not None:
        tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
        tp_mm_type = config.get("tp_mm_type", "TensorParallel")
        return MM_WEIGHT_REGISTER[tp_mm_type](
            weight_name=f"{name}.weight",
            bias_name=f"{name}.bias" if bias else None,
            mm_type=kind,
            tp_group=tp_group,
            tp_rank=dist.get_rank(tp_group),
            tp_size=dist.get_world_size(tp_group),
            split_dim=tp_split,
        )
    return MM_WEIGHT_REGISTER[kind](f"{name}.weight", f"{name}.bias" if bias else None)


def _rms(config, name, eps):
    return RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")](name, eps=eps)


class MiniMaxH3PostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        col = "col"
        self.add_module(
            "norm_out",
            _rms(config, "norm_out.norm.weight", eps=float(config.get("final_norm_eps", 1e-5))),
        )
        if not config.get("use_adaln_cache", False):
            # ADALN CACHE SYNC: The offline builder reads this key and persists
            # its output; update the offline builder if its definition changes.
            self.add_module(
                "norm_out_linear",
                _linear("norm_out.linear", bias=True, force_fp32=False, config=config, tp_split=col),
            )
        self.add_module(
            "proj_out",
            _linear("proj_out", bias=True, force_fp32=True, config=config, tp_split=col),
        )
        self.add_module(
            "audio_proj_out",
            _linear("audio_proj_out", bias=True, force_fp32=True, config=config, tp_split=col),
        )
