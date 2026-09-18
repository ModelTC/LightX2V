import torch.distributed as dist

from lightx2v.common.ops.mm.reproducible import MMWeightTPReproducible
from lightx2v.models.networks.minimax_h3.fp8_f16_accum_policy import (
    DIT_FP8_F16_ACCUM_ACTIVATION_QMAX,
    FP8_F16_ACCUM_PROJECTION_SUFFIXES,
)
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


def make_linear(config, name, bias=False, create_cuda_buffer=False, tp_split=None, mm_type=None, lora_prefix="transformer_blocks"):
    quant_scheme = mm_type or config.get("dit_quant_scheme", "Default")
    use_tp = config.get("tensor_parallel", False)
    reproducible = config.get("tp_reproducible", False)
    if tp_split is not None and (use_tp or reproducible):
        tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p") if use_tp else None
        linear_cls = MMWeightTPReproducible if reproducible else MM_WEIGHT_REGISTER[config.get("tp_mm_type", "TensorParallel")]
        return linear_cls(
            weight_name=f"{name}.weight",
            bias_name=f"{name}.bias" if bias else None,
            mm_type=quant_scheme,
            tp_group=tp_group,
            tp_rank=dist.get_rank(tp_group) if use_tp else 0,
            tp_size=dist.get_world_size(tp_group) if use_tp else 1,
            split_dim=tp_split,
            lora_column_chunks=3 if reproducible and name.endswith(".to_qkv") else 2 if ".ff.net.0.proj" in name else 1,
            create_cuda_buffer=create_cuda_buffer,
            lora_prefix=lora_prefix,
        )

    linear = MM_WEIGHT_REGISTER[quant_scheme](
        f"{name}.weight",
        f"{name}.bias" if bias else None,
        create_cuda_buffer=create_cuda_buffer,
        lora_prefix=lora_prefix,
    )
    if quant_scheme == "fp8-f16-accum" and name.endswith(FP8_F16_ACCUM_PROJECTION_SUFFIXES):
        linear.enable_fp8_f16_accum(DIT_FP8_F16_ACCUM_ACTIVATION_QMAX)
    return linear
