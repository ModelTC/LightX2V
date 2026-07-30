from dataclasses import dataclass

from lightx2v.common.flashinfer_autotune import (
    FlashInferAutotune,
    fi_autotune_cache_path,
)

MOE_FI_CACHE_NAMESPACE = "hunyuan3d_moe"
MOE_FI_FORCE_RETUNE_ENV = "LIGHTX2V_HUNYUAN3D_MOE_FI_FORCE_RETUNE"


def _moe_intermediate(config) -> int:
    hidden = int(config["hidden_size"])
    for key in ("moe_intermediate_size", "intermediate_size"):
        if config.get(key):
            return int(config[key])
    return int(hidden * float(config.get("mlp_ratio", 4)))


def build_moe_model_sig(config) -> str:
    hidden = int(config["hidden_size"])
    intermediate = _moe_intermediate(config)
    num_experts = int(config.get("num_experts", 8))
    top_k = int(config.get("moe_top_k", 2))
    return f"hunyuan3d_moe_e{num_experts}_k{top_k}_h{hidden}_i{intermediate}_gelu_bias"


def moe_fi_autotune_cache(config) -> str:
    return fi_autotune_cache_path(MOE_FI_CACHE_NAMESPACE, build_moe_model_sig(config))


@dataclass
class MoeFiAutotune(FlashInferAutotune):
    tune_max_num_tokens: int = 8192

    @classmethod
    def from_hunyuan3d_config(cls, config) -> "MoeFiAutotune":
        fi_cfg = config.get("moe_flashinfer_setting") or {}
        tune_max = int(fi_cfg.get("tune_max_num_tokens", 8192))
        if str(config.get("moe_backend", "pytorch")).strip().lower() != "flashinfer":
            return cls(tune_max_num_tokens=tune_max)
        if not fi_cfg.get("autotune", False):
            return cls(tune_max_num_tokens=tune_max)
        return cls(
            enabled=True,
            cache_path=moe_fi_autotune_cache(config),
            tune_max_num_tokens=tune_max,
            force_retune_env=MOE_FI_FORCE_RETUNE_ENV,
            log_prefix="Hunyuan3D Flashinfer MoE autotune",
        )
