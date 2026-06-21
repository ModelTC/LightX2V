from dataclasses import dataclass

from lightx2v.common.flashinfer_autotune import (
    FlashInferAutotune,
    fi_autotune_cache_path,
)

MOE_FI_CACHE_NAMESPACE = "neopp_moe"
MOE_FI_FORCE_RETUNE_ENV = "LIGHTX2V_MOE_FI_FORCE_RETUNE"


def build_moe_model_sig(config) -> str:
    llm = config["llm_config"]
    hidden = int(llm["hidden_size"])
    intermediate = int(llm.get("moe_intermediate_size", llm.get("gen_moe_intermediate_size", 0)))
    num_experts = int(llm["gen_num_experts"])
    top_k = int(llm["num_experts_per_tok"])
    return f"neopp_moe_e{num_experts}_k{top_k}_h{hidden}_i{intermediate}_swiglu"


def moe_fi_autotune_cache(config) -> str:
    return fi_autotune_cache_path(MOE_FI_CACHE_NAMESPACE, build_moe_model_sig(config))


@dataclass
class MoeFiAutotune(FlashInferAutotune):
    tune_max_num_tokens: int = 8192

    @classmethod
    def from_neopp_config(cls, config) -> "MoeFiAutotune":
        fi_cfg = config.get("moe_flashinfer_setting") or {}
        tune_max = int(fi_cfg.get("tune_max_num_tokens", 8192))
        if config.get("version", "moe") != "moe" or config.get("moe_backend") != "flashinfer":
            return cls(tune_max_num_tokens=tune_max)
        if not fi_cfg.get("autotune", False):
            return cls(tune_max_num_tokens=tune_max)
        return cls(
            enabled=True,
            cache_path=moe_fi_autotune_cache(config),
            tune_max_num_tokens=tune_max,
            force_retune_env=MOE_FI_FORCE_RETUNE_ENV,
            log_prefix="Flashinfer MoE autotune",
        )
