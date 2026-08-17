SUPPORTED_TASKS = {"t2t", "t2i", "ti2t", "ti2i", "i2i"}
SUPPORTED_BOT_TASKS = {"image", "auto", "think", "recaption", "think_recaption"}


def _config_ints(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if value is None:
        return []
    return [int(value)]


def normalize_hunyuan_image3_phase_parallel(config, parallel_config):
    moe_backend = config["moe_backend"]

    phase_keys_present = any(key in parallel_config for key in ("storage_tensor_p_size", "ar", "denoise"))
    raw_phase_aware = parallel_config.get("phase_aware")
    if raw_phase_aware is None:
        phase_aware = phase_keys_present
    elif not isinstance(raw_phase_aware, bool):
        raise ValueError(f"HunyuanImage3 parallel.phase_aware must be a boolean, got {raw_phase_aware!r}.")
    else:
        phase_aware = raw_phase_aware

    if phase_keys_present and not phase_aware:
        raise ValueError("HunyuanImage3 phase-specific parallel fields require parallel.phase_aware=true.")
    if not phase_aware:
        if moe_backend == "multi_micro":
            raise ValueError("HunyuanImage3 moe_backend='multi_micro' requires parallel.phase_aware=true.")
        parallel_config["phase_aware"] = False
        return False

    ar_config = parallel_config.get("ar")
    denoise_config = parallel_config.get("denoise")
    if not isinstance(ar_config, dict) or not isinstance(denoise_config, dict):
        raise ValueError("HunyuanImage3 phase-aware parallelism requires parallel.ar and parallel.denoise objects.")
    ar_config = dict(ar_config)
    denoise_config = dict(denoise_config)

    legacy_tensor_p_size = parallel_config.get("tensor_p_size")
    legacy_seq_p_size = parallel_config.get("seq_p_size")
    denoise_tensor_p_size = int(denoise_config.get("tensor_p_size", legacy_tensor_p_size or 1))
    denoise_seq_p_size = int(denoise_config.get("seq_p_size", legacy_seq_p_size or 1))
    if legacy_tensor_p_size is not None and int(legacy_tensor_p_size) != denoise_tensor_p_size:
        raise ValueError(
            f"HunyuanImage3 phase-aware parallel.tensor_p_size is a denoise compatibility alias and must equal parallel.denoise.tensor_p_size ({denoise_tensor_p_size}), got {legacy_tensor_p_size}."
        )
    if legacy_seq_p_size is not None and int(legacy_seq_p_size) != denoise_seq_p_size:
        raise ValueError(f"HunyuanImage3 phase-aware parallel.seq_p_size is a denoise compatibility alias and must equal parallel.denoise.seq_p_size ({denoise_seq_p_size}), got {legacy_seq_p_size}.")

    storage_tensor_p_size = int(parallel_config.get("storage_tensor_p_size", denoise_tensor_p_size))
    ar_tensor_p_size = int(ar_config.get("tensor_p_size", storage_tensor_p_size * denoise_seq_p_size))
    ar_seq_p_size = int(ar_config.get("seq_p_size", 1))
    named_sizes = {
        "parallel.storage_tensor_p_size": storage_tensor_p_size,
        "parallel.ar.tensor_p_size": ar_tensor_p_size,
        "parallel.ar.seq_p_size": ar_seq_p_size,
        "parallel.denoise.tensor_p_size": denoise_tensor_p_size,
        "parallel.denoise.seq_p_size": denoise_seq_p_size,
    }
    invalid_sizes = {name: value for name, value in named_sizes.items() if value < 1}
    if invalid_sizes:
        raise ValueError(f"HunyuanImage3 phase parallel sizes must be >= 1, got {invalid_sizes}.")
    if ar_seq_p_size != 1:
        raise ValueError(f"HunyuanImage3 AR phase supports tensor parallel only; parallel.ar.seq_p_size must be 1, got {ar_seq_p_size}.")
    if storage_tensor_p_size != denoise_tensor_p_size:
        raise ValueError(f"HunyuanImage3 storage TP must equal denoise TP so denoise can directly reuse AB shards: storage={storage_tensor_p_size}, denoise_tp={denoise_tensor_p_size}.")
    if ar_tensor_p_size % storage_tensor_p_size:
        raise ValueError(f"HunyuanImage3 parallel.ar.tensor_p_size ({ar_tensor_p_size}) must be divisible by parallel.storage_tensor_p_size ({storage_tensor_p_size}).")

    micro_shard_count = ar_tensor_p_size // storage_tensor_p_size
    if micro_shard_count != denoise_seq_p_size:
        raise ValueError(f"HunyuanImage3 AR micro-shards must match denoise SP replicas: ar_tp/storage_tp={micro_shard_count}, denoise_sp={denoise_seq_p_size}.")
    if ar_tensor_p_size != denoise_tensor_p_size * denoise_seq_p_size:
        raise ValueError(f"HunyuanImage3 AR and denoise phases must cover the same ranks: ar_tp={ar_tensor_p_size}, denoise_tp={denoise_tensor_p_size}, denoise_sp={denoise_seq_p_size}.")

    if moe_backend == "multi_micro":
        if micro_shard_count != 2:
            raise ValueError(f"HunyuanImage3 moe_backend='multi_micro' requires exactly two micro shards; got {micro_shard_count}.")
        num_experts = _config_ints(config.get("num_experts"))
        top_k = _config_ints(config.get("moe_topk"))
        hidden_size = int(config.get("hidden_size", 0))
        intermediate_sizes = _config_ints(config.get("moe_intermediate_size"))
        if not num_experts or any(value != 64 for value in num_experts):
            raise ValueError(f"HunyuanImage3 multi_micro requires 64 experts, got {num_experts}.")
        if not top_k or any(value != 8 for value in top_k):
            raise ValueError(f"HunyuanImage3 multi_micro requires top-8 routing, got {top_k}.")
        if hidden_size != 4096:
            raise ValueError(f"HunyuanImage3 multi_micro requires hidden_size=4096, got {hidden_size}.")
        logical_tp_size = storage_tensor_p_size * micro_shard_count
        invalid_intermediate = [value for value in intermediate_sizes if value % logical_tp_size or value // logical_tp_size != 768]
        if not intermediate_sizes or invalid_intermediate:
            raise ValueError(f"HunyuanImage3 multi_micro requires moe_intermediate_size / (storage_tensor_p_size * micro_shard_count) = 768, got {intermediate_sizes}.")

    divisibility_checks = {
        "num_attention_heads": _config_ints(config.get("num_attention_heads") or config.get("num_heads")),
        "num_key_value_heads": _config_ints(config.get("num_key_value_heads") or config.get("num_attention_heads") or config.get("num_heads")),
        "intermediate_size": _config_ints(config.get("intermediate_size")),
        "moe_intermediate_size": _config_ints(config.get("moe_intermediate_size")),
        "vocab_size": _config_ints(config.get("vocab_size")),
    }
    shared_experts = _config_ints(config.get("num_shared_expert"))
    moe_intermediate = _config_ints(config.get("moe_intermediate_size"))
    if shared_experts and moe_intermediate:
        if len(shared_experts) == 1:
            shared_experts *= len(moe_intermediate)
        if len(moe_intermediate) == 1:
            moe_intermediate *= len(shared_experts)
        divisibility_checks["shared_mlp_intermediate_size"] = [experts * intermediate for experts, intermediate in zip(shared_experts, moe_intermediate)]

    for name, values in divisibility_checks.items():
        invalid = sorted({value for value in values if value % ar_tensor_p_size})
        if invalid:
            raise ValueError(f"HunyuanImage3 AR TP size {ar_tensor_p_size} must divide every {name}; invalid values: {invalid}.")

    ar_config["tensor_p_size"] = ar_tensor_p_size
    ar_config["seq_p_size"] = ar_seq_p_size
    denoise_config["tensor_p_size"] = denoise_tensor_p_size
    denoise_config["seq_p_size"] = denoise_seq_p_size
    parallel_config.update(
        phase_aware=True,
        storage_tensor_p_size=storage_tensor_p_size,
        micro_shard_count=micro_shard_count,
        ar=ar_config,
        denoise=denoise_config,
        tensor_p_size=denoise_tensor_p_size,
        seq_p_size=denoise_seq_p_size,
    )
    return True


def _normalize_parallel_config(config, parallel_config, task):
    phase_aware = normalize_hunyuan_image3_phase_parallel(config, parallel_config)
    tensor_p_size = int(parallel_config.get("tensor_p_size", 1))
    cfg_p_size = int(parallel_config.get("cfg_p_size", 1))
    seq_p_size = int(parallel_config.get("seq_p_size", 1))

    nested_pipeline_parallel = parallel_config.get("pipeline_parallel")
    legacy_pipeline_parallel = config.get("pipeline_parallel")
    if nested_pipeline_parallel is not None and legacy_pipeline_parallel is not None and nested_pipeline_parallel != legacy_pipeline_parallel:
        raise ValueError(f"Conflicting HunyuanImage3 pipeline settings: parallel.pipeline_parallel={nested_pipeline_parallel!r}, pipeline_parallel={legacy_pipeline_parallel!r}.")
    pipeline_parallel = nested_pipeline_parallel if nested_pipeline_parallel is not None else legacy_pipeline_parallel
    if pipeline_parallel is None:
        pipeline_parallel = True
    if not isinstance(pipeline_parallel, bool):
        raise ValueError(f"HunyuanImage3 parallel.pipeline_parallel must be a boolean, got {pipeline_parallel!r}.")

    nested_cfg_mode = parallel_config.get("cfg_mode")
    legacy_cfg_mode = config.get("hunyuan_cfg_mode")
    if nested_cfg_mode is not None and legacy_cfg_mode is not None:
        if str(nested_cfg_mode).strip().lower() != str(legacy_cfg_mode).strip().lower():
            raise ValueError(f"Conflicting HunyuanImage3 CFG modes: parallel.cfg_mode={nested_cfg_mode!r}, hunyuan_cfg_mode={legacy_cfg_mode!r}.")
    cfg_mode = nested_cfg_mode if nested_cfg_mode is not None else legacy_cfg_mode
    cfg_mode = str(cfg_mode or "batch").strip().lower()
    if cfg_mode not in ("batch", "serial", "parallel"):
        raise ValueError(f"HunyuanImage3 parallel.cfg_mode must be one of batch/serial/parallel, got {cfg_mode!r}.")

    if tensor_p_size < 1:
        raise ValueError(f"HunyuanImage3 parallel.tensor_p_size must be >= 1, got {tensor_p_size}.")
    if cfg_p_size not in (1, 2):
        raise ValueError(f"HunyuanImage3 parallel.cfg_p_size must be 1 or 2, got {cfg_p_size}.")
    if phase_aware and cfg_p_size != 1:
        raise ValueError("HunyuanImage3 phase-aware TP4/TP2+SP2 uses all ranks and requires parallel.cfg_p_size=1; use serial CFG.")
    if seq_p_size < 1:
        raise ValueError(f"HunyuanImage3 parallel.seq_p_size must be >= 1, got {seq_p_size}.")
    if tensor_p_size > 1:
        if pipeline_parallel:
            raise ValueError("HunyuanImage3 tensor parallel requires parallel.pipeline_parallel=false.")
    if cfg_p_size == 2 and not config.get("enable_cfg", False):
        raise ValueError("HunyuanImage3 parallel.cfg_p_size=2 requires enable_cfg=true.")
    if task in {"t2t", "ti2t"} and cfg_p_size != 1:
        raise ValueError(f"HunyuanImage3 task={task} requires parallel.cfg_p_size=1.")

    parallel_config.update(
        tensor_p_size=tensor_p_size,
        cfg_p_size=cfg_p_size,
        seq_p_size=seq_p_size,
        pipeline_parallel=pipeline_parallel,
        cfg_mode=cfg_mode,
    )
    if seq_p_size > 1:
        attn_type = str(parallel_config.get("seq_p_attn_type", "kv_all_gather")).strip().lower().replace("-", "_")
        if attn_type in ("kv_allgather", "kv_gather"):
            attn_type = "kv_all_gather"
        if attn_type not in ("kv_all_gather", "ulysses"):
            raise ValueError(f"HunyuanImage3 sequence parallel attention must be 'kv_all_gather' or 'ulysses', got {attn_type!r}.")
        parallel_config["seq_p_attn_type"] = attn_type

    if cfg_p_size == 2 and cfg_mode != "parallel":
        raise ValueError("HunyuanImage3 parallel.cfg_p_size=2 requires parallel.cfg_mode='parallel'.")
    if cfg_p_size == 1 and seq_p_size > 1 and config.get("enable_cfg", False) and cfg_mode != "serial":
        raise ValueError("HunyuanImage3 sequence parallel with cfg_p_size=1 requires parallel.cfg_mode='serial'.")

    if seq_p_size > 1 and parallel_config["seq_p_attn_type"] == "ulysses":
        q_heads = int(config.get("num_attention_heads") or config["num_heads"])
        kv_heads = int(config.get("num_key_value_heads") or q_heads)
        combined_head_parallel_size = tensor_p_size * seq_p_size
        if q_heads % combined_head_parallel_size or kv_heads % combined_head_parallel_size:
            raise ValueError(f"HunyuanImage3 Ulysses requires tensor_p_size * seq_p_size to divide Q and KV heads: Q={q_heads}, KV={kv_heads}, tensor_p_size={tensor_p_size}, seq_p_size={seq_p_size}.")

    return phase_aware, pipeline_parallel, cfg_mode


def normalize_hunyuan_image3_config(config):
    moe_backend = config["moe_backend"]
    task = str(config.get("task", "t2i")).strip().lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"HunyuanImage3 task must be one of {sorted(SUPPORTED_TASKS)}, got {task!r}.")

    bot_task = str(config.get("bot_task", "image")).strip().lower()
    if bot_task not in SUPPORTED_BOT_TASKS:
        raise ValueError(f"HunyuanImage3 bot_task must be one of {sorted(SUPPORTED_BOT_TASKS)}, got {bot_task!r}.")
    config["bot_task"] = bot_task

    if task in {"t2t", "ti2t"}:
        if config.get("enable_cfg", False):
            raise ValueError(f"HunyuanImage3 task={task} does not support diffusion CFG; set enable_cfg=false.")
        if bot_task == "image":
            raise ValueError(f"HunyuanImage3 task={task} requires a text bot_task such as 'auto' or 'think_recaption'.")

    if "vae_scale_factor" not in config:
        vae_downsample_factor = config.get("vae_downsample_factor")
        if isinstance(vae_downsample_factor, list) and vae_downsample_factor:
            config["vae_scale_factor"] = int(vae_downsample_factor[0])

    parallel_config = config.get("parallel")
    if not isinstance(parallel_config, dict):
        if moe_backend == "multi_micro":
            raise ValueError("HunyuanImage3 moe_backend='multi_micro' requires a phase-aware parallel configuration.")
        return config

    parallel_config = dict(parallel_config)
    phase_aware, pipeline_parallel, cfg_mode = _normalize_parallel_config(config, parallel_config, task)
    config["parallel"] = parallel_config
    config["pipeline_parallel"] = pipeline_parallel
    config["hunyuan_cfg_mode"] = cfg_mode
    config["hunyuan_image3_phase_aware_parallel"] = phase_aware
    return config


__all__ = [
    "normalize_hunyuan_image3_config",
    "normalize_hunyuan_image3_phase_parallel",
]
