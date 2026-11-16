"""Initialize device communication and parallel execution."""

import os

from loguru import logger


def configure_nccl_graph(config):
    """Configure Ulysses graph communication before creating process groups."""
    import torch

    from lightx2v_platform.base.global_var import AI_DEVICE

    parallel = config.get("parallel") or {}
    if AI_DEVICE != "cuda" or not config.get("use_cuda_graph", False) or parallel.get("seq_p_size", 1) <= 1 or parallel.get("seq_p_attn_type", "ulysses") != "ulysses":
        return

    # Older NCCL reuses incomplete IPC mappings for cuMem-backed buffer views.
    # https://github.com/pytorch/torchtitan/pull/5393
    # https://github.com/NVIDIA/nccl/pull/2362
    nccl_version = torch.cuda.nccl.version()
    if nccl_version < (2, 32, 3) and os.environ.get("NCCL_GRAPH_REGISTER") != "0":
        logger.warning(
            f"NCCL {'.'.join(map(str, nccl_version))} has an IPC registration bug with Ulysses CUDA Graphs; "
            "setting NCCL_GRAPH_REGISTER=0. Graph and compile remain enabled. NCCL 2.32.3 fixes this issue."
        )
        os.environ["NCCL_GRAPH_REGISTER"] = "0"


def _validate_pipefusion_config(config):
    """Reject unsupported PipeFusion combinations instead of silently misbehaving.

    PipeFusion is currently a narrow feature: Flux2 Klein, T2I only, CUDA only,
    no CFG, and no stacking with SP / TP / feature-caching / cpu-offload. The
    pipeline driver only implements that slice; anything else must fail loudly
    at config time rather than run incorrectly (e.g. dropping CFG) or crash.
    """
    from lightx2v_platform.base.global_var import AI_DEVICE

    model_cls = config.get("model_cls")
    is_klein = model_cls == "flux2_klein" or (model_cls == "flux2" and config.get("model_variant") == "klein")
    if not is_klein:
        raise ValueError(
            "PipeFusion is only supported for the Flux2 Klein model "
            "(model_cls='flux2_klein', or model_cls='flux2' with model_variant='klein'); "
            f"got model_cls={model_cls!r}, model_variant={config.get('model_variant')!r}."
        )
    if config.get("task") not in (None, "t2i"):
        raise ValueError(f"PipeFusion currently supports only the 't2i' task, got {config.get('task', 't2i')!r}.")
    if AI_DEVICE != "cuda":
        raise ValueError(f"PipeFusion requires CUDA, but AI_DEVICE={AI_DEVICE!r}.")
    if config.get("feature_caching", "NoCaching") not in ("NoCaching", "None"):
        raise ValueError(f"PipeFusion cannot be combined with feature_caching={config.get('feature_caching')!r}.")
    if config.get("cpu_offload", False):
        raise ValueError("PipeFusion cannot be combined with cpu_offload.")
    if config.get("unload_modules", False) or config.get("lazy_load", False):
        raise ValueError("PipeFusion does not support unload_modules / lazy_load.")
    if config.get("fls", {}).get("enable", False):
        raise ValueError("PipeFusion cannot be combined with FLS enhancement (fls.enable).")
    if config.get("enable_cfg", False) and config.get("sample_guide_scale", 1.0) > 1.0:
        raise ValueError("PipeFusion does not support CFG; set sample_guide_scale <= 1.0 or enable_cfg=False.")
    if config["parallel"].get("seq_p_size", 1) > 1:
        raise ValueError("PipeFusion cannot be combined with sequence parallel (seq_p_size > 1).")
    if config["parallel"].get("cfg_p_size", 1) > 1:
        raise ValueError("PipeFusion cannot be combined with CFG parallel (cfg_p_size > 1).")

    num_patch = int(config["parallel"].get("num_pipeline_patch", 4))
    if num_patch <= 0:
        raise ValueError(f"num_pipeline_patch must be >= 1, got {num_patch}.")
    warmup_steps = int(config["parallel"].get("pipeline_warmup_steps", 1))
    if warmup_steps <= 0:
        raise ValueError(f"pipeline_warmup_steps must be >= 1, got {warmup_steps}.")


def init_parallel(config):
    """Create the model's parallel mesh and warm up its communication."""
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor.device_mesh import init_device_mesh

    from lightx2v_platform.base.global_var import AI_DEVICE

    parallel = config.get("parallel", False)
    if not parallel or config.get("model_cls") == "swiftvr":
        # SwiftVR owns its chunk group and validates its parallel settings in the runner.
        return

    tensor_p_size = int(parallel.get("tensor_p_size", 1))
    cfg_p_size = int(parallel.get("cfg_p_size", 1))
    seq_p_size = int(parallel.get("seq_p_size", 1))
    pp_size = int(parallel.get("pp_size", 1))
    if cfg_p_size > 1 and not config.get("enable_cfg", False):
        raise ValueError("parallel.cfg_p_size > 1 requires enable_cfg=true")
    world_size = dist.get_world_size()
    expected_world_size = tensor_p_size * cfg_p_size * seq_p_size * pp_size
    if expected_world_size != world_size:
        raise ValueError(
            f"Parallel sizes must match the distributed world size: tensor_p_size ({tensor_p_size}) * cfg_p_size ({cfg_p_size}) * seq_p_size ({seq_p_size}) * pp_size ({pp_size}) != world_size ({world_size})."
        )

    if config.get("model_cls") == "hunyuan_image3" and parallel.get("phase_aware", False):
        from lightx2v.models.networks.hunyuan_image3.parallel import initialize_hunyuan_image3_parallel_runtime

        initialize_hunyuan_image3_parallel_runtime(config)
        config["pipefusion_parallel"] = False
    elif pp_size > 1:
        if tensor_p_size > 1:
            raise ValueError("PipeFusion pipeline parallelism cannot be combined with tensor parallelism")
        # PipeFusion pipeline parallelism: 3D mesh (cfg_p, pp, seq_p).
        config["device_mesh"] = init_device_mesh(AI_DEVICE, (cfg_p_size, pp_size, seq_p_size), mesh_dim_names=("cfg_p", "pp", "seq_p"))
        config["tensor_parallel"] = False
        config["seq_parallel"] = seq_p_size > 1
        config["cfg_parallel"] = bool(config.get("enable_cfg", False) and cfg_p_size > 1)
        config["pipefusion_parallel"] = True
        _validate_pipefusion_config(config)
        from lightx2v.models.networks.flux2.infer.pipefusion import init_pipeline_parallel_state

        pp_group = config["device_mesh"].get_group(mesh_dim="pp")
        init_pipeline_parallel_state(pp_group)
    else:
        # Keep the original CFG/SP dimensions without TP. With TP, omit unit
        # dimensions and place TP last so its ranks form contiguous groups.
        mesh_dims = [("cfg_p", cfg_p_size), ("seq_p", seq_p_size)]
        if tensor_p_size > 1:
            mesh_dims = [(name, size) for name, size in mesh_dims if size > 1]
            mesh_dims.append(("tensor_p", tensor_p_size))
        config["device_mesh"] = init_device_mesh(
            AI_DEVICE,
            tuple(size for _, size in mesh_dims),
            mesh_dim_names=tuple(name for name, _ in mesh_dims),
        )
        config["tensor_parallel"] = tensor_p_size > 1
        config["seq_parallel"] = seq_p_size > 1
        config["cfg_parallel"] = bool(config.get("enable_cfg", False) and cfg_p_size > 1)
        config["pipefusion_parallel"] = False

    warmup_device = f"cuda:{torch.cuda.current_device()}" if AI_DEVICE == "cuda" else AI_DEVICE
    warmup_tensor = torch.zeros([1], device=warmup_device)
    dist.all_reduce(warmup_tensor)
