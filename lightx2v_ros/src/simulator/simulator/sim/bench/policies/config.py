import json
from pathlib import Path

from omegaconf import OmegaConf


def resolve_policy_config(cfg, root, backend):
    model, evaluation = cfg.model, cfg.EVALUATION
    if not cfg.config_json:
        name = model.name or "realtimewam"
        backbone = model.backbone or ("fastwam" if name == "fastwam" else backend.DEFAULT_BACKBONE)
        filename = f"{backend.POLICY_PROFILE}_i2va.json" if name == "fastwam" else f"{backend.POLICY_PROFILE}_{backbone}_teacher.json"
        cfg.config_json = str(root / "configs" / ("fastwam" if name == "fastwam" else "realtimewam") / filename)
    profile = json.loads(Path(cfg.config_json).read_text())
    if profile["policy_profile"] != backend.POLICY_PROFILE:
        raise ValueError(f"Model profile {profile['policy_profile']} does not match benchmark {cfg.benchmark}")
    supported_samplers = {"flow_matching": "fastwam", "teacher_flow": "realtimewam"}
    if profile["sampler"] not in supported_samplers or profile["backbone"] not in ("fastwam", "fasterwam"):
        raise ValueError("Unsupported model profile sampler/backbone; baseline preconditioning is not enabled")
    policy_name = supported_samplers[profile["sampler"]]
    if model.name is None:
        model.name = policy_name
    if model.backbone is None:
        model.backbone = profile["backbone"]
    if model.sampler is None:
        model.sampler = profile["sampler"]
    mappings = {
        "model.action_horizon": "action_chunk_size",
        "model.num_video_frames": "num_video_frames",
        "EVALUATION.num_inference_steps": "action_infer_steps",
        "EVALUATION.action_infer_mode": "action_infer_mode",
        "EVALUATION.sigma_shift": "action_sample_shift",
        "EVALUATION.replan_steps": "actions_per_plan",
    }
    for key, native_key in mappings.items():
        if OmegaConf.select(cfg, key) is None:
            OmegaConf.update(cfg, key, profile[native_key])
        profile[native_key] = OmegaConf.select(cfg, key)

    if cfg.ckpt and (cfg.base_ckpt or cfg.lora_path):
        raise ValueError("Use base_ckpt + lora_path OR a merged ckpt; do not combine them")
    if cfg.lora_path and not cfg.base_ckpt:
        raise ValueError("lora_path requires base_ckpt")
    if not model.factory:
        if model.name != policy_name or model.backbone != profile["backbone"] or model.sampler != profile["sampler"]:
            raise ValueError("Model name/backbone/sampler conflict with config_json; select a matching model profile")
        if model.name == "fastwam" and (cfg.lora_path or model.backbone != "fastwam" or evaluation.action_infer_mode != "first_frame"):
            raise ValueError("model=fastwam requires flow_matching, dense backbone, first_frame and full weights (no LoRA)")
        if evaluation.compile_action_infer:
            raise ValueError("compile_action_infer is not implemented; use false")
    if evaluation.action_infer_mode not in ("first_frame", "one_pass_future_cache"):
        raise ValueError("Unsupported action_infer_mode")
    if evaluation.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    if model.lora_weights not in ("ema", "student"):
        raise ValueError("lora_weights must be ema/student")

    # Legacy interactive profiles contain settle steps; benchmark protocols own them.
    profile.pop("num_steps_wait", None)
    profile.update(
        model_cls="fastwam",
        task="i2va",
        device="cuda:0",
        model_path=model.model_path,
        adapter_model_path=cfg.base_ckpt or cfg.ckpt,
        lora_path=cfg.lora_path,
        lora_weights=model.lora_weights,
        dataset_stats_path=evaluation.dataset_stats_path,
        seed=cfg.seed,
        t5_cpu_offload=model.t5_cpu_offload,
        vae_cpu_offload=model.vae_cpu_offload,
    )
    cfg.model.native_config = profile
