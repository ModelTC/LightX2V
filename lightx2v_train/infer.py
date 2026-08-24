import argparse
import copy
from pathlib import Path

import torch
from loguru import logger

from lightx2v_train.data import build_data, build_sample_processor
from lightx2v_train.infer import build_inferencer
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import cleanup_distributed, init_distributed, load_config, setup_logger
from lightx2v_train.runtime.fsdp import apply_fsdp2, fsdp2_enabled


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained LightX2V model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def _load_full_checkpoint_for_infer(model, model_config):
    checkpoint_path = model_config.get("checkpoint_path")
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "model_state.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint_path not found: {path}")

    state_dict = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict):
        for key in ("model", "generator", "state_dict"):
            value = state_dict.get(key)
            if isinstance(value, dict):
                state_dict = value
                break

    fixed_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("_fsdp_wrapped_module.", "")
        key = key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("_orig_mod.", "")
        if key.startswith("model."):
            key = key[len("model.") :]
        if key.startswith("transformer."):
            key = key[len("transformer.") :]
        fixed_state_dict[key] = value

    strict = model_config.get("checkpoint_strict", True)
    incompatible = model.denoiser_module().load_state_dict(fixed_state_dict, strict=strict)
    logger.info("Loaded inference checkpoint from {} strict={}", path, strict)
    if not strict and incompatible:
        logger.warning("Checkpoint load incompatible keys: {}", incompatible)


def _build_low_model_for_dual_infer(config, reference_model):
    model_config = config.get("model", {})
    low_override = model_config.get("student_2")
    if not isinstance(low_override, dict):
        raise ValueError("wan_t2v_dual_infer requires model.student_2.")
    role_names = {
        "student",
        "fake",
        "teacher",
        "student_2",
        "fake_2",
        "teacher_2",
        "checkpoint_path",
        "checkpoint_strict",
    }
    low_config = copy.deepcopy(config)
    low_config["model"] = {key: copy.deepcopy(value) for key, value in model_config.items() if key not in role_names}
    low_config["model"].update(copy.deepcopy(low_override))
    low_model = build_model(low_config)
    low_model.load_components(
        load_transformer=True,
        load_vae=False,
        load_condition_encoder=False,
    )
    low_model.reuse_frozen_components_from(reference_model)
    _load_full_checkpoint_for_infer(
        low_model,
        low_config["model"],
    )
    apply_fsdp2(low_model, config)
    return low_model


def main():
    args = parse_args()
    config = load_config(args.config)
    init_distributed(config)
    setup_logger(config)

    try:
        sample_processor = build_sample_processor(config)
        model = build_model(config)
        model.load_components(
            load_transformer=True,
            load_vae=True,
            load_condition_encoder=True,
        )
        _load_full_checkpoint_for_infer(
            model,
            config.get("model", {}),
        )

        lora_config = config.get("inference", {}).get("lora_config", None)
        lora_path = lora_config.get("path", None) if lora_config else None
        if fsdp2_enabled(config) and lora_path:
            model.load_lora_for_infer(lora_path)
        apply_fsdp2(model, config)

        dataloader_val = build_data(
            config,
            train_or_val="val",
            sample_processor=sample_processor,
        )

        inferencer = build_inferencer(config)
        inferencer.set_model(model)
        if hasattr(inferencer, "set_low_model"):
            inferencer.set_low_model(_build_low_model_for_dual_infer(config, model))
        inferencer.set_data(dataloader_val)

        inferencer.infer()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
