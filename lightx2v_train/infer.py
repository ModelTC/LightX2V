import argparse
from pathlib import Path

import torch
from loguru import logger

from lightx2v_train.data import build_data
from lightx2v_train.infer import build_inferencer
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import cleanup_distributed, init_distributed, load_config, setup_logger
from lightx2v_train.runtime.fsdp import apply_fsdp2, fsdp2_enabled


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained LightX2V model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def _load_full_checkpoint_for_infer(model, config):
    model_config = config.get("model", {})
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


def main():
    args = parse_args()
    config = load_config(args.config)
    init_distributed(config)
    setup_logger(config)

    try:
        model = build_model(config)
        model.load_components()
        _load_full_checkpoint_for_infer(model, config)

        lora_config = config.get("inference", {}).get("lora_config", None)
        lora_path = lora_config.get("path", None) if lora_config else None
        if fsdp2_enabled(config) and lora_path:
            model.load_lora_for_infer(lora_path)
        apply_fsdp2(model, config)

        dataloader_val = build_data(config, train_or_val="val")

        inferencer = build_inferencer(config)
        inferencer.set_model(model)
        inferencer.set_data(dataloader_val)

        inferencer.infer()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
