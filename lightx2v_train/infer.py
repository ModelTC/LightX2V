import argparse

from lightx2v_train.data import build_data
from lightx2v_train.infer import build_inferencer
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import cleanup_distributed, init_distributed, load_config
from lightx2v_train.runtime.fsdp import apply_fsdp2, fsdp2_enabled


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained LightX2V model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    init_distributed(config)

    try:
        model = build_model(config)
        model.load_components()

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
