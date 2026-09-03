import argparse

import torch
from loguru import logger

from lightx2v_train.data import build_data, build_sample_processor
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import (
    cleanup_distributed,
    init_distributed,
    load_config,
    setup_logger,
)
from lightx2v_train.trainers import build_trainer
from lightx2v_train.utils.utils import check_val_is_enabled, is_train_cache_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train generation models with LightX2V.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    init_distributed(config)
    setup_logger(config)

    try:
        uses_cache_dataset = is_train_cache_dataset(config)
        val_is_enabled = check_val_is_enabled(config)

        sample_processor = build_sample_processor(config)
        dataloader_train = build_data(config, train_or_val="train", sample_processor=sample_processor)
        dataloader_val = build_data(config, train_or_val="val", sample_processor=sample_processor) if val_is_enabled else None

        model = build_model(config)
        model.load_components(
            load_transformer=True,
            load_vae=(not uses_cache_dataset) or val_is_enabled,
            load_condition_encoder=not uses_cache_dataset,
        )

        trainer = build_trainer(config)
        trainer.set_model(model)
        trainer.set_data(dataloader_train, dataloader_val)

        trainer.train()
    except Exception:
        logger.exception("Training failed")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
