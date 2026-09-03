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


def parse_args():
    parser = argparse.ArgumentParser(description="Build a dataset cache with LightX2V.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--output_dir", required=True, help="Directory for cache_data.jsonl, cache_meta.json, and cached tensors.")
    parser.add_argument(
        "--data_split",
        choices=("train", "val"),
        default="train",
        help="Dataset split to cache: train or val. Use a separate output_dir for each split.",
    )
    parser.add_argument("--save_dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_split = str(args.data_split)

    config["cache_build"] = {
        "output_dir": args.output_dir,
        "save_dtype": args.save_dtype,
        "seed": args.seed,
        "overwrite": args.overwrite,
        "data_split": data_split,
    }
    config["data"][data_split]["preserve_records"] = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    init_distributed(config)
    setup_logger(config)

    try:
        sample_processor = build_sample_processor(config)
        dataloader = build_data(config, train_or_val=data_split, sample_processor=sample_processor)

        model = build_model(config)
        model.load_components(
            load_transformer=False,
            load_vae=True,
            load_condition_encoder=True,
        )

        trainer = build_trainer(config)
        trainer.set_model(model)
        trainer.set_data(dataloader)
        trainer.train()
    except Exception:
        logger.exception("Cache build failed")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
