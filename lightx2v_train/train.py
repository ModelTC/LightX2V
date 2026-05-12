import argparse

from lightx2v_train.data import build_data
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train image/video generation models.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    model = build_model(config)
    model.load_components()

    dataloader_train = build_data(config, train_or_val="train")
    dataloader_eval = build_data(config, train_or_val="val")

    trainer = build_trainer(config)
    trainer.set_model(model)
    trainer.set_data(dataloader_train, dataloader_eval)

    trainer.train()


if __name__ == "__main__":
    main()
