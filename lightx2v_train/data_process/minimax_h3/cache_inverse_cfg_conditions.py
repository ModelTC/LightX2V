"""Encode empty text with the original I2AV image, once per paired manifest record."""

import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from build_minimax_h3_prompt_latents import encode_prompt, load_conditioner, load_prepared_images
from lightx2v_train.trainers.dense2moe.data import ManifestAVDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dataset = ManifestAVDataset(cfg.prompts, cfg.seed, cfg.negative_dir)
    dataset.negative_dir.mkdir(parents=True, exist_ok=True)
    jobs = {dataset.negative_name(row): row for row in dataset.records}
    encoder = None
    for name in sorted(jobs)[rank::world]:
        path = dataset.negative_dir / name
        if path.is_file():
            continue
        if encoder is None:
            encoder = load_conditioner(cfg.condition_model_path, str(device), torch.bfloat16)
        row = jobs[name]
        images = load_prepared_images({"first_frame": row["image"]}, row["height"], row["width"]) if row["task"] == "i2av" else []
        result = encode_prompt(*encoder, " ", images, device, torch.bfloat16)
        pending = path.with_suffix(f".{rank}.tmp")
        torch.save(result, pending)
        pending.replace(path)


if __name__ == "__main__":
    main()
