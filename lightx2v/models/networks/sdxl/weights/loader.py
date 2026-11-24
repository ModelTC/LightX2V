import json
from pathlib import Path

import torch

from lightx2v.models.networks.three_models.checkpoint import TensorWeights, load_tensors


class SDXLWeights:
    """Load the official base-1.0 directory, without a Diffusers pipeline."""

    def __init__(self, model_path):
        root = Path(model_path)
        self.configs = {name: json.loads((root / name / "config.json").read_text()) for name in ("unet", "vae", "text_encoder", "text_encoder_2")}
        self.scheduler_config = json.loads((root / "scheduler/scheduler_config.json").read_text())
        if self.configs["unet"]["block_out_channels"] != [320, 640, 1280]:
            raise ValueError("This adapter requires stabilityai/stable-diffusion-xl-base-1.0")
        for name in self.configs:
            dtype = torch.float32 if name == "vae" else torch.float16
            setattr(self, name, TensorWeights(load_tensors(root / name, dtype=dtype, variant="fp16"), native_requires_grad=True))
