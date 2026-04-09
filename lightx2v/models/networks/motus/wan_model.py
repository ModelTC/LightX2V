import json
import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from lightx2v.models.networks.motus.wan.model import WanModel
from lightx2v.models.networks.motus.wan.vae2_2 import Wan2_2_VAE

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

logger = logging.getLogger(__name__)


def _strip_known_prefixes_for_wan(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict):
        return sd
    if not any(key.startswith("dit.") for key in sd.keys()):
        return sd
    return {(key[4:] if key.startswith("dit.") else key): value for key, value in sd.items()}


class WanVideoModel(nn.Module):
    def __init__(self, model_config: Dict[str, Any], vae_path: str, device: str = "cuda", precision: str = "bfloat16"):
        super().__init__()
        self.device = torch.device(device)
        self.precision = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]
        self.wan_model = WanModel(**model_config)
        self.wan_model.to(device=self.device, dtype=self.precision)
        self.vae = Wan2_2_VAE(vae_pth=vae_path, device=self.device)

    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode(video_pixels)

    def decode_video(self, video_latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([self.vae.decode([video_latents[i]])[0] for i in range(video_latents.shape[0])], dim=0)
            # TODO: maybe can speed up with batch to tensor
            # return self.vae.model.decode(video_latents, self.vae.scale).float().clamp(-1, 1)

    @classmethod
    def from_config(cls, config_path: str, vae_path: str, device: str = "cuda", precision: str = "bfloat16"):
        config_json_path = os.path.join(config_path, "config.json")
        with open(config_json_path, "r") as file:
            model_config = json.load(file)
        return cls(model_config=model_config, vae_path=vae_path, device=device, precision=precision)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, vae_path: str, config_path: Optional[str] = None, device: str = "cuda", precision: str = "bfloat16"):
        config_path = config_path or checkpoint_path
        config_json_path = os.path.join(config_path, "config.json")
        with open(config_json_path, "r") as file:
            model_config = json.load(file)
        model = cls(model_config=model_config, vae_path=vae_path, device=device, precision=precision)

        if checkpoint_path.endswith(".pt"):
            loaded = torch.load(checkpoint_path, map_location="cpu")
            state_dict = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        elif checkpoint_path.endswith(".bin") or checkpoint_path.endswith(".safetensors"):
            if checkpoint_path.endswith(".safetensors"):
                if safe_load_file is None:
                    raise RuntimeError("safetensors is not installed")
                state_dict = safe_load_file(checkpoint_path, device="cpu")
            else:
                loaded = torch.load(checkpoint_path, map_location="cpu")
                state_dict = loaded.get("state_dict", loaded.get("model", loaded)) if isinstance(loaded, dict) else loaded
        else:
            loaded_model = WanModel.from_pretrained(checkpoint_path)
            model.wan_model.load_state_dict(loaded_model.state_dict(), strict=False)
            return model

        state_dict = _strip_known_prefixes_for_wan(state_dict)
        model.wan_model.load_state_dict(state_dict, strict=False)
        return model
