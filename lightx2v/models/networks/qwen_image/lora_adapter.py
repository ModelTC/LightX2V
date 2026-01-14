import os

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.utils.envs import *
from lightx2v.utils.lora_loader import LoRALoader
from lightx2v_platform.base.global_var import AI_DEVICE


class QwenImageLoraWrapper:
    def __init__(self, qwenimage_model):
        self.model = qwenimage_model
        self.lora_metadata = {}
        self.lora_loader = LoRALoader()
        self.device = torch.device(AI_DEVICE) if not self.model.config.get("cpu_offload", False) else torch.device("cpu")

    def load_lora(self, lora_path, lora_name=None):
        if lora_name is None:
            lora_name = os.path.basename(lora_path).split(".")[0]

        if lora_name in self.lora_metadata:
            logger.info(f"LoRA {lora_name} already loaded, skipping...")
            return lora_name

        self.lora_metadata[lora_name] = {"path": lora_path}
        logger.info(f"Registered LoRA metadata for: {lora_name} from {lora_path}")

        return lora_name

    def _load_lora_file(self, file_path):
        with safe_open(file_path, framework="pt") as f:
            tensor_dict = {key: f.get_tensor(key).to(GET_DTYPE()).to(self.device) for key in f.keys()}
        return tensor_dict

    def apply_lora(self, lora_name, strength=1.0):
        if lora_name not in self.lora_metadata:
            logger.info(f"LoRA {lora_name} not found. Please load it first.")

        if not hasattr(self.model, "original_weight_dict"):
            logger.error("Model does not have 'original_weight_dict'. Cannot apply LoRA.")
            return False

        lora_weights = self._load_lora_file(self.lora_metadata[lora_name]["path"])
        weight_dict = self.model.original_weight_dict
        self.lora_loader.apply_lora(
            weight_dict=weight_dict,
            lora_weights=lora_weights,
            strength=strength,
        )
        self.model._apply_weights(weight_dict)

        logger.info(f"Applied LoRA: {lora_name} with strength={strength}")
        del lora_weights
        return True
