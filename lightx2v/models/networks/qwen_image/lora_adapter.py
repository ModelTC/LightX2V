import gc

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

    def _load_lora_file(self, file_path):
        with safe_open(file_path, framework="pt") as f:
            tensor_dict = {key: f.get_tensor(key).to(GET_DTYPE()).to(self.device) for key in f.keys()}
        return tensor_dict

    def apply_lora(self, lora_configs):
        if not hasattr(self.model, "original_weight_dict"):
            logger.error("Model does not have 'original_weight_dict'. Cannot apply LoRA.")
            return False

        for lora_config in lora_configs:
            lora_weights = self._load_lora_file(lora_config["path"])
            lora_strength = lora_config.get("strength", 1.0)
            self.lora_loader.apply_lora(
                weight_dict=self.model.original_weight_dict,
                lora_weights=lora_weights,
                strength=lora_strength,
            )
            logger.info(f"Applied LoRA: {lora_config['path']} with strength={lora_strength}")
            del lora_weights
            gc.collect()

        self.model._apply_weights(self.model.original_weight_dict)
        gc.collect()
        torch.cuda.empty_cache()
