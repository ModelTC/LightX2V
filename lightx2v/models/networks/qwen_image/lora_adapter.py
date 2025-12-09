import os

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


def fuse_lora_weights(original_weight, lora_down, lora_up, alpha):
    rank = lora_down.shape[0]
    lora_delta = torch.mm(lora_up, lora_down)  # W_up × W_down
    scaling = alpha / rank
    lora_delta = lora_delta * scaling
    fused_weight = original_weight + lora_delta
    return fused_weight


class QwenImageLoraWrapper:
    def __init__(self, qwenimage_model):
        self.model = qwenimage_model
        self.lora_metadata = {}
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

    def apply_lora(self, lora_name, alpha=1.0):
        if lora_name not in self.lora_metadata:
            logger.info(f"LoRA {lora_name} not found. Please load it first.")

        if not hasattr(self.model, "original_weight_dict"):
            logger.error("Model does not have 'original_weight_dict'. Cannot apply LoRA.")
            return False

        lora_weights = self._load_lora_file(self.lora_metadata[lora_name]["path"])

        weight_dict = self.model.original_weight_dict

        weight_dict = self._apply_lora_weights(weight_dict, lora_weights, alpha)

        self.model._apply_weights(weight_dict)

        logger.info(f"Applied LoRA: {lora_name} with alpha={alpha}")
        del lora_weights
        return True

    @torch.no_grad()
    def _apply_lora_weights(self, weight_dict, lora_weights, alpha):
        lora_prefixs = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_add_out",
            "attn.to_out.0",
            "img_mlp.net.0.proj",
            "txt_mlp.net.0.proj",
            "txt_mlp.net.2",
        ]

        for prefix in lora_prefixs:
            for idx in range(self.model.config["num_layers"]):
                prefix_name = f"transformer_blocks.{idx}.{prefix}"
                lora_up = lora_weights[f"{prefix_name}.lora_up.weight"]
                lora_down = lora_weights[f"{prefix_name}.lora_down.weight"]
                lora_alpha = lora_weights[f"{prefix_name}.alpha"]
                origin = weight_dict[f"{prefix_name}.weight"]
                weight_dict[f"{prefix_name}.weight"] = fuse_lora_weights(origin, lora_down, lora_up, lora_alpha)

        return weight_dict
