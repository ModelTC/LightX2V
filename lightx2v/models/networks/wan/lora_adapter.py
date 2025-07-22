import os
import torch
from safetensors import safe_open
from loguru import logger
import gc
from lightx2v.utils.envs import *


class WanLoraWrapper:
    def __init__(self, wan_model):
        self.model = wan_model
        self.lora_metadata = {}
        self.override_dict = {}  # On CPU

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
        use_bfloat16 = GET_DTYPE() == "BF16"
        if self.model.config and hasattr(self.model.config, "get"):
            use_bfloat16 = self.model.config.get("use_bfloat16", True)
        with safe_open(file_path, framework="pt") as f:
            if use_bfloat16:
                tensor_dict = {key: f.get_tensor(key).to(torch.bfloat16) for key in f.keys()}
            else:
                tensor_dict = {key: f.get_tensor(key) for key in f.keys()}
        return tensor_dict

    def apply_lora(self, lora_name, alpha=1.0):
        if lora_name not in self.lora_metadata:
            logger.info(f"LoRA {lora_name} not found. Please load it first.")

        if not hasattr(self.model, "original_weight_dict"):
            logger.error("Model does not have 'original_weight_dict'. Cannot apply LoRA.")
            return False

        lora_weights = self._load_lora_file(self.lora_metadata[lora_name]["path"])
        weight_dict = self.model.original_weight_dict
        self._apply_lora_weights(weight_dict, lora_weights, alpha)
        self.model._init_weights(weight_dict)

        logger.info(f"Applied LoRA: {lora_name} with alpha={alpha}")
        del lora_weights  # 删除节约显存
        return True

    @torch.no_grad()
    def _apply_lora_weights(self, weight_dict, lora_weights, alpha):
        lora_pairs = {}
        lora_diffs = {}
        prefix = "diffusion_model."

        def try_lora_pair(key, suffix_a, suffix_b, target_suffix):
            if key.endswith(suffix_a):
                base_name = key[len(prefix) :].replace(suffix_a, target_suffix)
                pair_key = key.replace(suffix_a, suffix_b)
                if pair_key in lora_weights:
                    lora_pairs[base_name] = (key, pair_key)

        def try_lora_diff(key, suffix, target_suffix):
            if key.endswith(suffix):
                base_name = key[len(prefix) :].replace(suffix, target_suffix)
                lora_diffs[base_name] = key

        for key in lora_weights.keys():
            if not key.startswith(prefix):
                continue

            try_lora_pair(key, "lora_A.weight", "lora_B.weight", "weight")
            try_lora_pair(key, "lora_down.weight", "lora_up.weight", "weight")
            try_lora_diff(key, "diff", "weight")
            try_lora_diff(key, "diff_b", "bias")
            try_lora_diff(key, "diff_m", "modulation")

        applied_count = 0
        for name, param in weight_dict.items():
            if name in lora_pairs:
                if name not in self.override_dict:
                    self.override_dict[name] = param.clone().cpu()
                name_lora_A, name_lora_B = lora_pairs[name]
                lora_A = lora_weights[name_lora_A].to(param.device, param.dtype)
                lora_B = lora_weights[name_lora_B].to(param.device, param.dtype)
                if param.shape == (lora_B.shape[0], lora_A.shape[1]):
                    param += torch.matmul(lora_B, lora_A) * alpha
                    applied_count += 1
            elif name in lora_diffs:
                if name not in self.override_dict:
                    self.override_dict[name] = param.clone().cpu()

                name_diff = lora_diffs[name]
                lora_diff = lora_weights[name_diff].to(param.device, param.dtype)
                if param.shape == lora_diff.shape:
                    param += lora_diff * alpha
                    applied_count += 1

        logger.info(f"Applied {applied_count} LoRA weight adjustments")
        if applied_count == 0:
            logger.info(
                "Warning: No LoRA weights were applied. Expected naming conventions: 'diffusion_model.<layer_name>.lora_A.weight' and 'diffusion_model.<layer_name>.lora_B.weight'. Please verify the LoRA weight file."
            )

    @torch.no_grad()
    def remove_lora(self):
        logger.info(f"Removing LoRA ...")

        restored_count = 0
        for k, v in self.override_dict.items():
            self.model.original_weight_dict[k] = v.to(self.model.device)
            restored_count += 1

        logger.info(f"LoRA removed, restored {restored_count} weights")

        self.model._init_weights(self.model.original_weight_dict)

        torch.cuda.empty_cache()
        gc.collect()

        self.lora_metadata = {}
        self.override_dict = {}

    def list_loaded_loras(self):
        return list(self.lora_metadata.keys())
