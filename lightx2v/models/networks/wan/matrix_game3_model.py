import json
import os

import torch
from safetensors import safe_open

from lightx2v.models.networks.wan.infer.matrix_game3.pre_infer import WanMtxg3PreInfer
from lightx2v.models.networks.wan.infer.matrix_game3.post_infer import WanMtxg3PostInfer
from lightx2v.models.networks.wan.infer.matrix_game3.transformer_infer import WanMtxg3TransformerInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.matrix_game3.pre_weights import WanMtxg3PreWeights
from lightx2v.models.networks.wan.weights.matrix_game3.transformer_weights import WanMtxg3TransformerWeights
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


class WanMtxg3Model(WanModel):
    """Network model for Matrix-Game-3.0.

    Extends the base Wan2.2 DiT backbone with:
    - Per-block ActionModule weights for keyboard/mouse conditioning
    - Camera plucker ray injection layers (cam_injector, cam_scale, cam_shift)
    - Memory-aware self-attention with indexed RoPE
    - Global plucker embedding (patch_embedding_wancamctrl, c2ws_hidden_states_layer1/2)

    The model loads diffusers-format safetensors from the MG3.0 checkpoint
    directory (base_model/ or base_distilled_model/).
    """

    pre_weight_class = WanMtxg3PreWeights
    transformer_weight_class = WanMtxg3TransformerWeights

    def __init__(self, model_path, config, device, model_type="wan2.2", lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type, lora_path, lora_strength)

    def _init_infer_class(self):
        # Merge the official MG3 model config so that all dimension / action fields
        # are available for weight and infer construction.
        sub_model_folder = self.config.get("sub_model_folder", "base_distilled_model")
        config_path = os.path.join(self.config["model_path"], sub_model_folder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                model_config = json.load(f)
            for k in model_config.keys():
                self.config[k] = model_config[k]

        self.pre_infer_class = WanMtxg3PreInfer
        self.post_infer_class = WanMtxg3PostInfer
        self.transformer_infer_class = WanMtxg3TransformerInfer

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        """Load MG3.0 safetensors checkpoint.

        The MG3.0 checkpoint uses diffusers format with keys like
        ``model.blocks.0.self_attn.q.weight`` (prefixed with ``model.``).
        We strip the ``model.`` prefix so the keys match our weight module names.
        """
        sub_model_folder = self.config.get("sub_model_folder", "base_distilled_model")
        model_dir = os.path.join(self.config["model_path"], sub_model_folder)

        # Find safetensor files
        safetensor_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        if not safetensor_files:
            raise FileNotFoundError(
                f"No safetensors files found in {model_dir}. "
                "Please download the Matrix-Game-3.0 model weights."
            )

        weight_dict = {}
        for sf_file in sorted(safetensor_files):
            file_path = os.path.join(model_dir, sf_file)
            with safe_open(file_path, framework="pt", device=str(self.device)) as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Strip the common diffusers prefix if present
                    name = key
                    if name.startswith("model."):
                        name = name[len("model."):]
                    # Cast to appropriate dtype
                    if unified_dtype or all(s not in name for s in sensitive_layer):
                        weight_dict[name] = tensor.to(GET_DTYPE())
                    else:
                        weight_dict[name] = tensor.to(GET_SENSITIVE_DTYPE())
        return weight_dict
