import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from lightx2v.models.networks.wan.infer.matrix_game3.post_infer import WanMtxg3PostInfer
from lightx2v.models.networks.wan.infer.matrix_game3.pre_infer import WanMtxg3PreInfer
from lightx2v.models.networks.wan.infer.matrix_game3.transformer_infer import WanMtxg3TransformerInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.matrix_game3.pre_weights import WanMtxg3PreWeights
from lightx2v.models.networks.wan.weights.matrix_game3.transformer_weights import WanMtxg3TransformerWeights
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


@lru_cache(maxsize=1)
def _import_official_matrix_game3_wan_model():
    """Load the official Matrix-Game-3 WanModel implementation on demand."""
    official_root = Path(__file__).resolve().parents[4] / "Matrix-Game-3" / "Matrix-Game-3"
    if not official_root.is_dir():
        raise FileNotFoundError(f"Official Matrix-Game-3 source directory not found: {official_root}")
    official_root_str = str(official_root)
    if official_root_str not in sys.path:
        sys.path.insert(0, official_root_str)
    from wan.modules.model import WanModel as OfficialWanModel

    return OfficialWanModel


def _matrix_game3_forward_tensor_probe(tensor: torch.Tensor, head_values: int = 8) -> dict[str, Any]:
    tensor = tensor.detach()
    tensor_fp32 = tensor.to(dtype=torch.float32)
    flattened = tensor_fp32.reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor_fp32.min().item()),
        "max": float(tensor_fp32.max().item()),
        "mean": float(tensor_fp32.mean().item()),
        "std": float(tensor_fp32.std(unbiased=False).item()),
        "head": flattened[:head_values].cpu().tolist(),
    }


def _matrix_game3_forward_value_probe(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return _matrix_game3_forward_tensor_probe(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "head": list(value[:8]),
        }
    if isinstance(value, dict):
        return {k: _matrix_game3_forward_value_probe(v) for k, v in value.items()}
    return str(value)


class WanMtxg3OfficialBaseModel:
    """Base-model wrapper that delegates denoising to the official MG3 forward.

    The distilled MG3 path is numerically tolerant enough to run through the
    custom LightX2V weight/infer stack, but the base checkpoint is much more
    sensitive under 50-step CFG. Reusing the official DiT forward here removes
    the remaining block/head precision mismatches from the adaptation.
    """

    def __init__(self, model_path, config, device, model_type="wan2.2", lora_path=None, lora_strength=1.0):
        del model_type, lora_path, lora_strength
        self.model_path = model_path
        self.config = config
        self.device = device
        self.scheduler = None
        self.transformer_infer = None
        self._official_model = self._load_official_model()

    def _load_official_model(self):
        sub_model_folder = self.config.get("sub_model_folder", "base_model")
        model_dir = os.path.join(self.config["model_path"], sub_model_folder)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Matrix-Game-3 base checkpoint directory not found: {model_dir}")
        OfficialWanModel = _import_official_matrix_game3_wan_model()
        model = OfficialWanModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
        model = model.eval().requires_grad_(False)
        model.to(device=self.device, dtype=torch.bfloat16)
        return model

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _build_official_timestep(self, latents):
        t = self.scheduler.timestep_input
        if t is None:
            raise RuntimeError("Matrix-Game-3 base forward requested before scheduler.timestep_input was prepared")
        if t.numel() != 1:
            return t.reshape(1, -1).to(device=latents.device, dtype=latents.dtype)

        timestep_scalar = t.reshape(1).to(device=latents.device, dtype=latents.dtype)
        timestep = latents.new_full(
            (latents.shape[1], latents.shape[2] * latents.shape[3] // 4),
            timestep_scalar.squeeze(0),
        )
        mask = getattr(self.scheduler, "mask", None)
        if mask is not None:
            fixed_latent_frames = int((mask[0].flatten(1).sum(dim=1) == 0).sum().item())
            if fixed_latent_frames > 0:
                timestep[:fixed_latent_frames].zero_()
        return timestep.flatten().unsqueeze(0)

    def _build_forward_kwargs(self, inputs, infer_condition):
        if self.scheduler is None:
            raise RuntimeError("Matrix-Game-3 base model used before scheduler was attached")

        latents = self.scheduler.latents.unsqueeze(0)
        timestep = self._build_official_timestep(self.scheduler.latents)
        image_encoder_output = inputs.get("image_encoder_output", {})
        dit_cond_dict = image_encoder_output.get("dit_cond_dict") or {}

        if infer_condition:
            context = inputs["text_encoder_output"]["context"]
            plucker_emb = dit_cond_dict.get("plucker_emb_with_memory")
            if plucker_emb is None:
                plucker_emb = dit_cond_dict.get("c2ws_plucker_emb")
            mouse_cond = dit_cond_dict.get("mouse_cond")
            keyboard_cond = dit_cond_dict.get("keyboard_cond")
            x_memory = dit_cond_dict.get("x_memory")
            timestep_memory = dit_cond_dict.get("timestep_memory")
            mouse_cond_memory = dit_cond_dict.get("mouse_cond_memory")
            keyboard_cond_memory = dit_cond_dict.get("keyboard_cond_memory")
            memory_latent_idx = dit_cond_dict.get("memory_latent_idx")
        else:
            context = inputs["text_encoder_output"]["context_null"]
            mouse_source = dit_cond_dict.get("mouse_cond")
            keyboard_source = dit_cond_dict.get("keyboard_cond")
            plucker_emb = dit_cond_dict.get("c2ws_plucker_emb")
            mouse_cond = torch.ones_like(mouse_source) if mouse_source is not None else None
            keyboard_cond = -torch.ones_like(keyboard_source) if keyboard_source is not None else None
            x_memory = None
            timestep_memory = None
            mouse_cond_memory = None
            keyboard_cond_memory = None
            memory_latent_idx = None

        total_latent_frames = latents.shape[2] + (int(x_memory.shape[2]) if x_memory is not None else 0)
        patch_h, patch_w = tuple(self.config.get("patch_size", (1, 2, 2)))[1:]
        seq_len = total_latent_frames * latents.shape[3] * latents.shape[4] // (patch_h * patch_w)

        forward_kwargs = {
            "x": latents,
            "t": timestep,
            "context": context,
            "seq_len": seq_len,
            "mouse_cond": mouse_cond,
            "keyboard_cond": keyboard_cond,
            "x_memory": x_memory,
            "timestep_memory": timestep_memory,
            "mouse_cond_memory": mouse_cond_memory,
            "keyboard_cond_memory": keyboard_cond_memory,
            "plucker_emb": plucker_emb,
            "memory_latent_idx": memory_latent_idx,
            "predict_latent_idx": dit_cond_dict.get("predict_latent_idx"),
        }
        if self.scheduler is not None:
            debug_payload = {k: _matrix_game3_forward_value_probe(v) for k, v in forward_kwargs.items()}
            if infer_condition:
                self.scheduler.forward_kwargs_cond = debug_payload
            else:
                self.scheduler.forward_kwargs_uncond = debug_payload
        return forward_kwargs

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        self.scheduler.infer_condition = infer_condition
        noise_pred = self._official_model(**self._build_forward_kwargs(inputs, infer_condition))
        if isinstance(noise_pred, list):
            noise_pred = torch.stack(noise_pred)
        if noise_pred.dim() == 5 and noise_pred.shape[0] == 1:
            noise_pred = noise_pred.squeeze(0)
        return noise_pred.float()

    @torch.no_grad()
    def infer(self, inputs):
        if self.config.get("enable_cfg", False):
            noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
            noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)
            noise_pred_guided = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
            self.scheduler.noise_pred_cond = noise_pred_cond
            self.scheduler.noise_pred_uncond = noise_pred_uncond
            self.scheduler.noise_pred_guided = noise_pred_guided
            self.scheduler.noise_pred = noise_pred_guided
        else:
            noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
            self.scheduler.noise_pred_cond = noise_pred
            self.scheduler.noise_pred_uncond = None
            self.scheduler.noise_pred_guided = noise_pred
            self.scheduler.noise_pred = noise_pred


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
            raise FileNotFoundError(f"No safetensors files found in {model_dir}. Please download the Matrix-Game-3.0 model weights.")

        weight_dict = {}
        for sf_file in sorted(safetensor_files):
            file_path = os.path.join(model_dir, sf_file)
            with safe_open(file_path, framework="pt", device=str(self.device)) as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Strip the common diffusers prefix if present
                    name = key
                    if name.startswith("model."):
                        name = name[len("model.") :]
                    # Cast to appropriate dtype
                    if unified_dtype or all(s not in name for s in sensitive_layer):
                        weight_dict[name] = tensor.to(GET_DTYPE())
                    else:
                        weight_dict[name] = tensor.to(GET_SENSITIVE_DTYPE())
        return weight_dict
