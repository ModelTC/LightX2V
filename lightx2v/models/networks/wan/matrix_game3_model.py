import json
import os
import sys
from functools import lru_cache
from pathlib import Path

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


def _matrix_game3_compute_max_seq_len(config, lat_h: int, lat_w: int) -> int:
    first_clip_frame = int(config.get("first_clip_frame", config.get("target_video_length", 57)))
    vae_stride_t = int(tuple(config.get("vae_stride", (4, 16, 16)))[0])
    patch_h, patch_w = tuple(config.get("patch_size", (1, 2, 2)))[1:]
    max_lat_f = (first_clip_frame - 1) // vae_stride_t + 1
    max_mem_f = 5
    max_total_f = max_lat_f + max_mem_f
    max_seq_len = max_total_f * lat_h * lat_w // (patch_h * patch_w)
    sp_size = int(config.get("sp_size", 1))
    if sp_size > 1:
        max_seq_len = int(((max_seq_len + sp_size - 1) // sp_size) * sp_size)
    return max_seq_len


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
        patch_size = tuple(self.config.get("patch_size", (1, 2, 2)))
        patch_h = int(patch_size[1])
        patch_w = int(patch_size[2])
        latent_frames = int(latents.shape[1])
        latent_h = int(latents.shape[2])
        latent_w = int(latents.shape[3])
        tokens_per_frame = latent_h * latent_w // (patch_h * patch_w)
        timestep = latents.new_full(
            (latent_frames, tokens_per_frame),
            timestep_scalar.squeeze(0),
        )
        mask = getattr(self.scheduler, "mask", None)
        if mask is not None:
            fixed_latent_frames = int((mask.to(dtype=torch.float32).amax(dim=(0, 2, 3)) == 0).sum().item())
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
        memory_plucker = dit_cond_dict.get("plucker_emb_with_memory")
        camera_plucker = dit_cond_dict.get("c2ws_plucker_emb")

        if infer_condition:
            context = inputs["text_encoder_output"]["context"]
            plucker_emb = memory_plucker
            if plucker_emb is None:
                plucker_emb = camera_plucker
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

        seq_len = _matrix_game3_compute_max_seq_len(self.config, latents.shape[3], latents.shape[4])

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
            cond_pred = self._infer_cond_uncond(inputs, infer_condition=True)
            uncond_pred = self._infer_cond_uncond(inputs, infer_condition=False)
            self.scheduler.noise_pred = uncond_pred + self.scheduler.sample_guide_scale * (cond_pred - uncond_pred)
        else:
            self.scheduler.noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)


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
