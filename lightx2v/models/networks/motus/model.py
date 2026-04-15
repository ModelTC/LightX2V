import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from loguru import logger
from transformers import AutoProcessor

from lightx2v.models.networks.motus.core import Motus, MotusConfig
from lightx2v.models.networks.motus.image_utils import resize_with_padding
from lightx2v.models.networks.motus.infer.post_infer import MotusPostInfer
from lightx2v.models.networks.motus.infer.pre_infer import MotusPreInfer
from lightx2v.models.networks.motus.infer.transformer_infer import MotusTransformerInfer
from lightx2v.models.networks.motus.ops import LinearWithMM, TripleQKVProjector
from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel
from lightx2v.models.networks.motus.primitives import rope_apply
from lightx2v.models.schedulers.motus.scheduler import MotusScheduler


class MotusModel:
    """Thin LightX2V wrapper over Motus native inference."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.motus_root = Path(config.get("model_path", "")).expanduser().resolve()
        if not self.motus_root.exists():
            raise FileNotFoundError(f"Motus root not found: {self.motus_root}")

        self._motus_cls = Motus
        self._motus_config_cls = MotusConfig
        self._resize_with_padding = resize_with_padding
        self._rope_apply = rope_apply
        self._t5_encoder_cls = T5EncoderModel

        self.model = self._load_model().eval()
        self.t5_encoder = self._load_t5_encoder()
        self.vlm_processor = self._load_vlm_processor()
        self._load_normalization_stats()
        self._rope_cos_sin_cache = {}
        self._build_native_stack()

    def _build_native_stack(self):
        self.scheduler = MotusScheduler(self.config)
        self.pre_infer = MotusPreInfer(self, self.config)
        self.transformer_infer = MotusTransformerInfer(self, self.config)
        self.post_infer = MotusPostInfer(self, self.config)

        self.pre_infer.set_scheduler(self.scheduler)
        self.transformer_infer.set_scheduler(self.scheduler)
        self.post_infer.set_scheduler(self.scheduler)

    def _build_model_config(self):
        return self._motus_config_cls(
            wan_checkpoint_path=self.config["wan_path"],
            vae_path=os.path.join(self.config["wan_path"], "Wan2.2_VAE.pth"),
            wan_config_path=self.config["wan_path"],
            video_precision=self.config.get("video_precision", "bfloat16"),
            vlm_checkpoint_path=self.config["vlm_path"],
            und_expert_hidden_size=self.config.get("und_expert_hidden_size", 512),
            und_expert_ffn_dim_multiplier=self.config.get("und_expert_ffn_dim_multiplier", 4),
            und_expert_norm_eps=self.config.get("und_expert_norm_eps", 1e-5),
            und_layers_to_extract=self.config.get("und_layers_to_extract"),
            vlm_adapter_input_dim=self.config.get("vlm_adapter_input_dim", 2048),
            vlm_adapter_projector_type=self.config.get("vlm_adapter_projector_type", "mlp3x_silu"),
            num_layers=self.config.get("num_layers", 30),
            action_state_dim=self.config.get("action_state_dim", 14),
            action_dim=self.config.get("action_dim", 14),
            action_expert_dim=self.config.get("action_expert_dim", 1024),
            action_expert_ffn_dim_multiplier=self.config.get("action_expert_ffn_dim_multiplier", 4),
            action_expert_norm_eps=self.config.get("action_expert_norm_eps", 1e-6),
            global_downsample_rate=self.config.get("global_downsample_rate", 3),
            video_action_freq_ratio=self.config.get("video_action_freq_ratio", 2),
            num_video_frames=self.config.get("num_video_frames", 8),
            video_height=self.config.get("video_height", 384),
            video_width=self.config.get("video_width", 320),
            batch_size=1,
            training_mode=self.config.get("training_mode", "finetune"),
            load_pretrained_backbones=self.config.get("load_pretrained_backbones", False),
        )

    def _load_model(self):
        logger.info("Loading Motus model")
        model = self._motus_cls(self._build_model_config())
        self._patch_qwen3_vl_rope_index(model)
        model.to(self.device)
        model.load_checkpoint(self.config["checkpoint_path"], strict=False)
        self._apply_lightx2v_patches(model)
        return model

    def _load_t5_encoder(self):
        return self._t5_encoder_cls(
            text_len=512,
            dtype=torch.bfloat16,
            device=self.device,
            checkpoint_path=os.path.join(self.config["wan_path"], "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.config["wan_path"], "google", "umt5-xxl"),
        )

    def _load_vlm_processor(self):
        return AutoProcessor.from_pretrained(self.config["vlm_path"], trust_remote_code=True)

    def _patch_qwen3_vl_rope_index(self, root: Any):
        visited = set()

        def walk(obj: Any):
            obj_id = id(obj)
            if obj is None or obj_id in visited:
                return
            visited.add(obj_id)

            method = getattr(obj, "get_rope_index", None)
            if callable(method):
                try:
                    signature = inspect.signature(method)
                except (TypeError, ValueError):
                    signature = None

                if signature and "mm_token_type_ids" in signature.parameters:

                    def wrapped_get_rope_index(*args, __orig=method, **kwargs):
                        if "mm_token_type_ids" not in kwargs:
                            input_ids = kwargs.get("input_ids")
                            if input_ids is None and args:
                                input_ids = args[0]
                            if torch.is_tensor(input_ids):
                                kwargs["mm_token_type_ids"] = torch.zeros_like(input_ids, dtype=torch.long)
                        return __orig(*args, **kwargs)

                    setattr(obj, "get_rope_index", wrapped_get_rope_index)

            if isinstance(obj, torch.nn.Module):
                for child in obj.children():
                    walk(child)

            for attr in ("model", "language_model", "visual", "vlm", "backbone"):
                child = getattr(obj, attr, None)
                if child is not None and child is not obj:
                    walk(child)

        walk(root)

    def _load_normalization_stats(self):
        stat_path = self.motus_root / "utils" / "stat.json"
        if stat_path.exists():
            with open(stat_path, "r") as f:
                stat_data = json.load(f)
            stats = stat_data.get(self.config.get("stats_key", "robotwin2"), {})
            if stats:
                self.action_min = torch.tensor(stats["min"], dtype=torch.float32, device=self.device)
                self.action_max = torch.tensor(stats["max"], dtype=torch.float32, device=self.device)
                self.action_range = self.action_max - self.action_min
                return

        action_dim = self.config.get("action_dim", 14)
        self.action_min = torch.zeros(action_dim, dtype=torch.float32, device=self.device)
        self.action_max = torch.ones(action_dim, dtype=torch.float32, device=self.device)
        self.action_range = torch.ones(action_dim, dtype=torch.float32, device=self.device)

    def _quant_flags(self):
        quantized = bool(self.config.get("motus_quantized", self.config.get("dit_quantized", False)))
        quant_scheme = self.config.get("motus_quant_scheme", self.config.get("dit_quant_scheme", "Default"))
        return quantized, quant_scheme

    def _replace_linear_modules(self, module):
        quantized, quant_scheme = self._quant_flags()
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    LinearWithMM.from_linear(
                        child,
                        quant_scheme=quant_scheme,
                        quantized=quantized,
                        config=self.config,
                    ),
                )
            else:
                self._replace_linear_modules(child)

    def _attach_qkv_projectors(self, model):
        quantized, quant_scheme = self._quant_flags()
        for block in model.action_expert.blocks:
            block.wan_action_qkv_mm = TripleQKVProjector(
                block.wan_action_qkv.detach(),
                quant_scheme=quant_scheme,
                quantized=quantized,
                config=self.config,
            )
            block.wan_action_o = LinearWithMM.from_linear(
                block.wan_action_o,
                quant_scheme=quant_scheme,
                quantized=quantized,
                config=self.config,
            )

        for block in model.und_expert.blocks:
            block.wan_und_qkv_mm = TripleQKVProjector(
                block.wan_und_qkv.detach(),
                quant_scheme=quant_scheme,
                quantized=quantized,
                config=self.config,
            )
            block.wan_und_o = LinearWithMM.from_linear(
                block.wan_und_o,
                quant_scheme=quant_scheme,
                quantized=quantized,
                config=self.config,
            )

    def _apply_lightx2v_patches(self, model):
        self._replace_linear_modules(model.action_expert)
        self._replace_linear_modules(model.und_expert)
        self._attach_qkv_projectors(model)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        shape = actions.shape
        flat = actions.reshape(-1, shape[-1])
        restored = flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return restored.reshape(shape)

    def rope_apply(self, q: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return self._rope_apply(q, grid_sizes, freqs)

    def get_wan_freqs(self) -> torch.Tensor:
        freqs = self.model.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)
        return freqs

    def get_wan_rotary_cos_sin(self, grid_size: tuple[int, int, int]):
        if grid_size in self._rope_cos_sin_cache:
            return self._rope_cos_sin_cache[grid_size]

        freqs = self.get_wan_freqs()
        head_dim_half = freqs.shape[1]
        c_f = head_dim_half - 2 * (head_dim_half // 3)
        c_h = head_dim_half // 3
        c_w = head_dim_half // 3
        fpart, hpart, wpart = freqs.split([c_f, c_h, c_w], dim=1)
        f, h, w = grid_size

        freq_grid = torch.cat(
            [
                fpart[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                hpart[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                wpart[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, -1)

        cos_sin = (freq_grid.real.contiguous(), freq_grid.imag.contiguous())
        self._rope_cos_sin_cache[grid_size] = cos_sin
        return cos_sin

    def prepare_frame(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image).astype(np.float32) / 255.0
        resized_np = self._resize_with_padding(
            image_np,
            (self.config.get("video_height", 384), self.config.get("video_width", 320)),
        )
        if resized_np.dtype == np.uint8:
            resized_np = resized_np.astype(np.float32) / 255.0
        return torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def prepare_state(self, state_value) -> torch.Tensor:
        if isinstance(state_value, torch.Tensor):
            state = state_value.float()
        else:
            state = torch.tensor(state_value, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return state.to(self.device)

    def build_instruction(self, prompt: str) -> str:
        prefix = self.config.get(
            "scene_prefix",
            "The whole scene is in a realistic, industrial art style with three views: "
            "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
            "The aloha robot is currently performing the following task: ",
        )
        return f"{prefix}{prompt}"

    def build_t5_embeddings(self, instruction: str):
        if hasattr(self.t5_encoder, "infer"):
            t5_out = self.t5_encoder.infer([instruction])
        else:
            t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            return [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        return t5_out

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.float().clamp(0, 1)
        np_img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_img, mode="RGB")

    def build_vlm_inputs(self, instruction: str, first_frame: torch.Tensor):
        image = self._tensor_to_pil(first_frame.squeeze(0))
        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image", "image": image}]}]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors="pt")

        vlm_inputs = {}
        for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "video_grid_thw", "second_per_grid_ts", "mm_token_type_ids"):
            value = encoded.get(key)
            if torch.is_tensor(value):
                vlm_inputs[key] = value.to(self.device)
            elif value is not None:
                vlm_inputs[key] = value

        if "mm_token_type_ids" not in vlm_inputs and "input_ids" in vlm_inputs:
            vlm_inputs["mm_token_type_ids"] = torch.zeros_like(vlm_inputs["input_ids"], dtype=torch.long)
        return vlm_inputs

    @torch.no_grad()
    def encode_condition_frame(self, first_frame: torch.Tensor):
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        return self.model.video_model.encode_video(first_frame_norm.to(self.model.dtype))

    @torch.no_grad()
    def infer(self, image_path: str, prompt: str, state_value, num_inference_steps: int, seed: int | None = None):
        self.scheduler.infer_steps = num_inference_steps
        pre_infer_out = self.pre_infer.infer(image_path=image_path, prompt=prompt, state_value=state_value, seed=seed)
        video_latents, action_latents = self.transformer_infer.infer(None, pre_infer_out)
        post_infer_out = self.post_infer.infer(video_latents, action_latents)

        pred_frames = post_infer_out.pred_frames
        if pred_frames.dim() == 5:
            if pred_frames.shape[1] == 3:
                pred_frames = pred_frames.permute(0, 2, 1, 3, 4)
            pred_frames = pred_frames.squeeze(0)
        return pred_frames, post_infer_out.pred_actions.squeeze(0)
