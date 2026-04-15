import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

from lightx2v.models.networks.wan.infer.triton_ops import fuse_scale_shift_kernel
from lightx2v.models.networks.wan.infer.utils import sinusoidal_embedding_1d

from .action_expert import ActionExpert, ActionExpertConfig
from .und_expert import UndExpert, UndExpertConfig
from .wan_model import WanVideoModel

logger = logging.getLogger(__name__)


def _apply_modulation(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    scale = scale.squeeze(2)
    shift = shift.squeeze(2)
    if x.is_cuda:
        return fuse_scale_shift_kernel(x.float().contiguous(), scale.contiguous(), shift.contiguous())
    return x.float() * (1 + scale) + shift


@dataclass
class MotusConfig:
    wan_checkpoint_path: str
    vae_path: str
    wan_config_path: str
    video_precision: str = "bfloat16"
    vlm_checkpoint_path: str = ""
    und_expert_hidden_size: int = 512
    und_expert_ffn_dim_multiplier: int = 4
    und_expert_norm_eps: float = 1e-5
    und_layers_to_extract: List[int] = None
    vlm_adapter_input_dim: int = 2048
    vlm_adapter_projector_type: str = "mlp3x_silu"
    num_layers: int = 30
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024
    action_expert_ffn_dim_multiplier: int = 4
    action_expert_norm_eps: float = 1e-6
    global_downsample_rate: int = 3
    video_action_freq_ratio: int = 2
    num_video_frames: int = 8
    video_height: int = 384
    video_width: int = 320
    batch_size: int = 1
    training_mode: str = "finetune"
    load_pretrained_backbones: Optional[bool] = None

    def __post_init__(self):
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        if self.und_layers_to_extract is None:
            self.und_layers_to_extract = list(range(self.num_layers))


class VideoModule(nn.Module):
    def __init__(self, video_model, dtype, device, grid_sizes):
        super().__init__()
        self.video_model = video_model
        self.dtype = dtype
        self.device = device
        self.grid_sizes = grid_sizes

    def prepare_input(self, noisy_video_latent: torch.Tensor) -> torch.Tensor:
        return self.video_model.wan_model.patch_embedding(noisy_video_latent).flatten(2).transpose(1, 2)

    def preprocess_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        if isinstance(language_embeddings, list):
            text_len = self.video_model.wan_model.text_len
            padded = []
            for emb in language_embeddings:
                padded.append(torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])]) if emb.shape[0] <= text_len else emb[:text_len])
            t5_context_raw = torch.stack(padded, dim=0)
        else:
            t5_context_raw = language_embeddings
        return self.video_model.wan_model.text_embedding(t5_context_raw)

    def get_time_embedding(self, t_video: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if t_video.dim() == 1:
            t_video = t_video.unsqueeze(1).expand(t_video.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t_video.size(0)
            t_flat = t_video.flatten()
            t_emb = self.video_model.wan_model.time_embedding(sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float())
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, self.video_model.wan_model.dim))
        return t_emb, t_emb_proj

    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast("cuda", dtype=torch.float32):
            return (wan_layer.modulation.unsqueeze(0) + video_adaln_params).chunk(6, dim=2)

    def process_ffn(self, video_tokens: torch.Tensor, video_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        v_mod = video_adaln_modulation
        ffn_input = _apply_modulation(wan_layer.norm2(video_tokens), v_mod[4], v_mod[3])
        ffn_out = wan_layer.ffn(ffn_input)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            return video_tokens + ffn_out * v_mod[5].squeeze(2)

    def apply_output_head(self, video_tokens: torch.Tensor, video_time_emb: torch.Tensor) -> torch.Tensor:
        x = self.video_model.wan_model.head(video_tokens, video_time_emb)
        x = self.video_model.wan_model.unpatchify(x, self.grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)


class UndModule(nn.Module):
    def __init__(self, vlm_model, und_expert, config, dtype, device, image_context_adapter=None):
        super().__init__()
        self.vlm_model = vlm_model
        self.und_expert = und_expert
        self.config = config
        self.dtype = dtype
        self.device = device
        self.image_context_adapter = image_context_adapter

    def _parse_vision_outputs(self, vision_outputs):
        if hasattr(vision_outputs, "pooler_output"):
            image_embeds = vision_outputs.pooler_output
            deepstack_image_embeds = vision_outputs.get("hidden_states", None) if hasattr(vision_outputs, "get") else getattr(vision_outputs, "hidden_states", None)
        elif isinstance(vision_outputs, tuple):
            image_embeds = vision_outputs[0]
            deepstack_image_embeds = vision_outputs[1] if len(vision_outputs) > 1 else None
        else:
            image_embeds = vision_outputs
            deepstack_image_embeds = None

        if torch.is_tensor(image_embeds):
            return image_embeds.to(self.device, self.dtype), deepstack_image_embeds
        if isinstance(image_embeds, (list, tuple)):
            return torch.cat(list(image_embeds), dim=0).to(self.device, self.dtype), deepstack_image_embeds
        raise TypeError(f"Unsupported image feature output type: {type(image_embeds)}")

    def _process_vlm_inputs_to_tokens(self, vlm_inputs, batch: int):
        if isinstance(vlm_inputs, list):
            input_ids_batch = torch.cat([item["input_ids"] for item in vlm_inputs], dim=0).to(self.device)
            attention_mask_batch = torch.cat([item["attention_mask"] for item in vlm_inputs], dim=0).to(self.device)
            pixel_values_batch = torch.cat([item["pixel_values"] for item in vlm_inputs], dim=0).to(self.device)
            image_grid_thw_batch = torch.cat([item["image_grid_thw"] for item in vlm_inputs], dim=0).to(self.device)
        else:
            input_ids_batch = vlm_inputs["input_ids"].to(self.device)
            attention_mask_batch = vlm_inputs["attention_mask"].to(self.device)
            pixel_values_batch = vlm_inputs["pixel_values"].to(self.device)
            image_grid_thw_batch = vlm_inputs["image_grid_thw"].to(self.device)

        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids_batch)
        vision_outputs = self.vlm_model.get_image_features(pixel_values_batch, image_grid_thw_batch)
        image_embeds, deepstack_image_embeds = self._parse_vision_outputs(vision_outputs)
        image_mask, _ = self.vlm_model.model.get_placeholder_mask(input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]
        position_ids, _ = self.vlm_model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,
            attention_mask=attention_mask_batch,
        )
        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids

    def extract_und_features(self, vlm_inputs) -> torch.Tensor:
        batch = len(vlm_inputs) if isinstance(vlm_inputs, list) else vlm_inputs["input_ids"].shape[0]
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs, batch)
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": None,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if visual_pos_masks is not None:
            kwargs["visual_pos_masks"] = visual_pos_masks
        if deepstack_image_embeds is not None:
            kwargs["deepstack_visual_embeds"] = deepstack_image_embeds
        with torch.no_grad():
            vlm_output = self.vlm_model.model.language_model(**kwargs)
        return self.und_expert.vlm_adapter(vlm_output.hidden_states[-1])

    def extract_image_context(self, vlm_inputs) -> torch.Tensor | None:
        if self.image_context_adapter is None:
            return None

        if isinstance(vlm_inputs, list):
            pixel_values = torch.cat([item["pixel_values"] for item in vlm_inputs], dim=0).to(self.device)
            image_grid_thw = torch.cat([item["image_grid_thw"] for item in vlm_inputs], dim=0).to(self.device)
        else:
            pixel_values = vlm_inputs["pixel_values"].to(self.device)
            image_grid_thw = vlm_inputs["image_grid_thw"].to(self.device)

        with torch.no_grad():
            vision_outputs = self.vlm_model.get_image_features(pixel_values, image_grid_thw)
        image_embeds, _ = self._parse_vision_outputs(vision_outputs)
        return self.image_context_adapter(image_embeds)

    def process_ffn(self, und_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        block = self.und_expert.blocks[layer_idx]
        return und_tokens + block.ffn(block.norm2(und_tokens))


class ActionModule(nn.Module):
    def __init__(self, action_expert: ActionExpert, config, video_model, vlm_model, dtype, device):
        super().__init__()
        self.action_expert = action_expert
        self.config = config
        self.video_model = video_model
        self.vlm_model = vlm_model
        self.dtype = dtype
        self.device = device

    def get_time_embedding(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            a_e = self.action_expert.time_embedding(sinusoidal_embedding_1d(self.action_expert.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float())
            a_e0 = self.action_expert.time_projection(a_e).unflatten(2, (6, self.config.action_expert_dim))
        return a_e, a_e0

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        action_layer = self.action_expert.blocks[layer_idx]
        with torch.amp.autocast("cuda", dtype=torch.float32):
            return (action_layer.modulation.unsqueeze(0) + action_adaln_params).chunk(6, dim=2)

    def process_ffn(self, action_tokens: torch.Tensor, action_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        action_block = self.action_expert.blocks[layer_idx]
        a_mod = action_adaln_modulation
        ffn_input = _apply_modulation(action_block.norm2(action_tokens), a_mod[4], a_mod[3])
        ffn_out = action_block.ffn(ffn_input)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            return action_tokens + ffn_out * a_mod[5].squeeze(2)


class Motus(nn.Module):
    def __init__(self, config: MotusConfig):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16
        load_backbones = True if config.load_pretrained_backbones is None else bool(config.load_pretrained_backbones)

        if load_backbones:
            self.video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision,
            )
        else:
            self.video_model = WanVideoModel.from_config(
                config_path=config.wan_config_path,
                vae_path=config.vae_path,
                device="cuda",
                precision=config.video_precision,
            )

        if load_backbones:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.vlm_checkpoint_path,
                dtype=self.dtype,
                device_map="cuda",
                trust_remote_code=True,
            )
        else:
            vlm_cfg = AutoConfig.from_pretrained(config.vlm_checkpoint_path, trust_remote_code=True)
            self.vlm_model = Qwen3VLForConditionalGeneration._from_config(vlm_cfg, torch_dtype=self.dtype)
            self.vlm_model.to(device="cuda", dtype=self.dtype)

        for param in self.vlm_model.parameters():
            param.requires_grad = False

        wan_dim = getattr(self.video_model.wan_model.config, "dim", 3072)
        wan_num_heads = getattr(self.video_model.wan_model.config, "num_heads", 24)
        wan_head_dim = wan_dim // wan_num_heads
        vlm_dim = self.vlm_model.config.text_config.hidden_size
        vlm_num_heads = self.vlm_model.config.text_config.num_attention_heads
        vlm_num_kv_heads = getattr(self.vlm_model.config.text_config, "num_key_value_heads", vlm_num_heads)
        vlm_num_hidden_layers = self.vlm_model.config.text_config.num_hidden_layers

        wan_config = {"dim": wan_dim, "num_heads": wan_num_heads, "head_dim": wan_head_dim}
        vlm_config = {
            "hidden_size": vlm_dim,
            "num_attention_heads": vlm_num_heads,
            "num_key_value_heads": vlm_num_kv_heads,
            "head_dim": vlm_dim // vlm_num_heads,
            "num_hidden_layers": vlm_num_hidden_layers,
        }

        action_chunk_size_for_expert = config.action_chunk_size if config.training_mode == "pretrain" else config.action_chunk_size + 1
        num_registers = 0 if config.training_mode == "pretrain" else 4
        action_config = ActionExpertConfig(
            dim=config.action_expert_dim,
            ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            state_dim=config.action_state_dim,
            action_dim=config.action_dim,
            chunk_size=action_chunk_size_for_expert,
            num_registers=num_registers,
            video_feature_dim=wan_dim,
            causal=False,
            eps=config.action_expert_norm_eps,
            training_mode=config.training_mode,
        )
        self.action_expert = ActionExpert(action_config, wan_config)

        und_config = UndExpertConfig(
            dim=config.und_expert_hidden_size,
            ffn_dim=config.und_expert_hidden_size * config.und_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            vlm_input_dim=config.vlm_adapter_input_dim,
            vlm_projector_type=config.vlm_adapter_projector_type,
            eps=config.und_expert_norm_eps,
        )
        self.und_expert = UndExpert(und_config, wan_config, vlm_config)
        self.image_context_adapter = nn.Sequential(
            nn.Linear(vlm_dim, wan_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(wan_dim, wan_dim),
        )

        self.device = next(self.video_model.parameters()).device
        self.action_expert.to(device=self.device, dtype=self.dtype)
        self.und_expert.to(device=self.device, dtype=self.dtype)
        self.image_context_adapter.to(device=self.device, dtype=self.dtype)
        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)

        lat_t = 1 + config.num_video_frames // 4
        lat_h = config.video_height // 32
        lat_w = config.video_width // 32
        self.grid_sizes = torch.tensor([lat_t, lat_h, lat_w], dtype=torch.long, device=self.device).unsqueeze(0).expand(config.batch_size, -1)
        self.video_module = VideoModule(self.video_model, self.dtype, self.device, self.grid_sizes)
        self.und_module = UndModule(self.vlm_model, self.und_expert, self.config, self.dtype, self.device, image_context_adapter=self.image_context_adapter)
        self.action_module = ActionModule(self.action_expert, self.config, self.video_model, self.vlm_model, self.dtype, self.device)

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["module"]
        self.load_state_dict(state_dict, strict=strict)
        return {key: value for key, value in checkpoint.items() if key not in ["module", "config"]}
