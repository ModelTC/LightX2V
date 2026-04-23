import math
import inspect
import json
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.motus.image_utils import resize_with_padding
from lightx2v.models.networks.motus.infer.post_infer import MotusPostInfer
from lightx2v.models.networks.motus.infer.pre_infer import MotusPreInfer
from lightx2v.models.networks.motus.infer.transformer_infer import MotusTransformerInfer
from lightx2v.models.networks.wan.weights.motus import (
    MotusPostWeights,
    MotusPreWeights,
    MotusTransformerWeights,
    apply_mm,
    build_motus_expert_configs,
)
from lightx2v.models.networks.wan.weights.motus._shared import apply_time_embedding
from lightx2v.utils.custom_compiler import compiled_method
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v.utils.utils import load_weights


def unpatchify_video(x, grid_sizes, patch_size, out_dim):
    outputs = []
    for sample, sample_grid in zip(x, grid_sizes.tolist()):
        sample = sample[: math.prod(sample_grid)].view(*sample_grid, *patch_size, out_dim)
        sample = torch.einsum("fhwpqrc->cfphqwr", sample)
        sample = sample.reshape(out_dim, *[grid * patch for grid, patch in zip(sample_grid, patch_size)])
        outputs.append(sample)
    return torch.stack([sample.float() for sample in outputs], dim=0)


class MotusVideoBackbone:
    def __init__(self, config, pre_weights, transformer_weights):
        self.config = config
        self.pre_weights = pre_weights
        self.transformer_weights = transformer_weights
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        self.text_len = config["text_len"]
        self.freq_dim = config["freq_dim"]
        self.dim = config["dim"]
        self.num_heads = config["num_heads"]
        self.out_dim = config["out_dim"]

    def prepare_input(self, noisy_video_latent):
        video_tokens = self.pre_weights.patch_embedding.apply(noisy_video_latent)
        return video_tokens.flatten(2).transpose(1, 2).contiguous()

    def preprocess_t5_embeddings(self, language_embeddings):
        if isinstance(language_embeddings, list):
            padded = []
            for emb in language_embeddings:
                if emb.shape[0] <= self.text_len:
                    pad = emb.new_zeros(self.text_len - emb.shape[0], emb.shape[1])
                    padded.append(torch.cat([emb, pad], dim=0))
                else:
                    padded.append(emb[: self.text_len])
            context = torch.stack(padded, dim=0)
        else:
            context = language_embeddings
        hidden = apply_mm(self.pre_weights.text_embedding_0, context.float())
        hidden = torch.nn.functional.gelu(hidden, approximate="tanh")
        return apply_mm(self.pre_weights.text_embedding_2, hidden)

    def get_time_embedding(self, timestep, seq_len):
        return apply_time_embedding(
            timestep,
            seq_len,
            self.freq_dim,
            self.dim,
            self.pre_weights.time_embedding_0,
            self.pre_weights.time_embedding_2,
            self.pre_weights.time_projection_1,
        )

    def compute_adaln_modulation(self, video_adaln_params, layer_idx):
        block = self.transformer_weights.blocks[layer_idx]
        return (block.compute_phases[0].modulation.tensor.unsqueeze(0) + video_adaln_params).chunk(6, dim=2)

    def apply_output_head(self, video_tokens, video_time_emb, grid_sizes):
        modulation = self.transformer_weights.head_modulation.tensor
        head_shift, head_scale = (modulation.unsqueeze(0) + video_time_emb.unsqueeze(2)).chunk(2, dim=2)
        head_input = self.transformer_weights.norm.apply(video_tokens)
        head_input = head_input * (1 + head_scale.squeeze(2)) + head_shift.squeeze(2)
        head_output = apply_mm(self.transformer_weights.head, head_input)
        return unpatchify_video(head_output, grid_sizes, self.patch_size, self.out_dim)


class MotusActionExpert:
    def __init__(self, action_config, pre_weights, transformer_weights, post_weights):
        self.config = action_config
        self.pre_weights = pre_weights
        self.transformer_weights = transformer_weights
        self.post_weights = post_weights
        self.freq_dim = pre_weights.freq_dim

    def prepare_tokens(self, state_tokens, action_tokens):
        return self.pre_weights.apply_input_encoder(state_tokens, action_tokens)

    def get_time_embedding(self, timestep, seq_len):
        return self.pre_weights.get_time_embedding(timestep, seq_len)

    def compute_adaln_modulation(self, adaln_params, layer_idx):
        block = self.transformer_weights.blocks[layer_idx]
        return (block.modulation.tensor.unsqueeze(0) + adaln_params).chunk(6, dim=2)

    def apply_output(self, action_tokens, time_emb):
        return self.post_weights.apply_output(action_tokens, time_emb)


class MotusUndExpert:
    def __init__(self, pre_weights, transformer_weights, image_context_weights, vlm_model, device, dtype):
        self.pre_weights = pre_weights
        self.transformer_weights = transformer_weights
        self.image_context_weights = image_context_weights
        self.vlm_model = vlm_model
        self.device = device
        self.dtype = dtype

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

    def _process_vlm_inputs_to_tokens(self, vlm_inputs):
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

    def extract_und_features(self, vlm_inputs):
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs)
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
        return self.pre_weights.vlm_adapter.apply(vlm_output.hidden_states[-1].to(self.dtype))

    def extract_image_context(self, vlm_inputs):
        if self.image_context_weights is None:
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
        return self.image_context_weights.apply(image_embeds)


class MotusModel(BaseTransformerModel):
    pre_weight_class = MotusPreWeights
    transformer_weight_class = MotusTransformerWeights
    post_weight_class = MotusPostWeights

    def __init__(self, config, device):
        config = self._apply_motus_defaults(dict(config))
        model_path = config.get("model_path", config.get("wan_path", ""))
        # CompiledMethodsMixin discovers attributes during BaseTransformerModel init.
        # Seed config early so property access is safe before super() assigns it.
        self.config = config
        self._cached_vlm_state = None
        super().__init__(model_path=model_path, config=config, device=device, model_type="motus")
        self.motus_root = Path(self.model_path).expanduser().resolve()
        if self.motus_root.is_file():
            self.motus_root = self.motus_root.parent
        self.model_dtype = torch.bfloat16
        self.dtype = self.model_dtype
        self.expert_configs = build_motus_expert_configs(self.config)
        self.action_config = self.expert_configs.action
        self.und_config = self.expert_configs.und
        self._init_infer_class()
        logger.info("[Motus] Initializing weight containers")
        self._init_weights()
        logger.info("[Motus] Loading VLM model")
        self.vlm_model = self._load_vlm_model().eval()
        logger.info("[Motus] Loading VLM processor")
        self.vlm_processor = AutoProcessor.from_pretrained(self.config["vlm_path"], trust_remote_code=True)
        self._load_normalization_stats()
        self._rope_cos_sin_cache = {}
        self._patch_qwen3_vl_rope_index(self.vlm_model)
        logger.info("[Motus] Building Motus backbone helpers")
        self.video_backbone = MotusVideoBackbone(self.config, self.pre_weight.video, self.transformer_weights.video)
        self.action_backbone = MotusActionExpert(self.action_config, self.pre_weight.action, self.transformer_weights.action, self.post_weight.action)
        image_context_weights = self.pre_weight.image_context if getattr(self.pre_weight.image_context, "available", True) else None
        self.und_backbone = MotusUndExpert(
            self.pre_weight.und,
            self.transformer_weights.und,
            image_context_weights,
            self.vlm_model,
            self.device,
            self.model_dtype,
        )
        logger.info("[Motus] Initializing infer modules")
        self._init_infer()
        for param in self.vlm_model.parameters():
            param.requires_grad = False

        lat_t = 1 + self.config["num_video_frames"] // 4
        lat_h = self.config["video_height"] // 32
        lat_w = self.config["video_width"] // 32
        batch_size = int(self.config.get("batch_size", 1))
        self.grid_sizes = torch.tensor([lat_t, lat_h, lat_w], dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        self.scheduler = None

    @property
    def action_chunk_size(self):
        config = getattr(self, "config", None) or {}
        return config.get("num_video_frames", 0) * config.get("video_action_freq_ratio", 2)

    @property
    def action_dim(self):
        config = getattr(self, "config", None) or {}
        return config.get("action_dim", 14)

    @staticmethod
    def _apply_motus_defaults(config):
        config.setdefault("task", "t2v")
        config.setdefault("model_cls", "wan2.2")
        config.setdefault("freq_dim", 256)
        config.setdefault("cpu_offload", False)
        config.setdefault("seq_parallel", False)
        config.setdefault("parallel", {})
        config.setdefault("rms_norm_type", "torch")
        config.setdefault("self_attn_1_type", config.get("self_joint_attn_type", config.get("attention_type", "flash_attn2")))
        config.setdefault("cross_attn_1_type", config.get("cross_attn_type", config.get("attention_type", "flash_attn2")))
        config.setdefault("cross_attn_2_type", config.get("cross_attn_type", config.get("attention_type", "flash_attn2")))
        return config

    def _should_load_pretrained_backbones(self):
        load_backbones = self.config.get("load_pretrained_backbones")
        return True if load_backbones is None else bool(load_backbones)

    def _init_infer_class(self):
        self.pre_infer_class = MotusPreInfer
        self.transformer_infer_class = MotusTransformerInfer
        self.post_infer_class = MotusPostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self, self.config)
        self.transformer_infer = self.transformer_infer_class(self, self.config)
        self.post_infer = self.post_infer_class(self, self.config)

    def _move_weight_tree(self, node, move_to_cuda, non_blocking=False):
        if node is None:
            return

        modules = getattr(node, "_modules", None)
        parameters = getattr(node, "_parameters", None)
        if isinstance(modules, dict) or isinstance(parameters, dict):
            if isinstance(parameters, dict):
                for parameter in parameters.values():
                    self._move_weight_tree(parameter, move_to_cuda, non_blocking=non_blocking)
            if isinstance(modules, dict):
                for module in modules.values():
                    self._move_weight_tree(module, move_to_cuda, non_blocking=non_blocking)
            return

        move_fn = getattr(node, "to_cuda" if move_to_cuda else "to_cpu", None)
        if callable(move_fn):
            move_fn(non_blocking=non_blocking)

    def to_cuda(self):
        self._move_weight_tree(self.pre_weight, move_to_cuda=True)
        self._move_weight_tree(self.transformer_weights, move_to_cuda=True)
        if hasattr(self, "post_weight"):
            self._move_weight_tree(self.post_weight, move_to_cuda=True)

    def to_cpu(self):
        self._move_weight_tree(self.pre_weight, move_to_cuda=False)
        self._move_weight_tree(self.transformer_weights, move_to_cuda=False)
        if hasattr(self, "post_weight"):
            self._move_weight_tree(self.post_weight, move_to_cuda=False)

    def _ensure_pre_weights_on_device(self):
        if self.device.type == "cpu":
            return
        probe_weight = getattr(self.pre_weight.video.text_embedding_0, "weight", None)
        if probe_weight is None or probe_weight.device != self.device:
            self._move_weight_tree(self.pre_weight, move_to_cuda=True)

    def _ensure_runtime_weights_on_device(self):
        self._ensure_pre_weights_on_device()
        if self.device.type == "cpu":
            return
        probe_weight = getattr(self.transformer_weights.video.head, "weight", None)
        if probe_weight is None or probe_weight.device != self.device:
            self._move_weight_tree(self.transformer_weights, move_to_cuda=True)
            if hasattr(self, "post_weight"):
                self._move_weight_tree(self.post_weight, move_to_cuda=True)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)

    def _resolve_checkpoint_path(self):
        checkpoint_path = Path(self.model_path).expanduser().resolve()
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "mp_rank_00_model_states.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        return checkpoint_path

    def _load_checkpoint_state(self):
        checkpoint = torch.load(self._resolve_checkpoint_path(), map_location="cpu")
        if isinstance(checkpoint, dict) and "module" in checkpoint:
            return checkpoint["module"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def _merge_pretrained_video_weights(self, weight_dict):
        if not self._should_load_pretrained_backbones():
            return weight_dict
        if not self.config.get("wan_path"):
            return weight_dict
        wan_weight_dict = load_weights(
            self.config["wan_path"],
            cpu_offload=self.cpu_offload,
            load_from_rank0=self.config.get("load_from_rank0", False),
        )
        for key, value in wan_weight_dict.items():
            full_key = f"video_model.wan_model.{key}"
            weight_dict.setdefault(full_key, value)
        return weight_dict

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        logger.info("[Motus] Reading checkpoint for weight containers")
        state_dict = self._load_checkpoint_state()
        weight_dict = {}
        cached_vlm_state = {}
        for key in list(state_dict.keys()):
            tensor = state_dict.pop(key)
            if not torch.is_tensor(tensor):
                continue
            if key.startswith("vlm_model."):
                cached_vlm_state[key[len("vlm_model.") :]] = tensor
                continue
            if key.startswith("video_model.vae."):
                continue
            if tensor.is_floating_point():
                target_dtype = GET_DTYPE() if unified_dtype or all(pattern not in key for pattern in sensitive_layer) else GET_SENSITIVE_DTYPE()
                weight_dict[key] = tensor.to(target_dtype)
            else:
                weight_dict[key] = tensor
        self._cached_vlm_state = cached_vlm_state
        del state_dict
        gc.collect()
        return self._merge_pretrained_video_weights(weight_dict)

    def _load_vlm_model(self):
        load_backbones = self._should_load_pretrained_backbones()
        if load_backbones:
            if self.device.type == "cpu":
                vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.config["vlm_path"],
                    torch_dtype=self.model_dtype,
                    trust_remote_code=True,
                )
            else:
                vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.config["vlm_path"],
                    torch_dtype=self.model_dtype,
                    device_map="cuda",
                    trust_remote_code=True,
                )
        else:
            vlm_cfg = AutoConfig.from_pretrained(self.config["vlm_path"], trust_remote_code=True)
            vlm_model = Qwen3VLForConditionalGeneration._from_config(vlm_cfg, torch_dtype=self.model_dtype)
            vlm_model.to(device=self.device, dtype=self.model_dtype)

        vlm_state = self._cached_vlm_state
        if vlm_state is None:
            logger.info("[Motus] Reading checkpoint again for VLM weights")
            checkpoint_state = self._load_checkpoint_state()
            vlm_state = {key[len("vlm_model.") :]: value for key, value in checkpoint_state.items() if key.startswith("vlm_model.")}
        if vlm_state:
            logger.info("[Motus] Loading cached VLM state dict")
            vlm_model.load_state_dict(vlm_state, strict=False)
        self._cached_vlm_state = None
        gc.collect()
        return vlm_model

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

    def denormalize_actions(self, actions):
        shape = actions.shape
        flat = actions.reshape(-1, shape[-1])
        restored = flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return restored.reshape(shape)

    def get_wan_freqs(self):
        return self.pre_infer.freqs

    def get_wan_rotary_cos_sin(self, grid_size):
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

    def prepare_frame(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image).astype(np.float32) / 255.0
        resized_np = resize_with_padding(
            image_np,
            (self.config.get("video_height", 384), self.config.get("video_width", 320)),
        )
        if resized_np.dtype == np.uint8:
            resized_np = resized_np.astype(np.float32) / 255.0
        return torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def prepare_state(self, state_value):
        if isinstance(state_value, torch.Tensor):
            state = state_value.float()
        else:
            state = torch.tensor(state_value, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return state.to(self.device)

    def build_instruction(self, prompt):
        prefix = self.config.get(
            "scene_prefix",
            "The whole scene is in a realistic, industrial art style with three views: "
            "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
            "The aloha robot is currently performing the following task: ",
        )
        return f"{prefix}{prompt}"

    def _tensor_to_pil(self, tensor):
        tensor = tensor.float().clamp(0, 1)
        np_img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_img, mode="RGB")

    def build_vlm_inputs(self, instruction, first_frame):
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
    def prepare_runtime_inputs(self, inputs, image_path, prompt, state_value):
        self._ensure_pre_weights_on_device()
        first_frame = self.prepare_frame(image_path)
        state = self.prepare_state(state_value)
        instruction = self.build_instruction(prompt)
        t5_context = inputs["text_encoder_output"]["context"]
        processed_t5_context = self.video_backbone.preprocess_t5_embeddings(t5_context)
        vlm_inputs = [self.build_vlm_inputs(instruction, first_frame)]
        und_tokens = self.und_backbone.extract_und_features(vlm_inputs)
        image_context = self.und_backbone.extract_image_context(vlm_inputs)
        inputs.update(
            {
                "motus_first_frame": first_frame,
                "motus_state": state,
                "motus_instruction": instruction,
                "motus_t5_embeddings": t5_context,
                "motus_processed_t5_context": processed_t5_context,
                "motus_vlm_inputs": vlm_inputs,
                "motus_und_tokens": und_tokens,
                "motus_image_context": image_context,
            }
        )
        return inputs

    @compiled_method()
    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        del infer_condition
        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)
        video_velocity, action_velocity = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)
        self.scheduler.noise_pred = video_velocity.squeeze(0)
        self.scheduler.action_noise_pred = action_velocity
        return video_velocity

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        return x

    @torch.no_grad()
    def infer(self, inputs):
        if self.scheduler is None:
            raise RuntimeError("MotusModel requires a scheduler before infer().")
        self._ensure_runtime_weights_on_device()
        self._infer_cond_uncond(inputs, infer_condition=True)

    @torch.no_grad()
    def postprocess_actions(self):
        if self.scheduler is None:
            raise RuntimeError("MotusModel requires a scheduler before postprocess_actions().")
        return self.post_infer.infer(self.scheduler.action_latents, None)
