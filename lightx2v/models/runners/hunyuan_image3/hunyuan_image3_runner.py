import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from loguru import logger
from safetensors import safe_open
from tqdm.auto import tqdm
from transformers import GenerationConfig

from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3StaticKVCache
from lightx2v.models.networks.hunyuan_image3.model import HunyuanImage3Model
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.hunyuan_image3.flashinfer_autotune import (
    DistributedAutotuneContext,
    FlashInferAutotuneController,
)
from lightx2v.models.schedulers.hunyuan_image3.scheduler import HunyuanImage3Scheduler
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@dataclass(frozen=True)
class HunyuanImage3TextGenerationPlan:
    first_bot_task: str
    stage_transitions: list[tuple[int, list[int]]]
    final_stop_tokens: list[int]


@RUNNER_REGISTER("hunyuan_image3")
class HunyuanImage3Runner(DefaultRunner):
    model_cpu_offload_seq = "transformer"

    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = []
        self.image_encoder = None
        self.vae_encoder = None
        self.vae_decoder = None

    def load_transformer(self):
        logger.info("Loading native HunyuanImage3 transformer weights")
        return HunyuanImage3Model(
            model_path=self.config["model_path"],
            config=self.config,
            device=self.init_device,
        )

    def load_text_encoder(self):
        return []

    def load_image_encoder(self):
        return None

    def load_vae(self):
        return None, None

    def init_scheduler(self):
        super().init_scheduler()
        self.scheduler = HunyuanImage3Scheduler(self.config)

    def init_modules(self):
        super().init_modules()
        self.run_dit = self._run_dit_local

    def _run_dit_local(self, total_steps=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        return self.run_main(total_steps=total_steps)

    def _resolve_upstream_repo_path(self):
        candidates = [
            self.config.get("hunyuan_image3_repo_path"),
            os.environ.get("HUNYUAN_IMAGE3_REPO_PATH"),
        ]
        model_path = Path(self.config["model_path"]).resolve()
        candidates.extend(parent / "HunyuanImage-3.0" for parent in model_path.parents)
        candidates.append(Path.cwd().parent / "HunyuanImage-3.0")

        for candidate in candidates:
            if not candidate:
                continue
            candidate = Path(candidate).expanduser().resolve()
            if (candidate / "hunyuan_image_3" / "__init__.py").exists():
                return str(candidate)
        return None

    def _import_upstream_modules(self):
        try:
            importlib.import_module("hunyuan_image_3")
        except ModuleNotFoundError:
            repo_path = self._resolve_upstream_repo_path()
            if repo_path is None:
                raise ModuleNotFoundError("Cannot import hunyuan_image_3. Set HUNYUAN_IMAGE3_REPO_PATH or `hunyuan_image3_repo_path` to the HunyuanImage-3.0 project path.")
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        return {
            "AutoencoderKLConv3D": importlib.import_module("hunyuan_image_3.autoencoder_kl_3d").AutoencoderKLConv3D,
            "CachedRoPE": importlib.import_module("hunyuan_image_3.modeling_hunyuan_image_3").CachedRoPE,
            "HunyuanImage3Config": importlib.import_module("hunyuan_image_3.configuration_hunyuan_image_3").HunyuanImage3Config,
            "HunyuanImage3ImageProcessor": importlib.import_module("hunyuan_image_3.image_processor").HunyuanImage3ImageProcessor,
            "LightProjector": importlib.import_module("hunyuan_image_3.siglip2").LightProjector,
            "Siglip2VisionTransformer": importlib.import_module("hunyuan_image_3.siglip2").Siglip2VisionTransformer,
            "HunyuanImage3TokenizerFast": importlib.import_module("hunyuan_image_3.tokenization_hunyuan_image_3").HunyuanImage3TokenizerFast,
            "get_system_prompt": importlib.import_module("hunyuan_image_3.system_prompt").get_system_prompt,
        }

    def _ensure_pipeline_modules(self):
        if getattr(self, "_hunyuan_pipeline_modules_ready", False):
            return

        modules = self._import_upstream_modules()
        model_path = self.config["model_path"]
        self.hunyuan_config = modules["HunyuanImage3Config"].from_pretrained(model_path)
        try:
            self.hunyuan_generation_config = GenerationConfig.from_pretrained(model_path)
        except OSError:
            self.hunyuan_generation_config = GenerationConfig()

        self.hunyuan_tokenizer = modules["HunyuanImage3TokenizerFast"].from_pretrained(
            model_path,
            model_version=self.hunyuan_config.model_version,
        )
        self.hunyuan_image_processor = modules["HunyuanImage3ImageProcessor"](self.hunyuan_config)
        self.hunyuan_cached_rope = modules["CachedRoPE"](self.hunyuan_config)
        self.hunyuan_vae_cls = modules["AutoencoderKLConv3D"]
        self.hunyuan_vision_cls = modules["Siglip2VisionTransformer"]
        self.hunyuan_vision_aligner_cls = modules["LightProjector"]
        self.hunyuan_get_system_prompt = modules["get_system_prompt"]

        # self.vae_decoder is a complete VAE model (encoder + decoder). T2I only
        # decodes on rank 0, so other SP ranks do not need this extra allocation.
        load_vae_on_this_rank = not (self.config.get("task") == "t2i" and self._sequence_parallel_enabled() and not self._is_output_rank())
        if load_vae_on_this_rank:
            self.vae_decoder = self._load_vae_decoder()
        else:
            self.vae_decoder = None
            logger.info("Skipping HunyuanImage3 VAE load on non-output SP rank for T2I.")
        self.vision_model = None
        self.vision_aligner = None
        self._hunyuan_pipeline_modules_ready = True

    def _pipeline_latent_device(self):
        devices = getattr(self.model, "pipeline_devices", None)
        if devices:
            return torch.device(devices[0])
        return torch.device(self.init_device)

    def _vae_device(self):
        configured = self.config.get("hunyuan_image3_vae_device")
        if configured is not None:
            return torch.device(str(configured))
        devices = getattr(self.model, "pipeline_devices", None)
        if devices:
            return torch.device(devices[-1])
        return torch.device(self.init_device)

    def _vision_device(self):
        configured = self.config.get("hunyuan_image3_vision_device")
        if configured is not None:
            return torch.device(str(configured))
        return self._pipeline_latent_device()

    def _resolve_torch_dtype(self, value, default):
        if value in (None, "auto"):
            return default
        if isinstance(value, torch.dtype):
            return value
        return getattr(torch, str(value), default)

    def _iter_vae_weight_files(self):
        model_path = Path(self.config["model_path"])
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            import json

            weight_map = json.loads(index_path.read_text())["weight_map"]
            files = sorted({filename for key, filename in weight_map.items() if key.startswith("vae.")})
            return [model_path / filename for filename in files]
        return sorted(model_path.glob("*.safetensors"))

    def _iter_prefixed_weight_files(self, prefix):
        model_path = Path(self.config["model_path"])
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            import json

            weight_map = json.loads(index_path.read_text())["weight_map"]
            files = sorted({filename for key, filename in weight_map.items() if key.startswith(prefix)})
            return [model_path / filename for filename in files]
        return sorted(model_path.glob("*.safetensors"))

    def _load_vae_decoder(self):
        logger.info("Loading HunyuanImage3 VAE weights")
        vae = self.hunyuan_vae_cls.from_config(self.hunyuan_config.vae)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        loaded_keys = set()
        for file_path in self._iter_vae_weight_files():
            state_dict = {}
            with safe_open(str(file_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("vae."):
                        state_key = key.removeprefix("vae.")
                        state_dict[state_key] = f.get_tensor(key)
                        loaded_keys.add(state_key)
            if state_dict:
                vae.load_state_dict(state_dict, strict=False)

        missing = sorted(set(vae.state_dict()) - loaded_keys)
        if missing:
            raise RuntimeError(f"HunyuanImage3 VAE weights are incomplete; missing {len(missing)} keys, first key: {missing[0]}")

        vae_dtype = getattr(torch, self.hunyuan_config.vae_dtype, torch.float32)
        return vae.to(device=self._vae_device(), dtype=vae_dtype)

    def _load_prefixed_module_weights(self, module, prefix):
        loaded_keys = set()
        for file_path in self._iter_prefixed_weight_files(prefix):
            state_dict = {}
            with safe_open(str(file_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        state_key = key.removeprefix(prefix)
                        state_dict[state_key] = f.get_tensor(key)
                        loaded_keys.add(state_key)
            if state_dict:
                module.load_state_dict(state_dict, strict=False)

        missing = sorted(set(module.state_dict()) - loaded_keys)
        if missing:
            raise RuntimeError(f"HunyuanImage3 {prefix.rstrip('.')} weights are incomplete; missing {len(missing)} keys, first key: {missing[0]}")

    def _ensure_vision_encoder(self):
        if self.vision_model is not None and self.vision_aligner is not None:
            return

        logger.info("Loading HunyuanImage3 conditional vision encoder weights")
        vision_model = self.hunyuan_vision_cls(self.hunyuan_config.vit)
        vision_aligner = self.hunyuan_vision_aligner_cls(self.hunyuan_config.vit_aligner)
        vision_model.eval()
        vision_aligner.eval()
        for module in (vision_model, vision_aligner):
            for param in module.parameters():
                param.requires_grad = False

        self._load_prefixed_module_weights(vision_model, "vision_model.")
        self._load_prefixed_module_weights(vision_aligner, "vision_aligner.")

        vision_dtype = self._resolve_torch_dtype(self.config.get("hunyuan_image3_vision_dtype"), torch.bfloat16)
        vision_device = self._vision_device()
        self.vision_model = vision_model.to(device=vision_device, dtype=vision_dtype)
        self.vision_aligner = vision_aligner.to(device=vision_device, dtype=vision_dtype)

    def _resolve_image_size(self, input_info):
        if self.config.get("image_size"):
            return self.config["image_size"]
        target_shape = getattr(input_info, "target_shape", None) or self.config.get("target_shape")
        if target_shape:
            if len(target_shape) >= 2:
                return int(target_shape[-2]), int(target_shape[-1])
        return int(self.config.get("target_height", 1024)), int(self.config.get("target_width", 1024))

    def _build_batch_rope_image_info(self, output, sections):
        if self.hunyuan_config.rope_type == "default":
            return None
        if self.hunyuan_config.rope_type != "2d":
            raise NotImplementedError(f"HunyuanImage3 rope type {self.hunyuan_config.rope_type} is not supported.")

        rope_image_info = []
        for image_slices, sample_sections in zip(output.all_image_slices, sections):
            image_idx = 0
            sample_info = []
            for section in sample_sections:
                if section["type"] in ("gen_image", "cond_vae_image", "cond_vit_image"):
                    sample_info.append((image_slices[image_idx], (section["token_height"], section["token_width"])))
                    image_idx += 1
                elif section["type"] == "cond_joint_image":
                    if self.hunyuan_image_processor.cond_token_attn_type in ("full", "joint_full"):
                        sample_info.extend(
                            [
                                (image_slices[image_idx], (section["token_height"][0], section["token_width"][0])),
                                (image_slices[image_idx + 1], (section["token_height"][1], section["token_width"][1])),
                            ]
                        )
                    elif self.hunyuan_image_processor.cond_token_attn_type == "full_causal":
                        sample_info.append((image_slices[image_idx], (section["token_height"][0], section["token_width"][0])))
                    elif self.hunyuan_image_processor.cond_token_attn_type == "causal":
                        pass
                    else:
                        raise NotImplementedError(f"HunyuanImage3 cond_token_attn_type={self.hunyuan_image_processor.cond_token_attn_type!r} is not supported.")
                    image_idx += 2
            rope_image_info.append(sample_info)
        return rope_image_info

    def _build_attention_mask(self, input_ids, tokenizer_output):
        batch, seq_len = input_ids.shape
        attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device).tril(diagonal=0).repeat(batch, 1, 1)
        full_attn_slices = self._build_full_attn_slices(tokenizer_output, batch, seq_len=seq_len)
        for batch_idx in range(batch):
            for start, stop in full_attn_slices[batch_idx]:
                attention_mask[batch_idx, start:stop, start:stop] = True
        return attention_mask.unsqueeze(1)

    def _build_full_attn_slices(self, tokenizer_output, batch, seq_len=None):
        batch_slices = []
        for batch_idx in range(batch):
            sample_slices = []
            for image_slice in self.hunyuan_image_processor.prepare_full_attn_slices(tokenizer_output, batch_idx):
                start = 0 if image_slice.start is None else int(image_slice.start)
                stop = seq_len if image_slice.stop is None else int(image_slice.stop)
                if seq_len is not None:
                    start = max(0, min(start, seq_len))
                    stop = max(0, min(stop, seq_len))
                if stop > start:
                    sample_slices.append((start, stop))
            sample_slices.sort()
            batch_slices.append(sample_slices)
        return batch_slices

    def _hunyuan_kv_cache_enabled(self):
        return bool(self.config.get("enable_kv_cache", True))

    def _hunyuan_text_kv_cache_enabled(self):
        if "enable_text_kv_cache" in self.config:
            return bool(self.config["enable_text_kv_cache"])
        if "enable_kv_cache" in self.config:
            return bool(self.config["enable_kv_cache"])
        return hasattr(self, "hunyuan_config")

    def _hunyuan_num_layers(self):
        hunyuan_config = getattr(self, "hunyuan_config", None)
        fallback = self.config.get("num_hidden_layers", getattr(hunyuan_config, "num_hidden_layers", 1))
        return int(self.config.get("num_layers") or fallback)

    def _hunyuan_taylor_cache_enabled(self):
        return bool(self.config.get("use_taylor_cache", False))

    def _build_flashinfer_autotune_controller(self):
        sequence_parallel_active = self._sequence_parallel_enabled()
        distributed_context = DistributedAutotuneContext(
            coordination_required=sequence_parallel_active,
            process_group=self._parallel_control_group() if sequence_parallel_active else None,
            status_device_resolver=self._pipeline_latent_device if sequence_parallel_active else None,
            is_cache_writer=self._is_output_rank() if sequence_parallel_active else False,
        )
        return FlashInferAutotuneController.from_config(
            config=self.config,
            distributed_context=distributed_context,
        )

    def _sequence_parallel_enabled(self):
        return bool(self.config.get("seq_parallel", False) and dist.is_available() and dist.is_initialized() and getattr(self.model, "seq_p_group", None) is not None)

    def _parallel_control_group(self):
        """Return the group used to keep replicated runner state identical.

        Attention collectives remain scoped to ``seq_p_group`` and CFG prediction
        exchange remains scoped to ``cfg_p_group``. Runner state such as sampled
        tokens and latents is synchronized only for sequence-parallel runs.
        Hybrid runs use WORLD so the state also spans the CFG dimension. Pure
        CFG deliberately returns no runner control group: its only collective
        is the prediction all-gather in the model, matching the known-good CFG
        implementation and avoiding cfg_p collectives on both ends of a
        multi-device pipeline.
        """
        if not self._sequence_parallel_enabled():
            return None
        if self._cfg_parallel_enabled():
            return dist.group.WORLD
        return self.model.seq_p_group

    def _is_output_rank(self):
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

    def _broadcast_parallel_tensor(self, tensor):
        group = self._parallel_control_group()
        if group is None:
            return tensor
        source_rank = dist.get_global_rank(group, 0)
        dist.broadcast(tensor, src=source_rank, group=group)
        return tensor

    def _parallel_barrier(self):
        group = self._parallel_control_group()
        if group is None:
            return
        if dist.get_backend(group) == "nccl":
            barrier_device = self._pipeline_latent_device()
            torch.cuda.set_device(barrier_device)
            device_index = barrier_device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            dist.barrier(group=group, device_ids=[device_index])
        else:
            dist.barrier(group=group)

    def _build_taylor_cache_dic(self, num_steps):
        return {
            "counter": 0,
            "current_step": 0,
            "cache_interval": int(self.config.get("taylor_cache_interval", 5)),
            "max_order": int(self.config.get("taylor_cache_order", 2)),
            "num_steps": int(num_steps),
            "enable_first_enhance": bool(self.config.get("taylor_cache_enable_first_enhance", False)),
            "first_enhance_steps": int(self.config.get("taylor_cache_first_enhance_steps", 3)),
            "enable_tailing_enhance": bool(self.config.get("taylor_cache_enable_tailing_enhance", False)),
            "tailing_enhance_steps": int(self.config.get("taylor_cache_tailing_enhance_steps", 1)),
            "low_freqs_order": int(self.config.get("taylor_cache_low_freqs_order", 2)),
            "high_freqs_order": int(self.config.get("taylor_cache_high_freqs_order", 2)),
            "enable_force_control": False,
            "force_compute": False,
        }

    def _build_denoise_cache_position_ids(self, prepared_inputs):
        image_mask = prepared_inputs["image_mask"]
        batch, seq_len = image_mask.shape
        device = image_mask.device
        batch_index = torch.arange(batch, device=device)
        token_index = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch, 1)
        position_parts = []

        timesteps_index = prepared_inputs.get("timesteps_index")
        if timesteps_index is not None:
            position_parts.append(token_index[batch_index, timesteps_index[:, -1]].unsqueeze(-1))

        guidance_index = prepared_inputs.get("guidance_index")
        if guidance_index is not None:
            position_parts.append(token_index[batch_index, guidance_index[:, -1]].unsqueeze(-1))

        timesteps_r_index = prepared_inputs.get("timesteps_r_index")
        if timesteps_r_index is not None:
            position_parts.append(token_index[batch_index, timesteps_r_index[:, -1]].unsqueeze(-1))

        image_position_ids = token_index.masked_select(image_mask.bool()).reshape(batch, -1)
        position_parts.append(image_position_ids)
        return torch.cat(position_parts, dim=1)

    def _slice_denoise_cache_attention_mask(self, attention_mask, position_ids):
        if attention_mask is None:
            return None
        mask_parts = []
        for sample_attention_mask, sample_position_ids in zip(attention_mask, position_ids):
            mask_parts.append(sample_attention_mask.index_select(dim=1, index=sample_position_ids.reshape(-1)))
        return torch.stack(mask_parts, dim=0)

    def _build_denoise_cache_local_indices(self, prepared_inputs, position_ids):
        image_tokens = int(prepared_inputs["image_mask"][0].sum().item())
        batch, local_seq_len = position_ids.shape
        device = position_ids.device
        special_tokens = local_seq_len - image_tokens
        local_image_mask = torch.zeros((batch, local_seq_len), dtype=torch.bool, device=device)
        local_image_mask[:, special_tokens:] = True

        local_index = 0
        local_inputs = {
            "image_mask": local_image_mask,
            "timesteps_index": None,
            "guidance_index": None,
            "timesteps_r_index": None,
        }
        if prepared_inputs.get("timesteps_index") is not None:
            local_inputs["timesteps_index"] = torch.full((batch, 1), local_index, dtype=torch.long, device=device)
            local_index += 1
        if prepared_inputs.get("guidance_index") is not None:
            local_inputs["guidance_index"] = torch.full((batch, 1), local_index, dtype=torch.long, device=device)
            local_index += 1
        if prepared_inputs.get("timesteps_r_index") is not None:
            local_inputs["timesteps_r_index"] = torch.full((batch, 1), local_index, dtype=torch.long, device=device)
        return local_inputs

    def _build_custom_pos_emb(self, position_ids, rope_image_info):
        return self.hunyuan_cached_rope(
            self.config.get("max_position_embeddings", self.hunyuan_config.max_position_embeddings),
            position_ids.device,
            rope_image_info=rope_image_info,
            position_ids=position_ids,
        )

    def _cfg_parallel_enabled(self):
        return bool(self.config.get("cfg_parallel", False)) and bool(self.config.get("enable_cfg", False))

    def _is_distributed_nonzero_rank(self):
        return dist.is_available() and dist.is_initialized() and dist.get_rank() != 0

    def _cfg_parallel_rank(self):
        if not self._cfg_parallel_enabled():
            return 0
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("HunyuanImage3 cfg_parallel requires torch.distributed to be initialized.")
        cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
        assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
        return dist.get_rank(cfg_p_group)

    def _slice_cfg_branch_value(self, value, branch_idx, cfg_batch_size):
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and value.shape[0] == cfg_batch_size:
                return value[branch_idx : branch_idx + 1]
            return value
        if isinstance(value, list):
            if len(value) == cfg_batch_size:
                return [value[branch_idx]]
            return [self._slice_cfg_branch_value(item, branch_idx, cfg_batch_size) for item in value]
        if isinstance(value, tuple):
            if len(value) == cfg_batch_size:
                return (value[branch_idx],)
            return tuple(self._slice_cfg_branch_value(item, branch_idx, cfg_batch_size) for item in value)
        if isinstance(value, dict):
            return {k: self._slice_cfg_branch_value(v, branch_idx, cfg_batch_size) for k, v in value.items()}
        return value

    def _resolve_denoise_cfg_mode(self, prepared_inputs):
        if not prepared_inputs.get("do_cfg", False):
            return "none"

        mode = str(self.config.get("hunyuan_cfg_mode", "batch")).lower()
        if mode not in ("batch", "serial", "parallel"):
            raise ValueError(f"HunyuanImage3 hunyuan_cfg_mode must be one of batch/serial/parallel, got {mode!r}.")
        if mode == "parallel" and not self._cfg_parallel_enabled():
            raise ValueError("HunyuanImage3 hunyuan_cfg_mode='parallel' requires enable_cfg=true and a distributed config with parallel.cfg_p_size=2.")
        return mode

    def _prepare_cfg_parallel_branch_inputs(self, prepared_inputs, cfg_p_rank, mark_parallel_branch=True):
        cfg_batch_size = int(prepared_inputs["input_ids"].shape[0])
        if cfg_batch_size != 2:
            raise ValueError(f"HunyuanImage3 CFG branch inputs expect packed cond/uncond batch size 2, got {cfg_batch_size}.")

        branch_inputs = {}
        for key, value in prepared_inputs.items():
            if key in ("custom_pos_emb",):
                continue
            branch_inputs[key] = self._slice_cfg_branch_value(value, cfg_p_rank, cfg_batch_size)

        branch_inputs["batch_size"] = 1
        branch_inputs["do_cfg"] = False
        if mark_parallel_branch:
            branch_inputs["_cfg_parallel_branch"] = True
        branch_inputs["custom_pos_emb"] = self._build_custom_pos_emb(branch_inputs["position_ids"], branch_inputs.get("rope_image_info"))
        return branch_inputs

    def _build_denoise_kv_state(self, prepared_inputs, use_kv_cache):
        if not use_kv_cache:
            return {
                "kv_cache": None,
                "cache_position_ids": None,
                "cache_local_inputs": None,
            }
        cache_position_ids = self._build_denoise_cache_position_ids(prepared_inputs)
        return {
            "kv_cache": HunyuanImage3StaticKVCache(
                num_layers=self._hunyuan_num_layers(),
                max_cache_len=prepared_inputs["input_ids"].shape[1],
            ),
            "cache_position_ids": cache_position_ids,
            "cache_local_inputs": self._build_denoise_cache_local_indices(prepared_inputs, cache_position_ids),
        }

    def _build_denoise_model_inputs(self, prepared_inputs, latent_model_input, timestep, step_index, use_kv_cache, kv_state, guidance_scale):
        timestep_input = timestep.repeat(latent_model_input.shape[0])
        first_step = step_index == 0 or not use_kv_cache
        if first_step:
            model_inputs = {
                "input_ids": prepared_inputs["input_ids"],
                "attention_mask": prepared_inputs["attention_mask"],
                "full_attn_slices": prepared_inputs.get("full_attn_slices"),
                "position_ids": prepared_inputs["position_ids"],
                "custom_pos_emb": prepared_inputs["custom_pos_emb"],
                "images": latent_model_input,
                "image_mask": prepared_inputs["image_mask"],
                "timesteps": timestep_input,
                "timesteps_index": prepared_inputs["timesteps_index"],
                "guidance_index": prepared_inputs["guidance_index"],
                "timesteps_r_index": prepared_inputs["timesteps_r_index"],
                "first_step": True,
            }
            if prepared_inputs.get("cond_vae_images") is not None:
                model_inputs["cond_vae_images"] = prepared_inputs["cond_vae_images"]
                model_inputs["cond_vae_image_mask"] = prepared_inputs.get("cond_vae_image_mask")
                model_inputs["cond_timesteps"] = prepared_inputs.get("cond_timesteps")
                model_inputs["cond_timestep_index"] = prepared_inputs.get("cond_timestep_index")
            if prepared_inputs.get("cond_vit_embeds") is not None:
                model_inputs["cond_vit_embeds"] = prepared_inputs["cond_vit_embeds"]
                model_inputs["cond_vit_image_mask"] = prepared_inputs.get("cond_vit_image_mask")
        else:
            cache_position_ids = kv_state["cache_position_ids"]
            cache_local_inputs = kv_state["cache_local_inputs"]
            model_inputs = {
                "input_ids": None,
                "attention_mask": self._slice_denoise_cache_attention_mask(prepared_inputs["attention_mask"], cache_position_ids),
                "full_attn_slices": prepared_inputs.get("full_attn_slices"),
                "position_ids": cache_position_ids,
                "custom_pos_emb": self._build_custom_pos_emb(cache_position_ids, prepared_inputs.get("rope_image_info")),
                "images": latent_model_input,
                "image_mask": cache_local_inputs["image_mask"],
                "timesteps": timestep_input,
                "timesteps_index": cache_local_inputs["timesteps_index"],
                "guidance_index": cache_local_inputs["guidance_index"],
                "timesteps_r_index": cache_local_inputs["timesteps_r_index"],
                "first_step": False,
            }

        if prepared_inputs.get("_cfg_parallel_branch", False):
            model_inputs["_cfg_parallel_branch"] = True
        if use_kv_cache:
            model_inputs["past_key_values"] = kv_state["kv_cache"]
            model_inputs["use_cache"] = True
        if self.config.get("cfg_distilled", False):
            model_inputs["guidance"] = torch.tensor([1000.0 * guidance_scale], device=latent_model_input.device, dtype=torch.bfloat16)
        if self.config.get("use_meanflow", False):
            raise NotImplementedError("HunyuanImage3 native meanflow sampling is not implemented yet.")
        return model_inputs

    def _attach_tokenizer_cond_masks(self, cond_inputs, tokenizer_output, device):
        if not cond_inputs:
            return cond_inputs
        cond_inputs = dict(cond_inputs)
        cond_inputs["cond_vae_image_mask"] = None if getattr(tokenizer_output, "vae_image_mask", None) is None else tokenizer_output.vae_image_mask.to(device)
        cond_inputs["cond_vit_image_mask"] = None if getattr(tokenizer_output, "vit_image_mask", None) is None else tokenizer_output.vit_image_mask.to(device)
        cond_inputs["cond_timestep_index"] = None if getattr(tokenizer_output, "cond_timestep_scatter_index", None) is None else tokenizer_output.cond_timestep_scatter_index.to(device)
        return cond_inputs

    def _resolve_bot_task(self):
        bot_task = self.config.get("bot_task", getattr(self.hunyuan_generation_config, "bot_task", "image"))
        return bot_task or "image"

    def _resolve_system_prompt(self, bot_task):
        system_prompt_type = self.config.get("use_system_prompt", getattr(self.hunyuan_generation_config, "use_system_prompt", "None"))
        system_prompt = self.config.get("system_prompt")
        if hasattr(self, "hunyuan_get_system_prompt"):
            return self.hunyuan_get_system_prompt(system_prompt_type, bot_task, system_prompt)
        if system_prompt_type == "custom":
            return system_prompt
        return system_prompt if system_prompt_type not in (None, "None") else None

    def _resolve_text_generation_plan(self, bot_task):
        tokenizer = self.hunyuan_tokenizer
        if bot_task == "recaption":
            return HunyuanImage3TextGenerationPlan(
                first_bot_task="recaption",
                stage_transitions=[],
                final_stop_tokens=[tokenizer.end_of_recaption_token_id],
            )
        if bot_task == "think_recaption":
            recaption_token_id = tokenizer.convert_tokens_to_ids(tokenizer.recaption_token)
            return HunyuanImage3TextGenerationPlan(
                first_bot_task="think",
                stage_transitions=[(tokenizer.end_of_think_token_id, [recaption_token_id])],
                final_stop_tokens=[tokenizer.end_of_recaption_token_id],
            )
        if bot_task == "think":
            return HunyuanImage3TextGenerationPlan(
                first_bot_task="think",
                stage_transitions=[],
                final_stop_tokens=[tokenizer.end_of_think_token_id, tokenizer.end_of_recaption_token_id],
            )
        raise NotImplementedError(f"HunyuanImage3 native runner does not support bot_task={bot_task!r}.")

    def _expand_sequence_mask(self, mask, seq_len):
        if mask is None:
            return None
        if mask.shape[1] == seq_len:
            return mask
        if mask.shape[1] > seq_len:
            return mask[:, :seq_len]
        pad = torch.zeros((mask.shape[0], seq_len - mask.shape[1]), dtype=mask.dtype, device=mask.device)
        return torch.cat([mask, pad], dim=1)

    def _build_text_model_inputs(
        self,
        input_ids,
        tokenizer_output,
        cond_inputs=None,
        position_ids=None,
        attention_mask=None,
        build_attention_mask=True,
        past_key_values=None,
        use_cache=False,
    ):
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device)[None].expand(input_ids.shape[0], -1)
        position_max = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else input_ids.shape[1]
        max_position = max(int(self.config.get("max_position_embeddings", self.hunyuan_config.max_position_embeddings)), position_max)
        custom_pos_emb = self.hunyuan_cached_rope(max_position, device, rope_image_info=None, position_ids=position_ids)
        if build_attention_mask and attention_mask is None:
            attention_mask = self._build_attention_mask(input_ids, tokenizer_output)
        full_attn_slices = self._build_full_attn_slices(tokenizer_output, input_ids.shape[0], seq_len=input_ids.shape[1]) if attention_mask is not None else None
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "custom_pos_emb": custom_pos_emb,
            "full_attn_slices": full_attn_slices,
        }
        if use_cache:
            model_inputs["past_key_values"] = past_key_values
            model_inputs["use_cache"] = True
        if cond_inputs:
            model_inputs.update(
                {
                    "cond_vae_images": cond_inputs.get("cond_vae_images"),
                    "cond_vae_image_mask": self._expand_sequence_mask(cond_inputs.get("cond_vae_image_mask"), input_ids.shape[1]),
                    "cond_timesteps": cond_inputs.get("cond_timesteps"),
                    "cond_timestep_index": cond_inputs.get("cond_timestep_index"),
                    "cond_vit_embeds": cond_inputs.get("cond_vit_embeds"),
                    "cond_vit_image_mask": self._expand_sequence_mask(cond_inputs.get("cond_vit_image_mask"), input_ids.shape[1]),
                }
            )
        return model_inputs

    def _prepare_text_generation_inputs(self, prompt, bot_task, system_prompt, batch_cond_images=None, cond_inputs=None):
        generation_config = self.hunyuan_generation_config
        tokenizer_out = self.hunyuan_tokenizer.apply_chat_template(
            batch_prompt=[prompt],
            mode="gen_text",
            batch_cond_images=batch_cond_images,
            batch_system_prompt=[system_prompt],
            max_length=getattr(generation_config, "max_length", self.config.get("max_position_embeddings")),
            bot_task=bot_task,
            image_base_size=self.hunyuan_image_processor.vae_reso_group.base_size,
            sequence_template=getattr(generation_config, "sequence_template", "pretrain"),
            cfg_factor=1,
            drop_think=getattr(generation_config, "drop_think", False),
        )
        latent_device = self._pipeline_latent_device()
        output = tokenizer_out["output"]
        return {
            "input_ids": output.tokens.to(latent_device),
            "tokenizer_output": output,
            "cond_inputs": self._attach_tokenizer_cond_masks(cond_inputs, output, latent_device),
        }

    def _sample_text_token(self, logits, generator):
        do_sample = bool(self.config.get("text_do_sample", getattr(self.hunyuan_generation_config, "do_sample", False)))
        logits = logits.float()
        if not do_sample:
            return torch.argmax(logits, dim=-1)

        temperature = float(self.config.get("text_temperature", getattr(self.hunyuan_generation_config, "temperature", 1.0)))
        if temperature > 0:
            logits = logits / temperature

        top_k = int(self.config.get("text_top_k", getattr(self.hunyuan_generation_config, "top_k", 0)) or 0)
        if top_k > 0 and top_k < logits.shape[-1]:
            kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth_values, torch.finfo(logits.dtype).min)

        top_p = float(self.config.get("text_top_p", getattr(self.hunyuan_generation_config, "top_p", 1.0)))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(sorted_remove).scatter(dim=-1, index=sorted_indices, src=sorted_remove)
            logits = logits.masked_fill(remove, torch.finfo(logits.dtype).min)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    def _text_stream_enabled(self):
        return bool(self.config.get("text_stream_output", True))

    def _decode_generated_text_token(self, token_id):
        token_tensor = torch.tensor([token_id], dtype=torch.long)
        return self.hunyuan_tokenizer.decode(token_tensor)

    def _print_text_generation_chunk(self, text="", end=""):
        if self._is_output_rank():
            print(text, end=end, flush=True)

    @torch.no_grad()
    def _generate_text_tokens(self, input_ids, tokenizer_output, plan, stream_output=False, cond_inputs=None):
        device = input_ids.device
        seed = int(self.config.get("text_seed", self.config.get("seed", 42)))
        generator = torch.Generator(device=device).manual_seed(seed)
        max_new_tokens = int(self.config.get("max_new_tokens", getattr(self.hunyuan_generation_config, "max_new_tokens", 2048)))
        transition_map = {stop_id: list(append_ids) for stop_id, append_ids in plan.stage_transitions}
        completed_transitions = set()
        pending_tokens = []
        generated = []
        use_kv_cache = self._hunyuan_text_kv_cache_enabled()
        kv_cache = None
        cache_filled_length = 0
        if use_kv_cache:
            kv_cache = HunyuanImage3StaticKVCache(
                num_layers=self._hunyuan_num_layers(),
                max_cache_len=input_ids.shape[1] + max_new_tokens + sum(len(tokens) for tokens in transition_map.values()),
                dynamic=True,
            )

        for _ in range(max_new_tokens):
            if pending_tokens:
                next_token_id = pending_tokens.pop(0)
                next_token = torch.tensor([next_token_id], device=device, dtype=input_ids.dtype)
            else:
                if use_kv_cache:
                    model_input_ids = input_ids[:, cache_filled_length:]
                    position_ids = torch.arange(cache_filled_length, input_ids.shape[1], dtype=torch.long, device=device)[None].expand(input_ids.shape[0], -1)
                    first_cache_step = cache_filled_length == 0
                    model_inputs = self._build_text_model_inputs(
                        model_input_ids,
                        tokenizer_output,
                        cond_inputs=cond_inputs if first_cache_step else None,
                        position_ids=position_ids,
                        build_attention_mask=first_cache_step,
                        past_key_values=kv_cache,
                        use_cache=True,
                    )
                    cache_filled_length = input_ids.shape[1]
                else:
                    if cond_inputs is None:
                        model_inputs = self._build_text_model_inputs(input_ids, tokenizer_output)
                    else:
                        model_inputs = self._build_text_model_inputs(input_ids, tokenizer_output, cond_inputs=cond_inputs)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = self.model.infer(model_inputs)["logits"][:, -1, :]
                next_token = self._sample_text_token(logits, generator)
                next_token = next_token.to(device=device, dtype=input_ids.dtype)
            next_token = self._broadcast_parallel_tensor(next_token)
            next_token_id = int(next_token.item())

            generated.append(next_token_id)
            if stream_output:
                self._print_text_generation_chunk(self._decode_generated_text_token(next_token_id))
            input_ids = torch.cat([input_ids, next_token.reshape(1, 1)], dim=1)

            if next_token_id in transition_map and next_token_id not in completed_transitions:
                completed_transitions.add(next_token_id)
                pending_tokens.extend(transition_map[next_token_id])
                continue
            if next_token_id in plan.final_stop_tokens and not pending_tokens:
                break

        return generated

    def _decode_cot_text(self, generated_tokens, plan):
        if not generated_tokens:
            return None
        token_tensor = torch.tensor(generated_tokens, dtype=torch.long)
        generated_text = self.hunyuan_tokenizer.decode(token_tensor)
        if plan.first_bot_task == "think":
            return self.hunyuan_tokenizer.think_token + generated_text
        return self.hunyuan_tokenizer.recaption_token + generated_text

    def _generate_cot_text(self, prompt, image_size, batch_cond_images=None, cond_inputs=None):
        bot_task = self._resolve_bot_task()
        if bot_task == "image":
            return None
        # Generate COT text for bot_task in ("think", "recaption", "think_recaption"), default is "think_recaption"
        plan = self._resolve_text_generation_plan(bot_task)
        system_prompt = self._resolve_system_prompt(plan.first_bot_task)
        if batch_cond_images is None and cond_inputs is None:
            prepared = self._prepare_text_generation_inputs(prompt, plan.first_bot_task, system_prompt)
        else:
            prepared = self._prepare_text_generation_inputs(
                prompt,
                plan.first_bot_task,
                system_prompt,
                batch_cond_images=batch_cond_images,
                cond_inputs=cond_inputs,
            )
        stream_output = self._text_stream_enabled()
        if stream_output:
            prefix = self.hunyuan_tokenizer.think_token if plan.first_bot_task == "think" else self.hunyuan_tokenizer.recaption_token
            self._print_text_generation_chunk(prefix)
        generated_tokens = self._generate_text_tokens(
            prepared["input_ids"],
            prepared["tokenizer_output"],
            plan,
            stream_output=stream_output,
            cond_inputs=prepared.get("cond_inputs"),
        )
        if stream_output:
            self._print_text_generation_chunk("\n")
        cot_text = self._decode_cot_text(generated_tokens, plan)
        if cot_text:
            logger.info(f"HunyuanImage3 generated COT text for bot_task={bot_task}: {cot_text}")
        return cot_text

    def _prepare_text_to_image_inputs(self, prompt, image_size, seed, cot_text=None, batch_cond_images=None, cond_inputs=None):
        do_cfg = bool(self.config.get("enable_cfg", False)) and not bool(self.config.get("cfg_distilled", False))
        cfg_factor = 2 if do_cfg else 1
        gen_image_info = self.hunyuan_image_processor.build_gen_image_info(
            image_size,
            add_guidance_token=bool(self.config.get("cfg_distilled", False)),
            add_timestep_r_token=bool(self.config.get("use_meanflow", False)),
        )
        generation_config = self.hunyuan_generation_config
        tokenizer_out = self.hunyuan_tokenizer.apply_chat_template(
            batch_prompt=[prompt],
            mode="gen_image",
            batch_gen_image_info=[gen_image_info],
            batch_cond_images=batch_cond_images,
            batch_system_prompt=[self._resolve_system_prompt("image")],
            batch_cot_text=[cot_text] if cot_text else None,
            max_length=getattr(generation_config, "max_length", self.config.get("max_position_embeddings")),
            bot_task="image",
            image_base_size=self.hunyuan_image_processor.vae_reso_group.base_size,
            sequence_template=getattr(generation_config, "sequence_template", "pretrain"),
            cfg_factor=cfg_factor,
            drop_think=getattr(generation_config, "drop_think", False),
        )

        latent_device = self._pipeline_latent_device()
        output = tokenizer_out["output"]
        input_ids = output.tokens.to(latent_device)
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=latent_device)[None].expand(input_ids.shape[0], -1)
        rope_image_info = self._build_batch_rope_image_info(output, tokenizer_out["sections"])
        custom_pos_emb = self._build_custom_pos_emb(position_ids, rope_image_info)
        full_attn_slices = self._build_full_attn_slices(output, input_ids.shape[0], seq_len=input_ids.shape[1])

        generator = torch.Generator(device=latent_device).manual_seed(int(seed))
        prepared_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": self._build_attention_mask(input_ids, output),
            "full_attn_slices": full_attn_slices,
            "custom_pos_emb": custom_pos_emb,
            "rope_image_info": rope_image_info,
            "image_mask": output.gen_image_mask.to(latent_device),
            "timesteps_index": None if output.gen_timestep_scatter_index is None else output.gen_timestep_scatter_index.to(latent_device),
            "guidance_index": None if output.guidance_scatter_index is None else output.guidance_scatter_index.to(latent_device),
            "timesteps_r_index": None if output.gen_timestep_r_scatter_index is None else output.gen_timestep_r_scatter_index.to(latent_device),
            "generator": generator,
            "do_cfg": do_cfg,
            "batch_size": 1,
        }
        prepared_inputs.update(self._attach_tokenizer_cond_masks(cond_inputs, output, latent_device) or {})
        return prepared_inputs

    def _split_image_paths(self, image_path):
        if not image_path:
            raise ValueError("HunyuanImage3 ti2i requires --image_path with one or more comma-separated images.")
        if isinstance(image_path, (list, tuple)):
            return [str(path) for path in image_path if str(path)]
        return [path.strip() for path in str(image_path).split(",") if path.strip()]

    def _build_batch_cond_images(self, image_paths, infer_align_image_size):
        return [
            self.hunyuan_image_processor.build_cond_images(
                image_list=image_paths,
                infer_align_image_size=infer_align_image_size,
            )
        ]

    def _cond_vae_image(self, cond_image):
        return cond_image.vae_image if getattr(cond_image, "section_type", None) == "cond_joint_image" else cond_image

    def _cond_vit_image(self, cond_image):
        return cond_image.vit_image if getattr(cond_image, "section_type", None) == "cond_joint_image" else cond_image

    def _resolve_ti2i_image_size(self, requested_image_size, batch_cond_images):
        if requested_image_size != "auto":
            return requested_image_size
        if not batch_cond_images or not batch_cond_images[0]:
            return int(self.config.get("target_height", 1024)), int(self.config.get("target_width", 1024))
        first_cond_image = batch_cond_images[0][0]
        vae_image = self._cond_vae_image(first_cond_image)
        if hasattr(vae_image, "i"):
            return int(vae_image.i.image_height), int(vae_image.i.image_width)
        return int(self.config.get("target_height", 1024)), int(self.config.get("target_width", 1024))

    def _vae_encode_cond_tensor(self, image_tensor, generator=None):
        vae = self.vae_decoder
        vae_device = self._vae_device()
        image_tensor = image_tensor.unsqueeze(0).to(vae_device)
        autocast_dtype = getattr(torch, self.hunyuan_config.vae_autocast_dtype, torch.float16)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=vae_device.type == "cuda" and autocast_dtype != torch.float32):
            encoded = vae.encode(image_tensor)
            latents = encoded if torch.is_tensor(encoded) else encoded.latent_dist.sample(generator)
            if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
                latents = latents - vae.config.shift_factor
            if hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor:
                latents = latents * vae.config.scaling_factor
        if hasattr(vae, "ffactor_temporal"):
            latents = latents.squeeze(2)
        return latents.squeeze(0).to(self._pipeline_latent_device(), dtype=torch.bfloat16)

    def _encode_cond_vae_images(self, batch_cond_images, cfg_factor, seed):
        if not batch_cond_images or not batch_cond_images[0]:
            return None, None
        first_image = batch_cond_images[0][0]
        if getattr(first_image, "section_type", None) not in ("cond_vae_image", "cond_joint_image"):
            return None, None

        vae_device = self._vae_device()
        generator = torch.Generator(device=vae_device).manual_seed(int(seed))
        batch_cond_vae_images = []
        batch_cond_timesteps = []
        for cond_images in batch_cond_images:
            image_items = []
            timestep_items = []
            for cond_image in cond_images:
                vae_image = self._cond_vae_image(cond_image)
                latent = self._vae_encode_cond_tensor(vae_image, generator=generator)
                image_items.append(latent)
                timestep_items.append(torch.zeros(1, device=self._pipeline_latent_device(), dtype=torch.bfloat16))
            batch_cond_vae_images.append(image_items)
            batch_cond_timesteps.append(timestep_items)

        if all(len(items) == 1 for items in batch_cond_vae_images) and all(items[0].shape == batch_cond_vae_images[0][0].shape for items in batch_cond_vae_images):
            cond_vae_images = torch.stack([items[0] for items in batch_cond_vae_images], dim=0)
            cond_timesteps = torch.cat([items[0] for items in batch_cond_timesteps], dim=0)
            if cfg_factor > 1:
                cond_vae_images = cond_vae_images.repeat(cfg_factor, 1, 1, 1)
                cond_timesteps = cond_timesteps.repeat(cfg_factor)
            return cond_vae_images, cond_timesteps

        cond_vae_images = []
        cond_timesteps = [torch.cat(items, dim=0) for items in batch_cond_timesteps]
        for items in batch_cond_vae_images:
            if all(items[0].shape == item.shape for item in items):
                cond_vae_images.append(torch.stack(items, dim=0))
            else:
                cond_vae_images.append(items)
        if cfg_factor > 1:
            cond_vae_images = cond_vae_images * cfg_factor
            cond_timesteps = cond_timesteps * cfg_factor
        return cond_vae_images, cond_timesteps

    def _encode_cond_vit_embeds(self, batch_cond_images, cfg_factor):
        if not batch_cond_images or not batch_cond_images[0]:
            return None
        first_image = batch_cond_images[0][0]
        if getattr(first_image, "section_type", None) not in ("cond_vit_image", "cond_joint_image"):
            return None

        self._ensure_vision_encoder()
        vision_device = self._vision_device()
        cond_vit_embeds = []
        for cond_images in batch_cond_images:
            vit_images = [self._cond_vit_image(cond_image) for cond_image in cond_images]
            pixel_values = torch.stack(vit_images, dim=0).to(vision_device)
            image_kwargs = {}
            first_vit_image = vit_images[0]
            if hasattr(first_vit_image, "vision_encoder_kwargs") and first_vit_image.vision_encoder_kwargs:
                image_kwargs = {
                    "spatial_shapes": torch.stack([vit_image.vision_encoder_kwargs["spatial_shapes"] for vit_image in vit_images], dim=0).to(vision_device),
                    "attention_mask": torch.stack([vit_image.vision_encoder_kwargs["pixel_attention_mask"] for vit_image in vit_images], dim=0).to(vision_device),
                }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=vision_device.type == "cuda"):
                image_embeds = self.vision_model(pixel_values, **image_kwargs).last_hidden_state
                image_embeds = self.vision_aligner(image_embeds)
            cond_vit_embeds.append(image_embeds.to(self._pipeline_latent_device(), dtype=torch.bfloat16))

        if cfg_factor > 1:
            cond_vit_embeds = cond_vit_embeds * cfg_factor
        return cond_vit_embeds

    def _prepare_cond_inputs(self, batch_cond_images, cfg_factor, seed):
        cond_vae_images, cond_timesteps = self._encode_cond_vae_images(batch_cond_images, cfg_factor, seed)
        cond_vit_embeds = self._encode_cond_vit_embeds(batch_cond_images, cfg_factor)
        return {
            "cond_vae_images": cond_vae_images,
            "cond_timesteps": cond_timesteps,
            "cond_vit_embeds": cond_vit_embeds,
        }

    def _prepare_latents(self, batch_size, image_size, generator):
        latent_device = self._pipeline_latent_device()
        downsample = self.hunyuan_config.vae_downsample_factor
        if isinstance(downsample, int):
            downsample_h = downsample_w = downsample
        else:
            downsample_h, downsample_w = downsample[:2]
        height, width = image_size
        shape = (
            batch_size,
            self.hunyuan_config.vae["latent_channels"],
            height // downsample_h,
            width // downsample_w,
        )
        return torch.randn(shape, generator=generator, device=latent_device, dtype=torch.bfloat16)

    def _denoise_latents(self, prepared_inputs, image_size):
        cfg_mode = self._resolve_denoise_cfg_mode(prepared_inputs)
        serial_branch_inputs = None
        if cfg_mode == "parallel":
            prepared_inputs = self._prepare_cfg_parallel_branch_inputs(prepared_inputs, self._cfg_parallel_rank(), mark_parallel_branch=True)
        elif cfg_mode == "serial":
            serial_branch_inputs = [
                self._prepare_cfg_parallel_branch_inputs(prepared_inputs, 0, mark_parallel_branch=False),
                self._prepare_cfg_parallel_branch_inputs(prepared_inputs, 1, mark_parallel_branch=False),
            ]

        latents = self._prepare_latents(prepared_inputs["batch_size"], image_size, prepared_inputs["generator"])
        latents = self._broadcast_parallel_tensor(latents)
        num_steps = int(self.config.get("infer_steps", self.config.get("diff_infer_steps", 50)))
        self.scheduler.set_timesteps(num_steps, device=latents.device)
        guidance_scale = float(self.config.get("sample_guide_scale", self.config.get("diff_guidance_scale", 1.0)))
        use_kv_cache = self._hunyuan_kv_cache_enabled()
        taylor_cache_dic = self._build_taylor_cache_dic(num_steps) if self._hunyuan_taylor_cache_enabled() else None
        if taylor_cache_dic is not None and cfg_mode in ("serial", "parallel"):
            raise NotImplementedError("HunyuanImage3 Taylor cache currently supports hunyuan_cfg_mode='batch' only.")
        if taylor_cache_dic is not None and hasattr(self.model, "reset_taylor_cache"):
            self.model.reset_taylor_cache()
        if taylor_cache_dic is not None:
            logger.info(f"HunyuanImage3 Taylor cache enabled: {taylor_cache_dic}")

        if cfg_mode == "serial":
            kv_states = [self._build_denoise_kv_state(branch_inputs, use_kv_cache) for branch_inputs in serial_branch_inputs]
        else:
            kv_states = [self._build_denoise_kv_state(prepared_inputs, use_kv_cache)]
        if hasattr(self.model, "transformer_infer") and hasattr(self.model.transformer_infer, "reset_moe_profile"):
            self.model.transformer_infer.reset_moe_profile()

        denoise_steps = tqdm(
            self.scheduler.timesteps,
            total=len(self.scheduler.timesteps),
            desc="HunyuanImage3 denoise",
            dynamic_ncols=True,
            disable=bool(self.config.get("disable_progress_bar", False)) or not self._is_output_rank(),
        )
        autotune_controller = self._build_flashinfer_autotune_controller()
        with autotune_controller.context():
            for step_index, timestep in enumerate(denoise_steps):
                # ==================== CFG serial Processing ====================
                if cfg_mode == "serial":
                    cond_model_inputs = self._build_denoise_model_inputs(
                        serial_branch_inputs[0],
                        latents,
                        timestep,
                        step_index,
                        use_kv_cache,
                        kv_states[0],
                        guidance_scale,
                    )
                    uncond_model_inputs = self._build_denoise_model_inputs(
                        serial_branch_inputs[1],
                        latents,
                        timestep,
                        step_index,
                        use_kv_cache,
                        kv_states[1],
                        guidance_scale,
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=latents.device.type == "cuda"):
                        prediction = self.model.infer_cfg_serial(cond_model_inputs, uncond_model_inputs)["diffusion_prediction"].to(latents.device)

                elif cfg_mode == "batch":
                    # ==================== CFG Batch Processing ====================
                    # prepared_inputs 中是 packed CFG batch：
                    #   batch[0] = conditional
                    #   batch[1] = unconditional
                    #
                    # latents 原本是 batch=1，需要复制为 batch=2，
                    # 让 cond/uncond 在一次 transformer forward 中完成。
                    latent_model_input = torch.cat([latents, latents], dim=0)

                    model_inputs = self._build_denoise_model_inputs(
                        prepared_inputs,
                        latent_model_input,
                        timestep,
                        step_index,
                        use_kv_cache,
                        kv_states[0],
                        guidance_scale,
                    )

                    if taylor_cache_dic is not None:
                        taylor_cache_dic["current_step"] = step_index
                        model_inputs["cache_dic"] = taylor_cache_dic

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=latents.device.type == "cuda",
                    ):
                        prediction = self.model.infer(model_inputs)["diffusion_prediction"].to(latents.device)

                    # Transformer 返回 batch=2 prediction，
                    # 在 runner 中拆分并完成 CFG guidance。
                    pred_cond, pred_uncond = prediction.chunk(2, dim=0)

                    if hasattr(self.model, "combine_cfg_predictions"):
                        prediction = self.model.combine_cfg_predictions(
                            pred_cond,
                            pred_uncond,
                        )
                    else:
                        prediction = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                else:
                    # ==================== CFG Parallel Processing ====================
                    # cfg_mode == "parallel" 时：
                    #   prepared_inputs 已经按照 cfg_p_rank 切成 batch=1
                    #   cfg rank 0 负责 conditional
                    #   cfg rank 1 负责 unconditional
                    #
                    # cfg_mode == "none" 也会复用此单 batch forward 路径。
                    latent_model_input = latents

                    model_inputs = self._build_denoise_model_inputs(
                        prepared_inputs,
                        latent_model_input,
                        timestep,
                        step_index,
                        use_kv_cache,
                        kv_states[0],
                        guidance_scale,
                    )

                    # parallel 模式前面已经禁止 Taylor cache；
                    # 这里保留判断是为了兼容 cfg_mode == "none"。
                    if taylor_cache_dic is not None:
                        taylor_cache_dic["current_step"] = step_index
                        model_inputs["cache_dic"] = taylor_cache_dic

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=latents.device.type == "cuda",
                    ):
                        prediction = self.model.infer(model_inputs)["diffusion_prediction"].to(latents.device)

                    # CFG parallel 不需要在 runner 中 chunk/combine。
                    # model.infer() 内部已经通过 cfg_p_group：
                    #   1. all-gather cond/uncond prediction
                    #   2. 完成 CFG guidance
                    # 最终返回的 prediction 已经是 guided prediction。

                prediction = self._broadcast_parallel_tensor(prediction)
                sigma = self.scheduler.sigmas[step_index].to(latents.device)
                sigma_next = self.scheduler.sigmas[step_index + 1].to(latents.device)
                latents = latents.float() + prediction.float() * (sigma_next - sigma)

        self._parallel_barrier()
        if hasattr(self.model, "transformer_infer") and hasattr(self.model.transformer_infer, "print_moe_profile"):
            self.model.transformer_infer.print_moe_profile(reset=True)

        return latents

    def _decode_latents(self, latents, generator):
        vae = self.vae_decoder
        vae_device = self._vae_device()
        latents = latents.to(vae_device)
        if hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor:
            latents = latents / vae.config.scaling_factor
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
            latents = latents + vae.config.shift_factor
        if hasattr(vae, "ffactor_temporal"):
            latents = latents.unsqueeze(2)

        autocast_dtype = getattr(torch, self.hunyuan_config.vae_autocast_dtype, torch.float16)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=vae_device.type == "cuda" and autocast_dtype != torch.float32):
            image = vae.decode(latents, return_dict=False, generator=generator)[0]
        if hasattr(vae, "ffactor_temporal"):
            image = image.squeeze(2)

        image = (image / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).float().numpy()
        return [Image.fromarray((sample * 255.0).round().astype("uint8")) for sample in image]

    @torch.no_grad()
    def generate_t2i(self, input_info):
        self._ensure_pipeline_modules()
        prompt = getattr(input_info, "prompt_enhanced", None) or getattr(input_info, "prompt", "")
        image_size = self._resolve_image_size(input_info)
        seed = getattr(input_info, "seed", None) or self.config.get("seed", 42)
        # cot_text = self._generate_cot_text(prompt, image_size)
        prepared_inputs = self._prepare_text_to_image_inputs(
            prompt,
            image_size,
            seed,
            #  cot_text=cot_text
        )
        latents = self._denoise_latents(prepared_inputs, image_size)
        if not self._is_output_rank():
            return []
        return self._decode_latents(latents, prepared_inputs["generator"])

    @torch.no_grad()
    def generate_i2i(self, input_info):
        self._ensure_pipeline_modules()
        prompt = getattr(input_info, "prompt_enhanced", None) or getattr(input_info, "prompt", "")
        seed = getattr(input_info, "seed", None) or self.config.get("seed", 42)
        image_paths = self._split_image_paths(getattr(input_info, "image_path", None) or self.config.get("image_path"))
        infer_align_image_size = bool(getattr(input_info, "infer_align_image_size", self.config.get("infer_align_image_size", False)))

        batch_cond_images = self._build_batch_cond_images(image_paths, infer_align_image_size)
        image_size = self._resolve_ti2i_image_size(self._resolve_image_size(input_info), batch_cond_images)

        text_cond_inputs = self._prepare_cond_inputs(batch_cond_images, cfg_factor=1, seed=seed)
        # cot_text = self._generate_cot_text(prompt, image_size, batch_cond_images=batch_cond_images, cond_inputs=text_cond_inputs)

        # in default, we enable CFG for i2i/ti2i, but disable CFG if cfg_distilled is True
        do_cfg = bool(self.config.get("enable_cfg", False)) and not bool(self.config.get("cfg_distilled", False))
        gen_cond_inputs = self._prepare_cond_inputs(batch_cond_images, cfg_factor=2 if do_cfg else 1, seed=seed)
        prepared_inputs = self._prepare_text_to_image_inputs(
            prompt,
            image_size,
            seed,
            # cot_text=cot_text,
            batch_cond_images=batch_cond_images,
            cond_inputs=gen_cond_inputs,
        )
        latents = self._denoise_latents(prepared_inputs, image_size)
        if not self._is_output_rank():
            return []
        images = self._decode_latents(latents, prepared_inputs["generator"])
        return self.hunyuan_image_processor.postprocess_outputs(
            images,
            batch_cond_images=batch_cond_images,
            infer_align_image_size=infer_align_image_size,
        )

    def run_pipeline(self, input_info):
        self.input_info = input_info
        task = self.config.get("task")
        if task == "t2i":
            images = self.generate_t2i(input_info)
        elif task in ("i2i", "ti2i"):
            images = self.generate_i2i(input_info)
        else:
            raise NotImplementedError("HunyuanImage3 native runner currently supports task=t2i/i2i.")

        if not self._is_output_rank():
            return {"image": None}
        if getattr(input_info, "return_result_tensor", False):
            return {"image": images}

        save_result_path = getattr(input_info, "save_result_path", None)
        if save_result_path:
            Path(save_result_path).parent.mkdir(parents=True, exist_ok=True)
            images[0].save(save_result_path)
            logger.info(f"HunyuanImage3 image saved successfully to: {save_result_path}")
        return {"image": None}
