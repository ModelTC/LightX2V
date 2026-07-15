import json
import os
from contextlib import nullcontext

import torch
from loguru import logger
from peft import LoraConfig, inject_adapter_in_model
from peft.utils import set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file

from lightx2v_train.model_zoo.native.ltx2 import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    LTXV_MODEL_COMFY_RENAMING_MAP,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    LTXModelConfigurator,
    SingleGPUModelBuilder,
    convert_to_additive_mask,
    find_matching_file,
    module_ops_from_gemma_root,
)
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .base import BaseModel


@MODEL_REGISTER("ltx_t2av")
class LTX2T2AVModel(BaseModel):
    pipeline_cls = None

    def load_components(self, transformer_only=False, reference_model=None):
        model_config = self.config["model"]
        model_path = model_config["pretrained_model_name_or_path"]
        self.pretrained_model_path = os.path.abspath(os.path.expanduser(str(model_path)))
        self._fsdp2_activation_checkpointing = False

        self.transformer_param_dtype = get_running_dtype(model_config.get("transformer_param_dtype", "bf16"))
        self.embeddings_dtype = get_running_dtype(model_config.get("embeddings_dtype", "bf16"))
        self.text_encoder_dtype = get_running_dtype(model_config.get("text_encoder_dtype", "bf16"))
        self.load_transformer = bool(model_config.get("load_transformer", True))
        self.load_embeddings_processor = bool(model_config.get("load_embeddings_processor", True))
        self.load_text_encoder = bool(model_config.get("load_text_encoder", False))
        self.text_encoder_cpu = bool(model_config.get("text_encoder_cpu", False))
        self.load_text_encoder_in_8bit = bool(model_config.get("load_text_encoder_in_8bit", False))
        self.text_encoder_path = model_config.get("text_encoder_path", "/data/nvme0/gushiqiao/models/official_models/LTX-2")
        training_config = self.config.get("training", {})
        teacher_forcing_config = training_config.get("teacher_forcing", {})
        self.num_frame_per_chunk = int(model_config.get("num_frame_per_chunk", teacher_forcing_config.get("num_frame_per_chunk", 1)))
        self.local_attn_size = int(model_config.get("local_attn_size", training_config.get("dmd", {}).get("local_attn_size", -1)))
        self.sink_size = int(model_config.get("sink_size", training_config.get("dmd", {}).get("sink_size", 0)))
        if "defer_kv_cache_updates" in model_config:
            self.defer_kv_cache_updates = bool(model_config["defer_kv_cache_updates"])
        else:
            self.defer_kv_cache_updates = bool(model_config.get("defer_cache_updates", False))
        self.detach_kv_cache_updates = bool(model_config.get("detach_kv_cache_updates", False))

        if transformer_only:
            self.load_embeddings_processor = False
            self.load_text_encoder = False
            if reference_model is not None:
                self.embeddings_processor = reference_model.embeddings_processor
                self.text_encoder = getattr(reference_model, "text_encoder", None)
                self.text_encoder_path = getattr(reference_model, "text_encoder_path", self.text_encoder_path)
                self.text_encoder_cpu = getattr(reference_model, "text_encoder_cpu", self.text_encoder_cpu)

        if self.load_transformer:
            self.transformer = self._load_transformer(model_path)
            self._configure_causal_transformer()
        else:
            self.transformer = None

        self.embeddings_processor = getattr(self, "embeddings_processor", None)
        self.text_encoder = getattr(self, "text_encoder", None)
        if self.load_embeddings_processor:
            self.embeddings_processor = self._load_embeddings_processor(model_path)
            self.embeddings_processor.requires_grad_(False)
            self.embeddings_processor.eval()
            if hasattr(self.embeddings_processor, "feature_extractor") and not self.load_text_encoder:
                self.embeddings_processor.feature_extractor = None
        if self.load_text_encoder:
            self.text_encoder = self._load_text_encoder(self.text_encoder_path)
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

    def _load_transformer(self, model_path):
        builder = SingleGPUModelBuilder(
            model_path=str(model_path),
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        )
        self.transformer_checkpoint_config = builder.model_config()
        transformer = builder.build(device=self.device, dtype=self.transformer_param_dtype)
        return transformer.to(self.device, dtype=self.transformer_param_dtype)

    def prepare_consolidated_state_dict(self, state_dict):
        consolidated = {}
        for key, value in state_dict.items():
            while key.startswith("module.") or key.startswith("_fsdp_wrapped_module."):
                key = key.split(".", 1)[1]
            key = key.replace("._checkpoint_wrapped_module.", ".")
            if not key.startswith("model.diffusion_model."):
                key = f"model.diffusion_model.{key}"
            if key in consolidated:
                raise ValueError(f"Duplicate LTX2 consolidated checkpoint key: {key}")
            consolidated[key] = value

        if not os.path.isfile(self.pretrained_model_path):
            raise FileNotFoundError(f"Official LTX checkpoint not found: {self.pretrained_model_path}")
        added = 0
        with safe_open(self.pretrained_model_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in consolidated:
                    continue
                consolidated[key] = handle.get_tensor(key).contiguous()
                added += 1
        logger.info("[checkpoint] added {} non-trained component tensors from {}", added, self.pretrained_model_path)
        return consolidated

    def consolidated_safetensors_metadata(self):
        metadata = super().consolidated_safetensors_metadata()
        checkpoint_config = getattr(self, "transformer_checkpoint_config", None)
        if checkpoint_config:
            metadata["config"] = json.dumps(checkpoint_config, ensure_ascii=False, separators=(",", ":"))
        return metadata

    def _configure_causal_transformer(self):
        self.transformer.num_frame_per_block = self.num_frame_per_chunk
        self.transformer.local_attn_size = self.local_attn_size
        self.transformer.sink_size = self.sink_size
        self.transformer.defer_kv_cache_updates = self.defer_kv_cache_updates
        self.transformer.detach_kv_cache_updates = self.detach_kv_cache_updates

    def _load_embeddings_processor(self, model_path):
        processor = SingleGPUModelBuilder(
            model_path=str(model_path),
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
        ).build(device=self.device, dtype=self.embeddings_dtype)
        return processor.to(self.device, dtype=self.embeddings_dtype)

    def _load_text_encoder(self, text_encoder_path):
        if self.load_text_encoder_in_8bit:
            try:
                from ltx_trainer.gemma_8bit import load_8bit_gemma
            except ImportError as exc:
                raise ImportError("model.load_text_encoder_in_8bit=true requires ltx_trainer.gemma_8bit on PYTHONPATH. Disable it or add the LTX-2 trainer package.") from exc

            device = torch.device("cpu") if self.text_encoder_cpu else self.device
            return load_8bit_gemma(text_encoder_path, self.text_encoder_dtype, device=device)

        gemma_model_folder = find_matching_file(str(text_encoder_path), "model*.safetensors").parent
        gemma_weight_paths = tuple(str(path) for path in gemma_model_folder.rglob("*.safetensors"))
        device = torch.device("cpu") if self.text_encoder_cpu else self.device
        return SingleGPUModelBuilder(
            model_path=gemma_weight_paths,
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GEMMA_LLM_KEY_OPS,
            module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_root(str(text_encoder_path))),
        ).build(device=device, dtype=self.text_encoder_dtype)

    def denoiser_module(self):
        if self.transformer is None:
            raise RuntimeError("LTX2 transformer is not loaded. Set model.load_transformer=True for training.")
        return self.transformer

    def transformer_forward_context(self):
        if self.device.type == "cuda" and self.running_dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=self.running_dtype)
        return nullcontext()

    def add_lora(self, rank, alpha, target_modules):
        if not target_modules:
            raise ValueError("training.lora.target_modules must be set for LTX2 LoRA training.")
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self._inject_lora(lora_config)

    def _inject_lora(self, lora_config, adapter_name="default"):
        try:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer, adapter_name=adapter_name)
        except TypeError:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer)

    def load_lora_for_infer(self, lora_path, adapter_name=None):
        if adapter_name is None:
            adapter_name = "default"
        if not hasattr(self.denoiser_module(), "peft_config"):
            training_config = self.config.get("training", {})
            lora_config = training_config.get("lora", {})
            rank = int(lora_config.get("rank", 64))
            alpha = int(lora_config.get("alpha", rank))
            target_modules = lora_config.get("target_modules")
            self._inject_lora(
                LoraConfig(
                    r=rank,
                    lora_alpha=alpha,
                    init_lora_weights="gaussian",
                    target_modules=target_modules,
                ),
                adapter_name=adapter_name,
            )

        weight_path = lora_path
        if os.path.isdir(weight_path):
            weight_path = os.path.join(weight_path, "pytorch_lora_weights.safetensors")
        raw = load_file(weight_path)
        peft_state_dict = {}
        for key, value in raw.items():
            new_key = key.removeprefix("transformer.")
            new_key = new_key.replace(".lora.down.weight", ".lora_A.weight")
            new_key = new_key.replace(".lora.up.weight", ".lora_B.weight")
            peft_state_dict[new_key] = value

        incompatible = set_peft_model_state_dict(self.denoiser_module(), peft_state_dict)
        if incompatible and incompatible.unexpected_keys:
            logger.warning("Unexpected keys when loading LTX2 LoRA: {}", incompatible.unexpected_keys)
        self._infer_lora_adapter_name = adapter_name

    def enable_gradient_checkpointing(self):
        if self._fsdp2_activation_checkpointing:
            return
        if hasattr(self.denoiser_module(), "set_gradient_checkpointing"):
            self.denoiser_module().set_gradient_checkpointing(True)

    def _wrap_transformer_blocks_for_fsdp_checkpointing(self):
        if self._fsdp2_activation_checkpointing:
            return

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

        blocks = self.transformer.transformer_blocks
        for block_idx, block in enumerate(blocks):
            blocks[block_idx] = checkpoint_wrapper(
                block,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
        self._fsdp2_activation_checkpointing = True
        logger.info("Enabled FSDP2 activation checkpoint wrappers for {} LTX2 transformer blocks", len(blocks))

    def fsdp2_shard_plan(self, fsdp_config):
        training_config = self.config.get("training", {})
        if training_config.get("gradient_checkpointing", True):
            self._wrap_transformer_blocks_for_fsdp_checkpointing()
        reshard_config = fsdp_config.get(
            "reshard_after_forward",
            {
                "root_reshard": False,
                "block_reshard": True,
            },
        )
        return [
            {
                "modules": self.transformer.transformer_blocks,
                "reshard_after_forward": reshard_config.get("block_reshard", True),
            },
            {
                "module": self.transformer,
                "reshard_after_forward": reshard_config.get("root_reshard", False),
            },
        ]

    def prepare_text_condition(self, conditions):
        if self.embeddings_processor is None:
            raise RuntimeError("LTX2 embeddings processor is not loaded. Set model.load_embeddings_processor=True.")

        if "video_prompt_embeds" in conditions:
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds", video_features)
        else:
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]
        mask = conditions["prompt_attention_mask"]
        video_features = self._ensure_prompt_feature_batch(video_features, "video prompt embeds").to(device=self.device)
        if audio_features is not None:
            audio_features = self._ensure_prompt_feature_batch(audio_features, "audio prompt embeds").to(device=self.device)
        mask = self._ensure_prompt_mask_batch(mask).to(device=self.device)
        additive_mask = convert_to_additive_mask(mask, video_features.dtype)
        with torch.no_grad():
            video_embeds, audio_embeds, attention_mask = self.embeddings_processor.create_embeddings(
                video_features,
                audio_features,
                additive_mask,
            )
        return video_embeds.to(dtype=self.running_dtype), audio_embeds.to(dtype=self.running_dtype), attention_mask

    @staticmethod
    def _ensure_prompt_feature_batch(value, name):
        if not torch.is_tensor(value):
            raise TypeError(f"LTX2 {name} must be a tensor, got {type(value)!r}.")
        if value.ndim == 2:
            return value.unsqueeze(0)
        if value.ndim == 3:
            return value
        raise ValueError(f"LTX2 {name} must have shape [seq, dim] or [batch, seq, dim], got {tuple(value.shape)}.")

    @staticmethod
    def _ensure_prompt_mask_batch(value):
        if not torch.is_tensor(value):
            raise TypeError(f"LTX2 prompt_attention_mask must be a tensor, got {type(value)!r}.")
        if value.ndim == 1:
            return value.unsqueeze(0)
        if value.ndim == 2:
            return value
        raise ValueError(f"LTX2 prompt_attention_mask must have shape [seq] or [batch, seq], got {tuple(value.shape)}.")

    def encode_prompt_condition(self, prompt):
        if self.text_encoder is None:
            raise RuntimeError("LTX2 text encoder is not loaded. Set model.load_text_encoder=True for prompt-based training.")
        if self.embeddings_processor is None:
            raise RuntimeError("LTX2 embeddings processor is not loaded. Set model.load_embeddings_processor=True.")
        if getattr(self.embeddings_processor, "feature_extractor", None) is None:
            raise RuntimeError("LTX2 embeddings processor feature_extractor is required for prompt-based training.")

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        with torch.no_grad():
            encoded = self.text_encoder.encode(prompts, padding_side="left")

        hidden_states = []
        for layer_idx in range(len(encoded[0][0])):
            layer = torch.cat([item[0][layer_idx] for item in encoded], dim=0)
            hidden_states.append(layer.to(device=self.device, dtype=self.embeddings_dtype))
        attention_mask = torch.cat([item[1] for item in encoded], dim=0).to(device=self.device)

        with torch.no_grad():
            processed = self.embeddings_processor.process_hidden_states(
                tuple(hidden_states),
                attention_mask,
                padding_side="left",
            )
        audio_encoding = processed.audio_encoding
        if audio_encoding is None:
            audio_encoding = processed.video_encoding
        return (
            processed.video_encoding.to(dtype=self.running_dtype),
            audio_encoding.to(dtype=self.running_dtype),
            processed.attention_mask,
        )
