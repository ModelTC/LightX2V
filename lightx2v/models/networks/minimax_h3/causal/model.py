import glob
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from safetensors import safe_open

from lightx2v.models.networks.minimax_h3.autoregressive_model import H3AutoregressiveModel
from lightx2v.models.networks.minimax_h3.causal.kv_cache import FP8StepRollingKVCachePool
from lightx2v.models.networks.minimax_h3.infer.autoregressive.causal.post_infer import MiniMaxH3CausalPostInfer
from lightx2v.models.networks.minimax_h3.infer.autoregressive.causal.pre_infer import MiniMaxH3CausalPreInfer
from lightx2v.models.networks.minimax_h3.infer.autoregressive.causal.transformer_infer import MiniMaxH3CausalOffloadTransformerInfer, MiniMaxH3CausalTransformerInfer
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class MiniMaxH3CausalModel(H3AutoregressiveModel, MiniMaxH3Model):
    def initialize_kv_cache(self, capacity):
        if self.config["ar_config"].get("kv_cache_scheme", "rolling") != "fp8":
            return self._bind_rolling_kv_cache(capacity, step_cache=True)
        sequence_ranks = dist.get_world_size(self.seq_p_group) if self.seq_p_group is not None else 1
        cache = FP8StepRollingKVCachePool(
            num_steps=self.config["infer_steps"],
            num_layers=self.config["num_layers"],
            cache_size=capacity,
            num_heads=self.config["num_attention_heads"] // self.tp_size // sequence_ranks,
            head_dim=self.config["attention_head_dim"],
            dtype=GET_DTYPE(),
            device=torch.device(AI_DEVICE),
        )
        cache._init_kv_buffer()
        self.scheduler.cache = cache
        return cache

    def reset_autoregressive_state(self):
        super().reset_autoregressive_state()
        if self.config["ar_config"].get("kv_cache_scheme", "rolling") == "fp8":
            self.scheduler.cache = None

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = MiniMaxH3CausalPreInfer
        self.post_infer_class = MiniMaxH3CausalPostInfer
        self.transformer_infer_class = MiniMaxH3CausalOffloadTransformerInfer if self.cpu_offload else MiniMaxH3CausalTransformerInfer

    def _load_shared_cpu_weights(self, unified_dtype, sensitive_layer):
        from lightx2v.models.networks.minimax_h3.causal.shared_weights import load_shared_causal_weights

        return load_shared_causal_weights(self, sensitive_layer)

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        if os.path.splitext(file_path)[1] != ".pt":
            weights = super()._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
            return {key: value.to(torch.float32 if key.split(".", 1)[0] in sensitive_layer else torch.bfloat16) for key, value in weights.items()}
        source = torch.load(file_path, mmap=True, weights_only=True, map_location="cpu")
        weights = {}
        device = self._checkpoint_load_device()
        for key, tensor in source.items():
            if any(remove in key for remove in self.remove_keys):
                continue
            # Preserve sensitive tensors directly from the FP32 checkpoint;
            # converting through BF16 first would permanently round their values.
            dtype = torch.float32 if key.split(".", 1)[0] in sensitive_layer else torch.bfloat16
            weights[key] = self._select_tensor_parallel_shard(key, tensor).to(device=device, dtype=dtype)
        return weights

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        if self.use_tp or self.lazy_load or self.cpu_offload or not self.config.get("dit_compact_weights", False):
            return super()._load_quant_ckpt(unified_dtype, sensitive_layer)
        checkpoint_path = self.config["dit_quantized_ckpt"]
        files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors"))) if os.path.isdir(checkpoint_path) else [checkpoint_path]
        weights = {}
        for file_path in files:
            with safe_open(file_path, framework="pt", device="cpu") as source:
                self._record_checkpoint_metadata(source)
                weights.update({key: source.get_tensor(key) for key in source.keys() if not any(part in key for part in self.remove_keys)})

        if self.config.get("dit_quant_scheme") == "fp8-f16-accum":
            weights = {key: self._fp8_f16_checkpoint_tensor(key, tensor, sensitive_layer) for key, tensor in weights.items()}
        # Storage layout must not change checkpoint values, including FP32 scales.
        load_device = self._checkpoint_load_device()
        # Load each dtype into one allocation. Adjacent Q/K/V slices let
        # FusedQKVStorage reuse these bytes instead of allocating another copy.
        ordered = []
        for index in range(self.config["num_layers"]):
            for suffix in ("weight", "weight_scale"):
                for projection in ("q", "k", "v"):
                    key = f"transformer_blocks.{index}.attn.to_{projection}.{suffix}"
                    if key in weights:
                        ordered.append(key)
        qkv_keys = set(ordered)
        ordered.extend(key for key in weights if key not in qkv_keys)
        by_dtype = defaultdict(list)
        for key in ordered:
            by_dtype[weights[key].dtype].append(key)
        result = {}
        for dtype, keys in by_dtype.items():
            storage = torch.empty(sum(weights[key].numel() for key in keys), dtype=dtype, device=load_device)
            offset = 0
            for key in keys:
                source = weights[key]
                view = storage[offset : offset + source.numel()].view(source.shape)
                view.copy_(source)
                result[key] = view
                offset += source.numel()
        return result

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        pre = self.pre_infer.infer(self.pre_weight, inputs["text_encoder_output"]["prompt_embeds"])
        if self.seq_p_group is not None:
            pre = self._seq_parallel_pre_process(pre)
        hidden = self.transformer_infer.infer(self.transformer_weights, pre)
        # The causal video head projects sequence shards before gathering;
        # prefill returns immediately after the transformer has populated KV.
        return self.post_infer.infer(self.post_weight, hidden, pre)
