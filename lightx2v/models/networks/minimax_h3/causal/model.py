import os

import torch

from lightx2v.models.networks.minimax_h3.autoregressive_model import H3AutoregressiveModel
from lightx2v.models.networks.minimax_h3.infer.autoregressive.causal.pre_infer import MiniMaxH3CausalPreInfer
from lightx2v.models.networks.minimax_h3.infer.autoregressive.causal.transformer_infer import MiniMaxH3CausalOffloadTransformerInfer, MiniMaxH3CausalTransformerInfer
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model


class MiniMaxH3CausalModel(H3AutoregressiveModel, MiniMaxH3Model):
    def initialize_kv_cache(self, capacity):
        return self._bind_rolling_kv_cache(capacity, step_cache=True)

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = MiniMaxH3CausalPreInfer
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
