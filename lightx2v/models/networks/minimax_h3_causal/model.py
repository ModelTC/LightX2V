import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3SequenceParallelState
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3_causal.infer.pre_infer import MiniMaxH3CausalPreInfer
from lightx2v.models.networks.minimax_h3_causal.infer.transformer_infer import MiniMaxH3CausalOffloadTransformerInfer, MiniMaxH3CausalTransformerInfer


class MiniMaxH3CausalModel(MiniMaxH3Model):
    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = MiniMaxH3CausalPreInfer
        self.transformer_infer_class = MiniMaxH3CausalOffloadTransformerInfer if self.cpu_offload else MiniMaxH3CausalTransformerInfer

    def _load_shared_cpu_weights(self, unified_dtype, sensitive_layer):
        raise NotImplementedError("Causal H3 does not support shared_cpu_weights; set shared_cpu_weights=false to use model/block CPU offload")

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

    def _seq_parallel_pre_process(self, pre):
        size, rank = dist.get_world_size(self.seq_p_group), dist.get_rank(self.seq_p_group)
        padding = (-pre.hidden_states.shape[0]) % size
        hidden = F.pad(pre.hidden_states, (0, 0, 0, padding))
        pre.sequence_parallel_state = MiniMaxH3SequenceParallelState(0, hidden.shape[0] // size, pre.timestep_indices, pre.adaln_indices, pre.rotary_emb)
        pre.hidden_states = hidden.chunk(size)[rank].contiguous()
        pre.timestep_indices = F.pad(pre.timestep_indices, (0, padding)).chunk(size)[rank].contiguous()
        pre.adaln_indices = F.pad(pre.adaln_indices, (0, padding)).chunk(size)[rank].contiguous()
        # RoPE is applied after Ulysses restores the whole sequence per head.
        return pre

    def _seq_parallel_post_process(self, output, pre):
        return super()._seq_parallel_post_process(output, pre)[: pre.valid_sequence_length]
