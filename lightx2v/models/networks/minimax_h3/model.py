import os

import torch
import torch.distributed as dist
from safetensors import safe_open

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3SequenceParallelState
from lightx2v.models.networks.minimax_h3.infer.offload import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer
from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v.models.networks.minimax_h3.weights import (
    MiniMaxH3PostWeights,
    MiniMaxH3PreWeights,
    MiniMaxH3TransformerWeights,
)
from lightx2v.utils.envs import GET_DTYPE

H3_CHANNEL_QUANT_SCHEMES = {
    "fp8-q8f",
    "fp8-sgl",
    "fp8-torchao",
    "fp8-triton",
    "fp8-vllm",
    "int8-q8f",
    "int8-sgl",
    "int8-torchao",
    "int8-triton",
    "int8-vllm",
}


class MiniMaxH3Model(BaseTransformerModel):
    """LightX2V-native MiniMax-H3 joint audio/video transformer."""

    pre_weight_class = MiniMaxH3PreWeights
    transformer_weight_class = MiniMaxH3TransformerWeights
    post_weight_class = MiniMaxH3PostWeights

    def __init__(self, model_path, config, device):
        if GET_DTYPE() != torch.bfloat16:
            raise ValueError(
                "MiniMax-H3 requires DTYPE=BF16. The native loader preserves the released checkpoint's 626 BF16 tensors and 12 FP32 projection/time/head tensors without dtype conversion."
            )
        if config.get("tensor_parallel", False):
            raise NotImplementedError("MiniMax-H3 tensor parallel support is not implemented yet")
        if config.get("cfg_parallel", False) or config.get("enable_cfg", False):
            raise ValueError("MiniMax-H3 is guidance-distilled and does not have a CFG/unconditional branch")
        if config.get("dit_quantized", False):
            quant_scheme = config.get("dit_quant_scheme", "Default")
            if quant_scheme not in H3_CHANNEL_QUANT_SCHEMES:
                raise NotImplementedError(f"MiniMax-H3 quantized inference requires a per-output-channel FP8/INT8 scheme; got {quant_scheme!r}. Supported schemes: {sorted(H3_CHANNEL_QUANT_SCHEMES)}")
            if not config.get("dit_quantized_ckpt"):
                raise ValueError("MiniMax-H3 quantized inference requires dit_quantized_ckpt")
        elif config.get("dit_quant_scheme", "Default") != "Default":
            raise ValueError("MiniMax-H3 dit_quant_scheme requires a dit_quantized_ckpt")
        if config.get("lora_configs"):
            raise NotImplementedError("MiniMax-H3 LoRA loading is not implemented yet")
        if config.get("cpu_offload", False) and config.get("offload_granularity", "model") not in {"model", "block"}:
            raise NotImplementedError("MiniMax-H3 supports model and block CPU offload")

        transformer_path = config.get("dit_original_ckpt") or os.path.join(model_path, "transformer")
        super().__init__(transformer_path, config, device)
        if config.get("seq_parallel", False):
            parallel_type = config.get("parallel", {}).get("seq_p_attn_type", "ulysses")
            if parallel_type != "ulysses":
                raise NotImplementedError(f"MiniMax-H3 sequence parallel currently supports Ulysses, got {parallel_type!r}")
            world_size = dist.get_world_size(self.seq_p_group)
            num_heads = int(config.get("num_attention_heads", 56))
            if num_heads % world_size:
                raise ValueError(f"MiniMax-H3 Ulysses requires num_attention_heads ({num_heads}) to be divisible by seq_p_size ({world_size})")
        self.sensitive_layer = {
            "proj_in",
            "audio_proj_in",
            "time_embedder",
            "proj_out",
            "audio_proj_out",
        }
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        """Load the released mixed-precision tensors without generic dtype coercion."""
        del unified_dtype, sensitive_layer
        if os.path.splitext(file_path)[-1] != ".safetensors":
            raise ValueError(f"MiniMax-H3 native loading expects the released safetensors checkpoint; got {file_path}")
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []
        preserve_keys = self.preserved_keys if hasattr(self, "preserved_keys") else None
        with safe_open(file_path, framework="pt", device=str(self.device)) as source:
            return {
                key: source.get_tensor(key)
                for key in source.keys()
                if not any(remove_key in key for remove_key in remove_keys) and (preserve_keys is None or any(preserve_key in key for preserve_key in preserve_keys))
            }

    def _init_infer_class(self):
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("MiniMax-H3 feature caching is not implemented")
        self.pre_infer_class = MiniMaxH3PreInfer
        self.transformer_infer_class = MiniMaxH3OffloadTransformerInfer if self.cpu_offload else MiniMaxH3TransformerInfer
        self.post_infer_class = MiniMaxH3PostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        if not infer_condition:
            raise ValueError("MiniMax-H3 does not execute an unconditional pass")
        prompt_embeds = inputs["text_encoder_output"]["prompt_embeds"]
        pre = self.pre_infer.infer(self.pre_weight, prompt_embeds)
        if self.config.get("seq_parallel", False):
            pre = self._seq_parallel_pre_process(pre)
        hidden_states = self.transformer_infer.infer(self.transformer_weights, pre)
        if self.config.get("seq_parallel", False):
            hidden_states = self._seq_parallel_post_process(hidden_states, pre)
        return self.post_infer.infer(self.post_weight, hidden_states, pre)

    @torch.no_grad()
    def infer(self, inputs):
        block_offload = self.cpu_offload and self.offload_granularity == "block"
        if block_offload and self.scheduler.step_index == 0:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()
        output = self._infer_cond_uncond(inputs, infer_condition=True)
        self.scheduler.video_noise_pred = output.video
        self.scheduler.audio_noise_pred = output.audio
        if block_offload and self.scheduler.step_index == self.scheduler.infer_steps - 1:
            self.pre_weight.to_cpu()
            self.post_weight.to_cpu()

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        world_size = dist.get_world_size(self.seq_p_group)
        rank = dist.get_rank(self.seq_p_group)
        total_length = pre_infer_out.hidden_states.shape[0]
        text_length = pre_infer_out.text_indices.numel()
        expected_text_indices = torch.arange(text_length, device=pre_infer_out.text_indices.device)
        if not torch.equal(pre_infer_out.text_indices, expected_text_indices):
            raise ValueError("MiniMax-H3 sequence parallel requires the conditioner rows to be a contiguous prefix")

        # Keep the largest possible conditioner prefix replicated on every
        # rank, and shard the remaining packed sequence without padding. This
        # preserves exact dense-attention semantics for audio/video rows.
        remainder = total_length % world_size
        if text_length < remainder:
            raise ValueError(f"MiniMax-H3 cannot split packed length {total_length} over {world_size} ranks with only {text_length} prefix rows")
        aux_length = text_length - ((text_length - remainder) % world_size)
        main_length = total_length - aux_length
        if main_length <= 0 or main_length % world_size:
            raise ValueError(f"MiniMax-H3 sequence split is invalid: total={total_length}, aux={aux_length}, seq_p_size={world_size}")
        main_shard_length = main_length // world_size
        shard_start = aux_length + rank * main_shard_length
        shard_end = shard_start + main_shard_length

        def shard_sequence(tensor):
            return torch.cat((tensor[:aux_length], tensor[shard_start:shard_end]), dim=0)

        pre_infer_out.sequence_parallel_state = MiniMaxH3SequenceParallelState(
            aux_length=aux_length,
            main_shard_length=main_shard_length,
            timestep_indices=pre_infer_out.timestep_indices,
            adaln_indices=pre_infer_out.adaln_indices,
            rotary_emb=pre_infer_out.rotary_emb,
        )
        pre_infer_out.hidden_states = shard_sequence(pre_infer_out.hidden_states)
        pre_infer_out.timestep_indices = shard_sequence(pre_infer_out.timestep_indices)
        pre_infer_out.adaln_indices = shard_sequence(pre_infer_out.adaln_indices)
        pre_infer_out.rotary_emb = tuple(shard_sequence(tensor) for tensor in pre_infer_out.rotary_emb)
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, output, pre_infer_out):
        state = pre_infer_out.sequence_parallel_state
        if state is None:
            raise RuntimeError("MiniMax-H3 sequence-parallel metadata is missing")
        local_main = output[state.aux_length :]
        if local_main.shape[0] != state.main_shard_length:
            raise RuntimeError(f"MiniMax-H3 local sequence length changed from {state.main_shard_length} to {local_main.shape[0]}")
        world_size = dist.get_world_size(self.seq_p_group)
        gathered = [torch.empty_like(local_main) for _ in range(world_size)]
        dist.all_gather(gathered, local_main.contiguous(), group=self.seq_p_group)
        output = torch.cat((output[: state.aux_length], *gathered), dim=0)
        pre_infer_out.timestep_indices = state.timestep_indices
        pre_infer_out.adaln_indices = state.adaln_indices
        pre_infer_out.rotary_emb = state.rotary_emb
        pre_infer_out.sequence_parallel_state = None
        return output

    def to_cpu(self):
        super().to_cpu()
        if hasattr(self.transformer_infer, "offload_manager"):
            # Full teardown moves the active aliases away from the persistent
            # device buffers. Force buffer 0 to be populated again next run.
            self.transformer_infer.offload_manager.need_init_first_buffer = True
