"""Native Qwen3-VL conditioner for MiniMax-H3 AV inference.

MiniMax-H3 reads ``hidden_states[50]`` from the released Qwen3-VL
conditioner.  In Transformers' hidden-state convention that is the embedding
output after decoder layers 0 through 49, before the final RMSNorm.  This
module executes exactly that prefix with LightX2V weight and operator classes.
The vision tower is loaded lazily for keyframe/reference requests; the last
fourteen decoder layers, final norm, and LM head are never loaded.

Only the Hugging Face tokenizer and pixel processor are reused. By default,
model tensors are streamed directly from the official sharded ``text_encoder``
checkpoint. An optional FP8-SGL checkpoint can replace the language Linear
weights while config, tokenizer, and vision tensors keep using the original
model directory.
"""

import gc
import itertools
import json
import os
from collections import defaultdict
from contextlib import suppress
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.ops.attn.template import AttnWeightTemplate

# Import the concrete implementations so their registry decorators run even
# when this encoder is imported outside LightX2V's usual top-level entrypoint.
from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight as _TorchSDPAWeight  # noqa: F401
from lightx2v.common.ops.embedding.embedding_weight import EmbeddingWeight as _EmbeddingWeight  # noqa: F401
from lightx2v.common.ops.mm.mm_weight import MMWeight as _MMWeight  # noqa: F401
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightFP32Qwen as _RMSWeightFP32Qwen  # noqa: F401
from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_vision import MiniMaxH3Qwen3VLVisionTower
from lightx2v.models.networks.minimax_h3.packing import VIDEO_TAG
from lightx2v.models.networks.minimax_h3.packing_ref2av import (
    build_ref2av_presentation,
    sample_reference_video_frames,
)
from lightx2v.models.networks.minimax_h3.weights.tensor_parallel import MiniMaxH3TensorParallelLinear, unwrap_tp_linear
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    EMBEDDING_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
)
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, torch.device(AI_DEVICE).type)

try:
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
except (AttributeError, ImportError) as error:
    Qwen2TokenizerFast = None
    Qwen3VLProcessor = None
    _TOKENIZER_IMPORT_ERROR = error
else:
    _TOKENIZER_IMPORT_ERROR = None


MINIMAX_H3_TEXT_ENCODER_LAYER = 50
MINIMAX_H3_TEXT_HIDDEN_SIZE = 5120
MINIMAX_H3_TEXT_NUM_LAYERS = 64
MINIMAX_H3_TEXT_TAG = 1

_CHECKPOINT_PREFIX = "model.language_model"
_EXPECTED_RELEASE_CONFIG = {
    "hidden_size": MINIMAX_H3_TEXT_HIDDEN_SIZE,
    "intermediate_size": 25600,
    "num_hidden_layers": MINIMAX_H3_TEXT_NUM_LAYERS,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}


def _empty_device_cache():
    """Release cached accelerator allocations without assuming CUDA."""
    with suppress(Exception):
        device_module = getattr(torch, torch.device(AI_DEVICE).type)
        device_module.empty_cache()


def _rotate_half(x):
    midpoint = x.shape[-1] // 2
    return torch.cat((-x[..., midpoint:], x[..., :midpoint]), dim=-1)


def _repeat_kv(x, num_groups):
    """Repeat ``[tokens, kv_heads, dim]`` in Qwen's KV-head order."""
    if num_groups == 1:
        return x
    tokens, num_kv_heads, head_dim = x.shape
    return x[:, :, None, :].expand(tokens, num_kv_heads, num_groups, head_dim).reshape(tokens, num_kv_heads * num_groups, head_dim)


class _Qwen3VLVocabParallelEmbedding(_EmbeddingWeight):
    """Vocabulary-sharded embedding with a TP all-reduce on activations."""

    def __init__(self, weight_name, vocab_size, tp_group, tp_rank, tp_size):
        super().__init__(weight_name)
        if vocab_size % tp_size:
            raise ValueError(f"Qwen3-VL vocab_size ({vocab_size}) must be divisible by text TP size ({tp_size})")
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.vocab_per_rank = vocab_size // tp_size
        self.vocab_start = tp_rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank

    def apply(self, input_indices):
        outside = (input_indices < self.vocab_start) | (input_indices >= self.vocab_end)
        local_indices = (input_indices - self.vocab_start).masked_fill(outside, 0)
        output = super().apply(local_indices)
        output = output.masked_fill(outside.unsqueeze(-1), 0)
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return output


def _qwen_linear(config, weight_name, *, tp_group, tp_rank, tp_size, split_dim, create_cuda_buffer=False):
    mm_type = config.get("text_encoder_quant_scheme", "Default")
    if tp_size == 1:
        return MM_WEIGHT_REGISTER[mm_type](weight_name, bias_name=None, create_cuda_buffer=create_cuda_buffer)
    return MiniMaxH3TensorParallelLinear(
        weight_name=weight_name,
        bias_name=None,
        mm_type=mm_type,
        tp_group=tp_group,
        tp_rank=tp_rank,
        tp_size=tp_size,
        split_dim=split_dim,
        create_cuda_buffer=create_cuda_buffer,
    )


class _MiniMaxH3QwenSDPAWeight(AttnWeightTemplate):
    """Qwen SDPA preserving the released model's native grouped-query path."""

    def __init__(self):
        super().__init__(None)

    def apply(
        self,
        q,
        k,
        v,
        drop_rate=0,
        attn_mask=None,
        causal=False,
        softmax_scale=None,
        **kwargs,
    ):
        unbatched = q.ndim == 3
        if unbatched:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=drop_rate,
            is_causal=causal,
            scale=softmax_scale,
            enable_gqa=True,
        )
        output = output.transpose(1, 2)
        batch_size, sequence_length, num_heads, head_dim = output.shape
        output = output.reshape(batch_size, sequence_length, num_heads * head_dim)
        return output.squeeze(0) if unbatched else output


class _Qwen3VLMLPWeights(WeightModule):
    def __init__(self, prefix, config, tp_group=None, tp_rank=0, tp_size=1, create_cuda_buffer=False):
        super().__init__()
        for name, split_dim in (("gate_proj", "col"), ("up_proj", "col"), ("down_proj", "row")):
            self.add_module(
                name,
                _qwen_linear(
                    config,
                    f"{prefix}.{name}.weight",
                    tp_group=tp_group,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    split_dim=split_dim,
                    create_cuda_buffer=create_cuda_buffer,
                ),
            )

    def forward(self, hidden_states):
        gate = self.gate_proj.apply(hidden_states)
        up = self.up_proj.apply(hidden_states)
        return self.down_proj.apply(F.silu(gate) * up)


class _Qwen3VLAttentionWeights(WeightModule):
    def __init__(self, prefix, config, text_config, attn_type, tp_group=None, tp_rank=0, tp_size=1, create_cuda_buffer=False):
        super().__init__()
        self.num_heads = int(text_config["num_attention_heads"]) // tp_size
        self.num_key_value_heads = int(text_config["num_key_value_heads"]) // tp_size
        self.head_dim = int(text_config["head_dim"])
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.softmax_scale = self.head_dim**-0.5
        eps = float(text_config["rms_norm_eps"])

        for name, split_dim in (("q_proj", "col"), ("k_proj", "col"), ("v_proj", "col"), ("o_proj", "row")):
            self.add_module(
                name,
                _qwen_linear(
                    config,
                    f"{prefix}.{name}.weight",
                    tp_group=tp_group,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    split_dim=split_dim,
                    create_cuda_buffer=create_cuda_buffer,
                ),
            )
        self.add_module(
            "q_norm",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.q_norm.weight", create_cuda_buffer=create_cuda_buffer, eps=eps),
        )
        self.add_module(
            "k_norm",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.k_norm.weight", create_cuda_buffer=create_cuda_buffer, eps=eps),
        )
        self.native_gqa = attn_type == "torch_sdpa"
        if self.native_gqa:
            self.add_module("calculate", _MiniMaxH3QwenSDPAWeight())
        else:
            self.add_module("calculate", ATTN_WEIGHT_REGISTER[attn_type]())

    def forward(self, hidden_states, position_embeddings):
        sequence_length = hidden_states.shape[0]
        query = self.q_proj.apply(hidden_states).view(sequence_length, self.num_heads, self.head_dim)
        key = self.k_proj.apply(hidden_states).view(sequence_length, self.num_key_value_heads, self.head_dim)
        value = self.v_proj.apply(hidden_states).view(sequence_length, self.num_key_value_heads, self.head_dim)

        query = self.q_norm.apply(query)
        key = self.k_norm.apply(key)

        cos, sin = position_embeddings
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin

        # The released Qwen model keeps eight KV heads and uses torch SDPA's
        # native GQA specialization.  Preserve that CUDA kernel path for close
        # numerical parity; other common backends require materialized heads.
        if not self.native_gqa:
            key = _repeat_kv(key, self.num_key_value_groups)
            value = _repeat_kv(value, self.num_key_value_groups)
        attention_output = self.calculate.apply(
            query,
            key,
            value,
            causal=True,
            max_seqlen_q=sequence_length,
            max_seqlen_kv=sequence_length,
            softmax_scale=self.softmax_scale,
        )
        return self.o_proj.apply(attention_output)


class _Qwen3VLDecoderLayerWeights(WeightModule):
    def __init__(self, layer_index, config, text_config, attn_type, tp_group=None, tp_rank=0, tp_size=1, create_cuda_buffer=False):
        super().__init__()
        prefix = f"{_CHECKPOINT_PREFIX}.layers.{layer_index}"
        eps = float(text_config["rms_norm_eps"])
        self.add_module(
            "input_layernorm",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.input_layernorm.weight", create_cuda_buffer=create_cuda_buffer, eps=eps),
        )
        self.add_module(
            "post_attention_layernorm",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.post_attention_layernorm.weight", create_cuda_buffer=create_cuda_buffer, eps=eps),
        )
        self.add_module(
            "self_attn",
            _Qwen3VLAttentionWeights(
                f"{prefix}.self_attn",
                config,
                text_config,
                attn_type,
                tp_group=tp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                create_cuda_buffer=create_cuda_buffer,
            ),
        )
        self.add_module(
            "mlp",
            _Qwen3VLMLPWeights(
                f"{prefix}.mlp",
                config,
                tp_group=tp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                create_cuda_buffer=create_cuda_buffer,
            ),
        )

    def weight_modules(self):
        return (
            self.input_layernorm,
            self.post_attention_layernorm,
            self.self_attn.q_proj,
            self.self_attn.k_proj,
            self.self_attn.v_proj,
            self.self_attn.o_proj,
            self.self_attn.q_norm,
            self.self_attn.k_norm,
            self.mlp.gate_proj,
            self.mlp.up_proj,
            self.mlp.down_proj,
        )

    def forward(self, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm.apply(hidden_states)
        hidden_states = residual + self.self_attn.forward(hidden_states, position_embeddings)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.apply(hidden_states)
        return residual + self.mlp.forward(hidden_states)


class _Qwen3VLTextBackboneWeights(WeightModule):
    """Unbatched native prefix of Qwen3-VL's language backbone."""

    def __init__(self, config, text_config, num_layers=MINIMAX_H3_TEXT_ENCODER_LAYER, attn_type="torch_sdpa", block_offload=False, tp_group=None):
        super().__init__()
        self.config = config
        self.text_config = text_config
        self.num_layers = int(num_layers)
        self.attn_type = attn_type
        self.block_offload = bool(block_offload)
        self.offload_manager = None
        self.offload_cuda_buffers = None
        self._offload_completion_event = None
        self.hidden_size = int(text_config["hidden_size"])
        self.head_dim = int(text_config["head_dim"])
        self.rope_theta = float(text_config["rope_theta"])
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        if self.tp_size > 1:
            embed_tokens = _Qwen3VLVocabParallelEmbedding(
                f"{_CHECKPOINT_PREFIX}.embed_tokens.weight",
                int(text_config["vocab_size"]),
                tp_group,
                self.tp_rank,
                self.tp_size,
            )
        else:
            embed_tokens = EMBEDDING_WEIGHT_REGISTER["Default"](f"{_CHECKPOINT_PREFIX}.embed_tokens.weight")
        self.add_module("embed_tokens", embed_tokens)
        self.add_module(
            "layers",
            WeightModuleList(
                _Qwen3VLDecoderLayerWeights(
                    index,
                    self.config,
                    text_config,
                    attn_type,
                    tp_group=self.tp_group,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                for index in range(self.num_layers)
            ),
        )
        self._weight_modules = self._collect_weight_modules()

        # Qwen3-VL's text-only positions are equal on the temporal, height, and
        # width axes.  Its interleaved M-RoPE therefore reduces exactly to this
        # standard split-half RoPE frequency vector.
        self._inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).to(torch.float32) / self.head_dim))
        rope_config = text_config.get("rope_parameters") or text_config.get("rope_scaling") or {}
        self.mrope_section = tuple(rope_config.get("mrope_section", (24, 20, 20)))

    def _collect_weight_modules(self):
        modules = {self.embed_tokens.weight_name: self.embed_tokens}
        for layer in self.layers:
            modules.update((leaf.weight_name, leaf) for leaf in layer.weight_modules())
        return modules

    def select_tp_shard(self, name, tensor):
        if self.tp_size == 1:
            return tensor
        is_weight_scale = name.endswith(".weight_scale")
        linear_weight_name = name.removesuffix("_scale") if is_weight_scale else name
        if name == self.embed_tokens.weight_name:
            split_dim = 0
        elif any(
            pattern in linear_weight_name
            for pattern in (
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
                ".mlp.gate_proj.weight",
                ".mlp.up_proj.weight",
            )
        ):
            split_dim = 0
        elif ".self_attn.o_proj.weight" in linear_weight_name or ".mlp.down_proj.weight" in linear_weight_name:
            if is_weight_scale:
                return tensor
            split_dim = 1
        else:
            return tensor
        if tensor.shape[split_dim] % self.tp_size:
            raise ValueError(f"Cannot shard Qwen3-VL tensor {name} shape {tuple(tensor.shape)} across text TP size {self.tp_size}")
        return torch.chunk(tensor, self.tp_size, dim=split_dim)[self.tp_rank].contiguous()

    @staticmethod
    def _allocate_layer_buffer(buffer_layer, source_layer):
        """Allocate compute-layout device tensors without another checkpoint copy."""
        allocated_bytes = 0
        for buffer_weight, source_weight in zip(buffer_layer.weight_modules(), source_layer.weight_modules()):
            buffer_storage = unwrap_tp_linear(buffer_weight)
            source_storage = unwrap_tp_linear(source_weight)
            for _, attr_name, _ in buffer_storage.base_attrs:
                source_tensor = getattr(source_storage, f"pin_{attr_name}", None)
                if source_tensor is None:
                    source_tensor = getattr(source_storage, attr_name, None)
                if source_tensor is None:
                    raise RuntimeError(f"Cannot allocate Qwen3-VL offload buffer for {source_weight.weight_name}: missing {attr_name}")
                # Preserve the transposed compute-layout strides used by the
                # resident path, rather than relying on empty_like's memory-
                # format heuristics. This keeps GEMM inputs identical after
                # the H2D copy.
                device_buffer = torch.empty_strided(
                    source_tensor.size(),
                    source_tensor.stride(),
                    dtype=source_tensor.dtype,
                    device=AI_DEVICE,
                )
                setattr(buffer_storage, f"{attr_name}_cuda_buffer", device_buffer)
                allocated_bytes += device_buffer.numel() * device_buffer.element_size()
        return allocated_bytes

    def init_block_offload(self):
        if not self.block_offload or self.offload_manager is not None:
            return
        if self.num_layers < 2:
            raise ValueError("Qwen3-VL ping-pong block offload requires at least two decoder layers")

        buffers = WeightModuleList(
            _Qwen3VLDecoderLayerWeights(
                slot_index,
                self.config,
                self.text_config,
                self.attn_type,
                tp_group=self.tp_group,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                create_cuda_buffer=True,
            )
            for slot_index in range(2)
        )
        allocated_bytes = sum(self._allocate_layer_buffer(buffers[slot_index], self.layers[slot_index]) for slot_index in range(2))
        manager = EventSlotWeightAsyncStreamManager(offload_granularity="block")
        manager.init_cuda_buffer(blocks_cuda_buffer=buffers)
        self.offload_cuda_buffers = buffers
        self.offload_manager = manager
        logger.info(
            "MiniMax-H3 Qwen3-VL block offload allocated two ping-pong buffers ({:.2f} GiB total)",
            allocated_bytes / (1024**3),
        )

    def named_weight_modules(self):
        return self._weight_modules.items()

    @property
    def device(self):
        if self.block_offload:
            return torch.device(AI_DEVICE)
        return self.embed_tokens.weight.device

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    def _position_embeddings(self, hidden_states, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[0], device=hidden_states.device)[None].expand(3, -1)
        if position_ids.ndim != 2 or position_ids.shape[0] != 3:
            raise ValueError(f"Qwen3-VL position_ids must be [3,tokens], got {tuple(position_ids.shape)}")
        # Transformers' ``from_pretrained(dtype=...)`` keeps Qwen's
        # non-persistent RoPE buffers in FP32 even when parameters are BF16.
        # Generate phases and trig values in FP32, then cast at the same output
        # boundary as the released conditioner.
        inv_freq = self._inv_freq.to(device=hidden_states.device, dtype=torch.float32)
        frequencies = position_ids.to(torch.float32)[..., None] * inv_freq[None, None, :]
        frequencies_t = frequencies[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            frequencies_t[..., slice(offset, self.mrope_section[dim] * 3, 3)] = frequencies[dim, ..., slice(offset, self.mrope_section[dim] * 3, 3)]
        frequencies = frequencies_t
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(hidden_states.dtype), embeddings.sin().to(hidden_states.dtype)

    def _inject_vision_embeds(self, hidden_states, vision_mask, vision_embeds):
        if vision_embeds is None:
            return hidden_states
        if vision_mask is None or int(vision_mask.sum()) != vision_embeds.shape[0]:
            raise ValueError("Qwen3-VL vision placeholder count does not match vision embeddings")
        hidden_states = hidden_states.clone()
        hidden_states[vision_mask] = vision_embeds.to(hidden_states.device, hidden_states.dtype)
        return hidden_states

    @staticmethod
    def _inject_deepstack(hidden_states, layer_index, vision_mask, deepstack_embeds):
        if deepstack_embeds is None or layer_index >= len(deepstack_embeds):
            return hidden_states
        hidden_states = hidden_states.clone()
        hidden_states[vision_mask] += deepstack_embeds[layer_index].to(hidden_states.device, hidden_states.dtype)
        return hidden_states

    def _forward_with_block_offload(self, input_ids, position_ids, vision_mask, vision_embeds, deepstack_embeds):
        if self.offload_manager is None:
            raise RuntimeError("Qwen3-VL block-offload buffers were not initialized")
        manager = self.offload_manager
        caller_stream = torch_device_module.current_stream()

        # A prior request may still be completing asynchronously on the
        # caller stream. Protect both slots before resetting their metadata.
        if self._offload_completion_event is not None:
            manager.cuda_load_stream.wait_event(self._offload_completion_event)
            caller_stream.wait_event(self._offload_completion_event)
        manager.reset_slots()

        # Start layer 0 while the large embedding table is copied and applied
        # on the caller's stream. Only the embedding is temporarily resident.
        try:
            manager.prefetch_to_slot(0, 0, self.layers)
            self.embed_tokens.to_cuda(non_blocking=True)
            try:
                hidden_states = self.embed_tokens.apply(input_ids)
            finally:
                self.embed_tokens.weight = self.embed_tokens.pin_weight
            hidden_states = self._inject_vision_embeds(hidden_states, vision_mask, vision_embeds)
            position_embeddings = self._position_embeddings(hidden_states, position_ids)

            for layer_index in range(self.num_layers):
                slot_index = layer_index % 2
                manager.wait_ready(slot_index, caller_stream)
                if layer_index + 1 < self.num_layers:
                    manager.prefetch_to_slot((slot_index + 1) % 2, layer_index + 1, self.layers)

                # Keep all Qwen math on the caller's stream for numerical and
                # allocator parity with the resident/model-offload path. The
                # independent load stream still overlaps layer i+1 H2D with
                # layer i compute through the slot events above.
                hidden_states = manager.cuda_buffers[slot_index].forward(hidden_states, position_embeddings)
                hidden_states = self._inject_deepstack(hidden_states, layer_index, vision_mask, deepstack_embeds)
                manager.record_free(slot_index, caller_stream)

            completion_event = torch_device_module.Event()
            completion_event.record(caller_stream)
            self._offload_completion_event = completion_event
            return hidden_states
        except Exception:
            # Do not leave an in-flight copy able to overwrite a slot after an
            # exceptional exit or before a retry.
            torch_device_module.synchronize()
            manager.reset_slots()
            self._offload_completion_event = None
            raise

    def forward(self, input_ids, position_ids=None, vision_mask=None, vision_embeds=None, deepstack_embeds=None):
        if input_ids.ndim != 1:
            raise ValueError(f"MiniMax-H3's native Qwen3-VL backbone expects unbatched token IDs, got {tuple(input_ids.shape)}")
        if self.block_offload:
            return self._forward_with_block_offload(input_ids, position_ids, vision_mask, vision_embeds, deepstack_embeds)

        hidden_states = self.embed_tokens.apply(input_ids)
        hidden_states = self._inject_vision_embeds(hidden_states, vision_mask, vision_embeds)
        position_embeddings = self._position_embeddings(hidden_states, position_ids)
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer.forward(hidden_states, position_embeddings)
            hidden_states = self._inject_deepstack(hidden_states, layer_index, vision_mask, deepstack_embeds)
        return hidden_states

    def to_cpu(self, non_blocking=False):
        """Drop accelerator copies; checkpoint weights are immutable pinned tensors."""
        for _, module in self.named_weight_modules():
            storage = unwrap_tp_linear(module)
            pin_weight = getattr(storage, "pin_weight", None)
            if pin_weight is not None:
                storage.weight = pin_weight
                pin_weight_scale = getattr(storage, "pin_weight_scale", None)
                if pin_weight_scale is not None:
                    storage.weight_scale = pin_weight_scale
            elif getattr(storage, "weight", None) is not None:
                module.to_cpu(non_blocking=non_blocking)
        return self

    def to_cuda(self, non_blocking=False):
        super().to_cuda(non_blocking=non_blocking)
        return self


class MiniMaxH3Qwen3VLTextEncoder:
    """Encode with the native first 50 Qwen3-VL layers.

    ``text_encoder_offload_granularity=block`` keeps the immutable layer
    weights on CPU and streams them through two preallocated device buffers.
    When text TP is enabled, the language backbone uses the distributed world
    group.  This avoids replicating the conditioner across DiT SP/CFG lanes;
    set ``text_encoder_tensor_parallel=false`` to opt out.
    """

    def __init__(self, config):
        self.config = config
        if config.get("text_encoder_quantized", False) and not config.get("text_encoder_quantized_ckpt"):
            raise ValueError("MiniMax-H3 quantized text encoder requires text_encoder_quantized_ckpt")
        self.tensor_parallel = bool(config.get("text_encoder_tensor_parallel", config.get("tensor_parallel", False)))
        if self.tensor_parallel:
            if not dist.is_initialized():
                raise RuntimeError("MiniMax-H3 text encoder TP requires an initialized distributed process group")
            self.tp_group = dist.group.WORLD
            self.tp_size = dist.get_world_size(self.tp_group)
            self.tp_rank = dist.get_rank(self.tp_group)
            logger.info(
                "MiniMax-H3 text encoder uses world-size tensor parallel: rank {}/{}",
                self.tp_rank,
                self.tp_size,
            )
        else:
            self.tp_group = None
            self.tp_size = 1
            self.tp_rank = 0
        text_encoder_cpu_offload = bool(config.get("text_encoder_cpu_offload", config.get("cpu_offload", False)))
        if "qwen3vl_cpu_offload" in config and bool(config["qwen3vl_cpu_offload"]) != text_encoder_cpu_offload:
            raise ValueError("qwen3vl_cpu_offload cannot override text_encoder_cpu_offload for MiniMax-H3; the runner schedules the native conditioner through text_encoder_cpu_offload")
        self.cpu_offload = text_encoder_cpu_offload
        self.offload_granularity = config.get("text_encoder_offload_granularity", "model")
        if self.offload_granularity not in {"model", "block"}:
            raise ValueError(f"Unsupported text_encoder_offload_granularity={self.offload_granularity!r}; expected 'model' or 'block'")
        if self.offload_granularity == "block" and not self.cpu_offload:
            raise ValueError("text_encoder_offload_granularity='block' requires text_encoder_cpu_offload=true")
        self.block_offload = self.cpu_offload and self.offload_granularity == "block"
        self.local_files_only = config.get("local_files_only", True)
        self.text_encoder = None
        self.vision_encoder = None
        self.tokenizer = None
        self.processor = None
        if config.get("text_encoder_load_on_init", True):
            self.load()

    @staticmethod
    def _require_tokenizer():
        if Qwen2TokenizerFast is None:
            detail = f" Original import error: {_TOKENIZER_IMPORT_ERROR}" if _TOKENIZER_IMPORT_ERROR else ""
            raise ImportError("MiniMax-H3 text encoding requires Transformers' Qwen2TokenizerFast; the Qwen model itself is executed natively by LightX2V." + detail)

    def _component_path(self, config_key, subfolder):
        if config_key in self.config:
            return self.config[config_key]
        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError(f"MiniMax-H3 requires `model_path` or `{config_key}` in the config")
        return os.path.join(model_path, subfolder)

    @staticmethod
    def _read_text_config(text_encoder_path):
        config_path = Path(text_encoder_path) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"MiniMax-H3 text encoder config was not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        text_config = dict(raw_config.get("text_config", raw_config))

        rope_config = text_config.get("rope_parameters") or text_config.get("rope_scaling") or {}
        text_config["rope_theta"] = float(rope_config.get("rope_theta", text_config.get("rope_theta", 500000.0)))
        text_config.setdefault("rms_norm_eps", 1e-6)
        text_config.setdefault("attention_bias", False)
        text_config.setdefault("hidden_act", "silu")
        return text_config

    @staticmethod
    def _read_model_config(text_encoder_path):
        with (Path(text_encoder_path) / "config.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _validate_text_config(text_config):
        mismatches = {key: (text_config.get(key), expected) for key, expected in _EXPECTED_RELEASE_CONFIG.items() if text_config.get(key) != expected}
        if mismatches:
            details = ", ".join(f"{key}={actual!r} (expected {expected!r})" for key, (actual, expected) in mismatches.items())
            raise ValueError(f"MiniMax-H3 requires the released Qwen3-VL conditioner config; {details}")
        if text_config.get("hidden_act") != "silu":
            raise ValueError(f"MiniMax-H3's native Qwen3-VL path requires hidden_act='silu', got {text_config.get('hidden_act')!r}")
        if bool(text_config.get("attention_bias", False)):
            raise ValueError("MiniMax-H3's released Qwen3-VL attention projections do not use bias")

        num_heads = int(text_config["num_attention_heads"])
        num_kv_heads = int(text_config["num_key_value_heads"])
        head_dim = int(text_config["head_dim"])
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"Qwen3-VL attention heads ({num_heads}) must be divisible by KV heads ({num_kv_heads})")
        if head_dim % 2:
            raise ValueError(f"Qwen3-VL head_dim must be even for split-half RoPE, got {head_dim}")

    def _validate_text_tp(self, text_config):
        if self.tp_size == 1:
            return
        checks = {
            "num_attention_heads": int(text_config["num_attention_heads"]),
            "num_key_value_heads": int(text_config["num_key_value_heads"]),
            "intermediate_size": int(text_config["intermediate_size"]),
            "vocab_size": int(text_config["vocab_size"]),
        }
        invalid = {name: value for name, value in checks.items() if value % self.tp_size}
        if invalid:
            details = ", ".join(f"{name}={value}" for name, value in invalid.items())
            raise ValueError(f"MiniMax-H3 text TP size {self.tp_size} must divide {details}")

    @staticmethod
    def _resolve_attn_type(config):
        requested = config.get(
            "qwen3vl_attn_type",
            config.get("qwen_attn_implementation", config.get("qwen3vl_attn_implementation", "torch_sdpa")),
        )
        aliases = {
            "sdpa": "torch_sdpa",
            "flash_attention_2": "flash_attn2",
            "flash_attention_3": "flash_attn3",
        }
        attn_type = aliases.get(requested, requested)
        if attn_type not in ATTN_WEIGHT_REGISTER:
            available = ", ".join(sorted(ATTN_WEIGHT_REGISTER.keys()))
            raise ValueError(f"Unknown qwen3vl_attn_type={requested!r}; available LightX2V attention operators: {available}")
        return attn_type

    @staticmethod
    def _expected_weight_shapes(text_config):
        hidden_size = int(text_config["hidden_size"])
        intermediate_size = int(text_config["intermediate_size"])
        num_heads = int(text_config["num_attention_heads"])
        num_kv_heads = int(text_config["num_key_value_heads"])
        head_dim = int(text_config["head_dim"])
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        shapes = {
            f"{_CHECKPOINT_PREFIX}.embed_tokens.weight": (
                int(text_config["vocab_size"]),
                hidden_size,
            )
        }
        for layer_index in range(MINIMAX_H3_TEXT_ENCODER_LAYER):
            prefix = f"{_CHECKPOINT_PREFIX}.layers.{layer_index}"
            shapes.update(
                {
                    f"{prefix}.input_layernorm.weight": (hidden_size,),
                    f"{prefix}.post_attention_layernorm.weight": (hidden_size,),
                    f"{prefix}.self_attn.q_proj.weight": (q_size, hidden_size),
                    f"{prefix}.self_attn.k_proj.weight": (kv_size, hidden_size),
                    f"{prefix}.self_attn.v_proj.weight": (kv_size, hidden_size),
                    f"{prefix}.self_attn.o_proj.weight": (hidden_size, q_size),
                    f"{prefix}.self_attn.q_norm.weight": (head_dim,),
                    f"{prefix}.self_attn.k_norm.weight": (head_dim,),
                    f"{prefix}.mlp.gate_proj.weight": (intermediate_size, hidden_size),
                    f"{prefix}.mlp.up_proj.weight": (intermediate_size, hidden_size),
                    f"{prefix}.mlp.down_proj.weight": (hidden_size, intermediate_size),
                }
            )
        return shapes

    @staticmethod
    def _checkpoint_weight_map(text_encoder_path, required_names):
        root = Path(text_encoder_path)
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"Invalid safetensors index without a weight_map: {index_path}")
            return weight_map

        safetensor_paths = sorted(root.glob("*.safetensors"))
        if not safetensor_paths:
            raise FileNotFoundError(f"No model.safetensors.index.json or safetensors files found under {root}")
        required_names = set(required_names)
        weight_map = {}
        for checkpoint_path in safetensor_paths:
            with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
                for name in required_names.intersection(checkpoint.keys()):
                    weight_map[name] = checkpoint_path.name
        return weight_map

    @classmethod
    def _load_native_weights(cls, backbone, text_encoder_path, text_config):
        modules = dict(backbone.named_weight_modules())
        expected_shapes = cls._expected_weight_shapes(text_config)
        if modules.keys() != expected_shapes.keys():
            missing_native = sorted(expected_shapes.keys() - modules.keys())
            unexpected_native = sorted(modules.keys() - expected_shapes.keys())
            raise RuntimeError(f"Native Qwen3-VL weight declaration disagrees with its shape schema: missing={missing_native}, unexpected={unexpected_native}")

        root = Path(text_encoder_path)
        weight_map = cls._checkpoint_weight_map(root, modules)
        missing = sorted(modules.keys() - weight_map.keys())
        if missing:
            preview = ", ".join(missing[:8])
            raise KeyError(f"MiniMax-H3 text encoder checkpoint is missing {len(missing)} required tensors: {preview}")

        by_shard = defaultdict(list)
        for name in modules:
            by_shard[weight_map[name]].append(name)

        # Header-only preflight avoids allocating tens of GiB before discovering
        # a malformed or incompatible tensor near the end of the checkpoint.
        checkpoint_dtypes = set()
        for shard_name in sorted(by_shard):
            shard_path = root / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Safetensors shard from checkpoint index was not found: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
                shard_keys = set(checkpoint.keys())
                for name in by_shard[shard_name]:
                    if name not in shard_keys:
                        raise KeyError(f"Checkpoint index maps {name} to {shard_path}, but the tensor is absent")
                    tensor_slice = checkpoint.get_slice(name)
                    actual_shape = tuple(tensor_slice.get_shape())
                    if actual_shape != expected_shapes[name]:
                        raise ValueError(f"Unexpected checkpoint shape for {name}: {actual_shape}, expected {expected_shapes[name]}")
                    checkpoint_dtypes.add(tensor_slice.get_dtype())
        if len(checkpoint_dtypes) != 1:
            raise ValueError(f"MiniMax-H3 Qwen3-VL weights must use one floating dtype, got {sorted(checkpoint_dtypes)}")

        logger.info(
            "Loading {} native Qwen3-VL tensors (embedding + layers 0..{}) from {} shards",
            len(modules),
            MINIMAX_H3_TEXT_ENCODER_LAYER - 1,
            len(by_shard),
        )
        for shard_index, shard_name in enumerate(sorted(by_shard), start=1):
            shard_path = root / shard_name
            logger.info(
                "Streaming MiniMax-H3 text shard {}/{}: {}",
                shard_index,
                len(by_shard),
                shard_path.name,
            )
            with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
                for name in by_shard[shard_name]:
                    one_tensor = {name: backbone.select_tp_shard(name, checkpoint.get_tensor(name))}
                    module = modules[name]
                    if module is backbone.embed_tokens:
                        # The released embedding table is about 1.55 GiB.
                        # EmbeddingWeight's generic CPU path requires one
                        # monolithic pinned allocation, which some CUDA
                        # runtimes reject with cudaErrorInvalidValue. Keep the
                        # immutable host copy pageable; to_cuda()/F.embedding
                        # remain the same common-op path, only the transfer is
                        # synchronous when pinning is unavailable.
                        module.pin_weight = one_tensor.pop(name)
                    else:
                        module.load(one_tensor)
                    if one_tensor:
                        raise RuntimeError(f"LightX2V weight loader did not consume tensor {name}")

        # CPU-loaded common weights keep their canonical copy in pin_weight.
        # Activate those copies so the object is usable before/after offload.
        backbone.to_cpu()
        return checkpoint_dtypes.pop()

    @staticmethod
    def _load_quantized_weights(backbone, checkpoint_path):
        modules = dict(backbone.named_weight_modules())
        tensor_names = {}
        for primary_name, module in modules.items():
            storage = unwrap_tp_linear(module)
            attrs = getattr(storage, "base_attrs", ((primary_name, "weight", False),))
            tensor_names[primary_name] = tuple(name for name, _, _ in attrs)
        required_names = {name for names in tensor_names.values() for name in names}
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"MiniMax-H3 quantized text encoder checkpoint was not found: {checkpoint_path}")

        logger.info("Loading {} quantized Qwen3-VL tensors from {}", len(required_names), checkpoint_path)
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            missing = sorted(required_names - set(checkpoint.keys()))
            if missing:
                raise KeyError(f"MiniMax-H3 quantized text encoder checkpoint is missing tensors: {missing[:8]}")
            for primary_name, module in modules.items():
                tensors = {name: backbone.select_tp_shard(name, checkpoint.get_tensor(name)) for name in tensor_names[primary_name]}
                if module is backbone.embed_tokens:
                    module.pin_weight = tensors.pop(primary_name)
                else:
                    module.load(tensors)
                if tensors:
                    raise RuntimeError(f"LightX2V weight loader did not consume tensors for {primary_name}: {sorted(tensors)}")

        backbone.to_cpu()

    def load_tokenizer(self):
        """Load only the permitted Transformers tokenizer dependency."""
        self._require_tokenizer()
        if self.tokenizer is None:
            tokenizer_path = self._component_path("tokenizer_path", "tokenizer")
            logger.info(f"Loading MiniMax-H3 tokenizer from {tokenizer_path}")
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                tokenizer_path,
                local_files_only=self.local_files_only,
            )
        return self.tokenizer

    def unload_tokenizer(self):
        self.tokenizer = None

    def load_processor(self):
        if Qwen3VLProcessor is None:
            detail = f" Original import error: {_TOKENIZER_IMPORT_ERROR}" if _TOKENIZER_IMPORT_ERROR else ""
            raise ImportError("MiniMax-H3 image/video conditioning requires Transformers' Qwen3VLProcessor for pixel preprocessing." + detail)
        if self.processor is None:
            processor_path = self._component_path("processor_path", "processor")
            self.processor = Qwen3VLProcessor.from_pretrained(processor_path, local_files_only=self.local_files_only)
        return self.processor

    def unload_processor(self):
        self.processor = None

    def load_text_encoder(self):
        """Stream the native embedding and first 50 decoder layers."""
        if self.text_encoder is not None:
            return self.text_encoder

        text_encoder_path = self._component_path("text_encoder_path", "text_encoder")
        quantized = self.config.get("text_encoder_quantized", False)
        checkpoint_path = self.config["text_encoder_quantized_ckpt"] if quantized else text_encoder_path
        text_config = self._read_text_config(text_encoder_path)
        self._validate_text_config(text_config)
        self._validate_text_tp(text_config)
        attn_type = self._resolve_attn_type(self.config)
        logger.info(
            "Building native MiniMax-H3 Qwen3-VL prefix with weights from {}, attention operator {}, text TP {}/{}, and quantization {}",
            checkpoint_path,
            attn_type,
            self.tp_rank,
            self.tp_size,
            self.config.get("text_encoder_quant_scheme", "Default"),
        )
        text_encoder = _Qwen3VLTextBackboneWeights(
            self.config,
            text_config,
            num_layers=MINIMAX_H3_TEXT_ENCODER_LAYER,
            attn_type=attn_type,
            block_offload=self.block_offload,
            tp_group=self.tp_group,
        )
        if quantized:
            self._load_quantized_weights(text_encoder, checkpoint_path)
        else:
            self._load_native_weights(text_encoder, checkpoint_path, text_config)
        if self.block_offload:
            text_encoder.init_block_offload()
        elif not self.cpu_offload:
            text_encoder.to_cuda()
        self.text_encoder = text_encoder
        return self.text_encoder

    def load_vision_encoder(self):
        if self.vision_encoder is not None:
            return self.vision_encoder
        text_encoder_path = self._component_path("text_encoder_path", "text_encoder")
        model_config = self._read_model_config(text_encoder_path)
        vision_config = dict(model_config["vision_config"])
        logger.info(f"Building native MiniMax-H3 Qwen3-VL vision tower from {text_encoder_path}")
        self.vision_encoder = MiniMaxH3Qwen3VLVisionTower.from_pretrained(text_encoder_path, vision_config)
        return self.vision_encoder

    def unload_text_encoder(self):
        text_encoder = self.text_encoder
        self.text_encoder = None
        if text_encoder is not None:
            del text_encoder
            gc.collect()
            _empty_device_cache()

    def unload_vision_encoder(self):
        vision_encoder = self.vision_encoder
        self.vision_encoder = None
        if vision_encoder is not None:
            del vision_encoder
            gc.collect()
            _empty_device_cache()

    def to_cpu(self):
        if self.text_encoder is not None:
            self.text_encoder.to_cpu()
            _empty_device_cache()
            gc.collect()
        return self

    def load(self):
        self.load_tokenizer()
        self.load_text_encoder()
        return self

    def unload(self):
        self.unload_vision_encoder()
        self.unload_text_encoder()
        self.unload_processor()
        self.unload_tokenizer()

    def _ensure_loaded(self):
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.text_encoder is None:
            self.load_text_encoder()

    def _prepare_t2av_input_ids(self, prompt, device):
        if not isinstance(prompt, str):
            raise TypeError(f"MiniMax-H3 T2AV expects one prompt string, got {type(prompt).__name__}")

        # Match the upstream conditioner: prompt verbatim, no chat template,
        # normalization, padding, or tokenizer-added special tokens.
        token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError("MiniMax-H3 T2AV prompt must encode to at least one token")
        return torch.tensor(token_ids, dtype=torch.long, device=device)

    @staticmethod
    def _get_rope_index(input_ids, mm_token_type_ids, spatial_merge_size, image_grid_thw=None, video_grid_thw=None):
        """Unbatched Qwen3-VL M-RoPE index, matching the released conditioner."""
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
            video_grid_thw[:, 0] = 1
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        groups = []
        for key, group in itertools.groupby(enumerate(mm_token_type_ids.tolist()), lambda item: item[1]):
            group = list(group)
            groups.append((key, group[0][0], group[-1][0] + 1))
        current_position = 0
        positions = []
        for modality, start, end in groups:
            if modality == 0:
                length = end - start
                positions.append(torch.arange(length, device=input_ids.device).view(1, -1).expand(3, -1) + current_position)
                current_position += length
                continue
            grid = next(grid_iters[modality])
            grid_t = int(grid[0])
            grid_h = int(grid[1]) // spatial_merge_size
            grid_w = int(grid[2]) // spatial_merge_size
            temporal = torch.arange(grid_t, device=input_ids.device)
            height = torch.arange(grid_h, device=input_ids.device) + current_position
            width = torch.arange(grid_w, device=input_ids.device) + current_position
            t_grid, h_grid, w_grid = torch.meshgrid(temporal, height, width, indexing="ij")
            block = torch.stack((t_grid, h_grid, w_grid), dim=0).reshape(3, -1)
            block[0] += current_position
            positions.append(block)
            current_position += max(int(grid[1]), int(grid[2])) // spatial_merge_size
        result = torch.cat(positions, dim=1)
        if result.shape[1] != input_ids.shape[0]:
            raise RuntimeError(f"Qwen3-VL M-RoPE produced {result.shape[1]} positions for {input_ids.shape[0]} tokens")
        return result

    def _prepare_keyframe_inputs(self, prompt, images):
        processor = self.load_processor()
        vision = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = vision["image_grid_thw"]
        merge_unit = processor.image_processor.merge_size**2
        token_ids, token_tags = [], []
        for index in range(len(images)):
            count = int(image_grid_thw[index].prod()) // merge_unit
            label = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            block = [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            block += [self.tokenizer.convert_tokens_to_ids("<|image_pad|>")] * count
            block += [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            token_ids += label + block
            token_tags += [MINIMAX_H3_TEXT_TAG] * len(label) + [VIDEO_TAG] * len(block)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
        return token_ids, token_tags, vision["pixel_values"], image_grid_thw, None, None

    def _prepare_reference_inputs(self, prompt, references):
        processor = self.load_processor()
        pixel_values = image_grid_thw = None
        image_counts = []
        images = [reference.image for reference in references if reference.kind == "image"]
        merge_unit = processor.image_processor.merge_size**2
        if images:
            vision = processor.image_processor(images=images, return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            image_counts = [int(grid.prod()) // merge_unit for grid in image_grid_thw]
        pixel_values_videos = video_grid_thw = None
        video_counts = []
        videos = [reference for reference in references if reference.kind == "video"]
        if videos:
            import numpy as np

            sampled = [sample_reference_video_frames(reference.frames) for reference in videos]
            for reference, (_, timestamps) in zip(videos, sampled):
                reference.block_timestamps = timestamps
            vision = processor.video_processor(videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt")
            pixel_values_videos, video_grid_thw = vision["pixel_values_videos"], vision["video_grid_thw"]
            video_counts = [int(grid[1]) * int(grid[2]) // merge_unit for grid in video_grid_thw]
        token_ids, token_tags = build_ref2av_presentation(self.tokenizer, prompt, references, image_counts, video_counts)
        return token_ids, token_tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw

    def _encode_vision(self, input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw):
        if pixel_values is None and pixel_values_videos is None:
            return None, None, None
        vision_encoder = self.load_vision_encoder().to(AI_DEVICE)
        parameter = next(vision_encoder.parameters())
        image_features = image_deepstack = video_features = video_deepstack = None
        try:
            with torch.no_grad():
                if pixel_values is not None:
                    image_features, image_deepstack = vision_encoder(pixel_values.to(AI_DEVICE, parameter.dtype), image_grid_thw.to(AI_DEVICE))
                if pixel_values_videos is not None:
                    video_features, video_deepstack = vision_encoder(pixel_values_videos.to(AI_DEVICE, parameter.dtype), video_grid_thw.to(AI_DEVICE))
            image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            image_mask, video_mask = input_ids == image_token_id, input_ids == video_token_id
            vision_mask = image_mask | video_mask
            feature_dim = image_features.shape[-1] if image_features is not None else video_features.shape[-1]
            combined = torch.empty((int(vision_mask.sum()), feature_dim), device=AI_DEVICE, dtype=parameter.dtype)
            image_joint = image_mask[vision_mask]
            video_joint = video_mask[vision_mask]
            if image_features is not None:
                combined[image_joint] = image_features
            if video_features is not None:
                combined[video_joint] = video_features
            deepstack = []
            source = image_deepstack if image_deepstack is not None else video_deepstack
            for layer_index in range(len(source)):
                one = torch.empty_like(combined)
                if image_deepstack is not None:
                    one[image_joint] = image_deepstack[layer_index]
                if video_deepstack is not None:
                    one[video_joint] = video_deepstack[layer_index]
                deepstack.append(one.cpu())
            return vision_mask.cpu(), combined.cpu(), deepstack
        finally:
            if self.cpu_offload:
                vision_encoder.to("cpu")
                _empty_device_cache()
                gc.collect()

    @torch.inference_mode()
    def infer(self, prompt, image_list=None, references=None):
        """Return unbatched ``[tokens, 5120]`` conditioning and text tags."""
        self._ensure_loaded()
        try:
            # Input encoding happens before DefaultRunner enters its main-model
            # try/finally.  Keep migration here so a partially failed transfer
            # still reaches the conditioner-specific offload cleanup below.
            if references is not None:
                prepared = self._prepare_reference_inputs(prompt, references)
            elif image_list:
                prepared = self._prepare_keyframe_inputs(prompt, image_list)
            else:
                prepared = None
            if prepared is None:
                input_ids = self._prepare_t2av_input_ids(prompt, "cpu")
                token_tags = torch.full((input_ids.shape[0],), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
                position_ids = vision_mask = vision_embeds = deepstack = None
            else:
                token_ids, tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = prepared
                input_ids = torch.tensor(token_ids, dtype=torch.long)
                token_tags = torch.tensor(tags, dtype=torch.long)
                processor = self.load_processor()
                mm_types = torch.tensor(processor.create_mm_token_type_ids([token_ids])[0], dtype=torch.long)
                position_ids = self._get_rope_index(
                    input_ids,
                    mm_types,
                    processor.image_processor.merge_size,
                    image_grid_thw,
                    video_grid_thw,
                )
                vision_mask, vision_embeds, deepstack = self._encode_vision(input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
            if self.cpu_offload and not self.block_offload:
                self.text_encoder.to_cuda()
            device = self.text_encoder.device
            input_ids = input_ids.to(device)
            prompt_embeds = self.text_encoder.forward(
                input_ids,
                None if position_ids is None else position_ids.to(device),
                None if vision_mask is None else vision_mask.to(device),
                None if vision_embeds is None else vision_embeds.to(device),
                None if deepstack is None else [value.to(device) for value in deepstack],
            )
            expected_shape = (input_ids.shape[0], MINIMAX_H3_TEXT_HIDDEN_SIZE)
            if tuple(prompt_embeds.shape) != expected_shape:
                raise RuntimeError(f"MiniMax-H3 expected conditioner hidden shape {expected_shape}, but native Qwen3-VL returned {tuple(prompt_embeds.shape)}")

            prompt_embeds = prompt_embeds.to(device=AI_DEVICE, dtype=GET_DTYPE()).contiguous()
            return {
                "prompt_embeds": prompt_embeds,
                "text_token_tags": token_tags.to(prompt_embeds.device),
            }
        finally:
            if self.cpu_offload and not self.block_offload and self.text_encoder is not None:
                try:
                    self.text_encoder.to_cpu()
                except Exception as error:
                    logger.warning(f"Best-effort MiniMax-H3 text-encoder offload failed: {error}")
                _empty_device_cache()
                gc.collect()


MiniMaxH3TextEncoder = MiniMaxH3Qwen3VLTextEncoder


__all__ = [
    "MINIMAX_H3_TEXT_ENCODER_LAYER",
    "MINIMAX_H3_TEXT_HIDDEN_SIZE",
    "MINIMAX_H3_TEXT_NUM_LAYERS",
    "MINIMAX_H3_TEXT_TAG",
    "MiniMaxH3Qwen3VLTextEncoder",
    "MiniMaxH3TextEncoder",
]
