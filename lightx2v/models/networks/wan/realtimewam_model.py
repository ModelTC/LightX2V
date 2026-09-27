"""Native RealtimeWAM inference: teacher-distilled action flow + video KV cache.

Dense FastWAM and sparse FasterWAM backbones share the existing LightX2V weight
operators. Future-cache mode prefills noisy future tokens once, at video t=1000;
the observed frame has t=0 and cannot attend to future frames.
"""

import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.fastwam_model import FastWAMNativeModel
from lightx2v.models.networks.wan.infer.fastwam.pre_infer import FastWAMPreInfer
from lightx2v.models.networks.wan.infer.fastwam.transformer_infer import FastWAMTransformerInfer
from lightx2v.models.networks.wan.realtimewam_checkpoint import load_checkpoint
from lightx2v.models.networks.wan.weights.fastwam.transformer_weights import (
    FastWAMBlockWeights,
    FastWAMFFNWeights,
    FastWAMSelfAttentionWeights,
)
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, TENSOR_REGISTER


def condition_layers(config):
    layers = config.get("condition_layers")
    layers = list(range(int(config["num_layers"]))) if layers is None else list(layers)
    if not layers or layers != sorted(set(layers)) or any(type(i) is not int or not 0 <= i < config["num_layers"] for i in layers):
        raise ValueError("condition_layers must be nonempty, sorted, unique layer indices")
    return layers


class ActionOnlyWeights(WeightModule):
    def __init__(self, prefix, index, config):
        super().__init__()
        self.add_module("self_attn", FastWAMSelfAttentionWeights(prefix, index, config))
        self.add_module("ffn", FastWAMFFNWeights(prefix, index, config))


class RealtimeWAMTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        conditions = condition_layers(config)
        for name in ("video", "action"):
            expert = WeightModule()
            blocks = []
            for i in range(config["num_layers"]):
                cls = FastWAMBlockWeights if name == "video" or i in conditions else ActionOnlyWeights
                blocks.append(cls(f"mixtures.{name}", i, config))
            expert.add_module("blocks", WeightModuleList(blocks))
            self.add_module(name, expert)
        self.add_module("action_head", MM_WEIGHT_REGISTER["Default"]("mixtures.action.head.weight", "mixtures.action.head.bias"))
        if config.get("video_kv_fusion") == "interval_weighted_sum":
            self.add_module("fusion_logits", WeightModuleList([TENSOR_REGISTER["Default"](f"video_kv_fusion_logits.{i}") for i in range(len(conditions))]))


class RealtimeWAMPreInfer(FastWAMPreInfer):
    def infer_video(self, pre_weight, first_frame_latents, context, context_mask):
        pre = super().infer_video(pre_weight, first_frame_latents, context, context_mask)
        if self.config["action_infer_mode"] == "one_pass_future_cache":
            times = torch.full((len(pre.tokens),), 1000.0, device=pre.tokens.device, dtype=pre.tokens.dtype)
            times[: pre.tokens_per_frame] = 0
            pre.t, pre.t_mod = self._time_embedding(pre_weight.video, times, self.video_hidden_dim)
        return pre


class RealtimeWAMTransformerInfer(FastWAMTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.conditions = condition_layers(config)

    def _reshape_heads(self, x):
        # Sparse refinement layers have fewer heads, with the same head width.
        return x.reshape(x.shape[0], x.shape[-1] // self.head_dim, self.head_dim)

    def prefill_video_cache(self, weights, video_pre):
        x = video_pre.tokens
        mask = torch.ones((len(x), len(x)), device=x.device, dtype=torch.bool)
        mask[: video_pre.tokens_per_frame, video_pre.tokens_per_frame :] = False
        cache = [None] * self.num_layers
        interval = []
        for i in range(self.num_layers):
            block = weights.video.blocks[i]
            q, k, v, residual, gate, shift_mlp, scale_mlp, gate_mlp = self._build_self_attention_io(block, x, video_pre.freqs, video_pre.t_mod)
            interval.append({"k": k, "v": v})
            mixed = block.self_attn.attn.apply(q, k, v, attn_mask=mask)
            x = self._post_block(block, residual, mixed, gate, shift_mlp, scale_mlp, gate_mlp, video_pre.context, video_pre.context_mask)
            if i in self.conditions:
                if self.config.get("video_kv_fusion") == "interval_weighted_sum":
                    logits = weights.fusion_logits[self.conditions.index(i)].tensor.float()
                    if logits.numel() != len(interval):
                        raise ValueError("Fusion logits do not match configured condition intervals")
                    coefficients = logits.softmax(0).to(k.dtype)
                    cache[i] = {key: sum(c * item[key] for c, item in zip(coefficients, interval)) for key in ("k", "v")}
                else:
                    cache[i] = interval[-1]
                interval = []
        return cache

    def action_with_video_cache(self, weights, action_pre, video_kv_cache, video_seq_len, attention_mask):
        x = action_pre.tokens
        for i in range(self.num_layers):
            block = weights.action.blocks[i]
            q, k, v, residual, gate, shift_mlp, scale_mlp, gate_mlp = self._build_self_attention_io(block, x, action_pre.freqs, action_pre.t_mod)
            conditioned = i in self.conditions
            if conditioned:
                cached = video_kv_cache[i]
                k = torch.cat((cached["k"], k), dim=0)
                v = torch.cat((cached["v"], v), dim=0)
            mixed = block.self_attn.attn.apply(q, k, v)
            x = self._post_block(block, residual, mixed, gate, shift_mlp, scale_mlp, gate_mlp, action_pre.context if conditioned else None, action_pre.context_mask)
        return weights.action_head.apply(x)


class RealtimeWAMNativeModel(FastWAMNativeModel):
    transformer_weight_class = RealtimeWAMTransformerWeights

    def _init_infer_class(self):
        self.pre_infer_class = RealtimeWAMPreInfer
        self.transformer_infer_class = RealtimeWAMTransformerInfer

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        state = load_checkpoint(self.config["adapter_model_path"], self.config.get("lora_path"), self.config.get("lora_weights", "ema"))
        conditions = condition_layers(self.config)
        # Fail early on dense/sparse backbone mismatches, before executing actions.
        for i in range(self.config["num_layers"]):
            key = f"mixtures.action.blocks.{i}.cross_attn.q.weight"
            if (key in state) != (i in conditions):
                raise ValueError(f"condition_layers/checkpoint mismatch at action layer {i}")
        has_fusion = any(k.startswith("video_kv_fusion_logits.") for k in state)
        if has_fusion != (self.config.get("video_kv_fusion") == "interval_weighted_sum"):
            raise ValueError("video_kv_fusion/checkpoint mismatch")
        result = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                dtype = GET_DTYPE() if unified_dtype or all(s not in key for s in sensitive_layer) else GET_SENSITIVE_DTYPE()
                if "video_kv_fusion_logits" in key:
                    dtype = torch.float32
                result[key] = value.to(device=self.device, dtype=dtype if value.is_floating_point() else value.dtype)
        return result

    @torch.no_grad()
    def prepare_action_inputs(self, first_frame_latents, context, context_mask, action_chunk_size, robot_state=None, seed=None):
        if first_frame_latents.ndim == 4:
            first_frame_latents = first_frame_latents.unsqueeze(0)
        if self.config["action_infer_mode"] == "one_pass_future_cache":
            frames = int(self.config["num_video_frames"])
            if frames < 5 or frames % 4 != 1:
                raise ValueError("Future cache requires num_video_frames >= 5 and T % 4 == 1")
            shape = list(first_frame_latents.shape)
            shape[2] = (frames - 1) // 4 + 1
            generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
            video = torch.randn(shape, generator=generator, dtype=torch.float32).to(first_frame_latents)
            video[:, :, :1] = first_frame_latents
        else:
            video = first_frame_latents
        context = context.to(device=self.device, dtype=GET_DTYPE())
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        context, context_mask = self._append_robot_state_to_context(context, context_mask, robot_state)
        pre, cache = self._prepare_video_cache(video, context, context_mask)
        return {"context": context, "context_mask": context_mask, "video_kv_cache": cache, "video_seq_len": len(pre.tokens), "attention_mask": None}, (
            1,
            int(action_chunk_size),
            self.config["action_dim"],
        )
