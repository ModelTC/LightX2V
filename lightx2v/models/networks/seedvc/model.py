# SPDX-License-Identifier: GPL-3.0-only
# Seed-VC numerical compatibility; upstream: https://github.com/Plachtaa/seed-vc
"""Native Seed-VC V1 inference; no upstream DiT or pipeline is imported."""

import hashlib
import json
import math

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v.models.networks.base_model import BaseTransformerModel

from .length_regulator import LengthRegulator
from .transformer_infer import SeedVCTransformerInfer, adaptive_norm
from .weights import BlockOffload, SeedVCBlock, SeedVCRMS, linear, project


def fold_weight_norm(state):
    state = dict(state)
    for name in list(state):
        if name.endswith(".weight_g"):
            prefix = name.removesuffix(".weight_g")
            state[prefix + ".weight"] = torch._weight_norm(state.pop(prefix + ".weight_v"), state.pop(name), 0)
    return state


def merge_adapter(path, checkpoint, state, states, architecture):
    """Load the local seed-vc-lora-v1 format before TP partitioning."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta, weights = payload["metadata"], payload["adapter"]
    if meta["format"] != "seed-vc-lora-v1":
        raise ValueError("Expected a seed-vc-lora-v1 adapter")
    with open(checkpoint, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != meta["base_sha256"]:
        raise ValueError("LoRA base checkpoint SHA256 mismatch")
    params = architecture["model_params"]
    core = {key: params.get(key) for key in ("dit_type", "DiT", "length_regulator", "wavenet", "reg_loss_type")}
    core.update(style_dim=params["style_encoder"]["dim"], speech_tokenizer_type=params["speech_tokenizer"]["type"], vocoder_type=params["vocoder"]["type"])
    unused = ("t_embedder2.", "conv1.", "conv2.", "res_projection.", "final_layer.", "wavenet.")
    parameters = {
        "cfm": {key: list(value.shape) for key, value in states["cfm"].items() if not key.endswith(("input_pos", ".freqs")) and not key.startswith(tuple("estimator." + p for p in unused))},
        "length_regulator": {key: list(value.shape) for key, value in states["length_regulator"].items()},
    }
    identity = {"core": core, "preprocess": architecture["preprocess_params"], "parameters": parameters}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if signature != meta["architecture_signature"]:
        raise ValueError("LoRA model/config architecture mismatch")
    depth = params["DiT"]["depth"]
    targets = [f"estimator.transformer.layers.{i}.{name}" for i in range(depth) for name in ("attention.wqkv", "attention.wo", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3")]
    expected = {name + suffix for name in targets for suffix in (".lora_A", ".lora_B")}
    if set(weights) != expected or {item["name"] for item in meta["targets"]} != set(targets):
        raise ValueError("LoRA targets do not match the loaded DiT")
    rank, alpha = meta["lora"]["rank"], meta["lora"]["alpha"]
    if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid LoRA rank/alpha")
    for name in targets:
        key = name.removeprefix("estimator.") + ".weight"
        a, b = weights[name + ".lora_A"], weights[name + ".lora_B"]
        if a.shape != (rank, state[key].shape[1]) or b.shape != (state[key].shape[0], rank) or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Invalid LoRA tensors for {name}")
        state[key] = state[key] + (b.float() @ a.float()) * (alpha / rank)


class SeedVCModel(BaseTransformerModel):
    def __init__(self, config, device):
        super().__init__(config["model_path"], config, torch.device("cpu"))
        self.device = device
        self.params = config["architecture"]["model_params"]
        self.dit = self.params["DiT"]
        checkpoint = config["dit_original_ckpt"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)["net"]
        states = {group: {k.removeprefix("module."): v.float() if v.is_floating_point() else v for k, v in values.items()} for group, values in payload.items()}
        state = {key.removeprefix("estimator."): value for key, value in states["cfm"].items()}
        if config.get("adapter_path"):
            merge_adapter(config["adapter_path"], checkpoint, state, states, config["architecture"])
        state = fold_weight_norm(state)
        self.regulator = LengthRegulator(states["length_regulator"], self.params["length_regulator"], device)
        target = torch.device("cpu") if self.cpu_offload else device

        def make(i, destination):
            return SeedVCBlock(i, state, self.dit, destination, self.tp_size, self.tp_rank, self.tp_group)

        self.blocks = [make(i, target) for i in range(self.dit["depth"])]
        self.offload = BlockOffload(self.blocks, lambda: make(0, device), device) if self.cpu_offload else None
        self._init_infer()
        self.projections = {}
        names = ["cond_projection", "cond_x_merge_linear", "t_embedder.mlp.0", "t_embedder.mlp.2", "transformer.norm.project_layer"]
        if self.dit["style_as_token"]:
            names.append("style_in")
        if self.dit["long_skip_connection"]:
            names.append("skip_linear")
        if self.dit["final_layer_type"] == "wavenet":
            names += ["conv1", "res_projection", "t_embedder2.mlp.0", "t_embedder2.mlp.2", "final_layer.linear", "final_layer.adaLN_modulation.1"]
        else:
            names += ["final_mlp.0", "final_mlp.2"]
        for name in names:
            self.projections[name] = linear(name, state, device)
        self.final_norm = SeedVCRMS(state["transformer.norm.norm.weight"].to(device))
        self.head_state = {key: value.to(device) for key, value in state.items() if key.startswith(("wavenet.", "conv2."))}
        self.time_freq = torch.exp(-math.log(10000) * torch.arange(128, device=device, dtype=torch.float32) / 128)
        head_dim = self.dit["hidden_dim"] // self.dit["num_heads"]
        # Upstream builds RoPE on CPU in FP32, then transfers its cache.
        freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(16384), freq)
        complex_freq = torch.polar(torch.ones_like(angles), angles)
        self.rope = torch.stack((complex_freq.real, complex_freq.imag), -1).to(device)
        logger.info(
            "Seed-VC DiT loaded: dtype=FP32 rank={} TP={} local_heads={} layers={} offload={}", self.tp_rank, self.tp_size, self.dit["num_heads"] // self.tp_size, len(self.blocks), self.cpu_offload
        )

    def _init_infer_class(self):
        return SeedVCTransformerInfer

    def _init_infer(self):
        self.transformer_infer = SeedVCTransformerInfer(self.config)

    def _seq_parallel_pre_process(self, x):
        return x

    def _seq_parallel_post_process(self, x):
        return x

    def _infer_cond_uncond(self, inputs, infer_condition=True):
        return self.infer(**inputs)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.transformer_infer.set_scheduler(scheduler)

    def proj(self, name, x):
        return project(self.projections[name], x)

    def time_embedding(self, t, prefix="t_embedder"):
        angles = 1000 * t[:, None].float() * self.time_freq[None]
        emb = torch.cat((angles.cos(), angles.sin()), -1)
        return self.proj(prefix + ".mlp.2", F.silu(self.proj(prefix + ".mlp.0", emb)))

    def conv(self, name, x, dilation=1, reflect=False):
        weight, bias = self.head_state[name + ".weight"], self.head_state[name + ".bias"]
        if reflect and weight.shape[-1] > 1:
            pad = (weight.shape[-1] - 1) * dilation // 2
            x = F.pad(x, (pad, pad), mode="reflect")
        return F.conv1d(x, weight, bias, dilation=dilation)

    def wavenet(self, x, mask, t):
        channels = self.params["wavenet"]["hidden_dim"]
        count = self.params["wavenet"]["num_layers"]
        g = self.conv("wavenet.cond_layer.conv.conv", self.time_embedding(t, "t_embedder2").unsqueeze(2))
        result = torch.zeros_like(x)
        for i in range(count):
            value = self.conv(f"wavenet.in_layers.{i}.conv.conv", x, self.params["wavenet"]["dilation_rate"] ** i, reflect=True)
            a, b = (value + g[:, 2 * channels * i : 2 * channels * (i + 1)]).chunk(2, 1)
            value = self.conv(f"wavenet.res_skip_layers.{i}.conv.conv", a.tanh() * b.sigmoid())
            if i + 1 < count:
                residual, value = value.chunk(2, 1)
                x = (x + residual) * mask
            result = result + value
        return result * mask

    def infer(self, x, prompt, lengths, t, style, condition, check_stop=None):
        emb = self.time_embedding(t)
        residual = x.transpose(1, 2)
        pieces = [residual, prompt.transpose(1, 2), self.proj("cond_projection", condition)]
        if self.dit["style_condition"] and not self.dit["style_as_token"]:
            pieces.append(style[:, None].expand(-1, x.shape[-1], -1))
        h = self.proj("cond_x_merge_linear", torch.cat(pieces, -1))
        if self.dit["style_as_token"]:
            h = torch.cat((self.proj("style_in", style)[:, None], h), 1)
        if self.dit["time_as_token"]:
            h = torch.cat((emb[:, None], h), 1)
        extra = int(self.dit["time_as_token"]) + int(self.dit["style_as_token"])
        valid = torch.arange(h.shape[1], device=self.device)[None] < lengths[:, None] + extra
        mask = valid[:, None, None, :]
        if self.dit["is_causal"]:
            mask = mask & torch.ones((h.shape[1], h.shape[1]), dtype=torch.bool, device=self.device).tril()
        skips = []
        blocks = self.offload.iterate() if self.offload else enumerate(self.blocks)
        for index, block in blocks:
            if check_stop is not None:
                check_stop()
            receive = self.dit["uvit_skip_connection"] and index > len(self.blocks) // 2
            skip = skips.pop() if receive else None
            h = self.transformer_infer.infer(index, block, h, emb[:, None], self.rope[: h.shape[1]], mask, skip)
            if self.dit["uvit_skip_connection"] and index < len(self.blocks) // 2:
                skips.append(h)
        h = adaptive_norm(h, emb[:, None], self.final_norm, self.projections["transformer.norm.project_layer"])
        h = h[:, extra:]
        if self.dit["long_skip_connection"]:
            h = self.proj("skip_linear", torch.cat((h, residual), -1))
        if self.dit["final_layer_type"] == "mlp":
            return self.proj("final_mlp.2", F.silu(self.proj("final_mlp.0", h))).transpose(1, 2)
        value = self.wavenet(self.proj("conv1", h).transpose(1, 2), valid[:, None, extra:], t).transpose(1, 2) + self.proj("res_projection", h)
        shift, scale = self.proj("final_layer.adaLN_modulation.1", F.silu(emb)).chunk(2, 1)
        value = F.layer_norm(value, (value.shape[-1],), eps=1e-6) * (1 + scale[:, None]) + shift[:, None]
        return self.conv("conv2", self.proj("final_layer.linear", value).transpose(1, 2))

    def runtime_stats(self):
        return {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "local_heads": self.dit["num_heads"] // self.tp_size,
            "blocks_executed": self.transformer_infer.blocks_executed,
            "offload_transfers": self.offload.transfer_count if self.offload else 0,
            "offload_bytes": self.offload.transfer_bytes if self.offload else 0,
        }

    def close(self):
        if self.offload:
            self.offload.close()
        self.blocks.clear()
        self.projections.clear()
        self.head_state.clear()
        self.regulator = None
        self.rope = self.final_norm = None
        self.transformer_infer.compiled_blocks.clear()
