import json
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.ops.mm.mm_weight import MMWeight
from lightx2v.models.schedulers.longcat_video_avatar.scheduler import lora_strengths_by_step
from lightx2v.utils.registry_factory import CONV3D_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
from lightx2v.utils.utils import load_weights


class AvatarLoRALinear(MMWeight):
    """Mix the DMD and OPSD adapters at the current denoising step."""

    def __init__(self, prefix, groups=1, bias=True):
        super().__init__(f"{prefix}.weight", f"{prefix}.bias" if bias else None)
        self.groups = groups

    def load_adapters(self, checkpoints, merge_phases):
        module_name = self.weight_name.removesuffix(".weight")
        prefix = f"lora.{module_name}".replace(".", "___lorahyphen___")
        up_keys = [f"{prefix}.lora_up.weight"] if self.groups == 1 else [f"{prefix}.lora_up.blocks.{i}.weight" for i in range(self.groups)]
        self.adapters = []
        for checkpoint in checkpoints:
            down = checkpoint.get_tensor(f"{prefix}.lora_down.weight").to(self.weight)
            ups = [checkpoint.get_tensor(key).to(self.weight) for key in up_keys]
            scale = checkpoint.get_tensor(f"{prefix}.alpha_scale").item()
            self.adapters.append((down, ups, scale))

        self.merged_weights = {strengths: self._merged_weight(strengths) for strengths in merge_phases}
        if self.merged_weights:
            del self.weight, self.adapters

    def _merged_weight(self, strengths):
        # LoRA groups split the output rows of the checkpoint's [out, in] weight.
        merged = self.weight.T.float()
        for strength, (down, ups, scale) in zip(strengths, self.adapters):
            if strength:
                for dest, up, part in zip(merged.chunk(self.groups), ups, down.chunk(self.groups)):
                    dest.addmm_(up.float(), part.float(), alpha=strength * scale)
        return merged.to(self.weight.dtype)

    def apply(self, inputs, strengths):
        if strengths is None:
            return F.linear(inputs, self.active_weight, self.bias)
        result = super().apply(inputs)
        delta = None
        for strength, (down, ups, scale) in zip(strengths, self.adapters):
            if not strength:
                continue
            reduced = F.linear(inputs, down)
            if self.groups == 1:
                update = F.linear(reduced, ups[0])
            else:
                parts = reduced.chunk(self.groups, dim=-1)
                update = torch.cat([F.linear(part, up) for part, up in zip(parts, ups)], dim=-1)
            update.mul_(strength).mul_(scale)
            delta = update if delta is None else delta.add_(update)
        return result.add_(delta)


def linear(prefix, mm_type="Default", *, bias=True):
    return MM_WEIGHT_REGISTER[mm_type](f"{prefix}.weight", f"{prefix}.bias" if bias else None)


class AvatarFP8PhaseLinear(WeightModule):
    """Select the FP8 projection quantized with the active LoRA already merged."""

    def __init__(self, prefix, bias, phase_indices):
        super().__init__()
        self.merged_weights = {}
        for strengths, index in phase_indices.items():
            phase = f"phase_{index}.{prefix}"
            projection = linear(phase, "fp8-sgl", bias=bias)
            self.add_module(f"phase_{index}", projection)
            self.merged_weights[strengths] = projection

    def apply(self, inputs, _strengths):
        return self.active_weight.apply(inputs)


def layer_norm(prefix, *, fp32=False, eps=1e-6):
    norm = LN_WEIGHT_REGISTER["torch"](f"{prefix}.weight", f"{prefix}.bias", eps=eps)
    norm.sensitive_layer_dtype = torch.float32 if fp32 else norm.infer_dtype
    return norm


class AvatarPreWeights(WeightModule):
    def __init__(self, patch_size):
        super().__init__()
        self.add_module("patch", CONV3D_WEIGHT_REGISTER["Default"]("x_embedder.proj.weight", "x_embedder.proj.bias", stride=patch_size))
        self.add_module("time_in", linear("t_embedder.mlp.0", "Default-ForceFp32"))
        self.add_module("time_out", linear("t_embedder.mlp.2", "Default-ForceFp32"))
        self.add_module("text_in", linear("y_embedder.y_proj.0"))
        self.add_module("text_out", linear("y_embedder.y_proj.2"))
        self.add_module("audio_first", linear("audio_proj.proj1"))
        self.add_module("audio_later", linear("audio_proj.proj1_vf"))
        self.add_module("audio_hidden", linear("audio_proj.proj2"))
        self.add_module("audio_out", linear("audio_proj.proj3"))
        self.add_module("audio_norm", layer_norm("audio_proj.norm", eps=1e-5))


class AvatarAttentionWeights(WeightModule):
    def __init__(self, block_prefix, name, lora_linear, mm_type):
        super().__init__()
        prefix = f"{block_prefix}.{name}"
        if name == "attn":
            self.add_module("qkv", lora_linear(f"{prefix}.qkv", groups=3))
            self.add_module("out", lora_linear(f"{prefix}.proj"))
        elif name == "cross_attn":
            self.add_module("q", lora_linear(f"{prefix}.q_linear"))
            self.add_module("kv", lora_linear(f"{prefix}.kv_linear", groups=2))
            self.add_module("out", linear(f"{prefix}.proj", mm_type))
        else:
            self.add_module("q", linear(f"{prefix}.q_linear", mm_type))
            self.add_module("kv", linear(f"{prefix}.kv_linear", mm_type))
            self.add_module("out", linear(f"{prefix}.proj", mm_type))
        self.add_module("q_norm", RMS_WEIGHT_REGISTER["fp32_variance"](f"{prefix}.q_norm.weight"))
        self.add_module("k_norm", RMS_WEIGHT_REGISTER["fp32_variance"](f"{prefix}.k_norm.weight"))


class AvatarBlockWeights(WeightModule):
    def __init__(self, index, lora_linear, mm_type):
        super().__init__()
        prefix = f"blocks.{index}"
        self.add_module("modulation", linear(f"{prefix}.adaLN_modulation.1", "Default-ForceFp32"))
        self.add_module("audio_modulation", linear(f"{prefix}.audio_adaLN_modulation.1", "Default-ForceFp32"))
        self.add_module("text_norm", layer_norm(f"{prefix}.pre_crs_attn_norm", fp32=True))
        self.add_module("audio_norm", layer_norm(f"{prefix}.pre_video_crs_attn_norm", fp32=True))
        self.add_module("self_attn", AvatarAttentionWeights(prefix, "attn", lora_linear, mm_type))
        self.add_module("text_attn", AvatarAttentionWeights(prefix, "cross_attn", lora_linear, mm_type))
        self.add_module("audio_attn", AvatarAttentionWeights(prefix, "audio_cross_attn", lora_linear, mm_type))
        self.add_module("ffn_gate", lora_linear(f"{prefix}.ffn.w1", bias=False))
        self.add_module("ffn_out", lora_linear(f"{prefix}.ffn.w2", bias=False))
        self.add_module("ffn_up", lora_linear(f"{prefix}.ffn.w3", bias=False))


class AvatarWeights(WeightModule):
    def __init__(self, config, device):
        super().__init__()
        self.merged_projections = []
        self.active_strengths = None
        phases = dict.fromkeys(lora_strengths_by_step(config)) if config.get("premerge_lora") else ()
        quantized = config.get("dit_quantized", False)
        checkpoint_path = config["dit_quantized_ckpt"] if quantized else config["dit_original_ckpt"]
        if quantized:
            with safe_open(checkpoint_path, framework="pt") as file:
                metadata = file.metadata() or {}
                if "lora_phases" not in metadata:
                    raise ValueError(f"{checkpoint_path} requires LongCat Avatar FP8 weights with premerged LoRA phases")
                stored = [tuple(pair) for pair in json.loads(metadata["lora_phases"])]
            if any(pair not in stored for pair in phases):
                raise ValueError(f"FP8 checkpoint contains LoRA strengths {stored}, requested {list(phases)}")
            phase_indices = {pair: stored.index(pair) for pair in phases}

        def lora_linear(prefix, groups=1, bias=True):
            if quantized:
                projection = AvatarFP8PhaseLinear(prefix, bias, phase_indices)
            else:
                projection = AvatarLoRALinear(prefix, groups, bias)
            if phases:
                self.merged_projections.append(projection)
            return projection

        self.add_module("pre", AvatarPreWeights(tuple(config["patch_size"])))
        mm_type = config["dit_quant_scheme"] if quantized else "Default"
        self.add_module("blocks", WeightModuleList([AvatarBlockWeights(index, lora_linear, mm_type) for index in range(config["depth"])]))
        self.add_module("final_modulation", linear("final_layer.adaLN_modulation.1", "Default-ForceFp32"))
        self.add_module("final_out", linear("final_layer.linear", "Default-ForceFp32"))

        checkpoint = load_weights(checkpoint_path)
        default_dtype = None if quantized else torch.bfloat16
        with ExitStack() as files:
            if not quantized:
                root = Path(config["model_path"])
                adapters = [files.enter_context(safe_open(root / folder / "lora.safetensors", framework="pt")) for folder in ("loraA", "loraB")]
            for _, weight in self.named_weight_leaves():
                dtype = weight.sensitive_layer_dtype if isinstance(weight, LN_WEIGHT_REGISTER["torch"]) else default_dtype
                weight.load({name: checkpoint.pop(name).to(device=device, dtype=dtype) for name, _, _ in weight.base_attrs})
                if isinstance(weight, AvatarLoRALinear):
                    weight.load_adapters(adapters, phases)

    def select_lora(self, strengths):
        if not self.merged_projections:
            return strengths
        # Select the phase before entering the compiled block so both phases can share its graph.
        if self.active_strengths != strengths:
            for projection in self.merged_projections:
                projection.active_weight = projection.merged_weights[strengths]
            self.active_strengths = strengths
        return None
