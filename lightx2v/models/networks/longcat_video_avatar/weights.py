from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.ops.mm.mm_weight import MMWeight
from lightx2v.utils.registry_factory import CONV3D_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
from lightx2v.utils.utils import load_weights


class AvatarLoRALinear(MMWeight):
    """Mix the DMD and OPSD adapters at the current denoising step."""

    def __init__(self, prefix, groups=1, bias=True):
        super().__init__(f"{prefix}.weight", f"{prefix}.bias" if bias else None)
        self.groups = groups

    def load_adapters(self, checkpoints):
        module_name = self.weight_name.removesuffix(".weight")
        prefix = f"lora.{module_name}".replace(".", "___lorahyphen___")
        up_keys = [f"{prefix}.lora_up.weight"] if self.groups == 1 else [f"{prefix}.lora_up.blocks.{i}.weight" for i in range(self.groups)]
        self.adapters = []
        for checkpoint in checkpoints:
            down = checkpoint.get_tensor(f"{prefix}.lora_down.weight").to(self.weight)
            ups = [checkpoint.get_tensor(key).to(self.weight) for key in up_keys]
            scale = checkpoint.get_tensor(f"{prefix}.alpha_scale").item()
            self.adapters.append((down, ups, scale))

    def apply(self, inputs, strengths):
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


def linear(prefix, *, fp32=False):
    return MM_WEIGHT_REGISTER["Default-ForceFp32" if fp32 else "Default"](f"{prefix}.weight", f"{prefix}.bias")


def layer_norm(prefix, *, fp32=False, eps=1e-6):
    norm = LN_WEIGHT_REGISTER["torch"](f"{prefix}.weight", f"{prefix}.bias", eps=eps)
    norm.sensitive_layer_dtype = torch.float32 if fp32 else norm.infer_dtype
    return norm


class AvatarPreWeights(WeightModule):
    def __init__(self, patch_size):
        super().__init__()
        self.add_module("patch", CONV3D_WEIGHT_REGISTER["Default"]("x_embedder.proj.weight", "x_embedder.proj.bias", stride=patch_size))
        self.add_module("time_in", linear("t_embedder.mlp.0", fp32=True))
        self.add_module("time_out", linear("t_embedder.mlp.2", fp32=True))
        self.add_module("text_in", linear("y_embedder.y_proj.0"))
        self.add_module("text_out", linear("y_embedder.y_proj.2"))
        self.add_module("audio_first", linear("audio_proj.proj1"))
        self.add_module("audio_later", linear("audio_proj.proj1_vf"))
        self.add_module("audio_hidden", linear("audio_proj.proj2"))
        self.add_module("audio_out", linear("audio_proj.proj3"))
        self.add_module("audio_norm", layer_norm("audio_proj.norm", eps=1e-5))


class AvatarAttentionWeights(WeightModule):
    def __init__(self, block_prefix, name):
        super().__init__()
        prefix = f"{block_prefix}.{name}"
        if name == "attn":
            self.add_module("qkv", AvatarLoRALinear(f"{prefix}.qkv", groups=3))
            self.add_module("out", AvatarLoRALinear(f"{prefix}.proj"))
        elif name == "cross_attn":
            self.add_module("q", AvatarLoRALinear(f"{prefix}.q_linear"))
            self.add_module("kv", AvatarLoRALinear(f"{prefix}.kv_linear", groups=2))
            self.add_module("out", linear(f"{prefix}.proj"))
        else:
            self.add_module("q", linear(f"{prefix}.q_linear"))
            self.add_module("kv", linear(f"{prefix}.kv_linear"))
            self.add_module("out", linear(f"{prefix}.proj"))
        self.add_module("q_norm", RMS_WEIGHT_REGISTER["fp32_variance"](f"{prefix}.q_norm.weight"))
        self.add_module("k_norm", RMS_WEIGHT_REGISTER["fp32_variance"](f"{prefix}.k_norm.weight"))


class AvatarBlockWeights(WeightModule):
    def __init__(self, index):
        super().__init__()
        prefix = f"blocks.{index}"
        self.add_module("modulation", linear(f"{prefix}.adaLN_modulation.1", fp32=True))
        self.add_module("audio_modulation", linear(f"{prefix}.audio_adaLN_modulation.1", fp32=True))
        self.add_module("text_norm", layer_norm(f"{prefix}.pre_crs_attn_norm", fp32=True))
        self.add_module("audio_norm", layer_norm(f"{prefix}.pre_video_crs_attn_norm", fp32=True))
        self.add_module("self_attn", AvatarAttentionWeights(prefix, "attn"))
        self.add_module("text_attn", AvatarAttentionWeights(prefix, "cross_attn"))
        self.add_module("audio_attn", AvatarAttentionWeights(prefix, "audio_cross_attn"))
        self.add_module("ffn_gate", AvatarLoRALinear(f"{prefix}.ffn.w1", bias=False))
        self.add_module("ffn_out", AvatarLoRALinear(f"{prefix}.ffn.w2", bias=False))
        self.add_module("ffn_up", AvatarLoRALinear(f"{prefix}.ffn.w3", bias=False))


class AvatarWeights(WeightModule):
    def __init__(self, config, device):
        super().__init__()
        self.add_module("pre", AvatarPreWeights(tuple(config["patch_size"])))
        self.add_module("blocks", WeightModuleList([AvatarBlockWeights(index) for index in range(config["depth"])]))
        self.add_module("final_modulation", linear("final_layer.adaLN_modulation.1", fp32=True))
        self.add_module("final_out", linear("final_layer.linear", fp32=True))

        checkpoint = load_weights(config["dit_original_ckpt"])
        root = Path(config["model_path"])
        with safe_open(root / "loraA/lora.safetensors", framework="pt", device="cpu") as dmd, safe_open(root / "loraB/lora.safetensors", framework="pt", device="cpu") as opsd:
            for _, weight in self.named_weight_leaves():
                dtype = weight.sensitive_layer_dtype if isinstance(weight, LN_WEIGHT_REGISTER["torch"]) else torch.bfloat16
                weight.load({name: checkpoint.pop(name).to(device=device, dtype=dtype) for name, _, _ in weight.base_attrs})
                if isinstance(weight, AvatarLoRALinear):
                    weight.load_adapters((dmd, opsd))
