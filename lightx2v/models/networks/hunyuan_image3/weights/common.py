import math

import torch
import torch.nn.functional as F

import lightx2v.common.ops  # noqa: F401
from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


def _as_list(value, length):
    if isinstance(value, list):
        return value
    return [value for _ in range(length)]


def _moe_value(config, name, block_index, default=None):
    value = config.get(name, default)
    if isinstance(value, list):
        return value[block_index]
    return value


class TensorPairWeight:
    def __init__(self, weight_name, bias_name=None):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.weight = None
        self.bias = None

    def load(self, weight_dict):
        self.weight = weight_dict[self.weight_name]
        self.bias = weight_dict[self.bias_name] if self.bias_name is not None else None

    def apply_group_norm(self, x, eps=1e-5, groups=32):
        channels = x.shape[1]
        groups = math.gcd(groups, channels)
        return F.group_norm(x, groups, self.weight, self.bias, eps)

    def to_cpu(self, non_blocking=False):
        self.weight = self.weight.cpu(non_blocking=non_blocking)
        if self.bias is not None:
            self.bias = self.bias.cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=False):
        self.weight = self.weight.to(AI_DEVICE, non_blocking=non_blocking)
        if self.bias is not None:
            self.bias = self.bias.to(AI_DEVICE, non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.weight.detach().cpu().clone()
        if self.bias is not None:
            destination[self.bias_name] = self.bias.detach().cpu().clone()
        return destination


class HunyuanImage3Conv2dWeight:
    def __init__(self, weight_name, bias_name, stride=1, padding=0, dilation=1, groups=1):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = None
        self.bias = None

    def load(self, weight_dict):
        self.weight = weight_dict[self.weight_name]
        self.bias = weight_dict[self.bias_name] if self.bias_name is not None else None

    def apply(self, input_tensor):
        input_tensor = input_tensor.to(device=self.weight.device, dtype=self.weight.dtype)
        return torch.nn.functional.conv2d(
            input_tensor,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def to_cpu(self, non_blocking=False):
        self.weight = self.weight.cpu(non_blocking=non_blocking)
        if self.bias is not None:
            self.bias = self.bias.cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=False):
        self.weight = self.weight.to(AI_DEVICE, non_blocking=non_blocking)
        if self.bias is not None:
            self.bias = self.bias.to(AI_DEVICE, non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.weight.detach().cpu().clone()
        if self.bias is not None:
            destination[self.bias_name] = self.bias.detach().cpu().clone()
        return destination


class HunyuanImage3TimestepEmbedderWeights(WeightModule):
    def __init__(self, prefix):
        super().__init__()
        self.add_module("linear_1", MM_WEIGHT_REGISTER["Default"](f"{prefix}.mlp.0.weight", f"{prefix}.mlp.0.bias"))
        self.add_module("linear_2", MM_WEIGHT_REGISTER["Default"](f"{prefix}.mlp.2.weight", f"{prefix}.mlp.2.bias"))


class HunyuanImage3ResBlockWeights(WeightModule):
    def __init__(
        self,
        prefix,
        in_channels,
        out_channels,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_module("in_norm", TensorPairWeight(f"{prefix}.in_layers.0.weight", f"{prefix}.in_layers.0.bias"))
        self.add_module(
            "in_conv",
            HunyuanImage3Conv2dWeight(
                f"{prefix}.in_layers.2.weight",
                f"{prefix}.in_layers.2.bias",
                stride=1,
                padding=1,
            ),
        )
        self.add_module(
            "emb_proj",
            MM_WEIGHT_REGISTER["Default"](
                f"{prefix}.emb_layers.1.weight",
                f"{prefix}.emb_layers.1.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
            ),
        )
        self.add_module("out_norm", TensorPairWeight(f"{prefix}.out_layers.0.weight", f"{prefix}.out_layers.0.bias"))
        self.add_module(
            "out_conv",
            HunyuanImage3Conv2dWeight(
                f"{prefix}.out_layers.3.weight",
                f"{prefix}.out_layers.3.bias",
                stride=1,
                padding=1,
            ),
        )
        if in_channels != out_channels:
            self.add_module(
                "skip_connection",
                HunyuanImage3Conv2dWeight(
                    f"{prefix}.skip_connection.weight",
                    f"{prefix}.skip_connection.bias",
                    stride=1,
                    padding=0,
                ),
            )
        else:
            self.skip_connection = None


class HunyuanImage3UNetDownWeights(WeightModule):
    def __init__(self, prefix, config):
        super().__init__()
        patch_size = int(config.get("patch_size", 1) or 1)
        hidden_channels = int(config["patch_embed_hidden_dim"])
        out_channels = int(config["hidden_size"])
        in_channels = int(config.get("vae", {}).get("latent_channels", config.get("latent_channels", 32)))
        self.patch_size = patch_size
        self.add_module(
            "input_conv",
            HunyuanImage3Conv2dWeight(
                f"{prefix}.model.0.weight",
                f"{prefix}.model.0.bias",
                stride=1,
                padding=1,
            ),
        )

        block_count = 1 if patch_size == 1 else patch_size // 2
        blocks = []
        for i in range(block_count):
            block_in = hidden_channels
            block_out = out_channels if patch_size == 1 or (i + 1) * 2 == patch_size else hidden_channels
            blocks.append(HunyuanImage3ResBlockWeights(f"{prefix}.model.{i + 1}", block_in, block_out))
        self.blocks = WeightModuleList(blocks)
        self.add_module("blocks", self.blocks)
        self.in_channels = in_channels


class HunyuanImage3UNetUpWeights(WeightModule):
    def __init__(self, prefix, config):
        super().__init__()
        patch_size = int(config.get("patch_size", 1) or 1)
        hidden_channels = int(config["patch_embed_hidden_dim"])
        in_channels = int(config["hidden_size"])
        out_channels = int(config.get("vae", {}).get("latent_channels", config.get("latent_channels", 32)))
        self.patch_size = patch_size

        block_count = 1 if patch_size == 1 else patch_size // 2
        blocks = []
        for i in range(block_count):
            block_in = in_channels if i == 0 else hidden_channels
            blocks.append(HunyuanImage3ResBlockWeights(f"{prefix}.model.{i}", block_in, hidden_channels))
        self.blocks = WeightModuleList(blocks)
        self.add_module("blocks", self.blocks)

        last_index = block_count
        self.add_module("out_norm", TensorPairWeight(f"{prefix}.model.{last_index}.0.weight", f"{prefix}.model.{last_index}.0.bias"))
        self.add_module(
            "output_conv",
            HunyuanImage3Conv2dWeight(
                f"{prefix}.model.{last_index}.2.weight",
                f"{prefix}.model.{last_index}.2.bias",
                stride=1,
                padding=1,
            ),
        )
        self.out_channels = out_channels


class HunyuanImage3DenseMLPWeights(WeightModule):
    def __init__(
        self,
        prefix,
        mm_type,
        mlp_bias=False,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        lora_path=None,
    ):
        super().__init__()
        gate_and_up_bias = f"{prefix}.gate_and_up_proj.bias" if mlp_bias else None
        down_bias = f"{prefix}.down_proj.bias" if mlp_bias else None
        self.add_module(
            "gate_and_up_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.gate_and_up_proj.weight",
                gate_and_up_bias,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "down_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.down_proj.weight",
                down_bias,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=prefix,
                lora_path=lora_path,
            ),
        )


class HunyuanImage3MoEWeights(WeightModule):
    def __init__(
        self,
        prefix,
        block_index,
        config,
        mm_type,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        lora_path=None,
    ):
        super().__init__()
        self.num_experts = int(_moe_value(config, "num_experts", block_index, 1))
        self.moe_topk = int(_moe_value(config, "moe_topk", block_index, 1))
        self.add_module(
            "gate",
            MM_WEIGHT_REGISTER["Default-ForceFp32"](
                f"{prefix}.gate.wg.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=prefix,
                lora_path=lora_path,
            ),
        )
        if config.get("use_mixed_mlp_moe", False):
            self.add_module(
                "shared_mlp",
                HunyuanImage3DenseMLPWeights(
                    f"{prefix}.shared_mlp",
                    mm_type,
                    config.get("mlp_bias", False),
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_file,
                    lora_path,
                ),
            )
        else:
            self.shared_mlp = None
        self.experts = WeightModuleList(
            [
                HunyuanImage3DenseMLPWeights(
                    f"{prefix}.experts.{i}",
                    mm_type,
                    config.get("mlp_bias", False),
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_file,
                    lora_path,
                )
                for i in range(self.num_experts)
            ]
        )
        self.add_module("experts", self.experts)


class HunyuanImage3AttentionWeights(WeightModule):
    def __init__(
        self,
        block_prefix,
        block_index,
        config,
        mm_type,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        lora_path=None,
    ):
        super().__init__()
        prefix = f"{block_prefix}.{block_index}"
        rms_type = config.get("rms_norm_type", "fp32_variance")
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config.get("num_key_value_heads") or self.heads)
        self.head_dim = int(config.get("attention_head_dim", config["hidden_size"] // self.heads))
        self.add_module(
            "input_layernorm",
            RMS_WEIGHT_REGISTER[config.get("rms_norm_type", "fp32_variance")](
                f"{prefix}.input_layernorm.weight",
                eps=config.get("rms_norm_eps", 1e-5),
            ),
        )
        attn_bias = ".bias" if config.get("attention_bias", False) else None
        self.add_module(
            "qkv_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attn.qkv_proj.weight",
                attn_bias and f"{prefix}.self_attn.qkv_proj{attn_bias}",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "o_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attn.o_proj.weight",
                attn_bias and f"{prefix}.self_attn.o_proj{attn_bias}",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=prefix,
                lora_path=lora_path,
            ),
        )
        if config.get("use_qk_norm", True):
            self.add_module(
                "query_layernorm",
                RMS_WEIGHT_REGISTER[config.get("rms_norm_type", "fp32_variance")](
                    f"{prefix}.self_attn.query_layernorm.weight",
                    eps=config.get("rms_norm_eps", 1e-5),
                ),
            )
            self.add_module(
                "key_layernorm",
                RMS_WEIGHT_REGISTER[config.get("rms_norm_type", "fp32_variance")](
                    f"{prefix}.self_attn.key_layernorm.weight",
                    eps=config.get("rms_norm_eps", 1e-5),
                ),
            )
        else:
            self.query_layernorm = None
            self.key_layernorm = None


class HunyuanImage3MLPPhaseWeights(WeightModule):
    def __init__(
        self,
        block_prefix,
        block_index,
        config,
        mm_type,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        lora_path=None,
    ):
        super().__init__()
        prefix = f"{block_prefix}.{block_index}"
        self.add_module(
            "post_attention_layernorm",
            RMS_WEIGHT_REGISTER[config.get("rms_norm_type", "fp32_variance")](
                f"{prefix}.post_attention_layernorm.weight",
                eps=config.get("rms_norm_eps", 1e-5),
            ),
        )
        is_moe = int(_moe_value(config, "num_experts", block_index, 1)) > 1 and block_index >= int(config.get("moe_layer_num_skipped", 0))
        self.is_moe = is_moe
        if is_moe:
            self.add_module(
                "moe",
                HunyuanImage3MoEWeights(
                    f"{prefix}.mlp",
                    block_index,
                    config,
                    mm_type,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_file,
                    lora_path,
                ),
            )
            self.experts = self.moe.experts
        else:
            self.add_module(
                "dense_mlp",
                HunyuanImage3DenseMLPWeights(
                    f"{prefix}.mlp",
                    mm_type,
                    config.get("mlp_bias", False),
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_file,
                    lora_path,
                ),
            )
            self.gate_and_up_proj = self.dense_mlp.gate_and_up_proj
            self.down_proj = self.dense_mlp.down_proj
