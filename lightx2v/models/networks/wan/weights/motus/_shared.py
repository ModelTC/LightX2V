import copy
import re

import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.motus.primitives import sinusoidal_embedding_1d
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, TENSOR_REGISTER


def slice_prefixed_state_dict(weight_dict, prefix):
    return {key[len(prefix) :]: value for key, value in weight_dict.items() if key.startswith(prefix)}


def load_prefixed_submodules(root, weight_dict, prefix_by_name):
    for name, prefix in prefix_by_name.items():
        module = getattr(root, name)
        module_weights = weight_dict if prefix is None else slice_prefixed_state_dict(weight_dict, prefix)
        module.load(module_weights)


def get_motus_quant_flags(config):
    quantized = bool(config.get("motus_quantized", config.get("dit_quantized", False)))
    quant_scheme = config.get("motus_quant_scheme", config.get("dit_quant_scheme", "Default"))
    return quantized and quant_scheme != "Default", quant_scheme


def build_mm_weight(
    weight_name,
    bias_name,
    config,
    create_cuda_buffer=False,
    create_cpu_buffer=False,
    lazy_load=False,
    lazy_load_file=None,
):
    quantized, quant_scheme = get_motus_quant_flags(config)
    scheme = quant_scheme if quantized else "Default"
    module = MM_WEIGHT_REGISTER[scheme](
        weight_name,
        bias_name,
        create_cuda_buffer,
        create_cpu_buffer,
        lazy_load,
        lazy_load_file,
    )
    if hasattr(module, "set_config"):
        module_config = copy.deepcopy(config)
        if quantized:
            module_config["dit_quantized"] = True
            module_config["dit_quant_scheme"] = quant_scheme
            module_config.setdefault("weight_auto_quant", True)
        module.set_config(module_config)
    return module


def _resolve_mm_input_spec(weight_module, x):
    weight = getattr(weight_module, "weight", None)
    if torch.is_tensor(weight):
        target_device = weight.device
        if weight.is_floating_point():
            return target_device, weight.dtype
        return target_device, getattr(weight_module, "infer_dtype", x.dtype)

    pin_weight = getattr(weight_module, "pin_weight", None)
    if torch.is_tensor(pin_weight):
        target_device = pin_weight.device
        if pin_weight.is_floating_point():
            return target_device, pin_weight.dtype
        return target_device, getattr(weight_module, "infer_dtype", x.dtype)

    return x.device, getattr(weight_module, "infer_dtype", x.dtype)


def apply_mm(weight_module, x):
    target_device, target_dtype = _resolve_mm_input_spec(weight_module, x)
    if x.device != target_device or x.dtype != target_dtype:
        x = x.to(device=target_device, dtype=target_dtype)
    x_2d = x.reshape(-1, x.shape[-1])
    y_2d = weight_module.apply(x_2d)
    return y_2d.reshape(*x.shape[:-1], y_2d.shape[-1])


def projector_depth(projector_type):
    if projector_type == "linear":
        return 1
    match = re.match(r"^mlp(\d+)x_silu$", projector_type)
    if match is None:
        raise ValueError(f"Unknown projector type: {projector_type}")
    return int(match.group(1))


def projector_layer_prefixes(projector_type, base_prefix):
    if projector_type == "linear":
        return [base_prefix]
    return [f"{base_prefix}.{idx * 2}" for idx in range(projector_depth(projector_type))]


def apply_time_embedding(timestep, seq_len, freq_dim, hidden_dim, embedding_0, embedding_2, projection_1):
    if timestep.dim() == 1:
        timestep = timestep.unsqueeze(1).expand(timestep.size(0), seq_len)
    batch = timestep.size(0)
    timestep_embed = sinusoidal_embedding_1d(freq_dim, timestep.flatten()).unflatten(0, (batch, seq_len)).float()
    time_hidden = apply_mm(embedding_0, timestep_embed)
    time_hidden = torch.nn.functional.silu(time_hidden)
    time_hidden = apply_mm(embedding_2, time_hidden)
    adaln_hidden = torch.nn.functional.silu(time_hidden)
    adaln_hidden = apply_mm(projection_1, adaln_hidden).unflatten(2, (6, hidden_dim))
    return time_hidden, adaln_hidden


class SequentialLinearWeights(WeightModule):
    def __init__(
        self,
        layer_prefixes,
        activation,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
    ):
        super().__init__()
        self.activation = activation
        self.layers = WeightModuleList(
            [
                build_mm_weight(
                    f"{prefix}.weight",
                    f"{prefix}.bias",
                    config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=lazy_load,
                    lazy_load_file=lazy_load_file,
                )
                for prefix in layer_prefixes
            ]
        )
        self.add_module("layers", self.layers)

    def apply(self, x):
        out = x
        for layer_idx, layer in enumerate(self.layers):
            out = apply_mm(layer, out)
            if layer_idx == len(self.layers) - 1:
                continue
            if self.activation == "silu":
                out = torch.nn.functional.silu(out)
            elif self.activation == "gelu_tanh":
                out = torch.nn.functional.gelu(out, approximate="tanh")
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")
        return out


class PackedQKVWeights(WeightModule):
    def __init__(
        self,
        tensor_name,
        num_heads,
        in_features,
        head_dim,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
    ):
        super().__init__()
        self.tensor_name = tensor_name
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = head_dim
        self.out_features = self.num_heads * self.head_dim

        self.add_module(
            "q",
            build_mm_weight(
                f"{tensor_name}.q.weight",
                None,
                config,
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_file,
            ),
        )
        self.add_module(
            "k",
            build_mm_weight(
                f"{tensor_name}.k.weight",
                None,
                config,
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_file,
            ),
        )
        self.add_module(
            "v",
            build_mm_weight(
                f"{tensor_name}.v.weight",
                None,
                config,
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_file,
            ),
        )

    def load(self, weight_dict):
        if self.tensor_name not in weight_dict:
            return
        packed_qkv = weight_dict[self.tensor_name]
        q_weight = packed_qkv[0].permute(0, 2, 1).reshape(self.out_features, self.in_features).contiguous()
        k_weight = packed_qkv[1].permute(0, 2, 1).reshape(self.out_features, self.in_features).contiguous()
        v_weight = packed_qkv[2].permute(0, 2, 1).reshape(self.out_features, self.in_features).contiguous()
        self.q.load({f"{self.tensor_name}.q.weight": q_weight})
        self.k.load({f"{self.tensor_name}.k.weight": k_weight})
        self.v.load({f"{self.tensor_name}.v.weight": v_weight})

    def apply(self, x):
        q = apply_mm(self.q, x).reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        k = apply_mm(self.k, x).reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        v = apply_mm(self.v, x).reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        return q, k, v


class TensorAlias(WeightModule):
    def __init__(self, tensor_name):
        super().__init__()
        self.register_parameter("tensor", TENSOR_REGISTER["Default"](tensor_name))


class MotusJointExpertBlockWeights(WeightModule):
    def __init__(
        self,
        block_idx,
        config,
        expert_dim,
        wan_num_heads,
        wan_head_dim,
        attr_prefix,
        norm_eps,
        include_modulation=False,
    ):
        super().__init__()
        self.block_idx = block_idx

        self.add_module("norm1", LN_WEIGHT_REGISTER["torch"]())
        self.add_module("norm2", LN_WEIGHT_REGISTER["torch"]())
        self.add_module(
            f"{attr_prefix}_qkv",
            PackedQKVWeights(
                f"blocks.{block_idx}.{attr_prefix}_qkv",
                wan_num_heads,
                expert_dim,
                wan_head_dim,
                config,
            ),
        )
        self.add_module(
            f"{attr_prefix}_o",
            build_mm_weight(
                f"blocks.{block_idx}.{attr_prefix}_o.weight",
                None,
                config,
            ),
        )
        self.add_module(
            f"{attr_prefix}_norm_q",
            RMS_WEIGHT_REGISTER["torch"](f"blocks.{block_idx}.{attr_prefix}_norm_q.weight", eps=norm_eps),
        )
        self.add_module(
            f"{attr_prefix}_norm_k",
            RMS_WEIGHT_REGISTER["torch"](f"blocks.{block_idx}.{attr_prefix}_norm_k.weight", eps=norm_eps),
        )
        self.add_module("ffn_0", build_mm_weight(f"blocks.{block_idx}.ffn.0.weight", f"blocks.{block_idx}.ffn.0.bias", config))
        self.add_module("ffn_2", build_mm_weight(f"blocks.{block_idx}.ffn.2.weight", f"blocks.{block_idx}.ffn.2.bias", config))
        if include_modulation:
            self.register_parameter("modulation", TENSOR_REGISTER["Default"](f"blocks.{block_idx}.modulation"))


class MotusJointExpertTransformerWeights(WeightModule):
    def __init__(self, num_layers, block_factory):
        super().__init__()
        self.blocks = WeightModuleList([block_factory(block_idx) for block_idx in range(num_layers)])
        self.add_module("blocks", self.blocks)
