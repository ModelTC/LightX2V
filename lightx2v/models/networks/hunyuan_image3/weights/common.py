import math

import torch
import torch.nn.functional as F

import lightx2v.common.ops  # noqa: F401
from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.hunyuan_image3.weights.hybrid_tp import (
    FUSED_GATE_UP_LAYOUT,
    GROUPED_QKV_LAYOUT,
    PLAIN_LAYOUT,
    HunyuanImage3HybridTensorParallelLinear,
    resolve_micro_shard_count,
    resolve_storage_tp,
)
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


def _patch_size(config):
    patch_size = config.get("patch_size")
    if not patch_size or isinstance(patch_size, (list, tuple)):
        return 1
    return int(patch_size)


def _patch_embed_hidden_dim(config):
    hidden_dim = config.get("patch_embed_hidden_dim")
    return 1024 if hidden_dim is None else int(hidden_dim)


def hunyuan_image3_mm_weight(
    config,
    mm_type,
    weight_name,
    bias_name=None,
    split_dim=None,
    weight_layout=PLAIN_LAYOUT,
    reduce_output=True,
    create_cuda_buffer=False,
    create_cpu_buffer=False,
    lazy_load=False,
    lazy_load_file=None,
    lora_prefix="diffusion_model.blocks",
    lora_path=None,
):
    if config.get("tensor_parallel", False) and split_dim is not None:
        parallel_context, tp_group, tp_rank, tp_size = resolve_storage_tp(config)
        if parallel_context is not None:
            micro_shard_count = resolve_micro_shard_count(config, tp_size)
            qkv_group_width = None
            if weight_layout == GROUPED_QKV_LAYOUT:
                heads = int(config.get("num_attention_heads") or config["num_heads"])
                kv_heads = int(config.get("num_key_value_heads") or heads)
                head_dim = int(config.get("attention_head_dim", config["hidden_size"] // heads))
                if heads % kv_heads:
                    raise ValueError(f"HunyuanImage3 Q heads ({heads}) must be divisible by KV heads ({kv_heads}).")
                qkv_group_width = (heads // kv_heads + 2) * head_dim
            return HunyuanImage3HybridTensorParallelLinear(
                weight_name=weight_name,
                bias_name=bias_name,
                mm_type=mm_type,
                tp_group=tp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                split_dim=split_dim,
                parallel_context=parallel_context,
                micro_shard_count=micro_shard_count,
                weight_layout=weight_layout,
                qkv_group_width=qkv_group_width,
                hidden_act=config.get("hidden_act", "silu"),
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
                reduce_output=reduce_output,
            )
        return MM_WEIGHT_REGISTER["TensorParallel"](
            weight_name=weight_name,
            bias_name=bias_name,
            mm_type=mm_type,
            tp_group=tp_group,
            tp_rank=tp_rank,
            tp_size=tp_size,
            split_dim=split_dim,
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=lazy_load,
            lazy_load_file=lazy_load_file,
            lora_prefix=lora_prefix,
            lora_path=lora_path,
            reduce_output=reduce_output,
        )
    return MM_WEIGHT_REGISTER[mm_type](
        weight_name,
        bias_name,
        create_cuda_buffer,
        create_cpu_buffer,
        lazy_load,
        lazy_load_file,
        lora_prefix=lora_prefix,
        lora_path=lora_path,
    )


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
        patch_size = _patch_size(config)
        hidden_channels = _patch_embed_hidden_dim(config)
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
        patch_size = _patch_size(config)
        hidden_channels = _patch_embed_hidden_dim(config)
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
        config,
        mm_type,
        mlp_bias=False,
        defer_down_reduce=False,
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
            hunyuan_image3_mm_weight(
                config,
                mm_type,
                f"{prefix}.gate_and_up_proj.weight",
                gate_and_up_bias,
                split_dim="col",
                weight_layout=FUSED_GATE_UP_LAYOUT,
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_file,
                lora_prefix=prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "down_proj",
            hunyuan_image3_mm_weight(
                config,
                mm_type,
                f"{prefix}.down_proj.weight",
                down_bias,
                split_dim="row",
                reduce_output=not defer_down_reduce,
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_file,
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
        self.moe_impl = config.get("moe_impl", "eager")
        self.parallel_context, _, self.storage_tp_rank, self.storage_tp_size = resolve_storage_tp(config)
        self.micro_shard_count = resolve_micro_shard_count(config, self.storage_tp_size) if self.parallel_context is not None else 1
        self.flashinfer_logical_tp_size = self.storage_tp_size * self.micro_shard_count
        self.moe_weight = None
        self.moe_weight_2 = None
        self._flashinfer_weights_initialized = False
        self._flashinfer_weight_device = None
        self._flashinfer_weight_dtype = None
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
                    config,
                    mm_type,
                    config.get("mlp_bias", False),
                    True,
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
                    config,
                    mm_type,
                    config.get("mlp_bias", False),
                    True,
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

    @staticmethod
    def _unwrap_linear_weight(linear):
        """Return the concrete MM weight hidden behind TP wrappers."""
        current = linear
        visited = set()
        while getattr(current, "_mm", None) is not None:
            if id(current) in visited:
                raise RuntimeError("Detected a cycle while unwrapping a HunyuanImage3 MM weight.")
            visited.add(id(current))
            current = current._mm
        return current

    def _linear_weight_for_flashinfer(self, linear, device, dtype):
        if getattr(linear, "has_lora_branch", False):
            raise RuntimeError("HunyuanImage3 moe_impl='flashinfer' does not support dynamic LoRA branches.")
        if hasattr(linear, "weight_diff"):
            raise RuntimeError("HunyuanImage3 moe_impl='flashinfer' does not support runtime diff weights.")

        linear = self._unwrap_linear_weight(linear)
        if getattr(linear, "has_lora_branch", False):
            raise RuntimeError("HunyuanImage3 moe_impl='flashinfer' does not support dynamic LoRA branches.")
        if hasattr(linear, "weight_diff"):
            raise RuntimeError("HunyuanImage3 moe_impl='flashinfer' does not support runtime diff weights.")

        weight = getattr(linear, "weight", None)
        if weight is None:
            weight = getattr(linear, "pin_weight", None)
        if weight is None or weight.numel() == 0:
            raise RuntimeError("HunyuanImage3 FlashInfer MoE weights were already released and cannot be rebuilt.")

        bias = getattr(linear, "bias", None)
        pin_bias = getattr(linear, "pin_bias", None)
        if (bias is not None and bias.numel() > 0) or (pin_bias is not None and pin_bias.numel() > 0):
            raise RuntimeError("HunyuanImage3 moe_impl='flashinfer' expects bias-free expert MLP weights.")

        # LightX2V MM weights are stored as [in, out] for torch.mm(input, weight).
        # FlashInfer follows torch.nn.Linear/checkpoint layout [out, in].  For
        # the default loader the transpose is already a contiguous view of the
        # checkpoint tensor, so this does not allocate unless device/dtype
        # conversion is genuinely required during one-time initialization.
        return weight.t().to(device=device, dtype=dtype).contiguous()

    @classmethod
    def _release_linear_weight(cls, linear, device, dtype):
        linear = cls._unwrap_linear_weight(linear)
        empty = torch.empty(0, device=device, dtype=dtype)
        for attr in ("weight", "pin_weight", "bias", "pin_bias"):
            if hasattr(linear, attr):
                setattr(linear, attr, empty)

    def ensure_flashinfer_weights(self, device, dtype):
        """Build the single phase-neutral ``[micro, expert, ...]`` pack.

        AR selects one leading-axis view.  Denoising selects both views and
        sums their partial outputs before its storage-TP all-reduce.  The pack
        is never rebuilt or converted when the active phase changes.
        """

        device = torch.device(device)
        if dtype not in (torch.float16, torch.bfloat16):
            dtype = torch.bfloat16

        if self._flashinfer_weights_initialized:
            if self.moe_weight.device != device or self.moe_weight.dtype != dtype:
                raise RuntimeError(
                    "HunyuanImage3 phase-neutral FlashInfer weights cannot be converted after initialization: "
                    f"resident=({self.moe_weight.device}, {self.moe_weight.dtype}), requested=({device}, {dtype})."
                )
            if self.parallel_context is None:
                return self.moe_weight[0], self.moe_weight_2[0]
            return self.moe_weight, self.moe_weight_2

        first_gate_up = self._linear_weight_for_flashinfer(self.experts[0].gate_and_up_proj, device, dtype)
        first_down = self._linear_weight_for_flashinfer(self.experts[0].down_proj, device, dtype)
        if first_gate_up.ndim != 2 or first_gate_up.shape[0] % (2 * self.micro_shard_count):
            raise RuntimeError(f"Invalid HunyuanImage3 FlashInfer gate/up resident shape {tuple(first_gate_up.shape)} for {self.micro_shard_count} micro shards.")
        micro_intermediate = first_gate_up.shape[0] // (2 * self.micro_shard_count)
        hidden_size = first_gate_up.shape[1]
        if first_down.shape != (hidden_size, self.micro_shard_count * micro_intermediate):
            raise RuntimeError(f"HunyuanImage3 FlashInfer down weight does not match gate/up micro shards: gate_up={tuple(first_gate_up.shape)}, down={tuple(first_down.shape)}.")

        # Allocate the final resident pack once and copy experts into it
        # directly. This avoids retaining a second list of expert tensors;
        # sources are released together after the full pack is validated.
        self.moe_weight = torch.empty(
            self.micro_shard_count,
            self.num_experts,
            2 * micro_intermediate,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        self.moe_weight_2 = torch.empty(
            self.micro_shard_count,
            self.num_experts,
            hidden_size,
            micro_intermediate,
            device=device,
            dtype=dtype,
        )

        for expert_index, expert in enumerate(self.experts):
            if expert_index == 0:
                gate_up = first_gate_up
                down = first_down
            else:
                gate_up = self._linear_weight_for_flashinfer(expert.gate_and_up_proj, device, dtype)
                down = self._linear_weight_for_flashinfer(expert.down_proj, device, dtype)
            if gate_up.shape != first_gate_up.shape or down.shape != first_down.shape:
                raise RuntimeError(f"HunyuanImage3 expert {expert_index} shape differs while building the universal FlashInfer pack: gate_up={tuple(gate_up.shape)}, down={tuple(down.shape)}.")

            # gate_up was organized by the checkpoint loader as
            # [g_micro0,u_micro0,g_micro1,u_micro1,...].  Down remains
            # [hidden, intermediate_micro0, intermediate_micro1,...].
            gate_up_micro = gate_up.reshape(self.micro_shard_count, 2 * micro_intermediate, hidden_size)
            down_micro = down.reshape(hidden_size, self.micro_shard_count, micro_intermediate).permute(1, 0, 2)
            self.moe_weight[:, expert_index].copy_(gate_up_micro)
            self.moe_weight_2[:, expert_index].copy_(down_micro)

        if not all(self.moe_weight[index].is_contiguous() and self.moe_weight_2[index].is_contiguous() for index in range(self.micro_shard_count)):
            raise RuntimeError("HunyuanImage3 FlashInfer micro-shard views must be contiguous.")

        # Release source tensors only after every expert has been validated and
        # copied. A malformed checkpoint therefore cannot leave a half-built,
        # non-retryable pack behind.
        for expert in self.experts:
            self._release_linear_weight(expert.gate_and_up_proj, self.moe_weight.device, self.moe_weight.dtype)
            self._release_linear_weight(expert.down_proj, self.moe_weight_2.device, self.moe_weight_2.dtype)

        self._flashinfer_weights_initialized = True
        self._flashinfer_weight_device = self.moe_weight.device
        self._flashinfer_weight_dtype = self.moe_weight.dtype
        if self.parallel_context is None:
            return self.moe_weight[0], self.moe_weight_2[0]
        return self.moe_weight, self.moe_weight_2

    def _active_flashinfer_micro_shard_ids(self):
        if self.parallel_context is None or self.micro_shard_count == 1:
            return (0,)
        active_tp_size = int(getattr(self.parallel_context, "active_tp_size", getattr(self.parallel_context, "tp_size", self.storage_tp_size)))
        if active_tp_size == self.storage_tp_size:
            return tuple(range(self.micro_shard_count))
        if active_tp_size == self.flashinfer_logical_tp_size:
            micro_id = int(getattr(self.parallel_context, "local_micro_shard_id"))
            if not 0 <= micro_id < self.micro_shard_count:
                raise RuntimeError(f"Invalid HunyuanImage3 local FlashInfer micro shard id {micro_id}.")
            return (micro_id,)
        raise RuntimeError(f"HunyuanImage3 FlashInfer active TP size must be {self.storage_tp_size} or {self.flashinfer_logical_tp_size}; got {active_tp_size}.")

    def canonical_flashinfer_tp_rank(self, micro_shard_id):
        micro_shard_id = int(micro_shard_id)
        if not 0 <= micro_shard_id < self.micro_shard_count:
            raise ValueError(f"Invalid HunyuanImage3 FlashInfer micro shard id {micro_shard_id}.")
        return self.storage_tp_rank * self.micro_shard_count + micro_shard_id

    def active_flashinfer_weight_shards(self, device, dtype):
        """Return active zero-copy FI views as ``(micro, tp_rank, w1, w2)``."""

        self.ensure_flashinfer_weights(device, dtype)
        return tuple(
            (
                micro_id,
                self.canonical_flashinfer_tp_rank(micro_id),
                self.moe_weight[micro_id],
                self.moe_weight_2[micro_id],
            )
            for micro_id in self._active_flashinfer_micro_shard_ids()
        )

    def active_flashinfer_multi_micro_weights(self, device, dtype):
        """Return the resident multi-micro packs without copying or reordering.

        The multi-micro MoE path is valid only when every resident micro shard
        is active (the denoising TP2 phase in the phase-aware topology).  AR
        continues to select one zero-copy leading-axis view through
        :meth:`active_flashinfer_weight_shards` and uses the official
        FlashInfer operator.
        """

        if self.micro_shard_count != 2:
            raise RuntimeError(f"HunyuanImage3 multi-micro MoE requires exactly two resident micro shards; got {self.micro_shard_count}.")
        self.ensure_flashinfer_weights(device, dtype)
        active_micro_ids = self._active_flashinfer_micro_shard_ids()
        expected_micro_ids = tuple(range(self.micro_shard_count))
        if active_micro_ids != expected_micro_ids:
            raise RuntimeError(f"HunyuanImage3 multi-micro FlashInfer weights require all resident micro shards to be active: active={active_micro_ids}, resident={expected_micro_ids}.")
        return self.moe_weight, self.moe_weight_2


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
        self.heads = int(config.get("num_attention_heads") or config["num_heads"])
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
            hunyuan_image3_mm_weight(
                config,
                mm_type,
                f"{prefix}.self_attn.qkv_proj.weight",
                attn_bias and f"{prefix}.self_attn.qkv_proj{attn_bias}",
                split_dim="col",
                weight_layout=GROUPED_QKV_LAYOUT,
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_file,
                lora_prefix=prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "o_proj",
            hunyuan_image3_mm_weight(
                config,
                mm_type,
                f"{prefix}.self_attn.o_proj.weight",
                attn_bias and f"{prefix}.self_attn.o_proj{attn_bias}",
                split_dim="row",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_file,
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
                    config,
                    mm_type,
                    config.get("mlp_bias", False),
                    False,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_file,
                    lora_path,
                ),
            )
            self.gate_and_up_proj = self.dense_mlp.gate_and_up_proj
            self.down_proj = self.dense_mlp.down_proj
