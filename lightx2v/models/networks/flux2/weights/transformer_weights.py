import torch
import torch.distributed as dist

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.offload.block_slab import pack_cpu_block_slab
from lightx2v.common.ops.utils import move_transposed_weight_module_to_device
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, ROPE_REGISTER


def _resolve_resident_block_indices(value, num_blocks, policy, config_key):
    """Resolve a resident-block count into deterministic block indices.

    Resident blocks are opt-in.  A count of zero therefore preserves the
    original full block-streaming behaviour.  ``interleaved`` spreads the
    resident blocks over the whole transformer instead of concentrating them
    at the front, which gives the offload stream regular compute windows in
    which to prefetch the next non-resident block.
    """
    if value is None:
        value = 0
    if isinstance(value, str):
        if value.lower() != "all":
            raise ValueError(f"{config_key} must be an integer or 'all', got {value!r}")
        count = num_blocks
    elif isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{config_key} must be an integer or 'all', got {value!r}")
    else:
        count = value

    if not 0 <= count <= num_blocks:
        raise ValueError(f"{config_key} must be between 0 and {num_blocks}, got {count}")
    if count == 0:
        return frozenset()
    if count == num_blocks:
        return frozenset(range(num_blocks))

    if policy == "prefix":
        return frozenset(range(count))
    if policy == "interleaved":
        # floor(k * N / K) is duplicate-free for K <= N.  For example,
        # K=36 and N=48 leaves blocks 3, 7, ..., 47 for streaming.
        return frozenset((idx * num_blocks) // count for idx in range(count))
    raise ValueError(f"offload_resident_policy must be 'prefix' or 'interleaved', got {policy!r}")


def preserve_weight_module_cpu_tensors(module):
    """Keep CPU masters for attributes that do not have a pinned counterpart."""
    for child in getattr(module, "_modules", {}).values():
        if child is not None:
            preserve_weight_module_cpu_tensors(child)

    cpu_masters = getattr(module, "_offload_cpu_master_tensors", {})
    for lora_attr in getattr(module, "lora_attrs", {}):
        value = getattr(module, lora_attr, None)
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            cpu_masters[lora_attr] = value
    if cpu_masters:
        module._offload_cpu_master_tensors = cpu_masters


def release_weight_module_device_tensors(module):
    """Release immutable device weights while retaining their CPU masters.

    The generic ``to_cpu`` implementation copies device values back into the
    pinned CPU tensors.  Inference weights are immutable, so that D2H transfer
    is unnecessary.  Restore from the existing pinned master (by dropping the
    device reference) and only copy attributes which have no CPU master, such
    as dynamically registered LoRA tensors.
    """
    for child in getattr(module, "_modules", {}).values():
        if child is not None:
            release_weight_module_device_tensors(child)

    base_attrs = getattr(module, "base_attrs", ())
    for _, attr_name, _ in base_attrs:
        value = getattr(module, attr_name, None)
        pin_value = getattr(module, f"pin_{attr_name}", None)
        if pin_value is not None:
            # ``to_cuda`` can reconstruct this attribute from pin_value.
            setattr(module, attr_name, None)
        elif isinstance(value, torch.Tensor) and value.device.type != "cpu":
            # This path is required when weights were not originally loaded
            # through the CPU/pinned-memory offload path.
            setattr(module, attr_name, value.to("cpu"))

    cpu_masters = getattr(module, "_offload_cpu_master_tensors", {})
    for lora_attr in getattr(module, "lora_attrs", {}):
        value = getattr(module, lora_attr, None)
        if lora_attr in cpu_masters:
            setattr(module, lora_attr, cpu_masters[lora_attr])
        elif isinstance(value, torch.Tensor) and value.device.type != "cpu":
            setattr(module, lora_attr, value.to("cpu"))


def _tp_info(config):
    if not config.get("tensor_parallel", False):
        return None, 0, 1
    tp_group = config.get("device_mesh").get_group(mesh_dim="tensor_p")
    return tp_group, dist.get_rank(tp_group), dist.get_world_size(tp_group)


def _mm_weight(config, weight_name, bias_name=None, split_dim=None, create_cuda_buffer=False, create_cpu_buffer=False):
    mm_type = config.get("dit_quant_scheme", "Default")
    if config.get("tensor_parallel", False) and split_dim is not None:
        tp_group, tp_rank, tp_size = _tp_info(config)
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
        )
    return MM_WEIGHT_REGISTER[mm_type](
        weight_name,
        bias_name,
        create_cuda_buffer,
        create_cpu_buffer,
    )


def _rms_weight(config, weight_name, create_cuda_buffer=False, create_cpu_buffer=False):
    # Flux2 q/k RMSNorm weights are head_dim-sized, so TP over heads must replicate them.
    rms_norm_type = config.get("rms_norm_type", "torch")
    return RMS_WEIGHT_REGISTER[rms_norm_type](
        weight_name,
        create_cuda_buffer,
        create_cpu_buffer,
    )


class Flux2DoubleBlockWeights(WeightModule):
    """Weights for a single double-stream transformer block."""

    def __init__(self, config, block_idx, create_cuda_buffer=False, create_cpu_buffer=False):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.layer_norm_type = config.get("layer_norm_type", "torch")
        self.rms_norm_type = config.get("rms_norm_type", "torch")
        self.attn_type = config.get("attn_type", "flash_attn3")
        self.add_module(
            "rope",
            ROPE_REGISTER[config.get("rope_type", "flashinfer_rope")](layout="interleaved", compute_dtype=torch.float32),
        )

        p = f"transformer_blocks.{self.block_idx}"

        self.add_module("norm1", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=1e-5))
        self.add_module("norm1_context", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=1e-5))
        self.add_module("norm2", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=1e-5))
        self.add_module("norm2_context", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=1e-5))

        self.add_module("to_q", _mm_weight(config, f"{p}.attn.to_q.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("to_k", _mm_weight(config, f"{p}.attn.to_k.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("to_v", _mm_weight(config, f"{p}.attn.to_v.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_q", _rms_weight(config, f"{p}.attn.norm_q.weight", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_k", _rms_weight(config, f"{p}.attn.norm_k.weight", create_cuda_buffer, create_cpu_buffer))

        self.add_module("add_q_proj", _mm_weight(config, f"{p}.attn.add_q_proj.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("add_k_proj", _mm_weight(config, f"{p}.attn.add_k_proj.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("add_v_proj", _mm_weight(config, f"{p}.attn.add_v_proj.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_added_q", _rms_weight(config, f"{p}.attn.norm_added_q.weight", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_added_k", _rms_weight(config, f"{p}.attn.norm_added_k.weight", create_cuda_buffer, create_cpu_buffer))

        self.add_module("to_out", _mm_weight(config, f"{p}.attn.to_out.0.weight", None, "row", create_cuda_buffer, create_cpu_buffer))
        self.add_module("to_add_out", _mm_weight(config, f"{p}.attn.to_add_out.weight", None, "row", create_cuda_buffer, create_cpu_buffer))

        self.add_module("calculate", ATTN_WEIGHT_REGISTER[self.attn_type]())

        if self.config.get("seq_parallel", False):
            self.add_module(
                "calculate_parallel",
                ATTN_WEIGHT_REGISTER[self.config["parallel"].get("seq_p_attn_type", "ulysses")](),
            )

        self.add_module("ff_net_0", _mm_weight(config, f"{p}.ff.linear_in.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("ff_net_2", _mm_weight(config, f"{p}.ff.linear_out.weight", None, "row", create_cuda_buffer, create_cpu_buffer))
        self.add_module("ff_context_net_0", _mm_weight(config, f"{p}.ff_context.linear_in.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("ff_context_net_2", _mm_weight(config, f"{p}.ff_context.linear_out.weight", None, "row", create_cuda_buffer, create_cpu_buffer))

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                if self.mm_type == "Default":
                    # The fast path assumes Default's 2-D transpose views; quantized layouts need their own to_cuda().
                    move_transposed_weight_module_to_device(module, non_blocking=non_blocking)
                else:
                    module.to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)


class Flux2SingleBlockWeights(WeightModule):
    """Weights for a single single-stream transformer block."""

    def __init__(self, config, block_idx, create_cuda_buffer=False, create_cpu_buffer=False):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.layer_norm_type = config.get("layer_norm_type", "torch")
        self.rms_norm_type = config.get("rms_norm_type", "torch")
        self.attn_type = config.get("attn_type", "flash_attn3")
        self.add_module(
            "rope",
            ROPE_REGISTER[config.get("rope_type", "flashinfer_rope")](layout="interleaved", compute_dtype=torch.float32),
        )

        p = f"single_transformer_blocks.{self.block_idx}"

        self.add_module("norm", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=1e-5))

        self.add_module("to_qkv_mlp_proj", _mm_weight(config, f"{p}.attn.to_qkv_mlp_proj.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_q", _rms_weight(config, f"{p}.attn.norm_q.weight", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_k", _rms_weight(config, f"{p}.attn.norm_k.weight", create_cuda_buffer, create_cpu_buffer))
        self.add_module("to_out", _mm_weight(config, f"{p}.attn.to_out.weight", None, "row", create_cuda_buffer, create_cpu_buffer))

        self.add_module("calculate", ATTN_WEIGHT_REGISTER[self.attn_type]())

        if self.config.get("seq_parallel", False):
            self.add_module(
                "calculate_parallel",
                ATTN_WEIGHT_REGISTER[self.config["parallel"].get("seq_p_attn_type", "ulysses")](),
            )

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                if self.mm_type == "Default":
                    move_transposed_weight_module_to_device(module, non_blocking=non_blocking)
                else:
                    module.to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)


class Flux2TransformerWeights(WeightModule):
    """Complete transformer weights for Flux2 model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.get("num_layers", 10)
        self.num_single_layers = config.get("num_single_layers", 20)
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self._configure_resident_blocks(config)

        self.double_blocks = WeightModuleList([Flux2DoubleBlockWeights(config, i) for i in range(self.num_layers)])
        self.single_blocks = WeightModuleList([Flux2SingleBlockWeights(config, i) for i in range(self.num_single_layers)])
        self.register_offload_buffers(config)

        self.add_module("double_blocks", self.double_blocks)
        self.add_module("single_blocks", self.single_blocks)

        self.add_module("double_stream_modulation_img_linear", _mm_weight(config, "double_stream_modulation_img.linear.weight"))
        self.add_module("double_stream_modulation_txt_linear", _mm_weight(config, "double_stream_modulation_txt.linear.weight"))
        self.add_module("single_stream_modulation_linear", _mm_weight(config, "single_stream_modulation.linear.weight"))

    def _configure_resident_blocks(self, config):
        block_offload_enabled = config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "block"
        if not block_offload_enabled:
            double_setting = 0
            single_setting = 0
        else:
            double_setting = config.get("offload_resident_double_blocks", 0)
            single_setting = config.get("offload_resident_single_blocks", 0)

        resident_blocks_requested = double_setting not in (None, 0) or single_setting not in (None, 0)
        if resident_blocks_requested and config.get("dit_quantized", False):
            raise NotImplementedError("Flux2 resident block offload currently supports unquantized weights only")
        if resident_blocks_requested and config.get("lora_configs"):
            raise NotImplementedError("Flux2 resident block offload currently does not support LoRA weights")

        policy = config.get("offload_resident_policy", "prefix")
        self.resident_double_block_indices = _resolve_resident_block_indices(
            double_setting,
            self.num_layers,
            policy,
            "offload_resident_double_blocks",
        )
        self.resident_single_block_indices = _resolve_resident_block_indices(
            single_setting,
            self.num_single_layers,
            policy,
            "offload_resident_single_blocks",
        )

    def register_offload_buffers(self, config):
        if config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "block":
            if len(self.resident_double_block_indices) < self.num_layers:
                self.offload_double_block_cuda_buffers = WeightModuleList([Flux2DoubleBlockWeights(config, i, create_cuda_buffer=True) for i in range(2)])
                self.add_module("offload_double_block_cuda_buffers", self.offload_double_block_cuda_buffers)

            if len(self.resident_single_block_indices) < self.num_single_layers:
                self.offload_single_block_cuda_buffers = WeightModuleList([Flux2SingleBlockWeights(config, i, create_cuda_buffer=True) for i in range(2)])
                self.add_module("offload_single_block_cuda_buffers", self.offload_single_block_cuda_buffers)

    def is_double_block_resident(self, block_idx):
        return block_idx in self.resident_double_block_indices

    def is_single_block_resident(self, block_idx):
        return block_idx in self.resident_single_block_indices

    def get_resident_double_block(self, block_idx):
        if not self.is_double_block_resident(block_idx):
            return None
        return self.double_blocks[block_idx]

    def get_resident_single_block(self, block_idx):
        if not self.is_single_block_resident(block_idx):
            return None
        return self.single_blocks[block_idx]

    def resident_blocks_to_cuda(self, non_blocking=True):
        for block_idx in sorted(self.resident_double_block_indices):
            preserve_weight_module_cpu_tensors(self.double_blocks[block_idx])
            self.double_blocks[block_idx].to_cuda(non_blocking=non_blocking)
        for block_idx in sorted(self.resident_single_block_indices):
            preserve_weight_module_cpu_tensors(self.single_blocks[block_idx])
            self.single_blocks[block_idx].to_cuda(non_blocking=non_blocking)

    def release_resident_blocks(self):
        for block_idx in sorted(self.resident_double_block_indices):
            release_weight_module_device_tensors(self.double_blocks[block_idx])
        for block_idx in sorted(self.resident_single_block_indices):
            release_weight_module_device_tensors(self.single_blocks[block_idx])

    @staticmethod
    def _pack_offload_block_slabs(blocks, resident_indices):
        slabs = {}
        for block_idx, block in enumerate(blocks):
            if block_idx in resident_indices:
                continue

            full_state_dict = block.state_dict()
            required_names = set()
            visited = set()

            def collect_cpu_base_tensors(module):
                if module is None or id(module) in visited:
                    return
                visited.add(id(module))

                explicit_base_attrs = {attr_name for _, attr_name, _ in getattr(module, "base_attrs", ())}
                source_tensors = {}
                for attr_name in explicit_base_attrs:
                    pin_tensor = getattr(module, f"pin_{attr_name}", None)
                    base_tensor = getattr(module, attr_name, None)
                    tensor = pin_tensor if pin_tensor is not None else base_tensor
                    if tensor is None:
                        continue
                    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                        raise ValueError(f"Flux2 block slabs require every populated base attribute to have a CPU source; block {block_idx} attribute {attr_name!r} does not")
                    source_tensors[attr_name] = tensor

                # Platform weight templates do not all expose ``base_attrs``.
                # Their pinned masters are nevertheless the authoritative CPU
                # sources and are also what their state_dict implementations
                # return.
                for attr_name, tensor in vars(module).items():
                    if attr_name.startswith("pin_") and isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                        source_tensors.setdefault(attr_name.removeprefix("pin_"), tensor)

                for attr_name, tensor in source_tensors.items():
                    matching_names = [name for name, state_tensor in full_state_dict.items() if state_tensor is tensor]
                    if not matching_names:
                        raise ValueError(f"Flux2 block slab source state is missing CPU base attribute {attr_name!r} in block {block_idx}")
                    required_names.update(matching_names)

                for child in getattr(module, "_parameters", {}).values():
                    collect_cpu_base_tensors(child)
                for child in getattr(module, "_modules", {}).values():
                    collect_cpu_base_tensors(child)

            collect_cpu_base_tensors(block)
            state_dict = {name: tensor for name, tensor in full_state_dict.items() if name in required_names}
            unsupported = [name for name, tensor in state_dict.items() if tensor.dtype != torch.bfloat16]
            if unsupported or not state_dict:
                raise ValueError(f"Flux2 block slabs currently require CPU BF16 tensors; block {block_idx} has unsupported entries {unsupported or ['<no base tensors>']}")
            slabs[block_idx] = pack_cpu_block_slab(
                state_dict,
                pin_memory=True,
                strict_pin=True,
            )
        return slabs

    def prepare_offload_block_slabs(self):
        """Pack non-resident block weights after checkpoint loading."""
        if not self.config.get("offload_use_block_slab", False):
            return {}, {}
        if hasattr(self, "offload_double_block_slabs"):
            return self.offload_double_block_slabs, self.offload_single_block_slabs

        double_slabs = self._pack_offload_block_slabs(
            self.double_blocks,
            self.resident_double_block_indices,
        )
        single_slabs = self._pack_offload_block_slabs(
            self.single_blocks,
            self.resident_single_block_indices,
        )
        self.offload_double_block_slabs = double_slabs
        self.offload_single_block_slabs = single_slabs
        return double_slabs, single_slabs

    def non_block_weights_to_cuda(self, non_blocking=True):
        modules = (
            self.double_stream_modulation_img_linear,
            self.double_stream_modulation_txt_linear,
            self.single_stream_modulation_linear,
        )
        for module in modules:
            preserve_weight_module_cpu_tensors(module)
            if self.mm_type == "Default":
                move_transposed_weight_module_to_device(module, non_blocking=non_blocking)
            else:
                module.to_cuda(non_blocking=non_blocking)

    def non_block_weights_to_cpu(self, non_blocking=True):
        self.double_stream_modulation_img_linear.to_cpu(non_blocking=non_blocking)
        self.double_stream_modulation_txt_linear.to_cpu(non_blocking=non_blocking)
        self.single_stream_modulation_linear.to_cpu(non_blocking=non_blocking)

    def release_non_block_weights(self):
        release_weight_module_device_tensors(self.double_stream_modulation_img_linear)
        release_weight_module_device_tensors(self.double_stream_modulation_txt_linear)
        release_weight_module_device_tensors(self.single_stream_modulation_linear)

    def to_cuda(self, non_blocking=True):
        for block in self.double_blocks:
            block.to_cuda(non_blocking=non_blocking)
        for block in self.single_blocks:
            block.to_cuda(non_blocking=non_blocking)
        self.non_block_weights_to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=True):
        for block in self.double_blocks:
            block.to_cpu(non_blocking=non_blocking)
        for block in self.single_blocks:
            block.to_cpu(non_blocking=non_blocking)
        self.non_block_weights_to_cpu(non_blocking=non_blocking)


# Backward-compatible aliases
Flux2KleinDoubleBlockWeights = Flux2DoubleBlockWeights
Flux2KleinSingleBlockWeights = Flux2SingleBlockWeights
Flux2KleinTransformerWeights = Flux2TransformerWeights
