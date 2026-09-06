import gc
from contextlib import suppress

import torch
import torch.distributed as dist

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.minimax_h3.checkpoint import MiniMaxH3ShardCheckpoint
from lightx2v.models.networks.minimax_h3.infer.triton_ops import MiniMaxH3TritonRope  # noqa: F401
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, ROPE_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


def _resolve_streaming_block_name(name, block_index):
    block_prefix = "transformer_blocks."
    if not name.startswith(block_prefix):
        return name
    parts = name.split(".", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return f"{block_prefix}{int(block_index)}.{parts[2]}"
    return name


def _iter_base_attrs(module):
    if hasattr(module, "base_attrs"):
        yield from module.base_attrs
    for child in getattr(module, "_modules", {}).values():
        if child is not None:
            yield from _iter_base_attrs(child)


def _empty_device_cache():
    if not isinstance(AI_DEVICE, str):
        return
    device_module = getattr(torch, AI_DEVICE, None)
    if device_module is not None and hasattr(device_module, "empty_cache"):
        device_module.empty_cache()


def _linear(config, name, bias=False, create_cuda_buffer=False, tp_split=None):
    lora_prefix = "transformer_blocks"
    if config.get("tensor_parallel", False) and tp_split is not None:
        tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
        tp_mm_type = config.get("tp_mm_type", "TensorParallel")
        return MM_WEIGHT_REGISTER[tp_mm_type](
            weight_name=f"{name}.weight",
            bias_name=f"{name}.bias" if bias else None,
            mm_type=config.get("dit_quant_scheme", "Default"),
            tp_group=tp_group,
            tp_rank=dist.get_rank(tp_group),
            tp_size=dist.get_world_size(tp_group),
            split_dim=tp_split,
            lora_column_chunks=2 if ".ff.net.0.proj" in name else 1,
            create_cuda_buffer=create_cuda_buffer,
            lora_prefix=lora_prefix,
        )
    return MM_WEIGHT_REGISTER[config.get("dit_quant_scheme", "Default")](
        f"{name}.weight",
        f"{name}.bias" if bias else None,
        create_cuda_buffer=create_cuda_buffer,
        lora_prefix=lora_prefix,
    )


def _rms(config, name, eps, create_cuda_buffer=False):
    return RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")](
        name,
        create_cuda_buffer=create_cuda_buffer,
        eps=eps,
    )


class MiniMaxH3AttentionWeights(WeightModule):
    def __init__(self, prefix, config, create_cuda_buffer=False):
        super().__init__()
        self.add_module("to_q", _linear(config, f"{prefix}.to_q", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        self.add_module("to_k", _linear(config, f"{prefix}.to_k", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        self.add_module("to_v", _linear(config, f"{prefix}.to_v", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        qk_eps = float(config.get("qk_norm_eps", 1e-5))
        self.add_module(
            "norm_q",
            _rms(
                config,
                f"{prefix}.norm_q.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=qk_eps,
            ),
        )
        self.add_module(
            "norm_k",
            _rms(
                config,
                f"{prefix}.norm_k.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=qk_eps,
            ),
        )
        self.add_module(
            "rope",
            ROPE_REGISTER[config.get("rope_type", "torch_real_rope")](
                layout="split_half",
                # H3 requires BF16 Q/K; the reference rounds frequencies and
                # performs the rotation at that dtype, not in FP32.
                compute_dtype=torch.bfloat16,
            ),
        )
        attn_type = config.get("attn_type", "flash_attn3")
        attention_cls = ATTN_WEIGHT_REGISTER[attn_type]
        if attn_type == "dynamic_sparse_attn":
            calculate = attention_cls(config.get("dynamic_sparse_attn_setting", {}))
        else:
            calculate = attention_cls()
        if attn_type == "sol_attn":
            calculate.set_config(config.get("sol_attn_setting", {}))
        self.add_module("calculate", calculate)
        if config.get("seq_parallel", False):
            parallel = config.get("parallel", {})
            self.add_module(
                "calculate_parallel",
                ATTN_WEIGHT_REGISTER[parallel.get("seq_p_attn_type", "ulysses")](a2a_backend=parallel.get("seq_p_a2a_backend", "torch")),
            )
        self.add_module("to_out", _linear(config, f"{prefix}.to_out.0", create_cuda_buffer=create_cuda_buffer, tp_split="row"))


class MiniMaxH3FeedForwardWeights(WeightModule):
    def __init__(self, prefix, config, create_cuda_buffer=False):
        super().__init__()
        self.add_module("in_proj", _linear(config, f"{prefix}.net.0.proj", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        self.add_module("out_proj", _linear(config, f"{prefix}.net.2", create_cuda_buffer=create_cuda_buffer, tp_split="row"))


class MiniMaxH3TransformerBlockWeights(WeightModule):
    def __init__(self, index, config, create_cuda_buffer=False):
        super().__init__()
        prefix = f"transformer_blocks.{index}"
        eps = float(config.get("norm_eps", 1e-5))
        self.add_module(
            "norm1",
            _rms(
                config,
                f"{prefix}.norm1.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=eps,
            ),
        )
        self.add_module("attn", MiniMaxH3AttentionWeights(f"{prefix}.attn", config, create_cuda_buffer))
        self.add_module(
            "norm2",
            _rms(
                config,
                f"{prefix}.norm2.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=eps,
            ),
        )
        self.add_module("ff", MiniMaxH3FeedForwardWeights(f"{prefix}.ff", config, create_cuda_buffer))
        # AdaLN is the largest per-block projection in H3.  Its output is
        # column-sharded here and gathered once per block before modulation.
        self.add_module("adaln", _linear(config, f"{prefix}.adaln_proj.linear", bias=True, create_cuda_buffer=create_cuda_buffer, tp_split="col"))


class MiniMaxH3TransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.config = config
        self.num_layers = int(config.get("num_layers", 50))
        self.disk_streaming = bool(config.get("dit_disk_streaming", False))
        self.streaming_lora = None
        if self.disk_streaming:
            if config.get("lazy_load", False):
                raise NotImplementedError(
                    "MiniMax-H3 dit_disk_streaming reads the official sharded checkpoint directly and cannot be combined with converted lazy_load block shards."
                )
            if config.get("dit_quantized", False):
                raise NotImplementedError("MiniMax-H3 dit_disk_streaming does not support quantized DiT checkpoints yet.")
            if config.get("tensor_parallel", False):
                raise NotImplementedError("MiniMax-H3 dit_disk_streaming does not support tensor parallel inference yet.")
            if lora_path is not None:
                raise ValueError("Initialize MiniMax-H3 streamed LoRA through MiniMaxH3Model, not the weights constructor.")

            checkpoint_dir = config.get("dit_original_ckpt")
            if checkpoint_dir is None:
                raise ValueError("MiniMax-H3 dit_disk_streaming requires config['dit_original_ckpt'] to point to the official transformer checkpoint directory.")
            self.checkpoint = MiniMaxH3ShardCheckpoint(checkpoint_dir, config=config)
            if self.checkpoint.selected_reader is not None:
                # Raw config aliases (e.g. token_refiner_num_layers) must reach
                # the native pre/post constructors as well as the mapping plan.
                for name, value in self.checkpoint.config.items():
                    config.setdefault(name, value)
                self.num_layers = int(config["num_layers"])
            expected_block_indices = tuple(range(self.num_layers))
            if self.checkpoint.block_indices != expected_block_indices:
                raise ValueError(
                    "MiniMax-H3 dit_disk_streaming checkpoint block indices mismatch: "
                    f"expected {expected_block_indices}, found {self.checkpoint.block_indices}"
                )

            self.blocks = WeightModuleList([])
            self.streaming_block = None
            self.add_module("blocks", self.blocks)
            self._ensure_streaming_block()
            return

        if config.get("lazy_load", False):
            raise NotImplementedError(
                "MiniMax-H3 reads the official sharded checkpoint directly; disk lazy_load requires a converted block-sharded checkpoint and is not supported yet. Use lazy_load=false with model or block CPU offload."
            )
        self.blocks = WeightModuleList([MiniMaxH3TransformerBlockWeights(i, config) for i in range(self.num_layers)])
        if config.get("cpu_offload", False) and config.get("offload_granularity", "model") == "block":
            self.offload_block_cuda_buffers = WeightModuleList([MiniMaxH3TransformerBlockWeights(i, config, create_cuda_buffer=True) for i in range(2)])
            # Register device buffers before source blocks: buffer allocation
            # needs checkpoint metadata that normal CPU loading consumes.
            self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
            self.offload_phase_cuda_buffers = None
        self.add_module("blocks", self.blocks)

    @property
    def streaming_block_indices(self):
        if not self.disk_streaming:
            raise RuntimeError("MiniMax-H3 streaming_block_indices is only available when dit_disk_streaming=true.")
        return self.checkpoint.block_indices

    def load_streaming_block(self, block_index):
        if not self.disk_streaming:
            raise RuntimeError("MiniMax-H3 load_streaming_block requires dit_disk_streaming=true.")
        block_index = int(block_index)
        if block_index not in self.checkpoint.block_indices:
            raise IndexError(f"MiniMax-H3 checkpoint does not contain transformer block {block_index}.")

        self._ensure_streaming_block()
        if self.streaming_lora is not None:
            # Finish the previous use before either base weights or factors change.
            self.streaming_lora.clear(self.streaming_block)
        if self.checkpoint.selected_reader is not None:
            self.checkpoint.selected_reader.load_modules([self.streaming_block], device=AI_DEVICE, block_index=block_index, reusable=True)
            if self.streaming_lora is not None:
                self.streaming_lora.load_block(self.streaming_block, block_index)
            return self.streaming_block
        tensor_names = self.checkpoint.tensor_names_for_block(block_index)
        tensors = self.checkpoint.load_tensors(tensor_names, device="cpu")
        try:
            self.streaming_block.load_state_dict(self._prepare_streaming_state_dict(tensors, block_index), block_index)
        finally:
            del tensors
        if self.streaming_lora is not None:
            self.streaming_lora.load_block(self.streaming_block, block_index)
        return self.streaming_block

    def _ensure_streaming_block(self):
        if self.streaming_block is not None:
            return
        self.streaming_block = MiniMaxH3TransformerBlockWeights(0, self.config, create_cuda_buffer=True)
        self.add_module("streaming_block", self.streaming_block)
        if self.checkpoint.selected_reader is not None:
            self.checkpoint.selected_reader.load_modules([self.streaming_block], device=AI_DEVICE, block_index=0, reusable=True)
            return
        block0_tensors = self.checkpoint.load_tensors(self.checkpoint.tensor_names_for_block(0), device="cpu")
        try:
            self.streaming_block.load(block0_tensors)
            self.streaming_block.load_state_dict(self._prepare_streaming_state_dict(block0_tensors, 0), 0)
        finally:
            del block0_tensors
            gc.collect()
            _empty_device_cache()

    def release_disk_streaming_buffer(self):
        if not self.disk_streaming:
            return
        block = self.streaming_block
        if block is None:
            return
        if self.streaming_lora is not None:
            self.streaming_lora.clear(block)
        with suppress(Exception):
            device_module = getattr(torch, AI_DEVICE, None)
            if device_module is not None and hasattr(device_module, "synchronize"):
                device_module.synchronize()

        stack = [block]
        visited = set()
        while stack:
            module = stack.pop()
            if module is None or id(module) in visited:
                continue
            visited.add(id(module))
            for _, attr_name, _ in getattr(module, "base_attrs", ()):
                if hasattr(module, attr_name):
                    setattr(module, attr_name, None)
                buffer_attr = f"{attr_name}_cuda_buffer"
                if hasattr(module, buffer_attr):
                    setattr(module, buffer_attr, None)
            for attr_name in tuple(vars(module)):
                if attr_name.endswith("_cuda_buffer"):
                    setattr(module, attr_name, None)
            stack.extend(getattr(module, "_modules", {}).values())
            stack.extend(getattr(module, "_parameters", {}).values())

        self.streaming_block = None
        self._modules["streaming_block"] = None
        gc.collect()
        _empty_device_cache()

    def _prepare_streaming_state_dict(self, tensors, block_index):
        state_dict = dict(tensors)
        for name, _, transpose in _iter_base_attrs(self.streaming_block):
            if transpose:
                actual_name = _resolve_streaming_block_name(name, block_index)
                if actual_name in state_dict:
                    state_dict[actual_name] = state_dict[actual_name].t()
        return state_dict
