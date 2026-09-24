"""Qwen-Image-2.1 block weights backed by a process-shared pinned CPU arena."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.common.offload.checkpoint_metadata import checkpoint_content_digest, read_checkpoint_json, read_safetensors_header
from lightx2v.common.offload.shared_pinned_arena import SharedWeightManifest
from lightx2v.common.offload.shared_weight_coordinator import materialize_shared_weight_arena, validate_shared_weight_config
from lightx2v.common.offload.shared_weight_map import SharedWeightViewMap
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "F8_E4M3FN": torch.float8_e4m3fn}
_PRIVATE_KEYS = {
    "img_in.weight",
    "txt_in.in_layer.weight",
    "txt_in.out_layer.weight",
    "txt_in.text_norm.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "modulation.1.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
}


class QwenImage21SharedBlockAdapter:
    def __init__(self, model, unified_dtype, sensitive_layer):
        self.model = model
        self.config = model.config
        self.unified_dtype = unified_dtype
        self.sensitive_layer = sensitive_layer
        self.quantized = self.config.get("dit_quantized", False)
        self._validate_config()
        if self.quantized:
            path = self.config["dit_quantized_ckpt"]
        else:
            path = self.config.get("dit_original_ckpt") or self.config.get("transformer_model_path") or model.model_path
        self.entries, metadata, signatures = self._inspect_checkpoint(Path(path).resolve())
        signature = hashlib.sha256(
            json.dumps(
                {
                    "format": "qwen-image-21-shared-block-v1",
                    "files": signatures,
                    "tensors": [(name, list(t.shape), str(t.dtype)) for name, t in sorted(metadata.items())],
                    "load_dtypes": [(name, str(self._load_dtype(name, dtype))) for name, (_, _, dtype) in sorted(self.entries.items())],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.manifest = SharedWeightManifest.from_tensors(metadata, weight_signature=signature, alignment=4096)

    def _validate_config(self):
        validate_shared_weight_config(self.config)
        if not self.model.block_offload:
            raise ValueError("Qwen-Image-2.1 shared_cpu_weights requires cpu_offload=true and offload_granularity='block'")
        unsupported = ("tensor_parallel", "lazy_load", "unload_modules", "weight_auto_quant", "dummy_model", "lora_dynamic_apply", "lora_configs", "adapter_model_path")
        enabled = [key for key in unsupported if self.config.get(key)]
        if enabled:
            raise ValueError(f"Qwen-Image-2.1 shared CPU weights do not support {enabled}")
        if AI_DEVICE != "cuda":
            raise ValueError("Qwen-Image-2.1 shared CPU weights require CUDA")
        if GET_DTYPE() != torch.bfloat16:
            raise ValueError("Qwen-Image-2.1 shared CPU weights currently require DTYPE=BF16")
        scheme = self.config.get("dit_quant_scheme", "Default")
        if self.quantized:
            if scheme not in ("fp8-sgl", "fp8-f16-accum") or not self.config.get("dit_quantized_ckpt"):
                raise ValueError("Qwen-Image-2.1 shared FP8 weights require fp8-sgl or fp8-f16-accum and dit_quantized_ckpt")
        elif scheme != "Default":
            raise ValueError("dit_quant_scheme requires dit_quantized=true")

    def _block_shapes(self):
        head_dim = self.config["attention_head_dim"]
        hidden = self.config["num_attention_heads"] * head_dim
        intermediate = hidden * self.config["mlp_ratio"]
        shapes = {
            "attn.to_q.weight": (hidden, hidden),
            "attn.to_k.weight": (hidden, hidden),
            "attn.to_v.weight": (hidden, hidden),
            "attn.to_out.0.weight": (hidden, hidden),
            "img_mlp.proj.weight": (intermediate, hidden),
            "img_mlp.gate_layer.weight": (intermediate, hidden),
            "img_mlp.out.weight": (hidden, intermediate),
        }
        if self.quantized:
            shapes.update({name.removesuffix(".weight") + ".weight_scale": (shape[0], 1) for name, shape in list(shapes.items())})
        shapes.update({"attn.norm_q.weight": (head_dim,), "attn.norm_k.weight": (head_dim,)})
        return {f"transformer_blocks.{index}.{name}": shape for index in range(self.config["num_layers"]) for name, shape in shapes.items()}

    def _load_dtype(self, name, source_dtype):
        tensor = torch.empty((), dtype=source_dtype, device="meta")
        return self.model._checkpoint_dtype(name, tensor, self.unified_dtype, self.sensitive_layer, self.quantized) or source_dtype

    def _inspect_checkpoint(self, root):
        files = sorted(root.glob("*.safetensors")) if root.is_dir() else [root]
        if not files:
            raise FileNotFoundError(f"No Qwen-Image-2.1 checkpoint shards under {root}")
        expected = self._block_shapes()
        entries, metadata, signatures, indexed = {}, {}, [], {}
        for path in files:
            if path.suffix != ".safetensors":
                raise ValueError("Qwen-Image-2.1 shared CPU weights require safetensors checkpoints")
            for name, spec in read_safetensors_header(path).items():
                if name == "__metadata__":
                    continue
                if name in entries:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                dtype = _DTYPES[spec["dtype"]]
                shape = tuple(spec["shape"])
                source = torch.empty(shape, dtype=dtype, device="meta")
                if source.numel() * source.element_size() != spec["data_offsets"][1] - spec["data_offsets"][0]:
                    raise ValueError(f"Invalid checkpoint tensor byte extent: {name}")
                entries[name] = (path, shape, dtype)
                indexed[name] = path.name
                if name not in expected:
                    continue
                if shape != expected[name]:
                    raise ValueError(f"Qwen-Image-2.1 block shape mismatch: {name}: {shape}, expected {expected[name]}")
                is_fp8_weight = self.quantized and name.endswith(".weight") and source.ndim == 2
                if (dtype == torch.float8_e4m3fn) != is_fp8_weight:
                    raise ValueError(f"Unexpected Qwen-Image-2.1 block dtype: {name}: {dtype}")
                # Match the ordinary loader's cast before MM post_process.
                # Storing the final FP32 scale here avoids a private copy per rank.
                target_dtype = torch.float32 if name.endswith(".weight_scale") else self._load_dtype(name, dtype)
                metadata[name] = torch.empty(shape, dtype=target_dtype, device="meta")
            signatures.append((path.name, checkpoint_content_digest(path)))
        required = expected.keys() | _PRIVATE_KEYS
        if entries.keys() != required:
            raise ValueError(f"Qwen-Image-2.1 checkpoint keys differ: missing={sorted(required - entries.keys())}, unexpected={sorted(entries.keys() - required)}")
        if root.is_dir():
            for index in root.glob("*.safetensors.index.json"):
                if read_checkpoint_json(index).get("weight_map") != indexed:
                    raise ValueError(f"Qwen-Image-2.1 checkpoint index/header mismatch: {index}")
        return entries, metadata, signatures

    def _read(self, shared):
        by_file = defaultdict(list)
        for name, (path, shape, dtype) in self.entries.items():
            if name.startswith("transformer_blocks.") == shared:
                by_file[path].append((name, shape, dtype))
        for path, specs in sorted(by_file.items()):
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for name, shape, dtype in specs:
                    tensor = checkpoint.get_tensor(name)
                    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                        raise ValueError(f"Qwen-Image-2.1 checkpoint changed after preflight: {name}")
                    yield name, tensor.to(self._load_dtype(name, dtype))

    def load_private_weights(self):
        return {name: tensor.clone() for name, tensor in self._read(shared=False)}

    def _populate(self, views):
        logger.info("Qwen-Image-2.1 leader populating {:.3f} GiB shared block weights", self.manifest.nbytes / 1024**3)
        for name, tensor in self._read(shared=True):
            views[name].copy_(tensor)

    def materialize(self, private):
        allocation = materialize_shared_weight_arena(
            self.manifest,
            self._populate,
            scope=self.config.get("shared_cpu_weight_scope", "auto"),
            strict_numa=self.config.get("shared_cpu_weight_strict_numa", True),
            register_chunk_bytes=self.config.get("shared_cpu_weight_register_chunk_mb", 128) * 1024**2,
        )
        try:
            return SharedWeightViewMap(private, allocation.tensor_views(), owner=allocation)
        except BaseException:
            allocation.close()
            raise
