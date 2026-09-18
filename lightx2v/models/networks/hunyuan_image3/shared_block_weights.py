"""One shared CPU replica containing all storage-TP slices of Hunyuan blocks.

Every rank describes the same arena. TP selects disjoint regions, while SP/CFG
replicas of the same TP coordinate adopt the very same immutable tensor views.
"""

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
from lightx2v.models.networks.base_model import SAFETENSORS_DTYPE_MAP
from lightx2v.models.networks.hunyuan_image3.offload import iter_weight_leaves
from lightx2v.models.networks.hunyuan_image3.weights.transformer_weights import HunyuanImage3TransformerBlock
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE


class HunyuanImage3SharedBlockAdapter:
    FORMAT_VERSION = "hunyuan-image3-block-tp-v1"

    def __init__(self, model, unified_dtype, sensitive_layer):
        self.model = model
        self.config = model.config
        self.unified_dtype = unified_dtype
        self.sensitive_layer = sensitive_layer
        validate_shared_weight_config(self.config)
        root = Path(model.model_path).resolve()
        index = read_checkpoint_json(root / "model.safetensors.index.json")["weight_map"]
        files = defaultdict(set)
        for key, filename in index.items():
            path = (root / filename).resolve()
            if not path.is_relative_to(root) or path.suffix != ".safetensors":
                raise ValueError(f"Invalid HunyuanImage3 checkpoint shard: {filename}")
            files[path].add(key)
        self.entries = {}
        metadata = {}
        self.local_names = {}
        signatures = []
        logger.info("HunyuanImage3 shared checkpoint preflight: {} shards (content checks may read uncached files)", len(files))
        for path, indexed_names in sorted(files.items()):
            header = read_safetensors_header(path)
            entries = {key: value for key, value in header.items() if key != "__metadata__"}
            if entries.keys() != indexed_names:
                raise ValueError(f"HunyuanImage3 index/header mismatch: {path.name}")
            signatures.append((path.name, checkpoint_content_digest(path)))
            for name, spec in entries.items():
                if not any(name.startswith(prefix) for prefix in model.preserved_keys) or any(part in name for part in model.remove_keys):
                    continue
                source_dtype = SAFETENSORS_DTYPE_MAP[spec["dtype"]]
                shape = tuple(spec["shape"])
                tensor = torch.empty(shape, dtype=source_dtype, device="meta")
                if tensor.numel() * tensor.element_size() != spec["data_offsets"][1] - spec["data_offsets"][0]:
                    raise ValueError(f"Invalid HunyuanImage3 tensor byte extent: {name}")
                self.entries[name] = (path, shape, source_dtype)
                if not name.startswith("model.layers."):
                    continue
                split_type = model._tp_split_type(name)
                split = model.tp_size > 1 and split_type is not None and not (split_type == "row" and tensor.ndim == 1)
                dtype = torch.float32 if name.endswith(".mlp.gate.wg.weight") else self._load_dtype(name, source_dtype)
                for rank in range(model.tp_size) if split else (None,):
                    local = model._select_tensor_parallel_shard(name, tensor, tp_rank=rank) if split else tensor
                    arena_name = f"tp{rank}/{name}" if split else name
                    metadata[arena_name] = torch.empty(local.shape, device="meta", dtype=dtype)
                    if rank is None or rank == model.tp_rank:
                        self.local_names[name] = arena_name

        expected = set()
        for layer in range(int(self.config.get("num_layers") or self.config["num_hidden_layers"])):
            block = HunyuanImage3TransformerBlock(layer, self.config, "Default")
            expected.update(name for leaf in iter_weight_leaves(block) for name, _, _ in leaf.base_attrs)
        if expected != self.local_names.keys():
            raise ValueError(f"HunyuanImage3 block schema mismatch: missing={sorted(expected - self.local_names.keys())}, unexpected={sorted(self.local_names.keys() - expected)}")
        signature = hashlib.sha256(
            json.dumps(
                {
                    "format": self.FORMAT_VERSION,
                    "files": signatures,
                    "tp_size": model.tp_size,
                    "micro_shards": model.micro_shard_count,
                    "infer_dtype": str(GET_DTYPE()),
                    "sensitive_dtype": str(GET_SENSITIVE_DTYPE()),
                    "layouts": [(name, list(t.shape), str(t.dtype)) for name, t in sorted(metadata.items())],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.manifest = SharedWeightManifest.from_tensors(metadata, weight_signature=signature, alignment=4096)

    def _load_dtype(self, name, source_dtype):
        if not source_dtype.is_floating_point:
            return source_dtype
        return GET_DTYPE() if self.unified_dtype or all(part not in name for part in self.sensitive_layer) else GET_SENSITIVE_DTYPE()

    def _read(self, shared):
        files = defaultdict(list)
        for name, (path, shape, dtype) in self.entries.items():
            if name.startswith("model.layers.") == shared:
                files[path].append((name, shape, dtype))
        for index, (path, specs) in enumerate(sorted(files.items()), 1):
            if shared:
                logger.info("HunyuanImage3 shared payload shard {}/{}: {}", index, len(files), path.name)
            with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
                for name, shape, dtype in specs:
                    tensor = checkpoint.get_tensor(name)
                    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                        raise ValueError(f"HunyuanImage3 checkpoint changed after preflight: {name}")
                    yield name, tensor

    def load_private_weights(self):
        private = {}
        for name, tensor in self._read(shared=False):
            if self.model.tensor_parallel:
                tensor = self.model._select_tensor_parallel_shard(name, tensor)
            private[name] = tensor.to(device=self.model._tensor_target_device(name), dtype=self._load_dtype(name, tensor.dtype))
        return private

    def _populate(self, views):
        logger.info("HunyuanImage3 leader populating {:.3f} GiB shared block weights", self.manifest.nbytes / 1024**3)
        for name, tensor in self._read(shared=True):
            # Preserve the baseline cast *before* ForceFp32 router conversion.
            source = tensor.to(self._load_dtype(name, tensor.dtype))
            if name in views:
                # Storage TP=1 may still use multiple AR micro shards, whose
                # fused gate/up order differs from the checkpoint order.
                if self.model.tensor_parallel:
                    source = self.model._select_tensor_parallel_shard(name, source)
                views[name].copy_(source)
            else:
                for rank in range(self.model.tp_size):
                    views[f"tp{rank}/{name}"].copy_(self.model._select_tensor_parallel_shard(name, source, tp_rank=rank))

    def materialize(self, private):
        allocation = materialize_shared_weight_arena(
            self.manifest,
            self._populate,
            scope=self.config.get("shared_cpu_weight_scope", "auto"),
            strict_numa=self.config.get("shared_cpu_weight_strict_numa", True),
            register_chunk_bytes=self.config.get("shared_cpu_weight_register_chunk_mb", 128) * 1024**2,
        )
        try:
            views = allocation.tensor_views()
            weights = SharedWeightViewMap(private, {name: views[key] for name, key in self.local_names.items()}, owner=allocation)
            weights.manifest_names = self.local_names
            return weights
        except BaseException:
            allocation.close()
            raise


def validate_hunyuan_shared_views(weight_map, blocks):
    """Validate this TP coordinate, not the other TP slices in the same arena."""
    arena = weight_map.owner.arena
    if weight_map.consumed_shared_keys != weight_map.manifest_names.keys():
        raise RuntimeError("HunyuanImage3 operators did not consume every local shared tensor")
    count = 0
    for block in blocks:
        for leaf in iter_weight_leaves(block):
            for name, attr, transpose in leaf.base_attrs:
                spec = arena.manifest.by_name[weight_map.manifest_names[name]]
                expected = weight_map[name].t() if transpose else weight_map[name]
                actual = getattr(leaf, f"pin_{attr}", None)
                pointer = arena.address + spec.offset + spec.storage_offset * spec.itemsize
                if (
                    actual is None
                    or actual.device.type != "cpu"
                    or actual.data_ptr() != pointer
                    or actual.dtype != expected.dtype
                    or actual.shape != expected.shape
                    or actual.stride() != expected.stride()
                    or not actual.is_pinned()
                ):
                    raise RuntimeError(f"HunyuanImage3 shared tensor was copied, converted or misbound: {name}")
                count += 1
    logger.info(
        "[SharedCPUWeightsInfo] HunyuanImage3 validated {} local TP views; shmid={}, leader={}, replica_ranks={}, NUMA={}, arena={:.3f} GiB",
        count,
        arena.shmid,
        weight_map.owner.group.leader_rank,
        weight_map.owner.group.ranks,
        weight_map.owner.group.key.numa_node,
        arena.nbytes / 1024**3,
    )
