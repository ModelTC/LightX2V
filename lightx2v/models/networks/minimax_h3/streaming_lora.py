"""Selective dynamic LoRA for H3's reusable disk-streamed block.

The index owns metadata, not factor tensors. Arithmetic and alpha/rank scaling
remain in MMWeight.register_lora/apply_lora, as in ordinary dynamic H3 LoRA.
"""

import math
import re
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class LoraPair:
    down_key: str
    up_key: str
    down_shape: tuple
    up_shape: tuple
    rank: int
    alpha: float


class MiniMaxH3StreamingLora:
    def __init__(self, path, *, normalize_key, target_shapes, strength, alpha=None, dtype=torch.bfloat16):
        self.path = str(path)
        self.dtype = dtype
        self.strength = float(strength)
        if not math.isfinite(self.strength):
            raise ValueError("MiniMax-H3 LoRA strength must be finite")
        if alpha is not None and (not math.isfinite(float(alpha)) or float(alpha) <= 0):
            raise ValueError("MiniMax-H3 LoRA alpha must be finite and positive")
        self.pairs = {}
        self.blocks = {}
        self.resident = {}
        # Only pre/post factors may be cached on CPU, never main-block factors.
        self._resident_cpu = {}
        with safe_open(self.path, framework="pt", device="cpu") as source:
            keys = {}
            for key in sorted(source.keys()):
                name = normalize_key(key)
                if name is None:
                    raise ValueError(f"Unsupported MiniMax-H3 LoRA tensor: {key}")
                if name in keys:
                    raise ValueError(f"MiniMax-H3 LoRA keys collide after normalization: {name}")
                keys[name] = key
            down = {name.removesuffix(".lora_down.weight") for name in keys if name.endswith(".lora_down.weight")}
            up = {name.removesuffix(".lora_up.weight") for name in keys if name.endswith(".lora_up.weight")}
            if not down or down != up:
                raise ValueError("MiniMax-H3 LoRA has incomplete A/B pairs")
            if any(name.removesuffix(".alpha") not in down for name in keys if name.endswith(".alpha")):
                raise ValueError("MiniMax-H3 LoRA alpha has no matching pair")
            for name in sorted(down):
                weight_name = name + ".weight"
                if weight_name not in target_shapes:
                    raise ValueError(f"Unsupported MiniMax-H3 streamed LoRA target: {weight_name}")
                a_key, b_key = keys[name + ".lora_down.weight"], keys[name + ".lora_up.weight"]
                a, b = source.get_slice(a_key), source.get_slice(b_key)
                a_shape, b_shape = tuple(a.get_shape()), tuple(b.get_shape())
                base_shape = tuple(target_shapes[weight_name])
                if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] <= 0 or b_shape[1] != a_shape[0] or (b_shape[0], a_shape[1]) != base_shape:
                    raise ValueError(f"MiniMax-H3 LoRA shape mismatch for {weight_name}: A={a_shape}, B={b_shape}, base={base_shape}")
                if a.get_dtype() not in {"BF16", "F16", "F32"} or b.get_dtype() not in {"BF16", "F16", "F32"}:
                    raise ValueError(f"Unsupported MiniMax-H3 LoRA factor dtype: {weight_name}")
                pair_alpha = alpha
                if name + ".alpha" in keys:
                    # Only scalar metadata is read here; never materialize A/B.
                    alpha_key = keys[name + ".alpha"]
                    if math.prod(source.get_slice(alpha_key).get_shape()) != 1:
                        raise ValueError(f"MiniMax-H3 LoRA alpha must be scalar: {name}")
                    pair_alpha = float(source.get_tensor(alpha_key).item())
                if pair_alpha is None or not math.isfinite(float(pair_alpha)) or float(pair_alpha) <= 0:
                    raise ValueError(f"MiniMax-H3 LoRA requires finite positive alpha: {name}")
                cast_alpha = torch.tensor(pair_alpha, dtype=self.dtype)
                if not torch.isfinite(cast_alpha) or cast_alpha <= 0:
                    raise ValueError(f"MiniMax-H3 LoRA alpha is not representable in {self.dtype}: {name}")
                pair = LoraPair(a_key, b_key, a_shape, b_shape, a_shape[0], float(pair_alpha))
                self.pairs[weight_name] = pair
                match = re.fullmatch(r"transformer_blocks\.(\d+)\.(.+)", weight_name)
                if match:
                    self.blocks.setdefault(int(match[1]), {})["transformer_blocks.0." + match[2]] = pair
                else:
                    self.resident[weight_name] = pair

    @staticmethod
    def weights(root):
        stack, visited, weights = [root], set(), {}
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in visited:
                continue
            visited.add(id(obj))
            if hasattr(obj, "register_lora") and hasattr(obj, "weight_name"):
                weights[obj.weight_name] = obj
            stack.extend(getattr(obj, "_modules", {}).values())
            stack.extend(getattr(obj, "_parameters", {}).values())
        return weights

    @classmethod
    def clear(cls, root):
        weights = [weight for weight in cls.weights(root).values() if getattr(weight, "has_lora_branch", False)]
        devices = {weight.lora_down.device for weight in weights}
        for device in devices:
            if device.type != "cpu":
                torch.get_device_module(device).synchronize()
        for weight in weights:
            weight.remove_lora()

    def bind(self, root, pairs):
        self.clear(root)
        weights = self.weights(root)
        missing = pairs.keys() - weights.keys()
        if missing:
            raise ValueError(f"MiniMax-H3 streamed LoRA targets not found: {sorted(missing)}")
        try:
            with safe_open(self.path, framework="pt", device="cpu") as source:
                for name, pair in pairs.items():
                    weight = weights[name]
                    # Read only this target, with ordinary unpinned CPU storage.
                    tensors = self._resident_cpu.get(name) if name in self.resident else None
                    if tensors is None:
                        tensors = (
                            source.get_tensor(pair.down_key).to(dtype=self.dtype, copy=True),
                            source.get_tensor(pair.up_key).to(dtype=self.dtype, copy=True),
                            torch.tensor(pair.alpha, dtype=self.dtype),
                        )
                        if name in self.resident:
                            self._resident_cpu[name] = tensors
                    factors = dict(zip((weight.lora_down_name, weight.lora_up_name, weight.lora_alpha_name), tensors))
                    weight.register_lora(factors, self.strength)
                    del factors, tensors
                    if not getattr(weight, "has_lora_branch", False):
                        raise RuntimeError(f"MiniMax-H3 streamed LoRA registration failed: {name}")
        except Exception:
            self.clear(root)
            raise

    def load_block(self, block, index):
        self.bind(block, self.blocks.get(index, {}))

    @contextmanager
    def resident_scope(self, root):
        names = self.weights(root)
        pairs = {name: pair for name, pair in self.resident.items() if name in names}
        if not pairs:
            yield
            return
        self.bind(root, pairs)
        try:
            yield
        finally:
            self.clear(root)


def streaming_target_shapes(checkpoint, block, *resident_roots):
    """Resolve supported MMWeight targets using existing base-checkpoint metadata."""
    names = set()
    for name in MiniMaxH3StreamingLora.weights(block):
        for index in checkpoint.block_indices:
            names.add(name.replace("transformer_blocks.0.", f"transformer_blocks.{index}.", 1))
    for root in resident_roots:
        names.update(MiniMaxH3StreamingLora.weights(root))
    shapes = {}
    if checkpoint.selected_reader is not None:
        for name in names:
            spec = checkpoint.targets[name][1]
            # Ordinary dynamic H3 LoRA uses BF16 factors. Sensitive FP32
            # linears require a different execution contract and are excluded.
            if spec.dtype == "BF16":
                shapes[name] = spec.shape
    else:
        by_shard = {}
        for name in names:
            by_shard.setdefault(checkpoint.weight_map[name], []).append(name)
        for shard, shard_names in by_shard.items():
            with safe_open(checkpoint.checkpoint_dir / shard, framework="pt", device="cpu") as source:
                for name in shard_names:
                    tensor = source.get_slice(name)
                    if tensor.get_dtype() == "BF16":
                        shapes[name] = tuple(tensor.get_shape())
    return shapes
