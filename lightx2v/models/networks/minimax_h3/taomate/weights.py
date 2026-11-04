"""Load the released TaoMate-H3 adapter into native MiniMax-H3 weights.

The Stage3 adapter is trained against the FL2VA checkpoint. LightX2V's H3
checkpoint has the same tensors with split Q/K/V rows and the opposite order
of the two SwiGLU projection halves. These transforms also apply to LoRA B.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

_BLOCK_COUNT = 50
_REFINER_COUNT = 2
_HIDDEN_SIZE = 5376
_ATTENTION_SIZE = 7168
_FFN_SIZE = 14336
_PROJECTION_SHAPES = {
    "attn.qkv_proj": (3 * _ATTENTION_SIZE, _HIDDEN_SIZE),
    "attn.out_proj": (_HIDDEN_SIZE, _ATTENTION_SIZE),
    "mlp.fc1": (2 * _FFN_SIZE, _HIDDEN_SIZE),
    "mlp.fc2": (_HIDDEN_SIZE, _FFN_SIZE),
}


@dataclass(frozen=True)
class TaoMateAdapterInfo:
    path: Path
    rank: int
    alpha: float
    target_count: int
    tensor_count: int


def _targets():
    for prefix, count in (("blocks", _BLOCK_COUNT), ("token_refiner.blocks", _REFINER_COUNT)):
        for index in range(count):
            for projection in _PROJECTION_SHAPES:
                yield f"{prefix}.{index}.{projection}", projection


def inspect_taomate_adapter(adapter_path):
    """Validate the entire released adapter inventory without loading tensors."""

    path = Path(adapter_path).expanduser().resolve()
    config_path = path / "config.json"
    checkpoint_path = path / "adapter_model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = config["rank"]
    alpha = config["alpha"]
    if type(rank) is not int or rank <= 0 or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("TaoMate-H3 adapter rank and alpha must be positive")
    targets = tuple(_targets())
    expected = {f"{target}.lora_{leaf}" for target, _ in targets for leaf in ("a", "b")}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as source:
        actual = set(source.keys())
        if actual != expected:
            raise ValueError(f"TaoMate-H3 adapter inventory differs: missing={len(expected - actual)}, unexpected={len(actual - expected)}")
        for target, projection in targets:
            output_size, input_size = _PROJECTION_SHAPES[projection]
            for leaf, shape in (("a", (rank, input_size)), ("b", (output_size, rank))):
                name = f"{target}.lora_{leaf}"
                tensor = source.get_slice(name)
                if tuple(tensor.get_shape()) != shape or tensor.get_dtype() != "F32":
                    raise ValueError(f"TaoMate-H3 adapter tensor {name} must be FP32 with shape {shape}")
    return TaoMateAdapterInfo(path, rank, float(alpha), len(targets), len(expected))


def _x2v_prefix(target):
    if target.startswith("blocks."):
        index, projection = target[len("blocks.") :].split(".", 1)
        return f"transformer_blocks.{index}", projection
    if target.startswith("token_refiner.blocks."):
        index, projection = target[len("token_refiner.blocks.") :].split(".", 1)
        return f"token_refiner.refiner_blocks.{index}", projection
    raise ValueError(f"Unknown TaoMate-H3 adapter target: {target}")


def _mapped_projections(target, down, up):
    prefix, projection = _x2v_prefix(target)
    if projection == "attn.qkv_proj":
        # The checkpoint's base QKV rows are grouped per head, but the
        # released adapter acts on the already reordered [Q, K, V] output.
        for part, output in zip(("q", "k", "v"), up.chunk(3, dim=0)):
            yield f"{prefix}.attn.to_{part}", down, output
    elif projection == "attn.out_proj":
        yield f"{prefix}.attn.to_out.0", down, up
    elif projection == "mlp.fc1":
        first, second = up.chunk(2, dim=0)
        yield f"{prefix}.ff.net.0.proj", down, torch.cat((second, first), dim=0)
    elif projection == "mlp.fc2":
        yield f"{prefix}.ff.net.2", down, up


def load_taomate_lora(adapter_path, *, dtype=torch.bfloat16):
    """Return tensors understood by MiniMaxH3Model's LoRA weight wrappers.

    The upstream runtime converts the FP32 adapter to the BF16 projection
    dtype before each LoRA GEMM. We do the same once during loading.
    """

    info = inspect_taomate_adapter(adapter_path)
    weights = {}
    alpha = torch.tensor(info.alpha, dtype=dtype)
    checkpoint = info.path / "adapter_model.safetensors"
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        for target, _ in _targets():
            down = source.get_tensor(f"{target}.lora_a").to(dtype)
            up = source.get_tensor(f"{target}.lora_b").to(dtype)
            for prefix, mapped_down, mapped_up in _mapped_projections(target, down, up):
                weights[f"{prefix}.lora_down.weight"] = mapped_down
                weights[f"{prefix}.lora_up.weight"] = mapped_up
                weights[f"{prefix}.alpha"] = alpha
    return weights


__all__ = [
    "TaoMateAdapterInfo",
    "inspect_taomate_adapter",
    "load_taomate_lora",
]
