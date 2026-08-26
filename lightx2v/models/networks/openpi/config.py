"""Configuration objects for the native PyTorch pi0.5 backend.

The numerical values mirror Physical Intelligence's released ``pi05_libero``
configuration.  This module is intentionally pure Python: importing it does
not require JAX, Flax, Orbax, or the OpenPI source tree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


GemmaVariant = Literal["dummy", "gemma_300m", "gemma_2b"]


def get_config(variant: GemmaVariant) -> GemmaConfig:
    """Return the Gemma dimensions used by the official OpenPI model."""
    if variant == "dummy":
        return GemmaConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    if variant == "gemma_300m":
        return GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b":
        return GemmaConfig(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
    raise ValueError(f"Unsupported OpenPI Gemma variant: {variant!r}")


@dataclass(frozen=True)
class Pi0Config:
    """Minimal model config consumed by :class:`PI0Pytorch`.

    Defaults are the released pi0.5-LIBERO checkpoint.  ``action_dim`` is the
    padded internal dimension; the LIBERO environment still consumes 7-D
    actions.
    """

    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    paligemma_variant: GemmaVariant = "gemma_2b"
    action_expert_variant: GemmaVariant = "gemma_300m"
    pi05: bool = True
    discrete_state_input: bool = False
    pytorch_compile_mode: str | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "Pi0Config":
        return cls(
            action_dim=config["action_dim"],
            action_horizon=config["action_horizon"],
            max_token_len=config["max_token_len"],
            dtype=config["dtype"],
            paligemma_variant=config["paligemma_variant"],
            action_expert_variant=config["action_expert_variant"],
            pi05=config["pi05"],
            discrete_state_input=config["discrete_state_input"],
            pytorch_compile_mode=config["pytorch_compile_mode"],
        )

    def validate_pi05_libero(self) -> None:
        expected = {
            "pi05": True,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "action_dim": 32,
            "action_horizon": 10,
            "max_token_len": 200,
            "discrete_state_input": False,
        }
        actual = {name: getattr(self, name) for name in expected}
        wrong = {name: (actual[name], value) for name, value in expected.items() if actual[name] != value}
        if wrong:
            details = ", ".join(f"{name}={got!r} (expected {want!r})" for name, (got, want) in wrong.items())
            raise ValueError(f"Configuration does not match the released pi05_libero checkpoint: {details}")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"Unsupported OpenPI dtype: {self.dtype!r}")
