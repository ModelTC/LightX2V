"""Attention execution policies shared by Diffusers model adapters."""

from __future__ import annotations

from collections.abc import Mapping

from lightx2v_train.runtime.backend import get_backend


def configured_attention_backend(config: Mapping | None) -> str | None:
    """Prefer runtime operator policy and retain the existing model option."""
    config = config or {}
    policy = config.get("runtime", {}).get("ops", {}).get("attention")
    if isinstance(policy, Mapping):
        policy = policy.get("backend", "auto")
    if policy is None:
        policy = config.get("model", {}).get("attention_backend")
    return policy


def prepare_diffusers_attention(module, config: Mapping | None = None):
    """Set module-local attention while restoring Diffusers' global default."""
    backend = get_backend().resolve_attention_backend(configured_attention_backend(config))
    if backend is None:
        return module
    from diffusers.models.attention_dispatch import attention_backend

    # ModelMixin.set_attention_backend also changes the process-wide default.
    # The context restores that default while retaining processor-local choices.
    with attention_backend(backend):
        module.set_attention_backend(backend)
    return module
