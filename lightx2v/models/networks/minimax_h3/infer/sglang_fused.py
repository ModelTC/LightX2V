import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

from lightx2v.common.ops.mm.mm_weight import unwrap_tp_weight
from lightx2v.models.networks.minimax_h3.infer.sglang_parity import _linear_weight

_fused_qknorm = None
_fused_qknorm_rope = None
_indexed_scale_shift = None
_indexed_gate = None
_silu_mul = None
_vae_silu_mul = None
_vae_scaled_residual_add = None
_configured_root = None


def _preload_orjson(sglang_root: str) -> None:
    if importlib.util.find_spec("orjson") is not None:
        return
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    package_dir = os.path.join(sglang_root, ".venv", "lib", py_version, "site-packages", "orjson")
    init_path = os.path.join(package_dir, "__init__.py")
    if not os.path.isfile(init_path):
        raise RuntimeError(f"SGLang parity could not find orjson at {init_path}")
    spec = importlib.util.spec_from_file_location("orjson", init_path, submodule_search_locations=[package_dir])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"SGLang parity could not load an import spec for {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["orjson"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("orjson", None)
        raise


def _module_belongs_to(module, source_root: str) -> bool:
    candidates = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        candidates.append(module_file)
    candidates.extend(str(path) for path in (getattr(module, "__path__", None) or ()))
    source_root = os.path.realpath(source_root)
    return bool(candidates) and all(os.path.commonpath((source_root, os.path.realpath(candidate))) == source_root for candidate in candidates)


def _select_sglang_source(python_path: str) -> None:
    loaded = sys.modules.get("sglang")
    if loaded is not None and not _module_belongs_to(loaded, python_path):
        # Discard a pip namespace before loading the requested checkout.
        for name in sorted(
            (name for name in sys.modules if name == "sglang" or name.startswith("sglang.")),
            key=len,
            reverse=True,
        ):
            sys.modules.pop(name, None)
    sys.path[:] = [entry for entry in sys.path if os.path.realpath(entry or os.curdir) != os.path.realpath(python_path)]
    sys.path.insert(0, python_path)
    importlib.invalidate_caches()


def configure_sglang_fused_ops(sglang_root: str | None) -> None:
    global _configured_root, _fused_qknorm, _fused_qknorm_rope, _indexed_gate, _indexed_scale_shift, _silu_mul, _vae_silu_mul, _vae_scaled_residual_add
    requested_root = None if sglang_root is None else os.path.realpath(sglang_root)
    if _fused_qknorm_rope is not None:
        if requested_root != _configured_root:
            raise RuntimeError(f"SGLang parity ops are already loaded from {_configured_root}, cannot switch to {requested_root} in the same process")
        return
    if not requested_root:
        raise RuntimeError("h3_sglang_parity_ops=true requires h3_sglang_root")
    python_path = os.path.join(requested_root, "python")
    if not os.path.isdir(python_path):
        raise RuntimeError(f"SGLang parity could not find the Python source directory {python_path}")
    _preload_orjson(requested_root)
    _select_sglang_source(python_path)
    from sglang.kernels.ops.activation.activation import (
        silu_and_mul_with_activation_rounding,
        silu_and_mul_with_activation_rounding_,
    )
    from sglang.kernels.ops.diffusion.common import platform as diffusion_platform

    # Importing the multimodal registry here would re-register sgl_kernel fake ops.
    original_platform_key = diffusion_platform.platform_key
    original_is_cuda = diffusion_platform.is_cuda
    diffusion_platform.platform_key = lambda: "cuda" if torch.cuda.is_available() else "cpu"
    diffusion_platform.is_cuda = torch.cuda.is_available
    try:
        from sglang.kernels.ops.diffusion.modulate.scale_shift_triton import try_fused_scaled_residual_add_exact
    finally:
        diffusion_platform.platform_key = original_platform_key
        diffusion_platform.is_cuda = original_is_cuda
    from sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton import (
        indexed_gate_bf16_,
        indexed_scale_shift_bf16_,
    )
    from sglang.kernels.ops.diffusion.rope.qknorm_rope_jit import fused_inplace_qknorm_rope
    from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

    sglang_module = sys.modules.get("sglang")
    if sglang_module is None or not _module_belongs_to(sglang_module, python_path):
        raise RuntimeError(f"SGLang parity imported an unexpected sglang package instead of {python_path}")
    _configured_root = requested_root
    _fused_qknorm = fused_inplace_qknorm
    _fused_qknorm_rope = fused_inplace_qknorm_rope
    _indexed_scale_shift = indexed_scale_shift_bf16_
    _indexed_gate = indexed_gate_bf16_
    _silu_mul = silu_and_mul_with_activation_rounding_
    _vae_silu_mul = silu_and_mul_with_activation_rounding
    _vae_scaled_residual_add = try_fused_scaled_residual_add_exact


def _norm_weights(q_norm, k_norm) -> tuple[torch.Tensor, torch.Tensor]:
    return q_norm._get_actual_weight(), k_norm._get_actual_weight()


def apply_qk_norm_sglang(q: torch.Tensor, k: torch.Tensor, q_norm, k_norm) -> tuple[torch.Tensor, torch.Tensor]:
    q_weight, k_weight = _norm_weights(q_norm, k_norm)
    _fused_qknorm(q, k, q_weight, k_weight, eps=q_norm.eps, head_dim=q.shape[-1])
    return q, k


def apply_qk_norm_rope_sglang(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm,
    k_norm,
    rope_cache: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    q_weight, k_weight = _norm_weights(q_norm, k_norm)
    cos_sin_cache, positions = rope_cache
    _fused_qknorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=True,
        eps=q_norm.eps,
        head_dim=q.shape[-1],
        rope_dim=cos_sin_cache.shape[-1],
        round_norm_before_rope=True,
    )
    return q, k


def indexed_scale_shift_sglang(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return _indexed_scale_shift(x, shift, scale, indices)


def indexed_gate_sglang(x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return _indexed_gate(x, gate, other, indices)


def apply_mlp_sglang(weights, hidden_states: torch.Tensor) -> torch.Tensor:
    cache = getattr(weights, "_sglang_parity_mlp_cache", None)
    if cache is None:
        source_weight = _linear_weight(weights.in_proj)
        if source_weight.shape[1] % 2:
            raise ValueError(f"Invalid H3 fused MLP weight shape {tuple(source_weight.shape)}")
        value_weight, gate_weight = source_weight.chunk(2, dim=1)
        fused_weight = torch.cat((gate_weight.t(), value_weight.t()), dim=0).contiguous()
        cache = fused_weight
        weights._sglang_parity_mlp_cache = cache
        # The merged matrix replaces the Diffusers [value, gate] weight.
        unwrap_tp_weight(weights.in_proj).weight = None
    hidden = F.linear(hidden_states, cache)
    hidden = _silu_mul(hidden)
    return weights.out_proj.apply(hidden)


def apply_vae_silu_mul_sglang(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.is_cuda and hidden_states.dtype in (torch.float16, torch.bfloat16) and hidden_states.is_contiguous() and hidden_states.shape[-1] % 32 == 0:
        return _vae_silu_mul(hidden_states)
    gate, value = hidden_states.chunk(2, dim=-1)
    return F.silu(gate).mul_(value)


def scaled_residual_add_vae_sglang(
    residual: torch.Tensor,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    fused = _vae_scaled_residual_add(residual, hidden_states, scale)
    return residual + hidden_states * scale if fused is None else fused


def prepare_vae_rope_sglang(
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    cos, sin = rotary_emb
    if (
        not cos.is_cuda
        or dtype not in (torch.float16, torch.bfloat16)
        or cos.shape != sin.shape
        or cos.dim() != 4
        or cos.shape[0] != 1
        or cos.shape[2] != 1
        or cos.shape[-1] % 2
        or torch.compiler.is_compiling()
    ):
        return cos, sin

    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)
    half = cos.shape[-1] // 2
    cache = torch.cat((cos[0, :, 0, :half], sin[0, :, 0, :half]), dim=-1).contiguous()
    positions = torch.arange(cos.shape[1], dtype=torch.long, device=cos.device)
    return cos, sin, cache, positions


def _apply_vae_rope_fallback(
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    cos, sin = rotary_emb[:2]
    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    scaled = rotary * cos
    scaled.add_(torch.cat((-second, first), dim=-1) * sin)
    if rotary_dim < hidden_states.shape[-1]:
        return torch.cat((scaled, passthrough), dim=-1)
    return scaled


def apply_vae_rope_sglang(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(rotary_emb) == 4:
        _, _, cache, positions = rotary_emb
        import sgl_kernel

        query = query.contiguous()
        key = key.contiguous()
        sgl_kernel.rotary_embedding(
            positions,
            query.view(query.shape[1], -1),
            key.view(key.shape[1], -1),
            query.shape[-1],
            cache,
            True,
        )
        return query, key

    return (
        _apply_vae_rope_fallback(query, rotary_emb),
        _apply_vae_rope_fallback(key, rotary_emb),
    )
