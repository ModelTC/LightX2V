"""CUDA-graph-aware custom all-reduce for HunyuanImage3 AR.

The vLLM backend uses a Gloo process group to exchange CUDA IPC metadata and
CUDA kernels to perform the reduction.  Denoising deliberately does not use
this wrapper; its TP2 reductions continue to use the active NCCL group.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_SUPPORTED_AR_TP_SIZES = frozenset((2, 4))
_VLLM_SKIP_P2P_CHECK_ENV = "VLLM_SKIP_P2P_CHECK"


class HunyuanImage3CustomAllReduceError(RuntimeError):
    """Raised when a required AR custom all-reduce cannot be honored."""


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


@dataclass(frozen=True)
class HunyuanImage3CustomAllReduceConfig:
    enabled: bool = False
    backend: str = "vllm"
    required: bool = False
    max_size_bytes: int = 8 * 1024 * 1024
    skip_p2p_check: bool = True
    graph_mode: str = "direct"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> HunyuanImage3CustomAllReduceConfig:
        values: Mapping[str, Any] = {} if config is None else config
        if not isinstance(values, Mapping):
            raise TypeError(f"custom all-reduce config must be a mapping, got {type(values).__name__}.")

        parsed = cls(
            enabled=_as_bool(values.get("enable_ar_custom_all_reduce", False), name="enable_ar_custom_all_reduce"),
            backend=str(values.get("ar_custom_all_reduce_backend", "vllm")).strip().lower(),
            required=_as_bool(values.get("ar_custom_all_reduce_required", False), name="ar_custom_all_reduce_required"),
            max_size_bytes=int(values.get("ar_custom_all_reduce_max_size_bytes", 8 * 1024 * 1024)),
            skip_p2p_check=_as_bool(
                values.get("ar_custom_all_reduce_skip_p2p_check", True),
                name="ar_custom_all_reduce_skip_p2p_check",
            ),
            graph_mode=str(values.get("ar_custom_all_reduce_graph_mode", "direct")).strip().lower(),
        )
        if parsed.required and not parsed.enabled:
            raise ValueError("ar_custom_all_reduce_required=true requires enable_ar_custom_all_reduce=true.")
        if parsed.enabled and parsed.backend != "vllm":
            raise ValueError(f"Unsupported AR custom all-reduce backend {parsed.backend!r}; expected 'vllm'.")
        if parsed.max_size_bytes <= 0:
            raise ValueError("ar_custom_all_reduce_max_size_bytes must be positive.")
        if parsed.graph_mode not in {"direct", "workspace"}:
            raise ValueError("ar_custom_all_reduce_graph_mode must be 'direct' or 'workspace'.")
        return parsed


def _create_vllm_backend(**kwargs: Any) -> Any:
    # vLLM remains optional for configurations that do not enable custom AR.
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    return CustomAllreduce(**kwargs)


class HunyuanImage3CustomAllReduce:
    """Dispatch eligible AR SUM reductions to vLLM custom all-reduce."""

    def __init__(
        self,
        *,
        metadata_group: dist.ProcessGroup,
        fallback_group: dist.ProcessGroup | Callable[[], dist.ProcessGroup],
        config: Mapping[str, Any] | HunyuanImage3CustomAllReduceConfig | None,
        device: torch.device | str | int,
        phase_getter: Callable[[], str],
    ) -> None:
        self.config = config if isinstance(config, HunyuanImage3CustomAllReduceConfig) else HunyuanImage3CustomAllReduceConfig.from_mapping(config)
        self.metadata_group = metadata_group
        self._fallback_group = fallback_group
        self.device = torch.device(device)
        self._phase_getter = phase_getter
        self._backend: Any | None = None
        self._initialization_attempted = False
        self._available = False
        self._closed = False
        self._capture_depth = 0
        self._unavailable_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def required(self) -> bool:
        return self.config.required

    @property
    def available(self) -> bool:
        return self._available and not self._closed

    @staticmethod
    def _backend_name(group: dist.ProcessGroup) -> str:
        return str(dist.get_backend(group)).strip().lower()

    def _initialization_failure(self, reason: str, error: BaseException | None = None) -> bool:
        self._available = False
        self._unavailable_reason = reason
        if self.required:
            message = f"Required HunyuanImage3 AR custom all-reduce initialization failed: {reason}."
            if error is None:
                raise HunyuanImage3CustomAllReduceError(message)
            raise HunyuanImage3CustomAllReduceError(message) from error
        return False

    def initialize(self) -> bool:
        """Collectively initialize the vLLM backend on full-world AR ranks."""

        if self._closed:
            raise HunyuanImage3CustomAllReduceError("Cannot initialize a closed custom all-reduce.")
        if not self.enabled:
            return False
        if self._initialization_attempted:
            if self.required and not self._available:
                raise HunyuanImage3CustomAllReduceError(
                    f"Required HunyuanImage3 AR custom all-reduce is unavailable: {self._unavailable_reason}."
                )
            return self._available

        self._initialization_attempted = True
        if self.metadata_group is None:
            return self._initialization_failure("metadata_group_missing")
        if not dist.is_available() or not dist.is_initialized():
            return self._initialization_failure("torch_distributed_not_initialized")

        try:
            metadata_backend = self._backend_name(self.metadata_group)
            if "gloo" not in metadata_backend:
                return self._initialization_failure(f"metadata_group_backend_{metadata_backend or 'unknown'}")
            world_size = int(dist.get_world_size(group=self.metadata_group))
            if world_size not in _SUPPORTED_AR_TP_SIZES:
                return self._initialization_failure(f"unsupported_ar_tp_size_{world_size}")

            previous_skip_check = os.environ.get(_VLLM_SKIP_P2P_CHECK_ENV)
            if self.config.skip_p2p_check:
                os.environ[_VLLM_SKIP_P2P_CHECK_ENV] = "1"
            try:
                backend = _create_vllm_backend(
                    group=self.metadata_group,
                    device=self.device,
                    max_size=self.config.max_size_bytes,
                    symm_mem_enabled=False,
                )
            finally:
                if self.config.skip_p2p_check:
                    if previous_skip_check is None:
                        os.environ.pop(_VLLM_SKIP_P2P_CHECK_ENV, None)
                    else:
                        os.environ[_VLLM_SKIP_P2P_CHECK_ENV] = previous_skip_check

            self._backend = backend
            if backend is None:
                return self._initialization_failure("backend_factory_returned_none")
            if bool(getattr(backend, "disabled", False)):
                return self._initialization_failure("vllm_backend_disabled")
        except HunyuanImage3CustomAllReduceError:
            raise
        except Exception as error:
            return self._initialization_failure(type(error).__name__, error)

        self._available = True
        self._unavailable_reason = None
        return True

    @staticmethod
    def _weakly_contiguous(tensor: torch.Tensor) -> bool:
        if tensor.is_contiguous():
            return True
        try:
            storage_bytes = tensor.untyped_storage().nbytes()
            offset_bytes = tensor.storage_offset() * tensor.element_size()
            return storage_bytes - offset_bytes == tensor.numel() * tensor.element_size()
        except (AttributeError, RuntimeError):
            return False

    def _tensor_eligibility(self, tensor: torch.Tensor) -> tuple[bool, str]:
        if tensor.device.type != "cuda":
            return False, "not_cuda"
        if tensor.dtype not in _SUPPORTED_DTYPES:
            return False, f"unsupported_dtype_{tensor.dtype}"
        if tensor.numel() == 0:
            return False, "empty_tensor"
        num_bytes = tensor.numel() * tensor.element_size()
        if num_bytes % 16:
            return False, "size_not_16_byte_aligned"
        if num_bytes >= self.config.max_size_bytes:
            return False, "size_exceeds_custom_ar_limit"
        if not self._weakly_contiguous(tensor):
            return False, "not_weakly_contiguous"
        return True, "eligible"

    def _custom_decision(self, tensor: torch.Tensor) -> tuple[bool, str, bool]:
        locally_eligible, reason = self._tensor_eligibility(tensor)
        if not locally_eligible:
            return False, reason, False
        if not self.enabled:
            return False, "custom_ar_disabled", True
        if not self._initialization_attempted:
            return False, "custom_ar_not_initialized", True
        if not self.available or self._backend is None:
            return False, self._unavailable_reason or "custom_ar_unavailable", True
        try:
            if not bool(self._backend.should_custom_ar(tensor)):
                return False, "vllm_should_custom_ar_rejected", True
        except Exception as error:
            return False, f"vllm_should_custom_ar_{type(error).__name__}", True
        return True, "eligible", True

    def _nccl_fallback(self, tensor: torch.Tensor, *, reason: str) -> torch.Tensor:
        group = self._fallback_group() if callable(self._fallback_group) else self._fallback_group
        if group is None:
            raise HunyuanImage3CustomAllReduceError(f"NCCL fallback group is missing for AR all-reduce ({reason}).")
        backend = self._backend_name(group)
        if "nccl" not in backend:
            raise HunyuanImage3CustomAllReduceError(f"HunyuanImage3 AR fallback must use NCCL, got backend {backend!r}.")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    def all_reduce(self, tensor: torch.Tensor, *, is_decode: bool = False) -> torch.Tensor:
        """SUM-reduce an AR tensor and fail closed for required decode work."""

        if self._closed:
            raise HunyuanImage3CustomAllReduceError("Cannot use a closed custom all-reduce.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"all_reduce expects torch.Tensor, got {type(tensor).__name__}.")
        use_custom, reason, locally_eligible = self._custom_decision(tensor)
        if use_custom:
            try:
                if self.config.graph_mode == "workspace" and self._capture_depth and torch.cuda.is_current_stream_capturing():
                    output = self._backend.all_reduce(tensor, registered=False)
                else:
                    output = self._backend.custom_all_reduce(tensor)
            except Exception as error:
                reason = f"vllm_custom_all_reduce_{type(error).__name__}"
                if self.required and is_decode and locally_eligible:
                    raise HunyuanImage3CustomAllReduceError(f"Required AR decode custom all-reduce failed: {reason}.") from error
            else:
                if output is not None:
                    return output
                reason = "vllm_custom_all_reduce_returned_none"

        if self.required and is_decode and locally_eligible:
            raise HunyuanImage3CustomAllReduceError(f"Required AR decode tensor cannot fall back to NCCL: {reason}.")
        return self._nccl_fallback(tensor, reason=reason)

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Register stable custom-all-reduce buffers during AR graph capture."""

        if self._closed:
            raise HunyuanImage3CustomAllReduceError("Cannot capture with a closed custom all-reduce.")
        if not self.enabled:
            yield
            return
        if self._phase_getter().strip().lower() != "ar":
            raise HunyuanImage3CustomAllReduceError("Custom all-reduce graph capture is only valid during the AR phase.")
        if not self.available or self._backend is None:
            if self.required:
                raise HunyuanImage3CustomAllReduceError(
                    f"Required AR custom all-reduce is unavailable during graph capture: {self._unavailable_reason or 'not_initialized'}."
                )
            yield
            return
        if self._capture_depth:
            raise HunyuanImage3CustomAllReduceError("Nested custom all-reduce capture contexts are not supported.")

        self._capture_depth = 1
        try:
            with self._backend.capture():
                yield
        finally:
            self._capture_depth = 0

    def close(self) -> None:
        """Release vLLM CUDA IPC resources."""

        if self._closed:
            return
        if self._capture_depth:
            raise HunyuanImage3CustomAllReduceError("Cannot close custom all-reduce during graph capture.")
        backend = self._backend
        self._backend = None
        self._available = False
        self._closed = True
        if backend is not None:
            backend.close()


__all__ = [
    "HunyuanImage3CustomAllReduce",
    "HunyuanImage3CustomAllReduceConfig",
    "HunyuanImage3CustomAllReduceError",
]
