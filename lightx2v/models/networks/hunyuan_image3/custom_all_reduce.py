"""vLLM custom all-reduce for HunyuanImage3 AR."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_SUPPORTED_AR_TP_SIZES = {2, 4}
_VLLM_SKIP_P2P_CHECK_ENV = "VLLM_SKIP_P2P_CHECK"


@dataclass(frozen=True)
class HunyuanImage3CustomAllReduceConfig:
    max_size_bytes: int = 8 * 1024 * 1024
    skip_p2p_check: bool = True
    graph_mode: str = "direct"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> HunyuanImage3CustomAllReduceConfig:
        parsed = cls(
            max_size_bytes=int(config.get("ar_custom_all_reduce_max_size_bytes", 8 * 1024 * 1024)),
            skip_p2p_check=config.get("ar_custom_all_reduce_skip_p2p_check", True),
            graph_mode=config.get("ar_custom_all_reduce_graph_mode", "direct"),
        )
        if parsed.max_size_bytes <= 0:
            raise ValueError("ar_custom_all_reduce_max_size_bytes must be positive.")
        if parsed.graph_mode not in {"direct", "workspace"}:
            raise ValueError("ar_custom_all_reduce_graph_mode must be 'direct' or 'workspace'.")
        return parsed


class HunyuanImage3CustomAllReduce:
    def __init__(
        self,
        *,
        metadata_group: dist.ProcessGroup,
        fallback_group: dist.ProcessGroup,
        config: Mapping[str, Any],
        device: torch.device | str | int,
        phase_getter: Callable[[], str],
    ) -> None:
        self.config = HunyuanImage3CustomAllReduceConfig.from_mapping(config)
        self.metadata_group = metadata_group
        self.fallback_group = fallback_group
        self.device = torch.device(device)
        self._phase_getter = phase_getter
        self._backend = None
        self._capture_depth = 0
        self._closed = False

        if self._backend_name(metadata_group) != "gloo":
            raise RuntimeError("AR custom all-reduce metadata must use Gloo.")
        if self._backend_name(fallback_group) != "nccl":
            raise RuntimeError("AR custom all-reduce fallback must use NCCL.")
        world_size = dist.get_world_size(group=metadata_group)
        if world_size not in _SUPPORTED_AR_TP_SIZES:
            raise RuntimeError(f"AR custom all-reduce supports TP2 or TP4, got TP{world_size}.")

    @staticmethod
    def _backend_name(group: dist.ProcessGroup) -> str:
        return str(dist.get_backend(group))

    def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot initialize a closed custom all-reduce.")
        if self._backend is not None:
            return

        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

        previous_skip_check = os.environ.get(_VLLM_SKIP_P2P_CHECK_ENV)
        if self.config.skip_p2p_check:
            os.environ[_VLLM_SKIP_P2P_CHECK_ENV] = "1"
        backend = None
        initialization_error = None
        try:
            backend = CustomAllreduce(
                group=self.metadata_group,
                device=self.device,
                max_size=self.config.max_size_bytes,
                symm_mem_enabled=False,
            )
        except Exception as error:
            initialization_error = error
        finally:
            if self.config.skip_p2p_check:
                if previous_skip_check is None:
                    os.environ.pop(_VLLM_SKIP_P2P_CHECK_ENV, None)
                else:
                    os.environ[_VLLM_SKIP_P2P_CHECK_ENV] = previous_skip_check

        available = torch.tensor(backend is not None and not backend.disabled, dtype=torch.int32)
        dist.all_reduce(available, op=dist.ReduceOp.MIN, group=self.metadata_group)
        if not available.item():
            if backend is not None:
                backend.close()
            if initialization_error is not None:
                raise RuntimeError("vLLM custom all-reduce initialization failed.") from initialization_error
            raise RuntimeError("vLLM custom all-reduce is unavailable on this topology.")
        self._backend = backend

    @staticmethod
    def _weakly_contiguous(tensor: torch.Tensor) -> bool:
        if tensor.is_contiguous():
            return True
        storage_bytes = tensor.untyped_storage().nbytes()
        offset_bytes = tensor.storage_offset() * tensor.element_size()
        return storage_bytes - offset_bytes == tensor.numel() * tensor.element_size()

    def _eligible(self, tensor: torch.Tensor) -> bool:
        num_bytes = tensor.numel() * tensor.element_size()
        return (
            tensor.device.type == "cuda"
            and tensor.dtype in _SUPPORTED_DTYPES
            and tensor.numel() > 0
            and num_bytes % 16 == 0
            and num_bytes < self.config.max_size_bytes
            and self._weakly_contiguous(tensor)
        )

    def _nccl_fallback(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.fallback_group)
        return tensor

    def all_reduce(self, tensor: torch.Tensor, *, is_decode: bool) -> torch.Tensor:
        if self._closed or self._backend is None:
            raise RuntimeError("AR custom all-reduce is not initialized.")

        use_custom = self._eligible(tensor) and self._backend.should_custom_ar(tensor)
        if use_custom:
            if self.config.graph_mode == "workspace" and self._capture_depth and torch.cuda.is_current_stream_capturing():
                output = self._backend.all_reduce(tensor, registered=False)
            else:
                output = self._backend.custom_all_reduce(tensor)
            if output is not None:
                return output

        if is_decode:
            raise RuntimeError("vLLM custom all-reduce rejected an AR decode tensor.")
        return self._nccl_fallback(tensor)

    @contextmanager
    def capture(self) -> Iterator[None]:
        if self._closed or self._backend is None:
            raise RuntimeError("AR custom all-reduce is not initialized.")
        if self._phase_getter() != "ar":
            raise RuntimeError("Custom all-reduce capture is only valid during AR.")
        if self._capture_depth:
            raise RuntimeError("Nested custom all-reduce capture is not supported.")

        self._capture_depth = 1
        try:
            with self._backend.capture():
                yield
        finally:
            self._capture_depth = 0

    def close(self) -> None:
        if self._closed:
            return
        if self._capture_depth:
            raise RuntimeError("Cannot close custom all-reduce during graph capture.")
        backend = self._backend
        self._backend = None
        self._closed = True
        if backend is not None:
            backend.close()


__all__ = ["HunyuanImage3CustomAllReduce", "HunyuanImage3CustomAllReduceConfig"]
