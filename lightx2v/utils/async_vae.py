from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from loguru import logger


def async_vae_decode_enabled(config: dict[str, Any]) -> bool:
    """Read the shared top-level asynchronous decode switch."""
    return bool(config.get("async_vae_decode", False))


def normalize_async_vae_config(config: dict[str, Any]) -> None:
    """Set the top-level decode switch to its explicit or default boolean."""
    config["async_vae_decode"] = async_vae_decode_enabled(config)


class AsyncVAEChunkDecoder:
    """Submit chunk VAE decodes on a side CUDA stream.

    Mirrors LongLive ``streaming-async`` mode (``causal_diffusion_inference.py``):
    after each chunk's diffusion on the default stream, decode is queued on a
    dedicated ``vae_stream`` so the next chunk's denoising can overlap.

    Requirements for real overlap (same as LongLive):
      - VAE must stay resident on GPU for the AR loop (no per-chunk to_cuda/to_cpu).
      - Decode must not call host-side ``cuda.synchronize()`` inside ``submit``.
      - Caller should use ``no_sync_profiling`` when measuring with profilers on.
    """

    def __init__(self, enabled: bool, device: torch.device | str | None = None, on_chunk: Callable | None = None, dispatch_in_worker: bool = False, collect_outputs: bool = True) -> None:
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self._device = torch.device(device) if device is not None else None
        if self._device is not None and self._device.type != "cuda":
            self.enabled = False
        self._stream: torch.cuda.Stream | None = None
        self._prev_vae_done: torch.cuda.Event | None = None
        self._chunks: list[Any] = []
        self._decode_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._num_submitted = 0
        self._sync_decode_ms = 0.0
        self._async_decode_ms = 0.0
        self._submit_wait_ms = 0.0
        self._finish_wait_ms = 0.0
        self._logged_decode_events = 0
        self._vae_decoder: Any | None = None
        self._saved_vae_cpu_offload = False
        self._on_chunk = on_chunk
        self._callback_executor = None
        self._callback_futures = []
        self._dispatch_in_worker = dispatch_in_worker
        self._collect_outputs = collect_outputs
        self._decode_executor = None
        self._decode_futures = []

        if bool(enabled) and not self.enabled:
            logger.warning("[AsyncVAEChunkDecoder] async VAE requested but CUDA is unavailable; falling back to sync decode.")

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        device: torch.device | str | None = None,
        vae_decoder: Any | None = None,
        on_chunk: Callable | None = None,
        dispatch_in_worker: bool = False,
        collect_outputs: bool = True,
    ) -> "AsyncVAEChunkDecoder":
        enabled = async_vae_decode_enabled(config)
        decoder = cls(enabled=enabled, device=device, on_chunk=on_chunk, dispatch_in_worker=dispatch_in_worker, collect_outputs=collect_outputs)
        if vae_decoder is not None:
            decoder.prepare_vae(vae_decoder)
        return decoder

    def prepare_vae(self, vae_decoder: Any) -> None:
        """Keep decoder weights on the GPU for this streaming decode session."""
        self._vae_decoder = vae_decoder
        if self.enabled:
            logger.info("[AsyncVAEChunkDecoder] async VAE decode enabled")
            if hasattr(vae_decoder, "enable_async_decode"):
                vae_decoder.enable_async_decode()
        if not getattr(vae_decoder, "cpu_offload", False) or not torch.cuda.is_available() or (self._device is not None and self._device.type != "cuda"):
            return
        logger.info("[AsyncVAEChunkDecoder] keeping offloaded VAE weights on GPU until the decode session finishes")
        self._saved_vae_cpu_offload = True
        vae_decoder.cpu_offload = False
        try:
            if hasattr(vae_decoder, "to_cuda"):
                vae_decoder.to_cuda()
        except BaseException:
            self._restore_vae()
            raise

    def _restore_vae(self) -> None:
        vae_decoder = self._vae_decoder
        if vae_decoder is None or not self._saved_vae_cpu_offload:
            return
        vae_decoder.cpu_offload = True
        if hasattr(vae_decoder, "to_cpu"):
            vae_decoder.to_cpu()
        self._saved_vae_cpu_offload = False

    @property
    def is_async(self) -> bool:
        return self.enabled

    def _resolve_device(self, args: tuple[Any, ...]) -> torch.device:
        if self._device is not None:
            return self._device
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "cuda":
                self._device = arg.device
                return arg.device
        self._device = torch.device("cuda")
        return self._device

    def _ensure_stream(self, device: torch.device) -> torch.cuda.Stream:
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        return self._stream

    def _sync_if_cuda(self, args: tuple[Any, ...]) -> None:
        if not torch.cuda.is_available():
            return
        device = self._resolve_device(args)
        if device.type != "cuda":
            return
        torch.cuda.synchronize(device)

    def submit(self, decode_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._num_submitted += 1
        if not self.enabled:
            self._sync_if_cuda(args)
            t0 = time.perf_counter()
            output = decode_fn(*args, **kwargs)
            if self._collect_outputs:
                self._chunks.append(output)
            self._sync_if_cuda(args)
            chunk_ms = (time.perf_counter() - t0) * 1000.0
            self._sync_decode_ms += chunk_ms
            logger.info(
                "[AsyncVAEChunkDecoder] sync VAE chunk {}/{} decode={:.6f} seconds",
                self._num_submitted,
                self._num_submitted,
                chunk_ms / 1000.0,
            )
            if self._on_chunk is not None:
                self._on_chunk(output, self._num_submitted - 1)
            return

        device = self._resolve_device(args)
        stream = self._ensure_stream(device)
        # LongLive: diffusion_done.record() then vae_stream.wait_event(diffusion_done)
        diffusion_done = torch.cuda.Event()
        diffusion_done.record(torch.cuda.current_stream(device))
        if self._dispatch_in_worker:
            for value in list(args) + list(kwargs.values()):
                if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                    value.record_stream(stream)
            if len(self._decode_futures) >= 2:
                self._decode_futures.pop(0).result()
            if self._decode_executor is None:
                self._decode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vae-decode")
            self._decode_futures.append(self._decode_executor.submit(self._decode_chunk, decode_fn, args, kwargs, diffusion_done, self._num_submitted - 1))
            return

        # LongLive waits for the previous chunk's VAE before queueing the next
        # decode (stateful cached VAE). Measure exposed wait here; do not block
        # the next segment's DiT — that overlap happens after submit() returns.
        if self._prev_vae_done is not None:
            t0 = time.perf_counter()
            self._prev_vae_done.synchronize()
            wait_ms = (time.perf_counter() - t0) * 1000.0
            self._submit_wait_ms += wait_ms
            self._log_new_async_chunks(wait_ms, wait_kind="submit_wait")

        with torch.cuda.stream(stream), torch.no_grad():
            stream.wait_event(diffusion_done)
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            decode_start.record(stream)
            output = decode_fn(*args, **kwargs)
            decode_end.record(stream)
            if self._collect_outputs:
                self._chunks.append(output)
            self._decode_events.append((decode_start, decode_end))
            self._prev_vae_done = torch.cuda.Event()
            self._prev_vae_done.record(stream)
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                value.record_stream(stream)
        if isinstance(output, torch.Tensor) and output.device.type == "cuda":
            output.record_stream(stream)
        if self._on_chunk is not None:
            if self._callback_executor is None:
                self._callback_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vae-chunks")
            self._callback_futures.append(self._callback_executor.submit(self._deliver_chunk, output, self._num_submitted - 1, decode_end))

    def _deliver_chunk(self, output, index, done):
        done.synchronize()
        with torch.cuda.device(output.device):
            self._on_chunk(output, index)

    def _decode_chunk(self, decode_fn, args, kwargs, diffusion_done, index):
        with torch.cuda.device(self._stream.device), torch.cuda.stream(self._stream), torch.no_grad():
            self._stream.wait_event(diffusion_done)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self._stream)
            output = decode_fn(*args, **kwargs)
            end.record(self._stream)
            if self._collect_outputs:
                self._chunks.append(output)
            if self._collect_outputs:
                self._decode_events.append((start, end))
            else:
                end.synchronize()
                self._async_decode_ms += start.elapsed_time(end)
            if self._on_chunk is not None:
                end.synchronize()
                self._on_chunk(output, index)

    def synchronize(self) -> None:
        """Join queued chunks without ending the current causal decode request."""
        first_error = None
        pending, self._decode_futures = self._decode_futures, []
        for future in pending:
            try:
                future.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if self._stream is not None:
            try:
                self._stream.synchronize()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        callbacks, self._callback_futures = self._callback_futures, []
        for future in callbacks:
            try:
                future.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def finish(self) -> list[Any]:
        started = time.perf_counter()
        first_error = None
        try:
            self.synchronize()
            self._finish_wait_ms += (time.perf_counter() - started) * 1000.0
            self._log_timing()
            return self._chunks
        except BaseException as error:
            first_error = error
            raise
        finally:
            cleanup_error = None
            for executor in (self._decode_executor, self._callback_executor):
                if executor is not None:
                    try:
                        executor.shutdown(wait=True)
                    except BaseException as error:
                        if cleanup_error is None:
                            cleanup_error = error
            self._decode_executor = None
            self._callback_executor = None
            self._chunks = []
            self._prev_vae_done = None
            self._decode_events = []
            self._num_submitted = 0
            self._sync_decode_ms = 0.0
            self._async_decode_ms = 0.0
            self._submit_wait_ms = 0.0
            self._finish_wait_ms = 0.0
            self._logged_decode_events = 0
            try:
                self._restore_vae()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            if first_error is None and cleanup_error is not None:
                raise cleanup_error

    def _log_new_async_chunks(self, exposed_wait_ms: float, wait_kind: str) -> None:
        if not self.enabled:
            return
        while self._logged_decode_events < len(self._decode_events):
            start, end = self._decode_events[self._logged_decode_events]
            decode_ms = start.elapsed_time(end)
            # The synchronize wait is the portion of this decode that was still
            # visible to the caller; the rest was hidden behind later DiT work.
            visible_ms = exposed_wait_ms if self._logged_decode_events == len(self._decode_events) - 1 else 0.0
            overlapped_ms = max(decode_ms - visible_ms, 0.0)
            overlap_ratio = overlapped_ms / decode_ms if decode_ms > 0 else 0.0
            logger.info(
                "[AsyncVAEChunkDecoder] async VAE chunk {}/{} gpu_decode={:.2f} ms, {}={:.2f} ms, overlapped={:.2f} ms ({:.1%})",
                self._logged_decode_events + 1,
                self._num_submitted,
                decode_ms,
                wait_kind,
                visible_ms,
                overlapped_ms,
                overlap_ratio,
            )
            self._logged_decode_events += 1

    def _log_timing(self) -> None:
        if self._num_submitted == 0:
            return
        if not self.enabled:
            logger.info(
                "[AsyncVAEChunkDecoder] sync VAE decode: chunks={}, total={:.2f} ms, avg={:.2f} ms",
                self._num_submitted,
                self._sync_decode_ms,
                self._sync_decode_ms / max(self._num_submitted, 1),
            )
            return

        decode_ms = self._async_decode_ms
        for start, end in self._decode_events:
            decode_ms += start.elapsed_time(end)
        exposed_ms = self._submit_wait_ms + self._finish_wait_ms
        overlapped_ms = max(decode_ms - exposed_ms, 0.0)
        overlap_ratio = overlapped_ms / decode_ms if decode_ms > 0 else 0.0
        logger.info(
            "[AsyncVAEChunkDecoder] async VAE decode: chunks={}, gpu_decode={:.2f} ms, submit_wait={:.2f} ms, finish_wait={:.2f} ms, exposed={:.2f} ms, overlapped={:.2f} ms ({:.1%})",
            self._num_submitted,
            decode_ms,
            self._submit_wait_ms,
            self._finish_wait_ms,
            exposed_ms,
            overlapped_ms,
            overlap_ratio,
        )
