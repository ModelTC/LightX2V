"""Bounded sampling of block loads, without synchronizing the loading path."""

from contextlib import contextmanager, nullcontext
from time import perf_counter_ns

import torch


class TransferTimer:
    def __init__(self, device_module, metadata, limit=160):
        self.metadata = metadata
        self.context = {}
        self.events = [(device_module.Event(enable_timing=True), device_module.Event(enable_timing=True)) for _ in range(limit)]
        # Materialize lazy device events before the measured request.
        for start, end in self.events:
            start.record()
            end.record()
        device_module.synchronize()
        self.pending = []

    def start(self, block_index, phase, stream):
        if len(self.pending) == len(self.events):
            return None
        start, end = self.events[len(self.pending)]
        start.record(stream)
        row = {**self.context, "block_index": block_index, "phase": phase, **self.metadata[block_index]}
        ticket = (start, end, row, stream, perf_counter_ns())
        self.pending.append(ticket)
        return ticket

    def stop(self, ticket):
        if ticket is not None:
            _, end, row, stream, start_ns = ticket
            row["submit_ms"] = (perf_counter_ns() - start_ns) / 1e6
            end.record(stream)

    def collect(self):
        """Call after the measured request has completed on the device."""
        rows = []
        for start, end, row, _, _ in self.pending:
            if not end.query():
                raise RuntimeError("Synchronize the measured request before collecting transfer times")
            row = {**row, "load_stream_ms": start.elapsed_time(end)}
            row["effective_GBps"] = row["h2d_bytes"] / (row["load_stream_ms"] * 1e6) if row["load_stream_ms"] else None
            rows.append(row)
        return rows


class InferenceTimer:
    """Diagnostic stream spans and host waits; nested spans must not be added."""

    def __init__(self, device_module, capacity):
        self.module = device_module
        self.events = [(device_module.Event(enable_timing=True), device_module.Event(enable_timing=True)) for _ in range(capacity)]
        for start, end in self.events:
            start.record()
            end.record()
        device_module.synchronize()
        self.context = {"step": 0, "branch": "", "block_index": None}
        self.pending = []
        self.event_index = 0
        self.enabled = False
        self.profiling = False

    @contextmanager
    def measure(self, stage, device_timing=True):
        if not self.enabled:
            yield
            return
        row = {**self.context, "stage": stage}
        start = end = None
        stream = self.module.current_stream()
        if device_timing:
            start, end = self.events[self.event_index]
            self.event_index += 1
        marker = torch.profiler.record_function(f"wan/{stage}") if self.profiling else nullcontext()
        with marker:
            if start is not None:
                start.record(stream)
            begin = perf_counter_ns()
            try:
                yield
            finally:
                row["host_ms"] = (perf_counter_ns() - begin) / 1e6
                if end is not None:
                    end.record(stream)
                self.pending.append((start, end, row))

    def collect(self):
        """Read only after the diagnostic step's device synchronization."""
        rows = []
        for start, end, row in self.pending:
            if end is not None and not end.query():
                raise RuntimeError("Synchronize the diagnostic step before collecting spans")
            rows.append({**row, "device_span_ms": start.elapsed_time(end) if start is not None else None})
        self.pending.clear()
        self.event_index = 0
        return rows
