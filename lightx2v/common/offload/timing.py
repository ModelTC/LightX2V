"""Bounded sampling of block loads, without synchronizing the loading path."""

from time import perf_counter_ns


class TransferTimer:
    def __init__(self, device_module, metadata, limit=160):
        self.metadata = metadata
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
        row = {"block_index": block_index, "phase": phase, **self.metadata[block_index]}
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
