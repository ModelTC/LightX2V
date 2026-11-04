from dataclasses import dataclass
from typing import NamedTuple


class RingSpan(NamedTuple):
    logical_start: int
    physical_start: int
    length: int


@dataclass(frozen=True, slots=True)
class KVCacheUpdate:
    write_start: int
    write_end: int
    attention_start: int
    global_end: int
    sink_tokens: int
    evicted_tokens: int


@dataclass(slots=True)
class RingLayout:
    """Token positions for a fixed sink followed by a ring of recent tokens."""

    capacity: int
    sink_tokens: int = 0
    recent_head: int = 0
    recent_length: int = 0
    active: bool = False

    @property
    def recent_capacity(self) -> int:
        capacity = self.capacity - self.sink_tokens
        if capacity <= 0:
            raise RuntimeError(f"invalid recent capacity: cache_size={self.capacity}, sink={self.sink_tokens}")
        return capacity

    def roll(self, sink_tokens: int, evicted_tokens: int, valid_end: int) -> None:
        if evicted_tokens <= 0:
            return
        if self.active:
            if self.sink_tokens != sink_tokens:
                raise RuntimeError(f"ring sink changed: old={self.sink_tokens}, new={sink_tokens}")
        else:
            if not 0 <= sink_tokens < self.capacity:
                raise RuntimeError(f"invalid sink_tokens={sink_tokens}, cache_size={self.capacity}")
            recent_length = max(0, valid_end - sink_tokens)
            if recent_length > self.capacity - sink_tokens:
                raise RuntimeError("recent_len exceeds recent ring capacity")
            self.active = True
            self.sink_tokens = sink_tokens
            self.recent_head = 0
            self.recent_length = recent_length
        if evicted_tokens > self.recent_length:
            raise RuntimeError(f"cannot evict {evicted_tokens} recent tokens, only {self.recent_length} available")
        self.recent_head = (self.recent_head + evicted_tokens) % self.recent_capacity
        self.recent_length -= evicted_tokens

    def record_write(self, end: int) -> None:
        if self.active and end > self.sink_tokens:
            recent_length = max(self.recent_length, end - self.sink_tokens)
            if recent_length > self.recent_capacity:
                raise RuntimeError(f"recent ring overflow: recent_len={recent_length}, capacity={self.recent_capacity}")
            self.recent_length = recent_length

    def spans(self, start: int, end: int) -> list[RingSpan]:
        if end <= start:
            return []
        if not self.active:
            return [RingSpan(start, start, end - start)]

        spans = []
        if start < self.sink_tokens:
            spans.append(RingSpan(start, start, min(end, self.sink_tokens) - start))
        recent_start = max(start, self.sink_tokens)
        if end > recent_start:
            capacity = self.recent_capacity
            recent_offset = recent_start - self.sink_tokens
            length = end - recent_start
            if recent_offset + length > capacity:
                raise RuntimeError(f"logical range [{start}, {end}) exceeds ring capacity; sink={self.sink_tokens}, capacity={capacity}")
            physical_offset = (self.recent_head + recent_offset) % capacity
            first_length = min(length, capacity - physical_offset)
            spans.append(RingSpan(recent_start, self.sink_tokens + physical_offset, first_length))
            if first_length < length:
                spans.append(RingSpan(recent_start + first_length, self.sink_tokens, length - first_length))
        return spans
