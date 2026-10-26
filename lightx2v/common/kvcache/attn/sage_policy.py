"""Deterministic Sage cache layouts; runtime inference never benchmarks kernels.

The SM120 Eden geometry has offline V-only measurements. Other profiles retain
previous raw-KV choices or compatibility defaults; other architectures use a
write-aligned layout until an offline measurement is added here.
"""

from math import gcd

import torch

# RTX 5090 measurements are kept with the reproduction reports under
# save_results. A profile is used only when its required geometry is legal.
SM120_SEGMENT_TOKENS = {
    "edenlive": 2640,
    "taomate": 4096,
    "streamrender": 1536,
    "minimax_h3": 256,
}

SM120_EDEN_GEOMETRY = (15840, 2640, 5280, 24, 128)
DEFAULT_RAW_SEGMENT_TOKENS = {"taomate": 256, "streamrender": 256, "minimax_h3": 256}


def select_segment_tokens(
    key,
    *,
    write_tokens,
    sink_tokens=0,
    profile="wan",
    value_only=False,
    default_segment_tokens=None,
    write_alignment=(),
    architecture=None,
    capacity=None,
    **unused,
):
    """Return a fixed legal segment length for this cache's write geometry.

    A cache that stores only quantized V cannot recover an overwritten partial
    segment. Its segment therefore divides the capacity, writes, sink and any
    additional physical write boundaries. Raw-value paths may keep tail parts.
    ``default_segment_tokens`` is internal support for existing raw-KV adapters,
    not a configuration option or a request to perform timing.
    """
    capacity = int(key.shape[0] if capacity is None else capacity)
    write_tokens, sink_tokens = int(write_tokens), int(sink_tokens)
    if capacity <= 0 or write_tokens <= 0 or sink_tokens < 0:
        raise ValueError("Sage layouts require positive capacity/writes and a nonnegative sink")
    if architecture is None and key.device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(key.device)
        architecture = major * 10 + minor
    preferred = SM120_SEGMENT_TOKENS.get(profile) if architecture == 120 else None
    if profile == "edenlive" and (capacity, write_tokens, sink_tokens, *key.shape[1:]) != SM120_EDEN_GEOMETRY:
        preferred = None
    if value_only:
        aligned = gcd(capacity, write_tokens, sink_tokens, *(int(offset) for offset in write_alignment))
        return preferred if preferred is not None and aligned % preferred == 0 else aligned
    if preferred is None:
        preferred = DEFAULT_RAW_SEGMENT_TOKENS.get(profile)
    if default_segment_tokens is not None:
        preferred = int(default_segment_tokens)
    if preferred is not None:
        if preferred <= 0:
            raise ValueError("Sage segment lengths must be positive")
        return min(preferred, capacity)
    return gcd(capacity, write_tokens, sink_tokens, *(int(offset) for offset in write_alignment))
