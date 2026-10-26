"""Optional warmup search for Sage segment sizes on scratch storage."""

from dataclasses import dataclass
from statistics import median

import torch
from loguru import logger


@dataclass(frozen=True)
class SageSegmentChoice:
    segment_tokens: int
    timings_ms: tuple[tuple[int, float], ...]


_SEGMENT_CHOICES = {}


def validate_segment_candidates(candidates):
    """Require a reproducible, explicit set of positive segment lengths."""
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("sage_segment_candidates must be a nonempty list of positive integers")
    if any(type(tokens) is not int or tokens <= 0 for tokens in candidates):
        raise ValueError("sage_segment_candidates must contain only positive integers")
    if len(set(candidates)) != len(candidates):
        raise ValueError("sage_segment_candidates must not contain duplicates")
    return tuple(candidates)


def legal_segment_candidates(candidates, capacity, write_tokens, sink_tokens, write_alignment=(), *, value_only=False):
    """Keep raw-KV tails; require whole physical segments for encoded-only V."""
    boundaries = (capacity, write_tokens, sink_tokens, *write_alignment)
    return tuple(
        tokens
        for tokens in candidates
        if tokens <= capacity and all(offset % tokens == 0 for offset in write_alignment) and (not value_only or all(offset % tokens == 0 for offset in boundaries))
    )


def _benchmark_segment(key, segment_tokens, *, write_tokens, query_tokens, visible_tokens, sink_tokens, logical_start, freqs, rotary_dim, rope_layout, rope_fma, refresh_keys):
    from ..kernel import store_kv_ring
    from .sage import allocate_sage_kv_cache, sage_kv_attention, update_sage_kv_cache

    capacity, heads, head_dim = key.shape
    device = key.device
    with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
        scratch_key = torch.randn(key.shape, device=device, dtype=key.dtype)
        scratch_value = torch.randn_like(scratch_key)
        query = torch.randn((query_tokens, heads, head_dim), device=device, dtype=key.dtype)
        current_key = torch.randn((write_tokens, heads, head_dim), device=device, dtype=key.dtype)
        current_value = torch.randn_like(current_key)
    end = visible_tokens + max(logical_start, 0)
    metadata = torch.tensor((end - write_tokens, min(sink_tokens, end), 0), device=device, dtype=torch.int32)
    state = allocate_sage_kv_cache(scratch_key, segment_tokens)

    def run(*, refresh=False):
        store_kv_ring(current_key, current_value, scratch_key, scratch_value, metadata)
        if refresh:
            state.key_valid.zero_()
        update_sage_kv_cache(
            scratch_key,
            scratch_value,
            metadata,
            state,
            write_tokens,
            visible_tokens,
            freqs=freqs,
            logical_start=logical_start,
            rotary_dim=rotary_dim,
            rope_layout=rope_layout,
            rope_fma=rope_fma,
        )
        return sage_kv_attention(query, state, visible_tokens)

    # One re-anchor followed by three denoising reuses is a representative
    # four-step raw-K workload. Replay must reset K validity for every cycle.
    steps = 4 if refresh_keys else 1

    def run_cycle():
        output = run(refresh=refresh_keys)
        for _ in range(steps - 1):
            output = run()
        return output

    for _ in range(3):
        run_cycle()
    torch.cuda.current_stream(device).synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_cycle()
    for _ in range(3):
        graph.replay()
    samples = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        start.record()
        for _ in range(10):
            graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / (10 * steps))
    del output, graph
    return median(samples)


def select_sage_segment_tokens(
    key,
    candidates,
    *,
    baseline,
    write_tokens,
    sink_tokens=0,
    visible_tokens=None,
    query_tokens=None,
    logical_start=-1,
    freqs=None,
    rotary_dim=None,
    rope_layout="interleaved",
    rope_fma=True,
    refresh_keys=False,
    write_alignment=(),
    value_only=False,
    capacity=None,
    **unused,
):
    """Measure requested candidates once per geometry, falling back to policy."""
    candidates = validate_segment_candidates(candidates)
    capacity = key.shape[0] if capacity is None else int(capacity)
    baseline = int(baseline)
    if key.device.type != "cuda":
        raise ValueError("Sage segment search requires a CUDA key cache")
    if capacity != key.shape[0]:
        raise ValueError("Sage segment search requires key storage matching its cache capacity")
    visible_tokens = capacity if visible_tokens is None else visible_tokens
    query_tokens = max(1, min(write_tokens, visible_tokens)) if query_tokens is None else query_tokens
    rotary_dim = key.shape[-1] if rotary_dim is None else rotary_dim
    choices = legal_segment_candidates(candidates, capacity, write_tokens, sink_tokens, write_alignment, value_only=value_only)
    if not choices:
        logger.warning("Sage segment warmup: no legal candidate for shape={}, retaining baseline={}", tuple(key.shape), baseline)
        return SageSegmentChoice(baseline, ())
    signature = (
        key.device,
        torch.cuda.get_device_capability(key.device),
        tuple(key.shape),
        key.dtype,
        choices,
        write_tokens,
        query_tokens,
        sink_tokens,
        visible_tokens,
        logical_start,
        None if freqs is None else (tuple(freqs.shape), freqs.dtype),
        rotary_dim,
        rope_layout,
        rope_fma,
        refresh_keys,
        value_only,
    )
    cached = _SEGMENT_CHOICES.get(signature)
    if cached is not None:
        return cached
    if len(choices) == 1:
        result = SageSegmentChoice(choices[0], ())
        timings = ()
    else:
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Sage segment search must finish during warmup, before compile and CUDA Graph capture")
        measured = []
        with torch.cuda.device(key.device):
            for tokens in choices:
                try:
                    elapsed = _benchmark_segment(
                        key,
                        tokens,
                        write_tokens=write_tokens,
                        query_tokens=query_tokens,
                        visible_tokens=visible_tokens,
                        sink_tokens=sink_tokens,
                        logical_start=logical_start,
                        freqs=freqs,
                        rotary_dim=rotary_dim,
                        rope_layout=rope_layout,
                        rope_fma=rope_fma,
                        refresh_keys=refresh_keys,
                    )
                except torch.OutOfMemoryError:
                    logger.warning("Sage segment warmup: skipping {} tokens for shape={} because scratch storage exceeds available CUDA memory", tokens, tuple(key.shape))
                    torch.cuda.empty_cache()
                    continue
                measured.append((tokens, elapsed))
        timings = tuple(measured)
        if timings:
            result = SageSegmentChoice(min(timings, key=lambda item: item[1])[0], timings)
        else:
            logger.warning("Sage segment warmup: no candidate fits scratch storage for shape={}, retaining baseline={} without a timing result", tuple(key.shape), baseline)
            result = SageSegmentChoice(baseline, ())
    logger.info(
        "Sage segment warmup: device={}, shape={}, visible={}, write={}, rope={}, selected={}, timings_ms={}",
        key.device,
        tuple(key.shape),
        visible_tokens,
        write_tokens,
        freqs is not None,
        result.segment_tokens,
        dict(timings),
    )
    _SEGMENT_CHOICES[signature] = result
    return result
