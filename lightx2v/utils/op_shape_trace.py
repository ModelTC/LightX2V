"""Logical-op shape JSONL buffer for block region profiling."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_seq = 0
_buffer: list[dict[str, Any]] = []
_path: Path | None = None
_recording = False


def is_recording() -> bool:
    return _recording


def begin_recording(path: Path | str | None = None) -> None:
    """Clear buffer and start accepting log entries."""
    global _seq, _path, _buffer, _recording
    with _lock:
        _seq = 0
        _buffer = []
        _recording = True
        _path = Path(path) if path is not None else Path("op_trace.jsonl")


def end_recording() -> None:
    global _recording
    _recording = False


def flush(path: Path | str | None = None) -> Path:
    """Write buffered records to JSONL."""
    out = Path(path) if path is not None else _path
    if out is None:
        out = Path("op_trace.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with out.open("w", encoding="utf-8") as f:
            for row in _buffer:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return out


def _write(entry: dict[str, Any]) -> None:
    global _seq
    if not is_recording():
        return
    with _lock:
        _seq += 1
        _buffer.append({"seq": _seq, **entry})


def _gemm_flops(m: int, n: int, k: int) -> float:
    return float(2 * m * n * k)


def _attn_flops(num_heads: int, sq: int, sk: int, head_dim: int) -> float:
    return float(4 * num_heads * sq * sk * head_dim)


def _moe_routed_flops(routed_tokens: int, hidden: int, intermediate: int, fc_schema: str) -> float:
    if fc_schema in {"gate_up_down", "flashinfer_swiglu"}:
        return float(6 * routed_tokens * hidden * intermediate)
    return float(4 * routed_tokens * hidden * intermediate)


def log_gemm(region: str, tag: str, m: int, n: int, k: int) -> None:
    _write(
        {
            "region": region,
            "kind": "GEMM",
            "tag": tag,
            "M": int(m),
            "N": int(n),
            "K": int(k),
            "flops": _gemm_flops(m, n, k),
        }
    )


def log_attn(
    region: str,
    tag: str,
    *,
    batch: int,
    num_heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
) -> None:
    _write(
        {
            "region": region,
            "kind": "ATTN",
            "tag": tag,
            "B": int(batch),
            "H": int(num_heads),
            "Sq": int(seq_q),
            "Sk": int(seq_k),
            "D": int(head_dim),
            "flops": _attn_flops(num_heads, seq_q, seq_k, head_dim),
        }
    )


def log_moe_routed(
    region: str,
    tag: str,
    *,
    num_tokens: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    backend: str = "pytorch_loop",
    fc_schema: str = "fc1_fc2",
    expert_tokens: list[int] | None = None,
) -> None:
    entry = {
        "region": region,
        "kind": "MOE",
        "tag": tag,
        "tokens": int(num_tokens),
        "top_k": int(top_k),
        "hidden": int(hidden),
        "intermediate": int(intermediate),
        "backend": backend,
        "fc_schema": fc_schema,
    }
    if expert_tokens is None:
        routed_tokens = int(num_tokens) * int(top_k)
    else:
        expert_tokens = [int(x) for x in expert_tokens]
        routed_tokens = sum(expert_tokens)
        entry["expert_tokens"] = expert_tokens
        entry["num_experts"] = len(expert_tokens)
    entry["routed_tokens"] = routed_tokens
    entry["flops"] = _moe_routed_flops(routed_tokens, hidden, intermediate, fc_schema)
    _write(entry)
