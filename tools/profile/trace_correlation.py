"""Chrome trace kernel correlation helpers for region_event_trace and peers.

Heuristic, model-agnostic utilities that evolve as new profiler exports / kernel
naming patterns appear.  Prefer extending this module over touching infer code.

Expected extension points:
  - add kernel-name shortening rules in ``simplify_kernel_name``;
  - add trace-id fields in ``build_correlation_indexes`` / ``related_events_for_kernel``
    if future profiler exports stop using ``correlation`` or ``External id``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence


def simplify_kernel_name(name: str) -> str:
    """Heuristically shorten long CUDA kernel names for layer trace reports."""
    original = name

    if name.startswith("void "):
        name = name[5:]

    if name.startswith("at::native::(anonymous namespace)::"):
        name = name[len("at::native::(anonymous namespace)::") :]
    elif name.startswith("at::native::"):
        name = name[len("at::native::") :]

    for prefix in ("flashinfer::", "cutlass::"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    m = re.search(r"vectorized_elementwise_kernel<\d+,\s*([^,]+)", original)
    if m:
        functor = m.group(1)
        if "Gelu" in original:
            return "vectorized_elementwise_kernel(Gelu)"
        if "CUDAFunctor_add" in original:
            return "vectorized_elementwise_kernel(Add)"
        if "MulFunctor" in original:
            return "vectorized_elementwise_kernel(Mul)"
        return f"vectorized_elementwise_kernel({functor})"

    if "elementwise_kernel" in original and "gpu_kernel_impl_nocast" in original:
        if "MulFunctor" in original:
            return "elementwise_kernel(Mul)"
        if "CUDAFunctor_add" in original:
            return "elementwise_kernel(add)"
        if "GeluCUDAKernelImpl" in original:
            return "elementwise_kernel(Gelu)"
        return "elementwise_kernel"

    m = re.search(r"FlashAttn\w+", original)
    if m:
        return f"cutlass::{m.group(0)}"

    m = re.search(r"device_kernel<(cutlass_3x_gemm_sm\d+_\w+)", original)
    if m:
        return f"cutlass::{m.group(1)}"

    if "qk_int8_sv_f8_attn_kernel" in original:
        return "qk_int8_sv_f8_attn_kernel"

    if "<" in name:
        name = name[: name.index("<")]

    if len(name) > 70:
        name = name[:67] + "..."

    return name


def build_correlation_indexes(
    events: Sequence[dict],
) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    """Index trace events by correlation / External id for kernel↔cpu_op linking."""
    by_correlation: dict[int, list[dict]] = defaultdict(list)
    by_external_id: dict[int, list[dict]] = defaultdict(list)
    for ev in events:
        args = ev.get("args", {})
        corr = args.get("correlation")
        ext_id = args.get("External id")
        if corr is not None:
            by_correlation[int(corr)].append(ev)
        if ext_id is not None:
            by_external_id[int(ext_id)].append(ev)
    return by_correlation, by_external_id


def related_events_for_kernel(
    ev: dict,
    by_correlation: dict[int, list[dict]],
    by_external_id: dict[int, list[dict]],
) -> list[dict]:
    """Walk correlation links from a GPU kernel to sibling trace events."""
    args = ev.get("args", {})
    corr = args.get("correlation")
    ext_id = args.get("External id")

    related: list[dict] = []
    seen: set[int] = set()

    def add_event(candidate: dict) -> None:
        uid = id(candidate)
        if uid not in seen:
            seen.add(uid)
            related.append(candidate)

    if ext_id is not None:
        for sibling in by_external_id.get(int(ext_id), []):
            add_event(sibling)
    if corr is not None:
        for sibling in by_correlation.get(int(corr), []):
            add_event(sibling)

    for candidate in list(related):
        sibling_args = candidate.get("args", {})
        sibling_corr = sibling_args.get("correlation")
        sibling_ext = sibling_args.get("External id")
        if sibling_corr is not None:
            for sibling in by_correlation.get(int(sibling_corr), []):
                add_event(sibling)
        if sibling_ext is not None:
            for sibling in by_external_id.get(int(sibling_ext), []):
                add_event(sibling)

    return related
