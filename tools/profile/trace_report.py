"""模型范围时间线报告：显式关联、区间并集、同次核心计算量。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from tools.profile.operator_facts import GEMM_OPERATORS, call_precision, demangle, ideal_gemm_bytes, normalize_dtype

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
CPU_CATEGORIES = {"cpu_op", "user_annotation", "cuda_runtime", "cuda_driver", "runtime"}


def union_us(intervals):
    """合并重叠区间，得到实际覆盖的时间；输入和返回单位均为微秒。"""
    total = 0.0
    right = float("-inf")
    for start, end in sorted(intervals):
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


def intervals(events, start=float("-inf"), end=float("inf")):
    return [(max(start, e["ts"]), min(end, e["ts"] + e["dur"])) for e in events if e["ts"] < end and e["ts"] + e["dur"] > start]


def cpu_self_times(events, indices):
    """同线程父区间扣所有被包含子区间的并集；不扣异步GPU时长。

    按起点排序，增量合并每个父范围的子区间。保留仍活动的交叉兄弟，
    所以A中重叠的B/C只扣一次；每个父范围只扣包含在自身内的区间。
    """
    threads = defaultdict(list)
    covered, right = defaultdict(float), {}
    for index in indices:
        event = events[index]
        threads[(event.get("pid"), event.get("tid"))].append(index)
    for thread in threads.values():
        active = []
        for index in sorted(thread, key=lambda i: (events[i]["ts"], -events[i]["dur"], i)):
            event = events[index]
            start, end = event["ts"], event["ts"] + event["dur"]
            active = [parent for parent in active if events[parent]["ts"] + events[parent]["dur"] > start]
            for parent in active:
                if end <= events[parent]["ts"] + events[parent]["dur"]:
                    covered[parent] += max(0, end - max(start, right.get(parent, start)))
                    right[parent] = max(right.get(parent, end), end)
            active.append(index)
    return {index: max(0, events[index]["dur"] - covered[index]) for index in indices}


def _gap_diagnostics(events, cpu, selected_gpu, links, devices):
    """同设备可见空隙与CPU同步等待交叉定位；时间相交不是因果证明。"""
    syncs = []
    for i in sorted(cpu):
        e = events[i]
        if e.get("cat") not in {"cuda_runtime", "cuda_driver", "runtime"} or "synchroniz" not in e["name"].lower():
            continue
        parents = [
            j
            for j in cpu
            if j != i
            and events[j].get("cat") in {"cpu_op", "user_annotation"}
            and (events[j].get("pid"), events[j].get("tid")) == (e.get("pid"), e.get("tid"))
            and events[j]["ts"] <= e["ts"]
            and e["ts"] + e["dur"] <= events[j]["ts"] + events[j]["dur"]
        ]
        syncs.append(
            {
                "event_index": i,
                "name": e["name"],
                "cpu_ms": e["dur"] / 1000,
                "ts_us": e["ts"],
                "pid": e.get("pid"),
                "tid": e.get("tid"),
                "target_device": e.get("args", {}).get("device"),
                "target_stream": e.get("args", {}).get("stream"),
                "callers": [events[j]["name"] for j in sorted(parents, key=lambda j: -events[j]["dur"])],
            }
        )
    result = {}
    for device in devices:
        selected = [i for i in selected_gpu if str(events[i].get("args", {}).get("device", "unknown")) == device]
        start = min(events[i]["ts"] for i in selected)
        end = max(events[i]["ts"] + events[i]["dur"] for i in selected)
        visible = sorted((i for i in links if str(events[i].get("args", {}).get("device", "unknown")) == device), key=lambda i: events[i]["ts"])
        right, previous = start, None
        gaps = []
        for i in visible:
            e = events[i]
            left, stop = max(start, e["ts"]), min(end, e["ts"] + e["dur"])
            if left >= stop:
                continue
            if left > right:
                gaps.append({"start_us": right, "end_us": left, "duration_ms": (left - right) / 1000, "previous_gpu_event": previous, "next_gpu_event": i})
            if stop > right:
                right, previous = stop, i
        if right < end:
            gaps.append({"start_us": right, "end_us": end, "duration_ms": (end - right) / 1000, "previous_gpu_event": previous, "next_gpu_event": None})
        overlaps = []
        sync_rows = []
        for sync in syncs:
            if sync["target_device"] is not None and str(sync["target_device"]) != device:
                continue
            e = events[sync["event_index"]]
            hits = [(max(g["start_us"], e["ts"]), min(g["end_us"], e["ts"] + e["dur"])) for g in gaps if g["start_us"] < e["ts"] + e["dur"] and g["end_us"] > e["ts"]]
            overlaps.extend(hits)
            sync_rows.append({**sync, "gap_overlap_ms": union_us(hits) / 1000, "window_overlap_ms": max(0, min(end, e["ts"] + e["dur"]) - max(start, e["ts"])) / 1000})
        for gap in gaps:
            gap["sync_event_indices"] = [s["event_index"] for s in sync_rows if s["ts_us"] < gap["end_us"] and s["ts_us"] + s["cpu_ms"] * 1000 > gap["start_us"]]
        result[device] = {
            "window_start_us": start,
            "window_end_us": end,
            "gaps": sorted(gaps, key=lambda g: -g["duration_ms"]),
            "synchronizations": sorted(sync_rows, key=lambda s: -s["cpu_ms"]),
            "sync_gap_overlap_ms": union_us(overlaps) / 1000,
        }
    return result


def _id(event, key):
    value = event.get("args", {}).get(key)
    return value if type(value) is int and value >= 0 else None


def associate(events):
    """沿 correlation / External id 建立 GPU事件 → CPU事件下标集合。

    correlation 连接 CUDA launch 与 GPU 活动，External id 连接 launch 与上层调用。
    跨进程 ID 冲突保留为未归属；不按时间距离猜。下标始终指向原始事件数组。
    graph launch 可能连到多个 kernel，但不能恢复图内未记录的算子。
    """
    correlations, external = defaultdict(list), defaultdict(list)
    for index, event in enumerate(events):
        if event.get("cat") not in CPU_CATEGORIES or event.get("ph") != "X":
            continue
        correlation, ext = _id(event, "correlation"), _id(event, "External id")
        if correlation is not None:
            correlations[correlation].append(index)
        if ext is not None:
            external[ext].append(index)
    links = {}
    ambiguous = []
    for index, event in enumerate(events):
        if event.get("cat") not in GPU_CATEGORIES or event.get("ph") != "X":
            continue
        related = set(correlations.get(_id(event, "correlation"), []))
        related.update(external.get(_id(event, "External id"), []))
        for cpu_index in list(related):
            related.update(external.get(_id(events[cpu_index], "External id"), []))
        if len({events[i].get("pid") for i in related}) > 1:
            ambiguous.append(index)
            related = set()
        links[index] = related
    return links, ambiguous


def _inside(inner, outer):
    return (inner.get("pid"), inner.get("tid")) == (outer.get("pid"), outer.get("tid")) and outer["ts"] <= inner["ts"] and inner["ts"] + inner["dur"] <= outer["ts"] + outer["dur"]


def _kernel_family(name, operators=()):
    """名称只分类，不据此计算shape/FLOPs；compiler pointwise始终是辅助。"""
    text = demangle(name).lower().split("<", 1)[0]
    if any(
        word in text
        for word in (
            "triton_poi",
            "triton_red",
            "elementwise",
            "quantize",
            "quantization",
            "softmax",
            "norm",
            "permute",
            "transpose",
            "reduce_kernel",
            "nchwtonhwc",
            "nhwctonchw",
            "copy",
            "gather",
            "scatter",
            "fill",
        )
    ):
        return "auxiliary"
    if any("convolution" in operator for operator in operators):
        return "convolution"
    if set(operators) & GEMM_OPERATORS or any(word in text for word in ("gemm", "gemv", "matmul", "cutlass", "mma_kernel")):
        return "gemm"
    if any(word in text for word in ("flash_fwd", "attn", "attention", "sage_fwd", "fwd_kernel")):
        return "attention"
    if "conv" in text:
        return "convolution"
    if any(word in text for word in ("moe", "expert")):
        return "moe"
    return "unknown"


def short_name(name):
    text = demangle(name).removeprefix("void ").replace("(anonymous namespace)::", "")
    gemm = re.search(r"cutlass::device_kernel<(cutlass_\w+)", text)
    if gemm:
        return gemm[1]
    functors = re.findall(r"(?:native::|internal::)(?:CUDAFunctor_(\w+)|(\w+)Functor)<", text)
    operation = " · " + next(word for word in functors[-1] if word) if functors else " · copy" if "direct_copy_kernel" in text else ""
    text = text.split("<", 1)[0].split("(", 1)[0] + operation
    return text if len(text) <= 84 else text[:81] + "…"


def _bound_calls(events, cpu, links, selected_gpu, samples):
    """真实标记 → 同线程CPU子调用 → correlation关联GPU；多kernel不丢算量。"""
    by_marker = defaultdict(list)
    for index, event in enumerate(events):
        if event.get("ph") == "X" and event.get("cat") == "user_annotation":
            by_marker[event["name"]].append(index)
    candidates, gaps = {}, []
    for row in samples:
        if row.get("kind") != "core_op_compute_stats" and not row.get("trace_marker"):
            continue  # backend区域事实已嵌入其core owner，不要求独立标记再计一次。
        if row.get("parent_core_call") is not None:
            continue
        matches = by_marker.get(row.get("trace_marker"), [])
        if len(matches) != 1:
            gaps.append({"marker": row.get("trace_marker"), "reason": "marker_missing_or_not_unique"})
            continue
        index = matches[0]
        if index not in cpu:
            continue  # 唯一标记存在于其他scope，不是本范围的缺失证据。
        candidates[index] = row
    owners = defaultdict(list)
    for gpu in selected_gpu:
        for index in candidates:
            if any(_inside(events[host], events[index]) for host in links[gpu]):
                owners[gpu].append(index)
    assigned = defaultdict(list)
    for gpu, indices in owners.items():
        # 组合父子只归最内层实际调用；同层歧义不按耗时或顺序猜。
        leaves = [i for i in indices if not any(i != j and _inside(events[j], events[i]) for j in indices)]
        if len(leaves) == 1:
            assigned[leaves[0]].append(gpu)
        else:
            gaps.append({"gpu_event": gpu, "reason": "multiple_core_owners"})
    rows = []
    for index, fact in candidates.items():
        if any(index != child and assigned[child] and _inside(events[child], events[index]) for child in candidates):
            # 父work覆盖子调用，不能除以剥走子GPU后剩下的padding/cat时间。
            # 只发布叶子计算量；父自身GPU活动仍保留于全局核心/辅助表。
            continue
        all_gpu = assigned[index]
        kernels = [i for i in all_gpu if events[i]["cat"] == "kernel"]
        if not kernels:
            gaps.append({"marker": fact["trace_marker"], "reason": "no_exclusive_gpu_kernel"})
            continue
        family = fact.get("operator_family")
        if family in {"dense_attention", "sparse_attention"}:
            family = "attention"
        compute_gpu = None
        if family == "moe":
            matches = by_marker.get(fact.get("compute_trace_marker"), [])
            if len(matches) == 1 and matches[0] in cpu and _inside(events[matches[0]], events[index]):
                compute_gpu = [i for i in all_gpu if any(_inside(events[host], events[matches[0]]) for host in links[i])]
            if not compute_gpu:
                compute_gpu = None
                gaps.append({"marker": fact["trace_marker"], "reason": "expert_compute_range_missing_or_invalid"})
        hints = {i: _kernel_family(events[i]["name"], {events[j]["name"] for j in links[i]}) for i in kernels}
        main = []
        for i in kernels:
            if family == "moe" and (compute_gpu is None or i not in compute_gpu):
                continue
            hint = hints[i]
            # Conv/MoE可使用GEMM实现；cudnn前缀本身不能证明是卷积主体。
            if hint == family or (family in {"moe", "convolution"} and hint == "gemm"):
                main.append(i)
        timed = main
        basis = "compute_kernel_group"
        if fact.get("auxiliary_gemm_present"):
            timed, basis = all_gpu, "whole_operator_including_lora"
        elif family == "moe" and compute_gpu is not None and (not main or any(hints[i] == "unknown" for i in kernels if i in compute_gpu)):
            timed, basis = compute_gpu, "compute_range_including_auxiliary"
        elif not main:
            # 未识别主体名也不丢实际shape；以完整算子覆盖为分母，清楚注明。
            timed, basis = all_gpu, "whole_operator_including_auxiliary"
        devices = {str(events[i].get("args", {}).get("device", "unknown")) for i in all_gpu}
        if len(devices) != 1:
            gaps.append({"marker": fact["trace_marker"], "reason": "multiple_devices_in_one_core_call"})
            continue
        work = fact.get("work") or {}
        flops = work.get("logical_flops", work.get("selected_logical_flops"))
        elapsed = union_us(intervals([events[i] for i in timed])) / 1000
        rate_available = elapsed > 0 and not fact.get("auxiliary_gemm_present") and not (family == "moe" and compute_gpu is None)
        input_dtypes = fact.get("actual_input_dtypes")
        backend_precision = fact.get("precision") or {}
        if input_dtypes is None:
            input_dtypes = {key.removesuffix("_dtype"): value for key, value in backend_precision.items() if key in {"input_dtype", "weight_dtype"}}
        output_dtype = fact.get("actual_output_dtype", backend_precision.get("output_dtype"))
        # backend直接行允许torch dtype字符串；与core行在消费边界统一一次。
        input_dtypes = {key: normalize_dtype(value) for key, value in input_dtypes.items()}
        output_dtype = normalize_dtype(output_dtype)
        input_dtype = next(iter(input_dtypes.values()), None)
        precision_main = [] if family == "moe" and basis != "compute_kernel_group" else main
        names = {events[j]["name"] for i in precision_main for j in links[i]}
        known = call_precision(names, [events[i]["name"] for i in precision_main], input_dtype, fact.get("execution_context", {}))
        shape = fact.get("shape") or {}
        ideal_bytes = None
        if family == "gemm" and all(key in shape for key in ("m", "n", "k")) and known.get("input_dtype") in {"bf16", "fp16", "fp32", "fp8", "fp4"}:
            ideal_bytes = ideal_gemm_bytes({"batch": 1, **shape}, known["input_dtype"], output_dtype)
        rows.append(
            {
                "marker": fact["trace_marker"],
                "event_index": index,
                "family": family,
                "backend": fact.get("backend", fact.get("observed_backend")),
                "weight_name": fact.get("weight_name"),
                "device": next(iter(devices)),
                "shape": shape,
                "work": work,
                "logical_flops": flops,
                "dense_equivalent_flops": work.get("dense_equivalent_flops"),
                "gpu_ms": elapsed,
                "operator_gpu_ms": union_us(intervals([events[i] for i in all_gpu])) / 1000,
                "time_basis": basis,
                "compute_tflops": flops / elapsed / 1e9 if flops is not None and rate_available else None,
                "dense_equivalent_tflops": work["dense_equivalent_flops"] / elapsed / 1e9 if work.get("dense_equivalent_flops") is not None and elapsed > 0 else None,
                "precision": {**known, "actual_input_dtypes": input_dtypes, "output_dtype": output_dtype},
                "ideal_gemm_bytes": ideal_bytes,
                "ideal_arithmetic_intensity": flops / ideal_bytes if flops is not None and ideal_bytes else None,
                "gpu_event_indices": sorted(all_gpu),
                "compute_gpu_event_indices": sorted(compute_gpu) if compute_gpu is not None else None,
                "main_gpu_event_indices": sorted(main),
                "timed_gpu_event_indices": sorted(timed),
                "kernels": dict(Counter(events[i]["name"] for i in main)),
                "facts": fact,
                "kernel_gpu_ms": {name: union_us(intervals([events[i] for i in main if events[i]["name"] == name])) / 1000 for name in {events[i]["name"] for i in main}},
            }
        )
    return sorted(rows, key=lambda row: -row["gpu_ms"]), gaps


def analyze(trace, *, scope=None, occurrence=1, samples=()):
    """先选CPU范围，再沿显式关联保留GPU异步尾部；不裁到host结束时刻。"""
    events = trace["traceEvents"]
    cpu = {i for i, e in enumerate(events) if e.get("ph") == "X" and e.get("cat") in CPU_CATEGORIES and e.get("dur", 0) >= 0}
    selected_scope = None
    if occurrence < 1:
        raise ValueError("occurrence must be positive")
    if scope is not None:
        matches = sorted((i for i in cpu if events[i].get("cat") == "user_annotation" and events[i]["name"] == scope), key=lambda i: events[i]["ts"])
        if len(matches) < occurrence:
            raise ValueError(f"scope {scope!r} occurrence {occurrence} not found")
        selected_scope = matches[occurrence - 1]
        cpu = {i for i in cpu if _inside(events[i], events[selected_scope])}
    links, ambiguous = associate(events)
    selected = {i for i, related in links.items() if scope is None or related & cpu}
    calls, binding_gaps = _bound_calls(events, cpu, links, selected, samples)
    classification, work_roles = {}, {}
    for row in calls:
        for index in row["main_gpu_event_indices"]:
            classification[index] = row["family"]
        if row["family"] == "moe":
            compute = row["compute_gpu_event_indices"]
            for index in row["gpu_event_indices"]:
                if events[index]["cat"] != "kernel":
                    continue
                if compute is None:
                    work_roles[index] = "unresolved"
                elif index not in compute:
                    work_roles[index] = "auxiliary"
    # This native op receives projected Q/K/V. Its GEMM builds the SLA routing
    # map; QK/PV runs in the fused attention kernel and excludes routing FLOPs.
    routing_scopes = [events[i] for i in cpu if events[i]["name"] == "lightx2v::dynamic_sparse_sage2"]
    devices, groups = {}, {}
    for index in selected:
        event = events[index]
        category = event["cat"]
        family = classification.get(index, _kernel_family(event["name"], {events[j]["name"] for j in links[index]})) if category == "kernel" else category
        role = work_roles.get(index, "core" if category == "kernel" and family not in {"auxiliary", "unknown"} else "auxiliary")
        if family == "gemm" and any(_inside(events[j], owner) for j in links[index] for owner in routing_scopes):
            role = "auxiliary"
        key = (str(event.get("args", {}).get("device", "unknown")), category, family, event["name"], role)
        groups.setdefault(key, []).append(index)
    for device in {key[0] for key in groups}:
        indices = [i for i in selected if str(events[i].get("args", {}).get("device", "unknown")) == device]
        visible = [events[i] for i in links if str(events[i].get("args", {}).get("device", "unknown")) == device]
        chosen = [events[i] for i in indices]
        start = min(e["ts"] for e in chosen)
        end = max(e["ts"] + e["dur"] for e in chosen)
        active = union_us(intervals(chosen))
        visible_active = union_us(intervals(visible, start, end))
        categories = defaultdict(list)
        for key, values in groups.items():
            if key[0] != device:
                continue
            category = "core" if key[1] == "kernel" and key[4] == "core" else "auxiliary" if key[1] == "kernel" else key[1]
            categories[category].extend(values)
        devices[device] = {
            "span_ms": (end - start) / 1000,
            "activity_union_ms": active / 1000,
            "observed_gap_ms": max(0, end - start - visible_active) / 1000,
            "observed_gap_fraction": max(0, end - start - visible_active) / (end - start) if end > start else 0,
            "other_or_unattributed_active_ms": max(0, visible_active - active) / 1000,
            "activity": [{"category": cat, "count": len(ids), "gpu_ms": union_us(intervals([events[i] for i in ids])) / 1000} for cat, ids in categories.items()],
        }
    gpu_groups = [
        {
            "device": key[0],
            "category": key[1],
            "family": key[2],
            "work_role": key[4],
            "work_role_gap": "expert_compute_not_separated" if key[4] == "unresolved" else None,
            "classification_gap": "unrecognized_kernel_and_native" if key[2] == "unknown" else None,
            "name": key[3],
            "short_name": short_name(key[3]),
            "count": len(ids),
            "event_indices": sorted(ids),
            "gpu_ms": union_us(intervals([events[i] for i in ids])) / 1000,
        }
        for key, ids in groups.items()
    ]
    bound = {i for row in calls for i in row["main_gpu_event_indices"]}
    unbound_core = []
    for row in gpu_groups:
        ids = set(row["event_indices"]) - bound
        if ids and row["category"] == "kernel" and row["work_role"] == "core":
            unbound_core.append({**row, "event_indices": sorted(ids), "count": len(ids), "gpu_ms": union_us(intervals([events[i] for i in ids])) / 1000})
    auxiliary_groups = []
    for row in gpu_groups:
        if row["category"] != "kernel" or row["work_role"] != "core":
            name = row["short_name"] + ("（类型待识别）" if row["family"] == "unknown" else "")
            if row["work_role"] == "unresolved":
                name += "（主体归属待核）"
            elif row["category"] == "kernel" and row["family"] not in {"auxiliary", "unknown"}:
                name += "（主体阶段外）"
            native = sorted({events[j]["name"] for i in row["event_indices"] for j in links[i] if events[j].get("cat") == "cpu_op"})
            if name.startswith("at::native::") and " · " not in name and native:
                name += " · " + "/".join(native[:2]) + (" 等" if len(native) > 2 else "")
            # 只缩短显示，仍按完整kernel身份分组，不能把pow/add/cast合成一行。
            auxiliary_groups.append({"device": row["device"], "name": name, "full_name": row["name"], "native_operators": native, "count": row["count"], "gpu_ms": row["gpu_ms"]})
    self_times = cpu_self_times(events, cpu)
    return {
        "kind": "scope_trace_facts",
        "schema_version": 1,
        "scope": {"name": scope, "occurrence": occurrence, "event_index": selected_scope, "host_ms": events[selected_scope]["dur"] / 1000 if selected_scope is not None else None},
        "capture": trace.get("lightx2v_capture", {}),
        "devices": devices,
        "hardware": [{**gpu, "major": gpu.get("major", gpu.get("computeMajor")), "minor": gpu.get("minor", gpu.get("computeMinor"))} for gpu in trace.get("deviceProperties", [])],
        "core_calls": calls,
        "gpu_groups": sorted(gpu_groups, key=lambda row: -row["gpu_ms"]),
        "unbound_core_groups": sorted(unbound_core, key=lambda row: -row["gpu_ms"]),
        "auxiliary_groups": sorted(auxiliary_groups, key=lambda row: -row["gpu_ms"]),
        "gap_diagnostics": _gap_diagnostics(events, cpu, selected, links, devices),
        "cpu_self_ms": {str(i): value / 1000 for i, value in self_times.items()},
        "coverage": {
            "trace_gpu_events": len(links),
            "selected_gpu_events": len(selected),
            "unlinked_gpu_events": sum(not related for related in links.values()),
            "ambiguous_process_gpu_events": ambiguous,
            "binding_gaps": binding_gaps,
        },
    }


def apply_peaks(report, peaks=None, platform=None):
    """共用bench的显式硬件合同，不共用其报告或计时语义。"""
    from tools.benchmarks.operator_bench import _hardware_platform, _peak_family, _select_peak

    records = [{"environment": {"gpu": gpu}} for gpu in report["hardware"]]
    profile = _hardware_platform(peaks, platform, records)
    report["peak_profile"] = profile
    for row in report["core_calls"]:
        known = row["precision"]
        peak, variant, status = _select_peak(profile, _peak_family(known), known)
        row["peak_status"] = status
        row["nominal_peak_tflops"] = peak["dense_rate"] if peak and peak["unit"] == "TFLOPS" else None
        rate = row["nominal_peak_tflops"]
        row["peak_fraction"] = row["compute_tflops"] / rate if rate and row["compute_tflops"] is not None else None
        row["dense_equivalent_peak_fraction"] = row["dense_equivalent_tflops"] / rate if rate and row["dense_equivalent_tflops"] is not None else None
        bandwidth = (profile or {}).get("memory_bandwidth_gbps")
        intensity = row["ideal_arithmetic_intensity"]
        row["ideal_roofline_tflops"] = min(rate, bandwidth * intensity / 1000) if rate and bandwidth and intensity else None


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def _shape_text(shape):
    if all(key in shape for key in ("m", "n", "k")):
        return "M/N/K=" + "/".join(str(shape[key]) for key in ("m", "n", "k"))
    if "seq_q" in shape:
        return "B/H/Q/K/D=" + "/".join(str(shape.get(key, "?")) for key in ("batch", "heads", "seq_q", "seq_kv", "head_dim"))
    if "q" in shape and "k" in shape:
        if "v" in shape and shape["q"] == shape["k"] == shape["v"]:
            return "Q/K/V=" + "×".join(map(str, shape["q"]))
        return "; ".join(name.upper() + "=" + "×".join(map(str, shape[name])) for name in ("q", "k", "v") if name in shape)
    return "; ".join(f"{key}={value}" for key, value in shape.items())


def read_statistics(root):
    """Read saved facts and coverage together; never treat a capped session as full coverage."""
    samples, sessions = [], []
    for directory in sorted(Path(root).glob("call_*.stats")):
        sessions.append(json.loads((directory / "session.json").read_text()))
        for name in ("core_ops.jsonl", "workloads.jsonl"):
            samples += [json.loads(line) for line in (directory / name).read_text().splitlines() if line]
    return samples, sessions


def statistics_markdown(sessions):
    """Keep collection gaps visible; missing samples are not evidence of absent execution."""
    incomplete = [item for item in sessions if item.get("status") != "complete" or item.get("coverage_gaps")]
    if not incomplete:
        return []
    lines = ["", "**采集目录中的计算量会话存在覆盖缺口；须按下列范围和入口调用核对，未采到不等于未执行。**", "", "<details><summary>计算量采样覆盖</summary>", ""]
    for item in incomplete:
        source = item.get("provenance", {})
        lines.append(f"- rank {source.get('rank', '—')}，范围 `{source.get('scope', '未记录')}`，入口调用 {source.get('scope_call', '未记录')}：会话状态 `{item.get('status', 'unknown')}`。")
        for reason, count in item.get("coverage_gaps", {}).items():
            lines.append(f"  - {reason}：{count}次。")
    return [*lines, "", "</details>"]


def markdown(report):
    from tools.benchmarks.operator_bench import hardware_markdown

    scope = report["scope"]
    name = scope["name"] or report["capture"].get("metadata", {}).get("target", "整个采集窗口")
    lines = ["# 模型范围性能分析", "", f"**范围：{_cell(name)}**" + (f"，第{scope['occurrence']}次。" if scope["name"] else "。"), ""]
    metadata = report["capture"].get("metadata", {})
    if metadata:
        lines += ["输入/配置：" + _cell(metadata.get("description", metadata.get("input", "详见采集metadata"))) + "。", ""]
    lines += hardware_markdown({"gpus": report["hardware"], "profile": report.get("peak_profile")})
    if report.get("trace"):
        lines += ["", f"[原始 TensorBoard trace]({quote(report['trace']['path'], safe='/:')})（logdir：`{report['trace']['logdir']}`）。"]
    if report["capture"].get("workloads"):
        lines += ["", "**本次联合采集计算量：统计会增加CPU开销，动态路由读取可能同步。时间线只用于诊断；判断生产空隙应对照关闭统计的采集。**"]
    lines += statistics_markdown(report.get("statistics_sessions", []))
    lines += ["", "## 总体情况", "", "| 设备 | GPU首末跨度 ms | GPU活动并集 ms | 可见空隙 ms | 空隙占比 |", "|---|---:|---:|---:|---:|"]
    for device, facts in report["devices"].items():
        lines.append(f"| GPU {device} | {_number(facts['span_ms'])} | {_number(facts['activity_union_ms'])} | {_number(facts['observed_gap_ms'])} | {facts['observed_gap_fraction']:.1%} |")
    lines += [
        "",
        "GPU活动并集合并重叠区间，只计一次；包含kernel、拷贝和填充。GPU首末跨度从首个到最后一个关联活动，包含中间空隙。可见空隙是在该跨度内，本trace未显示任何本设备活动的时间；不证明整卡物理空闲。",
        "",
        "| 设备 / 活动 | 次数 | GPU覆盖 ms | 占跨度 |",
        "|---|---:|---:|---:|",
    ]
    labels = {"core": "核心计算", "auxiliary": "辅助计算", "gpu_memcpy": "拷贝", "gpu_memset": "填充"}
    for device, facts in report["devices"].items():
        for row in facts["activity"]:
            share = row["gpu_ms"] / facts["span_ms"] if facts["span_ms"] else 0
            lines.append(f"| GPU {device} · {labels[row['category']]} | {row['count']} | {_number(row['gpu_ms'])} | {share:.1%} |")
        if facts["other_or_unattributed_active_ms"]:
            lines += ["", f"GPU {device} 同期另有 {_number(facts['other_or_unattributed_active_ms'])} ms 的范围外/未归属活动；没有把它算成空隙。"]
    lines += [
        "",
        "分类覆盖可能重叠，各行不能相加。分类使用实际native关联和kernel名称，不能据此发布FLOPs；未识别kernel归入辅助计算并注明；拷贝/填充按trace的memcpy/memset活动分类，kernel实现的布局或数据移动仍属辅助计算。",
        "",
        "## 核心计算",
        "",
    ]
    if report["core_calls"] or report["unbound_core_groups"]:
        lines += ["| 主体 / kernel | 输入 shape | 计算量 GFLOPs | GPU覆盖 ms | 计算效率 TFLOPS | 名义峰值比例 | 计算 / 累加精度 |", "|---|---|---:|---:|---:|---:|---|"]
        for row in report["core_calls"]:
            names = sorted(row["kernels"], key=lambda name: -row["kernel_gpu_ms"][name])
            kernel = short_name(names[0]) + (f" 等{len(names)}种" if len(names) > 1 else "") if names else "主体待定位"
            rate = _number(row["compute_tflops"], 2)
            fraction = f"{row['peak_fraction']:.1%}" if row.get("peak_fraction") is not None else "—"
            if row["dense_equivalent_tflops"] is not None and row["logical_flops"] != row["dense_equivalent_flops"]:
                rate += f" / {_number(row['dense_equivalent_tflops'], 2)}*"
                fraction += f" / {row['dense_equivalent_peak_fraction']:.1%}*" if row.get("dense_equivalent_peak_fraction") is not None else " / —*"
            known = row["precision"]
            dtype = known.get("input_dtype") or "未知"
            accum = known.get("accum_dtype") or "未知"
            label = ""
            if row["time_basis"] == "compute_range_including_auxiliary":
                label = "（专家阶段含辅助）"
            elif row["time_basis"] != "compute_kernel_group":
                label = "（含辅助）"
            lines.append(
                f"| {_cell(row['family'])} · {_cell(kernel)}{label} | {_cell(_shape_text(row['shape']))} | "
                f"{_number(row['logical_flops'] / 1e9 if row['logical_flops'] is not None else None)} | "
                f"{_number(row['gpu_ms'])} | {rate} | {fraction} | {dtype} / {accum} |"
            )
        for row in report["unbound_core_groups"]:
            lines.append(f"| {_cell(row['family'])} · {_cell(row['short_name'])}（{row['count']}次） | 未绑定计算量 | — | {_number(row['gpu_ms'])} | — | — | 未知 / 未知 |")
        lines += [
            "",
            "每行对应一次真实核心调用，多个主体kernel取并集，整次计算量只计一次；不把计算量重复分摊到kernel。标“含辅助”时尚未识别主体，分母为完整算子覆盖。第二项带 * 表示按稠密等价量计算，可能超过100%；它不是物理利用率。输入/输出实际dtype、累加合同与理想字节量见参考。",
        ]
        if any(row["time_basis"] == "compute_range_including_auxiliary" for row in report["core_calls"]):
            lines += ["", "“专家阶段含辅助”只计已绑定专家阶段的活动并集，不含阶段外路由合并；未识别部分不按已知子集发布纯主体精度。"]
        if any(row["time_basis"] == "whole_operator_including_lora" for row in report["core_calls"]):
            lines += ["", "含LoRA的行保留主线性计算量，耗时包含LoRA/辅助；尚未拆分主体时效率留空。"]
    else:
        lines += ["本次没有绑定的核心计算量。正常时间线仍可定位热点；需要shape/FLOPs时用 `--workloads` 联合诊断，查看实际core/backend接线及采样上限。不会从record_shapes或kernel名称猜计算量。"]
    lines += ["", "## 非核心与数据搬运", "", "| 名称 | 设备 | 次数 | GPU覆盖 ms |", "|---|---:|---:|---:|"]
    for row in report["auxiliary_groups"][:15]:
        lines.append(f"| {_cell(row['name'])} | {row['device']} | {row['count']} | {_number(row['gpu_ms'])} |")
    lines += ["", "## 空隙定位", ""]
    for device, facts in report["gap_diagnostics"].items():
        for sync in facts["synchronizations"][:3]:
            callers = " → ".join(sync["callers"][-3:]) or "未记录上层入口"
            lines.append(f"- GPU {device}：`{sync['name']}` CPU等待 {_number(sync['cpu_ms'])} ms，与可见空隙交集 {_number(sync['gap_overlap_ms'])} ms；{_cell(callers)}。")
        if not facts["synchronizations"]:
            lines.append(f"- GPU {device}：本范围未看到同步API。先看原始时间线中的CPU准备/启动，以及预热和采集开销。")
        for gap in facts["gaps"][:3]:
            lines.append(f"- GPU {device}：空隙 {_number(gap['duration_ms'])} ms，trace时间戳 {_number(gap['start_us'], 0)}–{_number(gap['end_us'], 0)} µs。")
    lines += [
        "",
        "同步等待可能只是在等正常GPU工作；与空隙相交是定位线索，不能自动认定为原因或删除同步。",
        "",
        "<details><summary>完整kernel与核心事实参考</summary>",
        "",
        "| kernel完整名称 | 设备 | 类型 | 次数 | GPU覆盖 ms |",
        "|---|---:|---|---:|---:|",
    ]
    for row in report["gpu_groups"]:
        lines.append(f"| {_cell(demangle(row['name']))} | {row['device']} | {row['family']} | {row['count']} | {_number(row['gpu_ms'])} |")
    for row in report["core_calls"]:
        known = row["precision"]
        lines += [
            "",
            f"`{_cell(row['backend'])}`；权重 `{_cell(row.get('weight_name'))}`；实际输入 {known['actual_input_dtypes']}，输出 {known.get('output_dtype')}；计算量口径：{_cell(row['work'].get('definition', '未知'))}。",
            f"精度依据：{_cell('; '.join(known.get('evidence', [])))}。",
        ]
        if row.get("ideal_gemm_bytes"):
            lines.append(
                f"理想GEMM字节量 {row['ideal_gemm_bytes']:,.0f} B，算术强度 {_number(row['ideal_arithmetic_intensity'], 2)} FLOPs/B；只计一次读A/B写输出，不含量化/scale/cache，不是实测DRAM流量。"
            )
            if row.get("ideal_roofline_tflops") is not None:
                lines.append(f"按名义计算/显存带宽的理想roofline {_number(row['ideal_roofline_tflops'], 2)} TFLOPS；不能单凭此上限认定带宽瓶颈。")
    coverage = report["coverage"]
    lines += [
        "",
        f"关联缺口：{coverage['unlinked_gpu_events']}个GPU活动无显式CPU关联；{len(coverage['ambiguous_process_gpu_events'])}个跨进程关联歧义；{len(coverage['binding_gaps'])}条核心绑定缺口。详情在facts.json，不能按相近时间或相同shape补造归属。",
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def read_trace(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="TensorBoard的.pt.trace.json[.gz]文件")
    parser.add_argument("--output", type=Path, required=True, help="新的报告目录")
    parser.add_argument("--scope", help="精确user_annotation名；省略分析整个trace")
    parser.add_argument("--occurrence", type=int, default=1)
    parser.add_argument("--stats-root", type=Path, help="默认取trace所在采集目录；仅绑定本trace唯一标记")
    parser.add_argument("--peaks", type=Path)
    parser.add_argument("--platform")
    parser.add_argument("--library-log-dir", type=Path)
    args = parser.parse_args(argv)
    trace = read_trace(args.trace)
    samples, sessions = read_statistics(args.stats_root or args.trace.parent)
    report = analyze(trace, scope=args.scope, occurrence=args.occurrence, samples=samples)
    report["statistics_sessions"] = sessions
    report["trace"] = {"path": str(args.trace.resolve()), "logdir": str(args.trace.parent.resolve()), "sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest()}
    apply_peaks(report, json.loads(args.peaks.read_text()) if args.peaks else None, args.platform)
    if args.library_log_dir:
        from tools.profile.library_precision import read_library_logs

        report["library_logs"] = read_library_logs(args.library_log_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "facts.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
