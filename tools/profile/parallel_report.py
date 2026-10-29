"""Extract local compute, collective payload and wrapper activity from per-rank captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tools.profile.trace_report import _inside, analyze, apply_peaks, associate, intervals, read_statistics, read_trace, union_us

# PyTorch record_param_comms ScalarType names; unknown types leave byte counts unset.
_DTYPE_BYTES = {"Byte": 1, "Char": 1, "Bool": 1, "Short": 2, "Half": 2, "BFloat16": 2, "Int": 4, "Float": 4, "Long": 8, "Double": 8}


def collective_facts(events, selected):
    """Bind communication through explicit IDs; count payload once per collective call."""
    links, _ = associate(events)
    calls = defaultdict(list)
    missing = []
    communication = set()
    for index in selected:
        event = events[index]
        parents = [i for i in links[index] if events[i].get("name") == "record_param_comms"]
        communication_hint = "nccl" in event.get("name", "").lower() or bool(event.get("args", {}).get("Collective name"))
        if not parents and not communication_hint:
            continue
        communication.add(index)
        if len(parents) != 1:
            missing.append({"gpu_event_index": index, "reason": "no unique explicitly linked record_param_comms"})
            continue
        calls[(parents[0], str(event.get("args", {}).get("device", "unknown")))].append(index)
    rows = []
    for (parent, device), indices in calls.items():
        args = events[parent].get("args", {})
        size = _DTYPE_BYTES.get(args.get("dtype"))
        count = args.get("In msg nelems")
        output = args.get("Out msg nelems")
        input_bytes = count * size if size and type(count) is int and count >= 0 else None
        output_bytes = output * size if size and type(output) is int and output >= 0 else None
        duration = union_us(intervals([events[i] for i in indices])) / 1000
        rows.append(
            {
                "cpu_event_index": parent,
                "host_start_us": events[parent]["ts"],
                "gpu_event_indices": sorted(indices),
                "device": device,
                "collective": args.get("Collective name"),
                "group_ranks": args.get("Process Group Ranks"),
                "group_description": args.get("Process Group Description"),
                "group_size": args.get("Group size"),
                "dtype": args.get("dtype"),
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "gpu_ms": duration,
                "logical_input_gbps": input_bytes / (duration * 1e6) if input_bytes is not None and duration else None,
                "byte_gap": None if input_bytes is not None and output_bytes is not None else "unknown dtype or element count",
                "raw_parameters": args,
            }
        )
    return sorted(rows, key=lambda row: (row["host_start_us"], row["cpu_event_index"])), communication, missing


def wrapper_facts(events, selected, samples, communication):
    """Assign nested markers to the deepest owner; do not reallocate parent-inclusive FLOPs."""
    markers = {row["trace_marker"]: row for row in samples if row.get("kind") == "core_op_compute_stats" and row.get("trace_marker")}
    owners = {i: markers[e["name"]] for i, e in enumerate(events) if e.get("ph") == "X" and e.get("cat") == "user_annotation" and e.get("name") in markers}
    links, _ = associate(events)
    own = defaultdict(list)
    has_children = {i for i in owners if any(i != j and _inside(events[j], events[i]) for j in owners)}
    for index in selected - communication:
        candidates = [i for i in owners if any(_inside(events[cpu], events[i]) for cpu in links[index])]
        deepest = [i for i in candidates if not any(i != j and _inside(events[j], events[i]) for j in candidates)]
        if len(deepest) == 1 and deepest[0] in has_children:
            own[deepest[0]].append(index)
    rows = []
    for owner, indices in own.items():
        devices = {str(events[i].get("args", {}).get("device", "unknown")) for i in indices}
        for device in sorted(devices):
            local = [i for i in indices if str(events[i].get("args", {}).get("device", "unknown")) == device]
            rows.append(
                {
                    "backend": owners[owner]["backend"],
                    "marker": events[owner]["name"],
                    "device": device,
                    "gpu_event_indices": local,
                    "own_activity_ms": union_us(intervals([events[i] for i in local])) / 1000,
                }
            )
    return rows


def analyze_capture(directory, *, peaks=None, platform=None):
    directory = Path(directory)
    session = json.loads((directory / "session.json").read_text())
    if session["status"] != "complete" or session["captured"] != 1 or len(session["trace_files"]) != 1:
        raise ValueError("each rank needs a complete single-active-call capture")
    calls = [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()]
    active = [row for row in calls if row["phase"] == "active" and row["status"] == "complete"]
    if len(active) != 1:
        raise ValueError("one successful active scope required")
    path = directory / session["trace_files"][0]
    trace = read_trace(path)
    metadata = session["metadata"]
    if metadata != trace.get("lightx2v_capture", {}).get("metadata"):
        raise ValueError("capture session and embedded trace identity differ")
    rank = metadata.get("rank")
    if (
        type(rank) is not int
        or not isinstance(metadata.get("identity"), dict)
        or any(not isinstance(metadata["identity"].get(key), dict) or not metadata["identity"][key] for key in ("target", "input", "source"))
    ):
        raise ValueError("parallel capture requires actual rank and common global identity")
    if any(not isinstance(metadata.get(key), list) or rank not in metadata[key] for key in ("tp_ranks", "sp_ranks")):
        raise ValueError("parallel capture requires actual TP/SP group members")
    samples, statistics_sessions = read_statistics(directory)
    timeline = analyze(trace, scope=active[0]["scope"], samples=samples)
    apply_peaks(timeline, peaks, platform)
    events = trace["traceEvents"]
    selected = {i for row in timeline["gpu_groups"] for i in row["event_indices"]}
    comm, communication, missing = collective_facts(events, selected)
    core = {i for row in timeline["gpu_groups"] if row["work_role"] == "core" for i in row["event_indices"]} - communication
    other = selected - core - communication
    auxiliary_activities = []
    for row in timeline["gpu_groups"]:
        indices = sorted(set(row["event_indices"]) & other)
        if indices:
            elapsed = union_us(intervals([events[i] for i in indices])) / 1000
            auxiliary_activities.append({**row, "event_indices": indices, "count": len(indices), "gpu_ms": elapsed})
    breakdown = {}
    for device, total in timeline["devices"].items():
        local = {i for i in selected if str(events[i].get("args", {}).get("device", "unknown")) == device}

        def duration(indices):
            return union_us(intervals([events[i] for i in local & indices])) / 1000

        breakdown[device] = {
            **total,
            "core_ms": duration(core),
            "communication_ms": duration(communication),
            "other_ms": duration(other),
            "auxiliary_compute_ms": duration({i for i in other if events[i]["cat"] == "kernel"}),
            "copy_ms": duration({i for i in other if events[i]["cat"] == "gpu_memcpy"}),
            "fill_ms": duration({i for i in other if events[i]["cat"] == "gpu_memset"}),
            "core_communication_overlap_ms": max(0.0, duration(core) + duration(communication) - duration(core | communication)),
        }
    return {
        "metadata": session["metadata"],
        "trace": {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "workloads": session["workloads"],
        "record_shapes": session["record_shapes"],
        "statistics_sessions": statistics_sessions,
        "devices": breakdown,
        "collectives": comm,
        "auxiliary_activities": sorted(auxiliary_activities, key=lambda row: -row["gpu_ms"]),
        "communication_binding_gaps": missing,
        "wrapper_own_activities": wrapper_facts(events, selected, samples, communication),
        "timeline": timeline,
    }


def validate_captures(rows):
    """Validate declared identities; matching inputs/degrees do not prove a shared run."""
    if any(row["metadata"].get("identity") != rows[0]["metadata"].get("identity") for row in rows):
        raise ValueError("rank captures must declare the same global identity")
    if len({row["metadata"]["rank"] for row in rows}) != len(rows):
        raise ValueError("duplicate rank capture; use a separate report for each parallel case")
    if len({(len(row["metadata"]["tp_ranks"]), len(row["metadata"]["sp_ranks"])) for row in rows}) != 1:
        raise ValueError("rank captures must describe one uniform TP/SP configuration")
    run_ids = [row["metadata"].get("run_id") for row in rows]
    if any(run_ids) and (not all(isinstance(value, str) and value for value in run_ids) or len(set(run_ids)) != 1):
        raise ValueError("parallel capture run IDs differ or are missing on some ranks")
    return run_ids[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="per-rank Capture directories from one parallel run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peaks", type=Path)
    parser.add_argument("--platform")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must not exist")
    peaks = json.loads(args.peaks.read_text()) if args.peaks else None
    rows = [analyze_capture(path, peaks=peaks, platform=args.platform) for path in args.captures]
    run_id = validate_captures(rows)
    report = {"kind": "parallel_profile_v1", "run_id": run_id, "captures": rows}
    args.output.mkdir(parents=True)
    (args.output / "facts.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output / "facts.json")


if __name__ == "__main__":
    main()
